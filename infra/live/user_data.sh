#!/bin/bash
set -eo pipefail
exec > /var/log/fl-bootstrap.log 2>&1

echo "=== MOON-FL Bootstrap Start: $(date) ==="

###############################################################################
# 1. Docker (official Docker apt repo)
###############################################################################
apt-get update -y
apt-get install -y ca-certificates curl gnupg unzip

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

ARCH=$(dpkg --print-architecture)
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
echo "deb [arch=$ARCH signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $CODENAME stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable --now docker
usermod -aG docker ubuntu

###############################################################################
# 2. AWS CLI v2
###############################################################################
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
rm -rf /tmp/awscliv2.zip /tmp/aws

###############################################################################
# 3. Fetch secrets from AWS Secrets Manager
###############################################################################
AWS_CLI=/usr/local/bin/aws
REGION=${aws_region}

fetch_secret() {
  $AWS_CLI secretsmanager get-secret-value \
    --region "$REGION" \
    --secret-id "$1" \
    --query SecretString \
    --output text
}

# TLS certificates
mkdir -p /home/ubuntu/certs
chmod 700 /home/ubuntu/certs

fetch_secret "moon-fl/certs/ca-crt"     > /home/ubuntu/certs/ca.crt
fetch_secret "moon-fl/certs/server-crt" > /home/ubuntu/certs/server.crt
fetch_secret "moon-fl/certs/server-key" > /home/ubuntu/certs/server.key

chmod 644 /home/ubuntu/certs/ca.crt
chmod 644 /home/ubuntu/certs/server.crt
chmod 600 /home/ubuntu/certs/server.key
chown -R root:root /home/ubuntu/certs

# Database password
DB_PASSWORD=$(fetch_secret "moon-fl/db/password")

###############################################################################
# 4. Write .env
###############################################################################
cat > /home/ubuntu/.env << ENVEOF
DB_USER=${db_user}
DB_PASSWORD=$DB_PASSWORD
DB_ENDPOINT=${db_endpoint}
DB_NAME=${db_name}
S3_BUCKET_NAME=${s3_bucket}
GIT_PYTHON_REFRESH=quiet
ENVEOF

chmod 600 /home/ubuntu/.env
chown root:root /home/ubuntu/.env

###############################################################################
# 5. Write docker-compose.prod.yml
#    superlink — no command override, entrypoint.sh handles TLS flags
###############################################################################
cat > /home/ubuntu/docker-compose.prod.yml << 'COMPOSEEOF'
services:
  mlflow:
    image: __ECR_REGISTRY__/moon-fl-mlflow:__ECR_TAG__
    ports:
      - "5000:5000"
    environment:
      - DB_USER=__DB_USER__
      - DB_PASSWORD=__DB_PASSWORD__
      - DB_ENDPOINT=__DB_ENDPOINT__
      - DB_NAME=__DB_NAME__
      - S3_BUCKET_NAME=__S3_BUCKET__
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://__DB_USER__:__DB_PASSWORD__@__DB_ENDPOINT__:5432/__DB_NAME__
      --default-artifact-root s3://__S3_BUCKET__/mlflow
    logging:
      driver: awslogs
      options:
        awslogs-group: /moon-fl/mlflow
        awslogs-region: __AWS_REGION__
        awslogs-stream: mlflow
    restart: unless-stopped

  superlink:
    image: __ECR_REGISTRY__/moon-fl-server:__ECR_TAG__
    ports:
      - "9091:9091"
      - "9092:9092"
      - "9093:9093"
    volumes:
      - /home/ubuntu/certs:/certs:ro
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
    logging:
      driver: awslogs
      options:
        awslogs-group: /moon-fl/superlink
        awslogs-region: __AWS_REGION__
        awslogs-stream: superlink
    restart: unless-stopped
  
  node-exporter:
    image: prom/node-exporter:v1.8.1
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    restart: unless-stopped
COMPOSEEOF

# Substitute all placeholders
sed -i \
  -e "s|__ECR_REGISTRY__|${aws_account_id}.dkr.ecr.${aws_region}.amazonaws.com|g" \
  -e "s|__ECR_TAG__|${ecr_image_tag}|g" \
  -e "s|__AWS_REGION__|${aws_region}|g" \
  -e "s|__DB_USER__|${db_user}|g" \
  -e "s|__DB_PASSWORD__|$DB_PASSWORD|g" \
  -e "s|__DB_ENDPOINT__|${db_endpoint}|g" \
  -e "s|__DB_NAME__|${db_name}|g" \
  -e "s|__S3_BUCKET__|${s3_bucket}|g" \
  /home/ubuntu/docker-compose.prod.yml

chown ubuntu:ubuntu /home/ubuntu/docker-compose.prod.yml

###############################################################################
# 6. ECR login + pull images
###############################################################################
ECR_REGISTRY="${aws_account_id}.dkr.ecr.${aws_region}.amazonaws.com"

/usr/local/bin/aws ecr get-login-password --region ${aws_region} | \
  docker login --username AWS --password-stdin "$ECR_REGISTRY"

docker pull "$ECR_REGISTRY/moon-fl-server:${ecr_image_tag}"
docker pull "$ECR_REGISTRY/moon-fl-mlflow:${ecr_image_tag}"

###############################################################################
# 7. Start services
###############################################################################
cd /home/ubuntu
docker compose -f /home/ubuntu/docker-compose.prod.yml up -d

echo "=== MOON-FL Bootstrap Done: $(date) ==="