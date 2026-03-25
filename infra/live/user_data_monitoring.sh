#!/bin/bash
set -eo pipefail
exec > /var/log/monitoring-bootstrap.log 2>&1

echo "=== Monitoring Bootstrap Start: $(date) ==="

###############################################################################
# 1. Docker
###############################################################################
apt-get update -y
apt-get install -y ca-certificates curl gnupg

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

ARCH=$(dpkg --print-architecture)
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
echo "deb [arch=$ARCH signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $CODENAME stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable --now docker

###############################################################################
# 2. Directories
###############################################################################
mkdir -p /home/ubuntu/monitoring/grafana/provisioning/datasources
chown -R ubuntu:ubuntu /home/ubuntu/monitoring

###############################################################################
# 3. prometheus.yml — fl_server_ip injected by Terraform templatefile
###############################################################################
cat > /home/ubuntu/monitoring/prometheus.yml << EOF
global:
  scrape_interval:     15s
  evaluation_interval: 15s

scrape_configs:

  # System metrics — CPU, memory, disk, network
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['${fl_server_ip}:9100']
        labels:
          instance: 'fl-server'

  # SuperLink connectivity — exec API + fleet API
  - job_name: 'blackbox_superlink'
    metrics_path: /probe
    params:
      module: [tcp_connect]
    static_configs:
      - targets:
          - ${fl_server_ip}:9091
          - ${fl_server_ip}:9092
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox:9115
EOF

###############################################################################
# 4. blackbox.yml
###############################################################################
cat > /home/ubuntu/monitoring/blackbox.yml << 'EOF'
modules:
  tcp_connect:
    prober: tcp
    timeout: 5s
EOF

###############################################################################
# 5. Grafana datasource provisioning — fl_server_ip injected
###############################################################################
cat > /home/ubuntu/monitoring/grafana/provisioning/datasources/datasources.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: MLflow
    type: yesoreyeram-infinity-datasource
    access: proxy
    url: http://${fl_server_ip}:5000
    editable: true
    jsonData:
      tlsSkipVerify: true
EOF

###############################################################################
# 6. .env — grafana_password injected by Terraform templatefile
###############################################################################
cat > /home/ubuntu/monitoring/.env << ENVEOF
GRAFANA_PASSWORD=${grafana_password}
ENVEOF
chmod 600 /home/ubuntu/monitoring/.env
chown root:root /home/ubuntu/monitoring/.env

###############################################################################
# 7. docker-compose.yml
###############################################################################
cat > /home/ubuntu/monitoring/docker-compose.yml << 'EOF'
services:
  prometheus:
    image: prom/prometheus:v2.52.0
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9090:9090"
    restart: unless-stopped

  blackbox:
    image: prom/blackbox-exporter:v0.25.0
    volumes:
      - ./blackbox.yml:/config/blackbox.yml:ro
    command:
      - '--config.file=/config/blackbox.yml'
    ports:
      - "9115:9115"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.4.2
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=$${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=yesoreyeram-infinity-datasource
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
EOF

###############################################################################
# 8. Start stack
###############################################################################
cd /home/ubuntu/monitoring
docker compose up -d

echo "=== Monitoring Bootstrap Done: $(date) ==="