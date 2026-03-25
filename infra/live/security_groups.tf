###############################################################################
# EC2 Security Group — FL SuperLink
# NOTE: All ports open to 0.0.0.0/0 for development/debugging.
#       Restrict SSH, 9093, and 5000 to your IP before production.
###############################################################################

resource "aws_security_group" "fl_server_sg" {
  name        = "moon-fl-server-sg"
  description = "MOON FL SuperLink security group"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Flower Exec API (SuperNode to SuperLink)"
    from_port   = 9091
    to_port     = 9091
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Flower Fleet API (SuperNode to SuperLink)"
    from_port   = 9092
    to_port     = 9092
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Flower REST Admin API (flwr run)"
    from_port   = 9093
    to_port     = 9093
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "MLflow tracking UI"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description     = "node_exporter — Prometheus scrape from monitoring EC2 only"
    from_port       = 9100
    to_port         = 9100
    protocol        = "tcp"
    security_groups = [aws_security_group.monitoring_sg.id]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "moon-fl-server-sg"
    Project = "MOON-FL"
  }
}

###############################################################################
# RDS Security Group — MLflow PostgreSQL
###############################################################################

resource "aws_security_group" "rds_sg" {
  name        = "moon-fl-rds-sg"
  description = "Allow PostgreSQL access from FL server only"

  ingress {
    description     = "PostgreSQL from FL server"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.fl_server_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "moon-fl-rds-sg"
    Project = "MOON-FL"
  }
}

###############################################################################
# Monitoring Security Group — Prometheus + Grafana
###############################################################################

resource "aws_security_group" "monitoring_sg" {
  name        = "moon-fl-monitoring-sg"
  description = "MOON FL monitoring - Prometheus + Grafana"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Grafana UI"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Prometheus UI"
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "moon-fl-monitoring-sg"
    Project = "MOON-FL"
  }
}
