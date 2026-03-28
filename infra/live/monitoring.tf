resource "aws_instance" "monitoring" {
  ami                    = var.ami_id
  instance_type          = var.monitoring_instance_type
  key_name               = var.key_pair_name
  subnet_id              = aws_subnet.public_a.id 
  vpc_security_group_ids = [aws_security_group.monitoring_sg.id]

  user_data = templatefile("${path.module}/user_data_monitoring.sh", {
    fl_server_ip     = aws_eip.fl_server_ip.public_ip
    grafana_password = var.grafana_password
    aws_region       = var.aws_region
  })

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
    encrypted   = true
  }

  # FL server must be running before monitoring bootstraps — Prometheus needs :9100
  depends_on = [aws_instance.fl_server]

  tags = {
    Name        = "moon-fl-monitoring"
    Environment = "production"
    Project     = "MOON-FL"
  }

  lifecycle {
    ignore_changes = [user_data]
  }
}

resource "aws_eip" "monitoring_ip" {
  instance = aws_instance.monitoring.id
  domain   = "vpc"

  tags = {
    Name    = "moon-fl-monitoring-eip"
    Project = "MOON-FL"
  }
}