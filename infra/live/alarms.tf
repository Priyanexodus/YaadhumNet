###############################################################################
# Performance Alerts — CloudWatch Alarms + SNS
# Resources monitored:
#   - EC2: fl_server (moon-fl-superlink)
#   - EC2: monitoring (moon-fl-monitoring)
#   - RDS: mlflow_db (moon-fl-mlflow-db)
###############################################################################

# ---------------------------------------------------------------------------
# SNS Topic — all alarm notifications funnel here
# ---------------------------------------------------------------------------

resource "aws_sns_topic" "performance_alerts" {
  name = "moon-fl-performance-alerts"

  tags = {
    Project     = "MOON-FL"
    Environment = "production"
  }
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.performance_alerts.arn
  protocol  = "email"

  # TODO: Replace with your actual alert email address
  endpoint = "priyadharshan.27csb@licet.ac.in"
}

###############################################################################
# EC2 — FL Server (moon-fl-superlink)
###############################################################################

# --- CPU Utilization > 85% for 10 consecutive minutes ---
resource "aws_cloudwatch_metric_alarm" "fl_server_cpu_high" {
  alarm_name          = "moon-fl-server-cpu-high"
  alarm_description   = "FL Server CPU has been above 85% for 10 minutes. Consider scaling up or investigating."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 5       # 5 × 2min = 10 minutes sustained
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 120
  statistic           = "Average"
  threshold           = 85.0
  treat_missing_data  = "notBreaching"

  dimensions = {
    InstanceId = aws_instance.fl_server.id
  }

  alarm_actions = [aws_sns_topic.performance_alerts.arn]
  ok_actions    = [aws_sns_topic.performance_alerts.arn]

  tags = { Project = "MOON-FL" }
}

# --- Status Check Failed — instance is unreachable / crashed ---
resource "aws_cloudwatch_metric_alarm" "fl_server_status_check" {
  alarm_name          = "moon-fl-server-status-check-failed"
  alarm_description   = "FL Server status check failed. The instance may be down or unresponsive."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Maximum"
  threshold           = 0
  treat_missing_data  = "breaching"

  dimensions = {
    InstanceId = aws_instance.fl_server.id
  }

  alarm_actions = [aws_sns_topic.performance_alerts.arn]
  ok_actions    = [aws_sns_topic.performance_alerts.arn]

  tags = { Project = "MOON-FL" }
}

# --- Disk Read Ops spike (may indicate heavy disk I/O bottleneck) ---
resource "aws_cloudwatch_metric_alarm" "fl_server_disk_read_ops" {
  alarm_name          = "moon-fl-server-disk-read-ops-high"
  alarm_description   = "FL Server disk read ops are unusually high. May indicate model I/O bottleneck."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "DiskReadOps"
  namespace           = "AWS/EC2"
  period              = 300 # 5-minute window
  statistic           = "Average"
  threshold           = 5000
  treat_missing_data  = "notBreaching"

  dimensions = {
    InstanceId = aws_instance.fl_server.id
  }

  alarm_actions = [aws_sns_topic.performance_alerts.arn]

  tags = { Project = "MOON-FL" }
}

###############################################################################
# EC2 — Monitoring Server (Prometheus + Grafana)
###############################################################################

# --- CPU Utilization > 80% for 10 minutes ---
resource "aws_cloudwatch_metric_alarm" "monitoring_cpu_high" {
  alarm_name          = "moon-fl-monitoring-cpu-high"
  alarm_description   = "Monitoring server CPU above 80% for 10 minutes. Grafana/Prometheus may be under pressure."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 5
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 120
  statistic           = "Average"
  threshold           = 80.0
  treat_missing_data  = "notBreaching"

  dimensions = {
    InstanceId = aws_instance.monitoring.id
  }

  alarm_actions = [aws_sns_topic.performance_alerts.arn]
  ok_actions    = [aws_sns_topic.performance_alerts.arn]

  tags = { Project = "MOON-FL" }
}

# --- Status Check failed on monitoring server ---
resource "aws_cloudwatch_metric_alarm" "monitoring_status_check" {
  alarm_name          = "moon-fl-monitoring-status-check-failed"
  alarm_description   = "Monitoring server status check failed. Grafana/Prometheus may be unreachable."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Maximum"
  threshold           = 0
  treat_missing_data  = "breaching"

  dimensions = {
    InstanceId = aws_instance.monitoring.id
  }

  alarm_actions = [aws_sns_topic.performance_alerts.arn]
  ok_actions    = [aws_sns_topic.performance_alerts.arn]

  tags = { Project = "MOON-FL" }
}

###############################################################################
# RDS — MLflow PostgreSQL 16 (moon-fl-mlflow-db)
###############################################################################

# --- CPU Utilization > 80% for 10 minutes ---
resource "aws_cloudwatch_metric_alarm" "rds_cpu_high" {
  alarm_name          = "moon-fl-rds-cpu-high"
  alarm_description   = "MLflow RDS CPU above 80% for 10 minutes. MLflow experiment logging may slow down."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 5
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 120
  statistic           = "Average"
  threshold           = 80.0
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.mlflow_db.identifier
  }

  alarm_actions = [aws_sns_topic.performance_alerts.arn]
  ok_actions    = [aws_sns_topic.performance_alerts.arn]

  tags = { Project = "MOON-FL" }
}


# --- Free Storage Space < 2 GB (10% of 20 GB allocated) ---
resource "aws_cloudwatch_metric_alarm" "rds_free_storage_low" {
  alarm_name          = "moon-fl-rds-low-storage"
  alarm_description   = "MLflow RDS free storage below 2 GB. Increase allocated_storage before the DB crashes."
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 2147483648 # 2 GB in bytes
  treat_missing_data  = "breaching"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.mlflow_db.identifier
  }

  alarm_actions = [aws_sns_topic.performance_alerts.arn]
  ok_actions    = [aws_sns_topic.performance_alerts.arn]

  tags = { Project = "MOON-FL" }
}

# --- Database Connections > 80 (db.t3.micro max is ~87) ---
resource "aws_cloudwatch_metric_alarm" "rds_connection_count_high" {
  alarm_name          = "moon-fl-rds-connections-high"
  alarm_description   = "MLflow RDS connection count exceeds 80. Approaching db.t3.micro limit of ~87."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = 60
  statistic           = "Average"
  threshold           = 80
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.mlflow_db.identifier
  }

  alarm_actions = [aws_sns_topic.performance_alerts.arn]
  ok_actions    = [aws_sns_topic.performance_alerts.arn]

  tags = { Project = "MOON-FL" }
}

# --- Read Latency > 200ms ---
resource "aws_cloudwatch_metric_alarm" "rds_read_latency_high" {
  alarm_name          = "moon-fl-rds-read-latency-high"
  alarm_description   = "MLflow RDS read latency above 200ms. Queries may be degraded."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "ReadLatency"
  namespace           = "AWS/RDS"
  period              = 60
  statistic           = "Average"
  threshold           = 0.2 # 200ms in seconds
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.mlflow_db.identifier
  }

  alarm_actions = [aws_sns_topic.performance_alerts.arn]

  tags = { Project = "MOON-FL" }
}

# --- Freeable Memory < 100 MB ---
resource "aws_cloudwatch_metric_alarm" "rds_low_memory" {
  alarm_name          = "moon-fl-rds-low-memory"
  alarm_description   = "MLflow RDS freeable memory below 100 MB. DB is under significant memory pressure."
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "FreeableMemory"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 104857600 # 100 MB in bytes
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.mlflow_db.identifier
  }

  alarm_actions = [aws_sns_topic.performance_alerts.arn]
  ok_actions    = [aws_sns_topic.performance_alerts.arn]

  tags = { Project = "MOON-FL" }
}
