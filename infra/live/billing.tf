###############################################################################
# Billing Alerts — AWS Budgets + Cost Anomaly Detection
# All alerts are sent to the same email as performance alerts.
###############################################################################

# NOTE: AWS Budgets alarms require CloudWatch billing alerts to be enabled.
# Enable at: AWS Console → Billing → Billing Preferences → "Receive Billing Alerts"
# Also note: aws_budgets_budget uses us-east-1 internally regardless of your region.

locals {
  alert_email = "priyadharshan.27csb@licet.ac.in"

  # Adjust this to your expected monthly AWS spend in USD
  monthly_budget_usd = "20.0"
}

# ---------------------------------------------------------------------------
# SNS Topic — billing-specific alerts (separate from performance alerts)
# ---------------------------------------------------------------------------

resource "aws_sns_topic" "billing_alerts" {
  name = "moon-fl-billing-alerts"

  tags = {
    Project     = "MOON-FL"
    Environment = "production"
  }
}

resource "aws_sns_topic_subscription" "billing_email" {
  topic_arn = aws_sns_topic.billing_alerts.arn
  protocol  = "email"
  endpoint  = local.alert_email
}

###############################################################################
# AWS Budgets — Monthly Cost Budget with tiered alerts
###############################################################################

resource "aws_budgets_budget" "monthly_total" {
  name         = "moon-fl-monthly-budget"
  budget_type  = "COST"
  limit_amount = local.monthly_budget_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Alert at 50% of budget (early warning)
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [local.alert_email]
  }

  # Alert at 80% of budget (action required soon)
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [local.alert_email]
  }

  # Alert at 100% of budget (threshold exceeded)
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [local.alert_email]
  }

  # Alert when forecasted spend is on track to exceed 100% of budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [local.alert_email]
  }
}

###############################################################################
# Per-service budgets — catch rogue individual services early
###############################################################################

# EC2 budget — covers both fl_server and monitoring instances
resource "aws_budgets_budget" "ec2_monthly" {
  name         = "moon-fl-ec2-monthly"
  budget_type  = "COST"
  limit_amount = "12.0" # t3.small + t3.micro ~$10-11/month
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "Service"
    values = ["Amazon Elastic Compute Cloud - Compute"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 90
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [local.alert_email]
  }
}

# RDS budget — covers MLflow PostgreSQL instance
resource "aws_budgets_budget" "rds_monthly" {
  name         = "moon-fl-rds-monthly"
  budget_type  = "COST"
  limit_amount = "5.0" # db.t3.micro ~$15/month (free tier eligible)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "Service"
    values = ["Amazon Relational Database Service"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 90
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [local.alert_email]
  }
}

# S3 budget — covers MLflow artifact storage
resource "aws_budgets_budget" "s3_monthly" {
  name         = "moon-fl-s3-monthly"
  budget_type  = "COST"
  limit_amount = "3.0"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "Service"
    values = ["Amazon Simple Storage Service"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 90
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [local.alert_email]
  }
}

###############################################################################
# Cost Anomaly Detection — ML-based unusual spend detection
###############################################################################

# Monitor all AWS services for unexpected spend patterns
resource "aws_ce_anomaly_monitor" "all_services" {
  name              = "moon-fl-all-services-monitor"
  monitor_type      = "DIMENSIONAL"
  monitor_dimension = "SERVICE"

  tags = { Project = "MOON-FL" }
}

# Alert if an anomaly impact exceeds $5 in total
resource "aws_ce_anomaly_subscription" "daily_digest" {
  name      = "moon-fl-anomaly-subscription"
  frequency = "DAILY"

  monitor_arn_list = [aws_ce_anomaly_monitor.all_services.arn]

  subscriber {
    type    = "EMAIL"
    address = local.alert_email
  }

  threshold_expression {
    dimension {
      key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
      values        = ["5"]  # Alert when anomaly is > $5
      match_options = ["GREATER_THAN_OR_EQUAL"]
    }
  }

  tags = { Project = "MOON-FL" }
}
