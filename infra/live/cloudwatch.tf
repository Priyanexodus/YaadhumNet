resource "aws_cloudwatch_log_group" "superlink" {
  name              = "/moon-fl/superlink"
  retention_in_days = 14

  tags = { Project = "MOON-FL" }
}

resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/moon-fl/mlflow"
  retention_in_days = 14

  tags = { Project = "MOON-FL" }
}
