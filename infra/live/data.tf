###############################################################################
# ECR repositories are managed in infra/ecr/ — referenced here as data sources
###############################################################################

data "aws_ecr_repository" "moon_fl_server" {
  name = "moon-fl-server"
}

data "aws_ecr_repository" "moon_fl_client" {
  name = "moon-fl-client"
}

data "aws_ecr_repository" "moon_fl_mlflow" {
  name = "moon-fl-mlflow"
}
