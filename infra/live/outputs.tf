output "fl_server_public_ip" {
  value       = aws_eip.fl_server_ip.public_ip
  description = "SuperLink IP — Fleet API: <ip>:9092 / Exec API: <ip>:9091"
}

output "mlflow_url" {
  value       = "http://${aws_eip.fl_server_ip.public_ip}:5000"
  description = "MLflow tracking UI"
}

output "rds_endpoint" {
  value       = aws_db_instance.mlflow_db.address
  description = "PostgreSQL endpoint for MLflow backend-store-uri"
}

output "ecr_server_url" {
  value = data.aws_ecr_repository.moon_fl_server.repository_url
}

output "ecr_client_url" {
  value = data.aws_ecr_repository.moon_fl_client.repository_url
}

output "ecr_mlflow_url" {
  value = data.aws_ecr_repository.moon_fl_mlflow.repository_url
}

output "s3_bucket" {
  value = aws_s3_bucket.fl_checkpoints.bucket
}
