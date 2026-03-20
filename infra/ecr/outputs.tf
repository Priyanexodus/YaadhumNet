output "ecr_server_url" {
  value       = aws_ecr_repository.moon_fl_server.repository_url
  description = "docker tag <image> <url>:latest && docker push <url>:latest"
}

output "ecr_client_url" {
  value       = aws_ecr_repository.moon_fl_client.repository_url
}

output "ecr_mlflow_url" {
  value       = aws_ecr_repository.moon_fl_mlflow.repository_url
}

output "ecr_registry" {
  value       = "${split("/", aws_ecr_repository.moon_fl_server.repository_url)[0]}"
  description = "Use for: aws ecr get-login-password | docker login --username AWS --password-stdin <registry>"
}
