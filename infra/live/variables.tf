variable "aws_region" {
  default = "ap-south-1"
}

variable "aws_account_id" {
  description = "AWS account ID — used for ECR image URLs"
  default     = "097155545354"
}

variable "instance_type" {
  description = "EC2 instance type. t3.small confirmed stable for ViT + 2 clients."
  default     = "t3.small"
}

variable "key_pair_name" {
  description = "EC2 SSH key pair name (must already exist in ap-south-1)"
}

variable "my_ip_cidr" {
  description = "Your IP for SSH/MLflow access e.g. 1.2.3.4/32"
}

variable "ami_id" {
  description = "Ubuntu 24.04 LTS in ap-south-1"
  default     = "ami-0f58b397bc5c1f2e8"
}

variable "s3_bucket_name" {
  description = "Existing S3 bucket created by bootstrap — adopted here for MLflow artifacts"
  default     = "moon-fl-checkpoints"
}

variable "db_instance_identifier" {
  default = "moon-fl-mlflow-db"
}

variable "db_name" {
  default = "mlflow"
}

variable "db_user" {
  default = "mlflow"
}

variable "db_password" {
  description = "RDS master password — store in tfvars, never commit"
  sensitive   = true
}

variable "ecr_image_tag" {
  description = "Docker image tag to pull on EC2 bootstrap"
  default     = "latest"
}
