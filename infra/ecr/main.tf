terraform {
  required_version = ">= 1.7"

  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }

  backend "s3" {
    bucket = "moon-fl-checkpoints"
    key    = "terraform-state/ecr/terraform.tfstate"
    region = "ap-south-1"
  }
}

provider "aws" {
  region = var.aws_region
}
