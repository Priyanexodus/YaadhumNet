###############################################################################
# ECR Repositories
###############################################################################

resource "aws_ecr_repository" "moon_fl_server" {
  name                 = "moon-fl-server"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = { Project = "MOON-FL" }
}

resource "aws_ecr_repository" "moon_fl_client" {
  name                 = "moon-fl-client"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = { Project = "MOON-FL" }
}

resource "aws_ecr_repository" "moon_fl_mlflow" {
  name                 = "moon-fl-mlflow"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = { Project = "MOON-FL" }
}

###############################################################################
# Lifecycle Policy — keep last 5 images per repo
###############################################################################

locals {
  ecr_lifecycle_policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_ecr_lifecycle_policy" "moon_fl_server" {
  repository = aws_ecr_repository.moon_fl_server.name
  policy     = local.ecr_lifecycle_policy
}

resource "aws_ecr_lifecycle_policy" "moon_fl_client" {
  repository = aws_ecr_repository.moon_fl_client.name
  policy     = local.ecr_lifecycle_policy
}

resource "aws_ecr_lifecycle_policy" "moon_fl_mlflow" {
  repository = aws_ecr_repository.moon_fl_mlflow.name
  policy     = local.ecr_lifecycle_policy
}
