###############################################################################
# EC2 Instance Role
###############################################################################

resource "aws_iam_role" "fl_server_role" {
  name = "moon-fl-server-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = { Project = "MOON-FL" }
}

###############################################################################
# Inline Policies
###############################################################################

resource "aws_iam_role_policy" "s3_access" {
  name = "moon-fl-s3-access"
  role = aws_iam_role.fl_server_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.fl_checkpoints.arn,
        "${aws_s3_bucket.fl_checkpoints.arn}/*"
      ]
    }]
  })
}

resource "aws_iam_role_policy" "ecr_pull" {
  name = "moon-fl-ecr-pull"
  role = aws_iam_role.fl_server_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = [
          data.aws_ecr_repository.moon_fl_server.arn,
          data.aws_ecr_repository.moon_fl_client.arn,
          data.aws_ecr_repository.moon_fl_mlflow.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "cloudwatch_logs" {
  name = "moon-fl-cloudwatch-logs"
  role = aws_iam_role.fl_server_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ]
      Resource = "arn:aws:logs:${var.aws_region}:${var.aws_account_id}:log-group:/moon-fl/*"
    }]
  })
}

###############################################################################
# Instance Profile
###############################################################################

resource "aws_iam_instance_profile" "fl_server_profile" {
  name = "moon-fl-server-profile"
  role = aws_iam_role.fl_server_role.name
}
