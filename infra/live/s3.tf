# Import the bucket bootstrap/ created — live/ references it for IAM + lifecycle only.
# Versioning, encryption, and public access block are owned by bootstrap/ — not redeclared here.
import {
  to = aws_s3_bucket.fl_checkpoints
  id = "moon-fl-checkpoints"
}

resource "aws_s3_bucket" "fl_checkpoints" {
  bucket = var.s3_bucket_name

  tags = {
    Name        = var.s3_bucket_name
    Project     = "MOON-FL"
    Environment = "production"
    ManagedBy   = "terraform-live"
  }
}

# Lifecycle rules only — MLflow artifact expiry is live/'s concern, not bootstrap/'s
resource "aws_s3_bucket_lifecycle_configuration" "fl_checkpoints" {
  bucket = aws_s3_bucket.fl_checkpoints.id

  rule {
    id     = "expire-mlflow-artifacts"
    status = "Enabled"

    filter {
      prefix = "mlflow/"
    }

    expiration {
      days = 30
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }

  rule {
    id     = "keep-terraform-state"
    status = "Enabled"

    filter {
      prefix = "terraform-state/"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}
