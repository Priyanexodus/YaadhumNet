###############################################################################
# Secrets Manager — TLS Certificates
#
# ignore_changes = [secret_string] on every version resource means:
#   - Terraform writes the value ONCE on first apply
#   - Subsequent applies never overwrite it
#   - Cert rotation requires manual: aws secretsmanager put-secret-value
###############################################################################

# ── CA Private Key ────────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "ca_key" {
  name        = "moon-fl/certs/ca-key"
  description = "YaadhumNet CA private key — never rotated by Terraform"

  tags = { Project = "MOON-FL" }
}

resource "aws_secretsmanager_secret_version" "ca_key" {
  secret_id     = aws_secretsmanager_secret.ca_key.id
  secret_string = tls_private_key.ca.private_key_pem

  lifecycle {
    ignore_changes = [secret_string]  # written once, never overwritten
  }
}

# ── CA Certificate ────────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "ca_crt" {
  name        = "moon-fl/certs/ca-crt"
  description = "YaadhumNet CA certificate — distributed to SuperNode clients"

  tags = { Project = "MOON-FL" }
}

resource "aws_secretsmanager_secret_version" "ca_crt" {
  secret_id     = aws_secretsmanager_secret.ca_crt.id
  secret_string = tls_self_signed_cert.ca.cert_pem

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# ── Server Private Key ────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "server_key" {
  name        = "moon-fl/certs/server-key"
  description = "SuperLink server private key"

  tags = { Project = "MOON-FL" }
}

resource "aws_secretsmanager_secret_version" "server_key" {
  secret_id     = aws_secretsmanager_secret.server_key.id
  secret_string = tls_private_key.server.private_key_pem

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# ── Server Certificate ────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "server_crt" {
  name        = "moon-fl/certs/server-crt"
  description = "SuperLink server certificate signed by YaadhumNet CA"

  tags = { Project = "MOON-FL" }
}

resource "aws_secretsmanager_secret_version" "server_crt" {
  secret_id     = aws_secretsmanager_secret.server_crt.id
  secret_string = tls_locally_signed_cert.server.cert_pem

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# ── Database Password ─────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "db_password" {
  name        = "moon-fl/db/password"
  description = "MLflow RDS PostgreSQL password"

  tags = { Project = "MOON-FL" }
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = var.db_password

  lifecycle {
    ignore_changes = [secret_string]
  }
}
