###############################################################################
# SSH Key Pair — EC2 access
###############################################################################

resource "tls_private_key" "fl_server" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "fl_server" {
  key_name   = var.key_pair_name
  public_key = tls_private_key.fl_server.public_key_openssh

  tags = {
    Name    = var.key_pair_name
    Project = "MOON-FL"
  }
}

resource "aws_s3_object" "fl_server_private_key" {
  bucket  = aws_s3_bucket.fl_checkpoints.bucket
  key     = "keys/${var.key_pair_name}.pem"
  content = tls_private_key.fl_server.private_key_pem

  server_side_encryption = "AES256"
  tags = { Project = "MOON-FL" }
}

###############################################################################
# TLS — CA Certificate Authority
###############################################################################

resource "tls_private_key" "ca" {
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "tls_self_signed_cert" "ca" {
  private_key_pem = tls_private_key.ca.private_key_pem

  subject {
    common_name  = "YaadhumNet CA"
    organization = "YaadhumNet"
  }

  validity_period_hours = 8760  # 1 year

  is_ca_certificate = true

  allowed_uses = [
    "cert_signing",
    "key_encipherment",
    "digital_signature",
  ]
}

###############################################################################
# TLS — Server Certificate (signed by CA)
###############################################################################

resource "tls_private_key" "server" {
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "tls_cert_request" "server" {
  private_key_pem = tls_private_key.server.private_key_pem

  subject {
    common_name  = "moon-fl-superlink"
    organization = "YaadhumNet"
  }

  # SAN — SuperNode verifies server using this IP
  ip_addresses = [aws_eip.fl_server_ip.public_ip]
}

resource "tls_locally_signed_cert" "server" {
  cert_request_pem   = tls_cert_request.server.cert_request_pem
  ca_private_key_pem = tls_private_key.ca.private_key_pem
  ca_cert_pem        = tls_self_signed_cert.ca.cert_pem

  validity_period_hours = 8760  # 1 year

  allowed_uses = [
    "key_encipherment",
    "digital_signature",
    "server_auth",
  ]
}

###############################################################################
# After apply, download SSH key:
#   aws s3 cp s3://moon-fl-checkpoints/keys/moon-fl-key.pem ~/.ssh/moon-fl-key.pem
#   chmod 400 ~/.ssh/moon-fl-key.pem
#
# Certs are stored in Secrets Manager — see secrets.tf
###############################################################################
