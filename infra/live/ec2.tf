resource "aws_instance" "fl_server" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  subnet_id              = aws_subnet.public_a.id 
  iam_instance_profile   = aws_iam_instance_profile.fl_server_profile.name
  vpc_security_group_ids = [aws_security_group.fl_server_sg.id]

  user_data = templatefile("${path.module}/user_data.sh", {
    aws_region     = var.aws_region
    aws_account_id = var.aws_account_id
    s3_bucket      = var.s3_bucket_name
    ecr_image_tag  = var.ecr_image_tag
    db_user        = var.db_user
    db_password    = var.db_password
    db_endpoint    = aws_db_instance.mlflow_db.address
    db_name        = var.db_name
  })

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
    encrypted   = true
  }

  # EC2 must wait for RDS to be available — MLflow needs the DB endpoint at boot
  depends_on = [aws_db_instance.mlflow_db]

  tags = {
    Name        = "moon-fl-superlink"
    Environment = "production"
    Project     = "MOON-FL"
  }

  lifecycle {
    # Prevents replacement when user_data changes after initial deploy.
    # To re-run bootstrap: terraform taint aws_instance.fl_server → terraform apply
    ignore_changes = [user_data]
  }
}

resource "aws_eip" "fl_server_ip" {
  instance = aws_instance.fl_server.id
  domain   = "vpc"

  tags = {
    Name    = "moon-fl-superlink-eip"
    Project = "MOON-FL"
  }
}
