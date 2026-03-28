###############################################################################
# RDS PostgreSQL 16 — MLflow backend store
###############################################################################

resource "aws_db_instance" "mlflow_db" {
  identifier        = var.db_instance_identifier
  engine            = "postgres"
  engine_version    = "16"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  storage_type      = "gp2"

  db_name  = var.db_name
  username = var.db_user
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds_sg.id]

  multi_az            = false
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name 
  publicly_accessible = false
  storage_encrypted   = true
  skip_final_snapshot = true
  deletion_protection = false

  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  tags = {
    Name        = "moon-fl-mlflow-db"
    Environment = "production"
    Project     = "MOON-FL"
  }
}
