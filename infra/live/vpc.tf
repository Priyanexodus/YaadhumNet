###############################################################################
# VPC — explicit network for all MOON-FL resources
###############################################################################

resource "aws_vpc" "moon_fl" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "moon-fl-vpc", Project = "MOON-FL" }
}

resource "aws_internet_gateway" "moon_fl" {
  vpc_id = aws_vpc.moon_fl.id
  tags   = { Name = "moon-fl-igw", Project = "MOON-FL" }
}

resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.moon_fl.id
  cidr_block              = var.public_subnet_a_cidr
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true
  tags = { Name = "moon-fl-public-a", Project = "MOON-FL" }
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.moon_fl.id
  cidr_block              = var.public_subnet_b_cidr
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true
  tags = { Name = "moon-fl-public-b", Project = "MOON-FL" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.moon_fl.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.moon_fl.id
  }

  tags = { Name = "moon-fl-public-rt", Project = "MOON-FL" }
}

resource "aws_route_table_association" "public_a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

# RDS requires a subnet group spanning at least 2 AZs
resource "aws_db_subnet_group" "mlflow" {
  name       = "moon-fl-mlflow-subnet-group"
  subnet_ids = [aws_subnet.public_a.id, aws_subnet.public_b.id]
  tags       = { Name = "moon-fl-mlflow-subnet-group", Project = "MOON-FL" }
}