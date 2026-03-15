#!/bin/bash

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=ap-south-1
ECR="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Tag all three
docker tag moon-fl-server:latest $ECR/moon-fl-server:latest
docker tag moon-fl-client:latest $ECR/moon-fl-client:latest
docker tag moon-fl-mlflow:latest $ECR/moon-fl-mlflow:latest

# Push all three
docker push $ECR/moon-fl-server:latest
docker push $ECR/moon-fl-client:latest
docker push $ECR/moon-fl-mlflow:latest
