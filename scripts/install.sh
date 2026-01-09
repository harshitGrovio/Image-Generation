#!/bin/bash
set -e

export ECR_REPO_URI=831926583882.dkr.ecr.eu-north-1.amazonaws.com/miniapp-image

# Login to ECR
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin $ECR_REPO_URI

# Pull latest image
docker pull $ECR_REPO_URI:latest
