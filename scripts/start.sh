#!/bin/bash
set -e

export ECR_REPO_URI=831926583882.dkr.ecr.eu-north-1.amazonaws.com/miniapp-image

# Stop existing container if running
docker stop grovio-app 2>/dev/null || true
docker rm grovio-app 2>/dev/null || true

# Run container with env file
docker run -d \
  --name grovio-app \
  --restart unless-stopped \
  -p 5005:5005 \
  --env-file /home/ec2-user/.env \
  $ECR_REPO_URI:latest
