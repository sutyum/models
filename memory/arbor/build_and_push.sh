#!/bin/bash
set -e

# Configuration
REGISTRY="docker.io"  # Change to your registry (docker.io, ghcr.io, etc.)
USERNAME=""  # Your Docker Hub username or GitHub username
IMAGE_NAME="arbor-runpod"
TAG="latest"

# Full image name
FULL_IMAGE_NAME="${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building and pushing Arbor Docker image..."
echo "Target: ${FULL_IMAGE_NAME}"

# Check if username is set
if [ -z "$USERNAME" ]; then
    echo "Error: Please set USERNAME in this script to your Docker Hub username"
    echo "Edit this file and set USERNAME=\"your-docker-hub-username\""
    exit 1
fi

# Build the image
echo "Building Docker image..."
docker build -t "${IMAGE_NAME}:${TAG}" .

# Tag for registry
docker tag "${IMAGE_NAME}:${TAG}" "${FULL_IMAGE_NAME}"

# Login to registry (if not already logged in)
echo "Logging in to ${REGISTRY}..."
docker login "${REGISTRY}"

# Push to registry
echo "Pushing to registry..."
docker push "${FULL_IMAGE_NAME}"

echo "Successfully pushed ${FULL_IMAGE_NAME}"
echo ""
echo "To use this image on RunPod:"
echo "1. Update deploy_runpod.py and set image_name to: ${FULL_IMAGE_NAME}"
echo "2. Run: python deploy_runpod.py deploy"
echo ""
echo "Or manually create a RunPod instance with image: ${FULL_IMAGE_NAME}"