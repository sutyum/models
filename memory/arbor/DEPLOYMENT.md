# Arbor RunPod Deployment Guide

This guide will help you build, test, and deploy the Arbor memory system on RunPod instances.

## Prerequisites

1. Docker installed locally
2. Docker Hub account (or other registry access)
3. RunPod account with API key
4. Python with `runpod` package installed

```bash
pip install runpod
```

## Step 1: Build and Test Locally

First, test the Docker image locally:

```bash
./test_docker.sh
```

This will:
- Build the Docker image
- Run it locally on port 8000
- Test basic connectivity
- Provide logs if there are issues

## Step 2: Push to Docker Registry

1. Edit `build_and_push.sh` and set your Docker Hub username:
   ```bash
   USERNAME="your-docker-hub-username"
   ```

2. Run the build and push script:
   ```bash
   ./build_and_push.sh
   ```

This will:
- Build the Docker image
- Tag it for your registry
- Push it to Docker Hub
- Provide the full image name for RunPod

## Step 3: Deploy to RunPod

1. Update `deploy_runpod.py` with your Docker image:
   ```python
   "image_name": "docker.io/your-username/arbor-runpod:latest"
   ```

2. Deploy to RunPod:
   ```bash
   python deploy_runpod.py deploy
   ```

3. List your pods:
   ```bash
   python deploy_runpod.py list
   ```

4. Stop a pod when done:
   ```bash
   python deploy_runpod.py stop <pod-id>
   ```

## Configuration Files

### Dockerfile
- Based on RunPod's PyTorch image
- Installs Arbor and dependencies
- Exposes port 8000
- Uses startup.sh as entrypoint

### startup.sh
- Sets up the environment
- Installs dependencies
- Starts Arbor server on 0.0.0.0:8000

### arbor.yaml
- Arbor configuration
- GPU assignments for inference and training

## RunPod Instance Configuration

The deployment script configures:
- **GPU**: NVIDIA RTX A5000 (adjustable)
- **Storage**: 50GB volume + 20GB container disk
- **Memory**: 15GB minimum
- **vCPU**: 4 cores minimum
- **Ports**: 8000/http exposed
- **SSH**: Enabled for debugging

## Troubleshooting

### Build Issues
- Large base image may cause timeouts
- Use `docker system prune` to free space
- Consider using a faster internet connection

### Deployment Issues
- Verify RunPod API key is correct
- Check GPU availability in your region
- Monitor pod logs through RunPod dashboard

### Runtime Issues
- SSH into the pod to debug
- Check logs: `docker logs <container-id>`
- Verify Arbor configuration in arbor.yaml

## Environment Variables

Key environment variables set in the container:
- `RUNPOD_API_KEY`: Your RunPod API key
- `CUDA_VISIBLE_DEVICES`: GPU assignment

## Costs

RunPod charges by:
- GPU hours used
- Storage allocated
- Network transfer

Monitor usage through the RunPod dashboard to control costs.

## Alternative Registries

Instead of Docker Hub, you can use:
- GitHub Container Registry (ghcr.io)
- Google Container Registry (gcr.io)
- AWS ECR

Update the `REGISTRY` variable in `build_and_push.sh` accordingly.