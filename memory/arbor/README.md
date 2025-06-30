# Arbor RunPod Deployment

This directory contains all the necessary files to deploy an Arbor server on RunPod.

## Files

- `deploy.py` - Main deployment script
- `deploy.sh` - Convenience wrapper script
- `startup.sh` - Container startup script
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container definition (for reference)
- `arbor.yaml` - Arbor configuration

## Usage

### Prerequisites

1. Set your RunPod API key:
   ```bash
   export RUNPOD_API_KEY=your_api_key_here
   ```

2. Ensure you have `runpod` Python package installed locally:
   ```bash
   pip install runpod
   ```

### Deploy

From the memory directory:

```bash
# Deploy with default settings
python arbor/deploy.py

# Deploy with specific GPU type
python arbor/deploy.py --gpu-type A100

# Deploy with custom configuration
python arbor/deploy.py --config path/to/custom-arbor.yaml
```

Or use the convenience script from anywhere:

```bash
./arbor/deploy.sh --gpu-type RTX3090
```

### Manage Pods

```bash
# List all pods
python arbor/deploy.py --list

# Terminate a pod
python arbor/deploy.py --terminate <pod-id>
```

## How It Works

1. The deployment script creates a tarball of the arbor directory
2. The tarball is base64-encoded and embedded in the startup script
3. RunPod executes the startup script which:
   - Extracts the arbor files
   - Sets up the Python environment
   - Installs dependencies from requirements.txt
   - Starts the Arbor server

## Customization

- Modify `requirements.txt` to add/remove dependencies
- Edit `startup.sh` to change the server startup behavior
- Update `arbor.yaml` for different Arbor configurations