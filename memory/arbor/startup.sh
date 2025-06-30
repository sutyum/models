#!/bin/bash
set -e

# Update system packages
apt-get update && apt-get install -y git curl wget

# Setup workspace
mkdir -p /workspace
cd /workspace

# Copy all files from arbor directory
cp -r /arbor/* /workspace/

# Create Python virtual environment
python -m venv /workspace/venv
source /workspace/venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Start Arbor server
python -m arbor.cli serve --arbor-config arbor.yaml --host 0.0.0.0 --port 8000