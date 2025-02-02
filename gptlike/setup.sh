#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Poetry
if ! command_exists poetry; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Set up Python environment
poetry install

# Install CUDA and cuDNN if GPU is available
if command_exists nvidia-smi; then
    echo "GPU detected. Installing CUDA and cuDNN..."
    # Note: This is a placeholder. You should replace this with the appropriate
    # commands to install CUDA and cuDNN for your specific system.
    # For example:
    # wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    # mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    # wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    # dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    # cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    # apt-get update
    # apt-get -y install cuda
else
    echo "No GPU detected. Skipping CUDA and cuDNN installation."
fi

# Run the main script
echo "Running the main script..."
poetry run python fineweb.py

echo "Setup complete. You can now use Git with GitHub and develop your GPT-2 project."

# Enter the virtual env
poetry shell