#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install packages
install_package() {
    if command_exists apt-get; then
        apt-get update && apt-get install -y "$1"
    elif command_exists yum; then
        yum install -y "$1"
    else
        echo "Unsupported package manager. Please install $1 manually."
        exit 1
    fi
}

# Check and install required packages
for pkg in git python3 python3-pip; do
    if ! command_exists $pkg; then
        echo "Installing $pkg..."
        install_package $pkg
    fi
done

# Prompt for email and name
read -p "Enter your GitHub email: " email
read -p "Enter your full name: " name

# Generate SSH key
ssh-keygen -t ed25519 -C "$email" -f ~/.ssh/id_ed25519 -N ""

# Start SSH agent and add key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Set Git config
git config --global user.email "$email"
git config --global user.name "$name"

# Display public key
echo "Add this public key to your GitHub account:"
cat ~/.ssh/id_ed25519.pub

# Test connection
echo "Testing connection to GitHub..."
ssh -T git@github.com || true  # The command may exit with non-zero status even on success

# Clone repository
git clone git@github.com:sutyum/gpt2.git
cd gpt2/