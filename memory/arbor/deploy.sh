#!/bin/bash
# Simple wrapper script to deploy Arbor on RunPod

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to parent directory (memory)
cd "$SCRIPT_DIR/.."

# Run the deploy script with all arguments passed through
python arbor/deploy.py "$@"