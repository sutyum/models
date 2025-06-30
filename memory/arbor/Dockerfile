FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Copy arbor directory contents
COPY . /arbor/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment and install dependencies
RUN python -m venv /workspace/venv && \
    . /workspace/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /arbor/requirements.txt

# Make startup script executable
RUN chmod +x /arbor/startup.sh

# Expose port for Arbor server
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/arbor/startup.sh"]