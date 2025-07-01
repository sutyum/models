FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt /arbor/requirements.txt

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /workspace/venv \
    && . /workspace/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r /arbor/requirements.txt

# Copy arbor directory contents
COPY . /arbor/

# Make startup script executable
RUN chmod +x /arbor/startup.sh

# Expose port for Arbor server
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/arbor/startup.sh"]