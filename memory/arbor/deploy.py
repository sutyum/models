#!/usr/bin/env python3
"""Deploy Arbor server on RunPod with clean configuration management."""

import os
import sys
import yaml
import time
import logging
import argparse
import base64
import tarfile
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field

import runpod


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class GPUType(Enum):
    """Supported GPU types for RunPod deployment."""

    RTX3090 = ("RTX3090", "NVIDIA GeForce RTX 3090")
    A6000 = ("A6000", "NVIDIA RTX A6000")
    A100 = ("A100", "NVIDIA A100-SXM4-80GB")

    def __init__(self, short_name: str, runpod_id: str):
        self.short_name = short_name
        self.runpod_id = runpod_id

    @classmethod
    def from_string(cls, gpu_type: str) -> "GPUType":
        """Get GPUType from string name."""
        for gpu in cls:
            if gpu.short_name == gpu_type:
                return gpu
        raise ValueError(
            f"Invalid GPU type: {gpu_type}. Must be one of: {[g.short_name for g in cls]}"
        )


@dataclass
class ArborConfig:
    """Arbor configuration parsed from YAML."""

    inference: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    raw_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ArborConfig":
        """Load Arbor configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        return cls(
            inference=raw_config.get("inference", {}),
            training=raw_config.get("training", {}),
            raw_config=raw_config,
        )

    def to_yaml(self) -> str:
        """Convert configuration back to YAML string."""
        return yaml.dump(self.raw_config, default_flow_style=False)

    def get_gpu_requirements(self) -> int:
        """Calculate total number of GPUs needed."""
        gpu_count = 0

        # Parse inference GPUs
        if "gpu_ids" in self.inference:
            inference_gpus = self._parse_gpu_ids(self.inference["gpu_ids"])
            if inference_gpus:
                gpu_count = max(gpu_count, max(inference_gpus) + 1)

        # Parse training GPUs
        if "gpu_ids" in self.training:
            training_gpus = self._parse_gpu_ids(self.training["gpu_ids"])
            if training_gpus:
                gpu_count = max(gpu_count, max(training_gpus) + 1)

        return max(gpu_count, 1)  # At least 1 GPU

    @staticmethod
    def _parse_gpu_ids(gpu_ids: Any) -> list[int]:
        """Parse GPU IDs from various formats."""
        if not gpu_ids:
            return []

        if isinstance(gpu_ids, str):
            return [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
        elif isinstance(gpu_ids, list):
            return [int(x) for x in gpu_ids]
        elif isinstance(gpu_ids, int):
            return [gpu_ids]
        else:
            logger.warning(f"Unknown GPU ID format: {type(gpu_ids)}")
            return []


@dataclass
class PodConfig:
    """Configuration for creating a RunPod pod."""

    name: str = "arbor-server"
    gpu_type: GPUType = GPUType.RTX3090
    gpu_count: int = 1
    disk_size_gb: int = 50
    volume_size_gb: int = 100
    image_name: str = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
    ports: str = "8000/http,22/tcp"
    support_public_ip: bool = True


@dataclass
class ConnectionInfo:
    """Pod connection information."""

    pod_id: str
    name: str
    arbor_url: str
    ssh_command: Optional[str] = None
    status: str = "Unknown"


class RunPodDeployer:
    """Handles deployment of Arbor server on RunPod."""

    def __init__(self, api_key: str):
        """Initialize deployer with RunPod API key."""
        self.api_key = api_key
        runpod.api_key = api_key

    def create_pod(self, arbor_config: ArborConfig, pod_config: PodConfig) -> str:
        """Create a RunPod pod with Arbor server."""
        # Update GPU count based on Arbor requirements
        pod_config.gpu_count = arbor_config.get_gpu_requirements()

        logger.info(
            f"Creating pod '{pod_config.name}' with {pod_config.gpu_count} {pod_config.gpu_type.short_name} GPU(s)"
        )

        # Create startup script that extracts and runs the arbor files
        startup_script = self._generate_startup_script(arbor_config)

        try:
            pod = runpod.create_pod(
                name=pod_config.name,
                image_name=pod_config.image_name,
                gpu_type_id=pod_config.gpu_type.runpod_id,
                gpu_count=pod_config.gpu_count,
                container_disk_in_gb=pod_config.disk_size_gb,
                volume_in_gb=pod_config.volume_size_gb,
                ports=pod_config.ports,
                docker_args=startup_script,
                support_public_ip=pod_config.support_public_ip,
            )

            logger.info(f"Pod created successfully with ID: {pod['id']}")
            return pod["id"]

        except Exception as e:
            logger.error(f"Failed to create pod: {e}")
            raise

    def wait_for_pod(self, pod_id: str, timeout: int = 300) -> ConnectionInfo:
        """Wait for pod to be ready and return connection info."""
        logger.info(f"Waiting for pod {pod_id} to be ready (timeout: {timeout}s)")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                pod = runpod.get_pod(pod_id)

                if self._is_pod_ready(pod):
                    return self._extract_connection_info(pod)

                elapsed = int(time.time() - start_time)
                status = pod.get("desiredStatus", "Unknown") if pod else "Unknown"
                logger.debug(f"Pod status: {status} ({elapsed}s/{timeout}s)")

            except Exception as e:
                logger.debug(f"Error checking pod status: {e}")

            time.sleep(10)

        raise TimeoutError(
            f"Pod {pod_id} did not become ready within {timeout} seconds"
        )

    def list_pods(self) -> list[dict[str, Any]]:
        """List all RunPod pods."""
        return runpod.get_pods()

    def terminate_pod(self, pod_id: str) -> None:
        """Terminate a RunPod pod."""
        logger.info(f"Terminating pod {pod_id}")
        runpod.terminate_pod(pod_id)
        logger.info("Pod terminated successfully")

    def _generate_startup_script(self, arbor_config: ArborConfig) -> str:
        """Generate startup script for the pod."""
        # Get the arbor directory path
        arbor_dir = Path(__file__).parent
        
        # Create a tarball of the arbor directory
        tar_content = self._create_arbor_tarball(arbor_dir, arbor_config)
        
        return f"""#!/bin/bash
set -e

echo "Starting Arbor server setup..."

# Create arbor directory
mkdir -p /arbor
cd /arbor

# Extract arbor files
echo "{tar_content}" | base64 -d | tar -xzf -

# Make startup script executable
chmod +x /arbor/startup.sh

# Execute startup script
/arbor/startup.sh
"""
    
    def _create_arbor_tarball(self, arbor_dir: Path, arbor_config: ArborConfig) -> str:
        """Create a base64-encoded tarball of the arbor directory."""
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            try:
                with tarfile.open(tmp_file.name, 'w:gz') as tar:
                    # Add all files from arbor directory
                    for file_path in arbor_dir.iterdir():
                        if file_path.name not in ['.git', '__pycache__', '*.pyc']:
                            if file_path.is_file():
                                tar.add(file_path, arcname=file_path.name)
                    
                    # Create a temporary arbor.yaml with the config
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as yaml_file:
                        yaml_file.write(arbor_config.to_yaml())
                        yaml_file.flush()
                        tar.add(yaml_file.name, arcname='arbor.yaml')
                        os.unlink(yaml_file.name)
                
                # Read and encode the tarball
                with open(tmp_file.name, 'rb') as f:
                    tar_content = base64.b64encode(f.read()).decode('utf-8')
                
                return tar_content
            finally:
                os.unlink(tmp_file.name)

    @staticmethod
    def _is_pod_ready(pod: dict[str, Any]) -> bool:
        """Check if pod is ready to accept connections."""
        if not pod:
            return False

        return (
            pod.get("desiredStatus") == "RUNNING"
            and pod.get("runtime") is not None
            and "ports" in pod.get("runtime", {})
        )

    @staticmethod
    def _extract_connection_info(pod: dict[str, Any]) -> ConnectionInfo:
        """Extract connection information from pod data."""
        runtime = pod["runtime"]

        # Find Arbor server port
        arbor_url = None
        ssh_command = None

        for port_info in runtime.get("ports", []):
            if port_info.get("privatePort") == 8000:
                public_ip = port_info.get("ip", pod.get("ip"))
                public_port = port_info.get("publicPort", 8000)
                if public_ip:
                    arbor_url = f"http://{public_ip}:{public_port}"

            elif port_info.get("privatePort") == 22:
                public_ip = port_info.get("ip", pod.get("ip"))
                ssh_port = port_info.get("publicPort")
                if public_ip and ssh_port:
                    ssh_command = f"ssh root@{public_ip} -p {ssh_port}"

        if not arbor_url:
            raise ValueError("Could not extract Arbor server URL from pod information")

        return ConnectionInfo(
            pod_id=pod["id"],
            name=pod["name"],
            arbor_url=arbor_url,
            ssh_command=ssh_command,
            status=pod.get("desiredStatus", "Unknown"),
        )


def main():
    """Main entry point for the deployment script."""
    parser = argparse.ArgumentParser(
        description="Deploy Arbor server on RunPod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Deploy with default settings:
    python arbor/deploy.py
  
  Deploy with A100 GPU:
    python arbor/deploy.py --gpu-type A100
  
  List all pods:
    python arbor/deploy.py --list
  
  Terminate a pod:
    python arbor/deploy.py --terminate <pod-id>
""",
    )

    # Deployment options
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("arbor.yaml"),
        help="Path to arbor.yaml config file",
    )
    parser.add_argument(
        "--gpu-type",
        choices=[gpu.short_name for gpu in GPUType],
        default="RTX3090",
        help="GPU type to use",
    )
    parser.add_argument("--name", default="arbor-server", help="Pod name")
    parser.add_argument(
        "--disk-size", type=int, default=50, help="Container disk size in GB"
    )
    parser.add_argument(
        "--volume-size", type=int, default=100, help="Volume size in GB"
    )

    # API and management options
    parser.add_argument(
        "--api-key", help="RunPod API key (or set RUNPOD_API_KEY env var)"
    )
    parser.add_argument(
        "--terminate", metavar="POD_ID", help="Terminate pod with given ID"
    )
    parser.add_argument("--list", action="store_true", help="List all pods")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        logger.error("RunPod API key not provided")
        print("Set RUNPOD_API_KEY environment variable or use --api-key flag")
        sys.exit(1)

    deployer = RunPodDeployer(api_key)

    # Handle management commands
    if args.list:
        pods = deployer.list_pods()
        print("\n📋 Your RunPod pods:")
        for pod in pods:
            status = pod.get("desiredStatus", "Unknown")
            print(f"  • {pod['name']} (ID: {pod['id']}) - Status: {status}")
        return

    if args.terminate:
        deployer.terminate_pod(args.terminate)
        return

    # Deploy new pod
    try:
        # Load Arbor configuration
        arbor_config = ArborConfig.from_yaml(args.config)
        logger.info(f"Loaded Arbor configuration from {args.config}")

        # Create pod configuration
        pod_config = PodConfig(
            name=args.name,
            gpu_type=GPUType.from_string(args.gpu_type),
            disk_size_gb=args.disk_size,
            volume_size_gb=args.volume_size,
        )

        # Deploy pod
        pod_id = deployer.create_pod(arbor_config, pod_config)

        # Wait for pod to be ready
        connection_info = deployer.wait_for_pod(pod_id)

        # Display connection information
        print("\n✅ Arbor server is running on RunPod!")
        print(f"\n🌐 Arbor URL: {connection_info.arbor_url}")
        if connection_info.ssh_command:
            print(f"🔐 SSH Access: {connection_info.ssh_command}")

        print(f"\n📝 To connect from DSPy:")
        print(f"   import dspy")
        print(f"   dspy.settings.configure(arbor_url='{connection_info.arbor_url}')")

        print(f"\n🛑 To terminate this pod:")
        print(f"   python arbor/deploy.py --terminate {pod_id}")
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        print("Please create an arbor.yaml file with your configuration")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
