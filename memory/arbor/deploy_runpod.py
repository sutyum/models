#!/usr/bin/env python3
"""
RunPod deployment script for Arbor
"""

import runpod
import os
import sys

# RunPod API Key from environment
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")

def deploy_arbor_pod():
    """Deploy Arbor on RunPod"""
    
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)
    
    # Set API key
    runpod.api_key = RUNPOD_API_KEY
    
    # Pod configuration
    pod_config = {
        "name": "arbor-memory-system",
        "image_name": "your-registry/arbor-runpod:latest",  # Replace with your registry
        "gpu_type_id": "NVIDIA RTX A5000",  # Adjust as needed
        "cloud_type": "ALL",
        "support_public_ip": True,
        "start_jupyter": False,
        "start_ssh": True,
        "gpu_count": 1,
        "volume_in_gb": 50,
        "container_disk_in_gb": 20,
        "min_vcpu_count": 4,
        "min_memory_in_gb": 15,
        "docker_args": "",
        "ports": "8000/http",
        "volume_mount_path": "/workspace",
        "env": {
            "RUNPOD_API_KEY": RUNPOD_API_KEY,
            "CUDA_VISIBLE_DEVICES": "0"
        }
    }
    
    try:
        # Create pod
        pod = runpod.create_pod(**pod_config)
        
        print(f"Pod created successfully!")
        print(f"Pod ID: {pod['id']}")
        print(f"Pod Status: {pod.get('desiredStatus', 'Unknown')}")
        
        # Get pod details
        pod_details = runpod.get_pod(pod['id'])
        if pod_details.get('runtime'):
            runtime = pod_details['runtime']
            if runtime.get('ports'):
                for port in runtime['ports']:
                    if port['privatePort'] == 8000:
                        print(f"Arbor Server URL: https://{pod['id']}-{port['publicPort']}.proxy.runpod.net")
        
        return pod
        
    except Exception as e:
        print(f"Error creating pod: {e}")
        return None

def list_pods():
    """List all RunPod instances"""
    runpod.api_key = RUNPOD_API_KEY
    
    try:
        pods = runpod.get_pods()
        print("\nCurrent RunPod instances:")
        for pod in pods:
            print(f"- {pod['name']} ({pod['id']}): {pod.get('desiredStatus', 'Unknown')}")
    except Exception as e:
        print(f"Error listing pods: {e}")

def stop_pod(pod_id):
    """Stop a RunPod instance"""
    runpod.api_key = RUNPOD_API_KEY
    
    try:
        result = runpod.stop_pod(pod_id)
        print(f"Pod {pod_id} stopped successfully")
        return result
    except Exception as e:
        print(f"Error stopping pod {pod_id}: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python deploy_runpod.py deploy    # Deploy new Arbor pod")
        print("  python deploy_runpod.py list      # List all pods")
        print("  python deploy_runpod.py stop <pod_id>  # Stop specific pod")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "deploy":
        deploy_arbor_pod()
    elif command == "list":
        list_pods()
    elif command == "stop" and len(sys.argv) > 2:
        stop_pod(sys.argv[2])
    else:
        print("Invalid command. Use 'deploy', 'list', or 'stop <pod_id>'")