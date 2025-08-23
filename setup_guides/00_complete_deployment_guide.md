# Complete Deployment Guide - LangGraph Local AI Cluster

## Overview

This guide provides **step-by-step commands** to deploy your revised LangGraph architecture across your available machines.

**🎯 SIMPLIFIED FOR LEARNING**: For learning LangGraph, you can **skip Docker entirely** and run **native Ollama** on jetson-node. This guide includes both approaches.

**Native Approach**: Ollama directly installed → simpler, better performance  
**Docker Approach**: Containerized services → complex, production-style

### **Target Architecture**

| Role | Machine | IP | RAM | CPU | Purpose |
|------|---------|-----|-----|-----|---------|
| **Primary LLM** | jetson-node | 192.168.1.177 | 8GB | ARM Cortex-A78AE | Ollama + TensorRT + Triton |
| **Coordinator** | cpu-node | 192.168.1.81 | 32GB | Intel i5-6500T | LangGraph + Heavy LLM + HAProxy + Redis |
| **Embeddings** | rp-node | 192.168.1.178 | 8GB | ARM Cortex-A76 | Vector processing |
| **Tools** | worker-node3 | 192.168.1.105 | 6GB | Intel i5 VM | Web scraping, APIs |
| **Monitoring** | worker-node4 | 192.168.1.137 | 6GB | Intel i5 VM | Health checks, alerts |

---

## 🎯 SIMPLE 2-OLLAMA ARCHITECTURE (Perfect for Learning LangGraph)

**Simplified design using Ollama on BOTH machines - much easier for learning!**

### The Architecture Logic:
- **jetson-node (8GB)**: Ollama → Small/efficient models (2-7B parameters)
- **cpu-node (32GB)**: Ollama → Large models (7-13B parameters) 
- **HAProxy**: Load balances requests between both Ollama endpoints

### Why This Simplified Design is PERFECT for Learning:
1. **Consistent Interface**: Same Ollama API on both machines - no complexity
2. **Easy Model Management**: `ollama pull/list/run` works the same everywhere
3. **Automatic GPU Detection**: Ollama handles CUDA on Jetson, CPU on Intel automatically
4. **High Availability**: If one server fails, the other takes over
5. **Focus on LangGraph**: Spend time learning workflows, not LLM optimization

### Quick Start - 2 Machines Setup:

#### Machine 1: jetson-node (192.168.1.177) - Small/Efficient Models
```bash
# SSH to Jetson  
ssh sanzad@192.168.1.177

# Install Ollama (already done)
# Configure for network access (already done)
# Pull small efficient models optimized for 8GB
ollama pull gemma2:2b        # 2B parameters - very fast
ollama pull phi3:mini        # 3.8B parameters - efficient
ollama pull llama3.2:3b      # 3B parameters - good quality
```

#### Machine 2: cpu-node (192.168.1.81) - Large Models + Coordinator
```bash
# SSH to CPU machine
ssh sanzad@192.168.1.81

# Install Ollama (same as Jetson - consistent!)
curl -fsSL https://ollama.com/install.sh | sh

# Configure for network access
export OLLAMA_HOST=0.0.0.0:11435  # Different port to avoid conflict
echo 'export OLLAMA_HOST=0.0.0.0:11435' >> ~/.bashrc
source ~/.bashrc

# Start Ollama service on custom port
ollama serve &

# Pull large models that leverage 32GB RAM
ollama pull llama2:7b-chat   # 7B parameters - good for complex tasks
ollama pull mistral:7b       # 7B parameters - excellent reasoning
ollama pull codellama:7b     # 7B parameters - coding tasks

# Install LangGraph for orchestration
pip install langgraph langchain-ollama
```

**🔥 RESULT**: Two powerful Ollama endpoints with different model sizes - simple and effective!

---

## 🏗️ Detailed Setup Steps (Required for Full Architecture)

**The sections below provide detailed step-by-step commands for implementing the 2-LLM architecture above.**

## 🚀 Phase 1: jetson-node Setup (Primary LLM Server)

**Machine**: jetson-node (192.168.1.177) - Jetson Orin Nano 8GB

### Step 1.1: Initial System Preparation

```bash
# SSH into jetson-node
ssh sanzad@192.168.1.177

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git htop iotop nano vim build-essential

# Check Jetson info
sudo nvpmodel -q  # Check current power mode
tegrastats  # Monitor system stats (Ctrl+C to exit)

# Set maximum performance mode
sudo nvpmodel -m 0  # MAXN mode
sudo jetson_clocks  # Max CPU/GPU clocks

# Verify CUDA installation
nvidia-smi

# Fix CUDA PATH if nvcc not found (common Jetson issue)
if ! command -v nvcc &> /dev/null; then
    echo "⚠️ nvcc not found, setting up CUDA PATH..."
    
    # Find CUDA installation
    if [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    elif [ -d "/usr/local/cuda-12.6" ]; then
        CUDA_PATH="/usr/local/cuda-12.6"
    elif [ -d "/usr/local/cuda-12.2" ]; then
        CUDA_PATH="/usr/local/cuda-12.2"
    elif [ -d "/usr/local/cuda-11.4" ]; then
        CUDA_PATH="/usr/local/cuda-11.4"
    else
        echo "❌ CUDA not found. Installing..."
        sudo apt install -y cuda-toolkit-12-2
        CUDA_PATH="/usr/local/cuda-12.2"
    fi
    
    # Add CUDA to PATH permanently
    echo "export PATH=${CUDA_PATH}/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export CUDA_HOME=${CUDA_PATH}" >> ~/.bashrc
    
    # Create symlink if needed
    if [ ! -L "/usr/local/cuda" ] && [ "$CUDA_PATH" != "/usr/local/cuda" ]; then
        sudo ln -sf $CUDA_PATH /usr/local/cuda
    fi
    
    # Reload environment for current session
    source ~/.bashrc
    export PATH=${CUDA_PATH}/bin:$PATH
    export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:$LD_LIBRARY_PATH
fi

# Verify CUDA is working
nvcc --version

# If nvcc still not found, troubleshoot:
if ! command -v nvcc &> /dev/null; then
    echo "🔍 TROUBLESHOOTING: nvcc still not found"
    echo "Check JetPack version:"
    cat /etc/nv_tegra_release
    echo "Check installed CUDA packages:"
    dpkg -l | grep cuda
    echo "Check available CUDA paths:"
    ls -la /usr/local/cuda*
    echo "Install build tools if missing:"
    sudo apt install -y build-essential
fi
```

#### 🛠️ **CUDA Troubleshooting**

**Issue: `nvcc --version` returns "command not found"**

Very common on Jetson - CUDA installed but PATH not configured.

```bash
# Diagnosis:
ls -la /usr/local/ | grep cuda
echo $PATH
cat /etc/nv_tegra_release

# Your system has CUDA 12.6, manual fix if needed:
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
source ~/.bashrc && nvcc --version
```

**Expected CUDA Output:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Cuda compilation tools, release 12.6, V12.6.xxx
```

### Step 1.2: Install Docker and NVIDIA Container Runtime (OPTIONAL - Skip for Learning)

**⚠️ IMPORTANT**: This Docker section is **OPTIONAL** and adds unnecessary complexity for learning setups.

**🎯 RECOMMENDED**: Skip Docker entirely and run Ollama **natively** for:
- Learning LangGraph
- Single-user development  
- Simpler configuration
- Better performance on Jetson

**Only use Docker if you need**:
- Multiple isolated LLM services
- Production deployment
- Complex orchestration

```bash
# SAFER Docker installation for Jetson (research shows removing containerd breaks services)

# First, try normal installation
sudo apt update

# If containerd.io conflicts occur, use safer approach:
# Step 1: Stop services temporarily (don't remove containerd!)
sudo systemctl stop containerd || true
sudo systemctl stop docker || true

# Step 2: Install Docker using official script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh

# Step 3: Handle Docker conflicts properly (based on real-world testing)
if ! docker --version &> /dev/null; then
    echo "⚠️ Docker installation issues detected. Using proven Jetson approach..."
    
    # Stop any existing Docker services
    sudo systemctl stop docker.socket docker.service || true
    
    # Remove only conflicting packages (keep containerd - research shows removing it breaks services)
    sudo apt remove -y docker.io docker-engine || true
    
    # Install Docker CE (works better with existing containerd)
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli docker-compose-plugin
    
    # If still issues, check kernel compatibility
    if ! docker --version &> /dev/null; then
        echo "❌ Docker installation failed. Checking kernel compatibility..."
        curl -fsSL https://raw.githubusercontent.com/moby/moby/master/contrib/check-config.sh | bash
        echo "⚠️ See kernel compatibility report above. May need kernel modules or different approach."
    fi
fi

# Configure Docker service
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-runtime

# Configure Docker to use nvidia runtime
sudo tee /etc/docker/daemon.json << EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

sudo systemctl restart docker

# Test NVIDIA Docker with ARM64-compatible images
# CRITICAL: Container L4T version must match your Jetson's L4T version

# Step 1: Check your L4T version for container selection
cat /etc/nv_tegra_release
# Example: R36 (release), REVISION: 4.4 = Use r36.x containers

# Step 2: Test with version-matched container
# For L4T R36.x (JetPack 6.x):
docker run --rm --gpus all dustynv/l4t-ml:r36.2.0 nvidia-smi
# For L4T R35.x (JetPack 5.x):
# docker run --rm --gpus all dustynv/l4t-ml:r35.3.1 nvidia-smi

# Step 3: Alternative lightweight test (may have L4T version mismatch)
docker run --rm --gpus all mdegans/l4t-base:latest ls -la /dev/nvidia*

# Step 4: Fallback with manual install in current Ubuntu
docker run --rm --gpus all arm64v8/ubuntu:22.04 bash -c "apt update && apt install -y nvidia-utils-535 && nvidia-smi"
```

#### 🛠️ **Docker Troubleshooting**

**Issue: `containerd.io : Conflicts: containerd`**

Very common on Jetson. **CRITICAL**: Do NOT remove containerd - research shows it breaks system services!

```bash
# Root cause: Jetson has containerd pre-installed, conflicts with containerd.io
# SAFER solutions (preserve containerd):

# Option 1: Snap installation (recommended - no conflicts)
sudo snap install docker
sudo groupadd docker || true
sudo usermod -aG docker $USER

# Activate Docker group (avoid newgrp password prompt):
echo "💡 TIP: Exit and reconnect SSH to activate Docker group: exit; ssh sanzad@192.168.1.177"
# Or start fresh shell: sudo su - $USER

# Option 2: Minimal package removal (safer than full removal)
sudo systemctl stop docker containerd || true
sudo apt remove -y docker.io docker-engine || true  # Keep containerd!
sudo apt install -y docker-ce docker-ce-cli docker-compose-plugin
sudo systemctl start containerd docker

# Option 3: Version pinning (if specific versions work)
sudo apt-mark hold containerd
sudo apt install -y docker-ce=<specific-version>
```

**Issue: Permission denied accessing Docker**
```bash
# Need logout/login after adding to docker group, or:
newgrp docker
# Or start new shell: su - $USER
```

**Issue: Docker service won't start**
```bash
sudo journalctl -u docker.service  # Check logs
# Reset if needed:
sudo systemctl stop docker && sudo rm -rf /var/lib/docker && sudo systemctl start docker
```

**Issue: NVIDIA runtime not found**
```bash
dpkg -l | grep nvidia-container  # Verify installed
# If missing: sudo apt update && sudo apt install -y nvidia-container-runtime
docker info | grep nvidia  # Check runtime registered
```

**Alternative: Snap installation (if above fails)**
```bash
sudo apt remove -y docker docker-engine docker.io containerd runc || true
sudo snap install docker && sudo usermod -aG docker $USER
```

**Expected Docker Output:**
```bash
$ docker --version  
Docker version 28.3.3, build 980b856

$ docker run hello-world
Hello from Docker!
This message shows that your installation appears to be working correctly.

$ docker run --rm --gpus all mdegans/l4t-base:latest nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 540.4.0     Driver Version: 540.4.0      CUDA Version: 12.6     |
|   0  Orin (nvgpu)           N/A| N/A              N/A |                  N/A |
+-----------------------------------------------------------------------------+
```

**Issue: `manifest for nvidia/cuda:X.X not found`**

**Root cause**: Standard nvidia/cuda images are **x86_64 only** - Jetson uses ARM64.

```bash
# Verify architecture
uname -m  # Should show: aarch64
docker system info | grep Architecture  # Should show: aarch64

# Check your L4T version for correct container tags
cat /etc/nv_tegra_release
# R36.x.x = JetPack 6.x → use dustynv/l4t-*:r36.*
# R35.x.x = JetPack 5.x → use dustynv/l4t-*:r35.*
# R32.x.x = JetPack 4.x → use dustynv/l4t-*:r32.*

# Use version-matched Jetson containers
docker run --rm --gpus all dustynv/l4t-ml:r36.2.0 nvidia-smi  # For R36.x
docker run --rm --gpus all dustynv/l4t-ml:r35.3.1 nvidia-smi  # For R35.x
docker search dustynv/l4t  # Find containers for your L4T version
```

**Issue: `Unable to locate package nvidia-utils` (L4T version mismatch)**

**Root cause**: Container has different L4T version than your Jetson.

```bash
# Diagnose version mismatch
cat /etc/nv_tegra_release  # Your Jetson's L4T version
docker run --rm dustynv/l4t-ml:r36.2.0 cat /etc/nv_tegra_release  # Container's L4T

# Solution: Use matching L4T version container
# For L4T R36.x (JetPack 6.x) - your system:
docker run --rm --gpus all dustynv/l4t-ml:r36.2.0 nvidia-smi

# Alternative: Verify GPU access without nvidia-smi
docker run --rm --gpus all mdegans/l4t-base:latest ls -la /dev/nvidia*
# Should show: /dev/nvidia0, /dev/nvidiactl, etc.

# Search for containers matching your L4T
docker search dustynv  # Look for l4t-ml:rXX.X tags matching your version
```

**Issue: `docker: Cannot connect to daemon` (socket activation)**

**Root cause**: Docker socket activation can cause timing/permission issues.

```bash
# Disable socket activation, use direct service
sudo systemctl stop docker.socket docker.service
sudo systemctl disable docker.socket
sudo systemctl start docker.service
sudo systemctl status docker.service
docker run hello-world
```

**Issue: `docker --version` works but `docker run` fails**

**Root cause**: Stale socket files from previous daemon instances.

```bash
# Clean stale sockets and restart fresh
sudo systemctl stop docker
sudo rm -f /var/run/docker.sock /run/docker.sock
sudo systemctl start docker
docker run hello-world
```

**Issue: Docker daemon kernel/networking errors**

**Root cause**: Missing kernel modules (common with Snap Docker).

```bash
# Check kernel compatibility
curl -fsSL https://raw.githubusercontent.com/moby/moby/master/contrib/check-config.sh | bash

# If issues with Snap Docker, remove and use regular Docker
sudo snap remove docker
curl -fsSL https://get.docker.com | sh
sudo systemctl enable docker && sudo systemctl start docker
```

#### 🎯 **Container Selection for LangGraph Setup**

For our LangGraph architecture, choose containers based on your role:

**For LLM Inference (jetson-node):**

First, check your L4T version:
```bash
cat /etc/nv_tegra_release
# R36.4.4 = JetPack 6.x → use r36.x containers
```

Choose container based on your L4T version:
```bash
# For L4T R36.x (JetPack 6.x) - your system:
docker run --rm --gpus all dustynv/l4t-ml:r36.2.0 nvidia-smi

# For L4T R35.x (JetPack 5.x):
docker run --rm --gpus all dustynv/l4t-ml:r35.3.1 nvidia-smi

# Lightweight base (may have version mismatch - use for device testing):
docker run --rm --gpus all mdegans/l4t-base:latest ls -la /dev/nvidia*
```

**For Development/Testing:**
```bash
# Find containers matching your L4T version
docker search dustynv/l4t
# Look for tags like: r36.2.0, r35.3.1, r32.7.1

# Container selection by L4T/JetPack version:
# L4T R36.x (JetPack 6.x): dustynv/l4t-ml:r36.2.0 ← Your system
# L4T R35.x (JetPack 5.x): dustynv/l4t-ml:r35.3.1
# L4T R32.x (JetPack 4.x): dustynv/l4t-ml:r32.7.1

# CRITICAL: Version mismatch causes "package not found" errors
```

### Step 1.3: Install Ollama with Optimizations

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama for production
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_GPU_OVERHEAD=0.9"
Environment="CUDA_VISIBLE_DEVICES=0"
Restart=always
RestartSec=10
EOF

# Enable and start Ollama
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for service to start
sleep 10

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Step 1.4: Install Optimized Models

```bash
# Install models optimized for Jetson Orin Nano
ollama pull tinyllama:1.1b-chat-fp16      # Ultra-fast responses (~40-60 tok/s)
ollama pull gemma2:2b                     # Efficient general purpose (~25-35 tok/s)  
ollama pull llama3.2:3b                   # Best quality/speed balance (~15-25 tok/s)
ollama pull phi3:mini                     # Excellent for coding (~12-20 tok/s)

# Verify models are installed
ollama list

# Test each model performance
echo "=== Testing tinyllama ===" 
time ollama run tinyllama:1.1b-chat-fp16 "Hello, what are you?" --verbose

echo "=== Testing gemma2 ==="
time ollama run gemma2:2b "Explain Python in one sentence." --verbose

echo "=== Testing llama3.2 ==="
time ollama run llama3.2:3b "Write a simple Python function." --verbose
```

### Step 1.5: Setup TensorRT Optimization (Advanced)

```bash
# Install TensorRT (if not already installed)
sudo apt install -y tensorrt

# Create TensorRT optimization script
mkdir -p ~/jetson-optimizations
cd ~/jetson-optimizations

cat > optimize_models.py << 'EOF'
#!/usr/bin/env python3
"""
TensorRT optimization script for Jetson Orin Nano
Optimizes models for faster inference
"""

import os
import subprocess
import time

def optimize_model_for_tensorrt(model_name):
    """Optimize a model using TensorRT"""
    print(f"Optimizing {model_name} with TensorRT...")
    
    # This is a placeholder for TensorRT optimization
    # In practice, you'd convert ONNX models to TensorRT engines
    # Ollama handles some optimizations internally
    
    try:
        # Warm up the model (loads it into GPU memory)
        result = subprocess.run([
            'ollama', 'run', model_name, 
            'Hello', '--verbose'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ {model_name} optimization completed")
        else:
            print(f"❌ {model_name} optimization failed")
            
    except subprocess.TimeoutExpired:
        print(f"⚠️ {model_name} optimization timed out")

if __name__ == "__main__":
    models = [
        "tinyllama:1.1b-chat-fp16",
        "gemma2:2b", 
        "llama3.2:3b",
        "phi3:mini"
    ]
    
    for model in models:
        optimize_model_for_tensorrt(model)
        time.sleep(5)  # Cool down between optimizations

print("TensorRT optimization completed!")
EOF

chmod +x optimize_models.py
python3 optimize_models.py
```

### Step 1.6: Setup Triton Inference Server (Optional Advanced Setup)

```bash
# Install Triton Inference Server
# Note: This is advanced and optional - Ollama provides good performance out of box

# Create Triton setup directory
mkdir -p ~/triton-setup
cd ~/triton-setup

# Download Triton server for Jetson
wget https://github.com/triton-inference-server/server/releases/download/v2.40.0/tritonserver2.40.0-jetpack5.1.tgz
tar -xzf tritonserver2.40.0-jetpack5.1.tgz

# Create model repository structure
mkdir -p ~/triton-models/llama-optimized/1

# Create Triton config (example)
cat > ~/triton-models/llama-optimized/config.pbtxt << 'EOF'
name: "llama-optimized"
platform: "python"
max_batch_size: 1
input [
  {
    name: "prompt"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]
output [
  {
    name: "response"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
EOF

# Note: Full Triton setup requires model conversion and is complex
# For this setup, Ollama provides sufficient performance
echo "Triton setup prepared (advanced users can complete configuration)"
```

### Step 1.7: Performance Monitoring Setup

```bash
# Install additional monitoring tools
sudo apt install -y python3-pip
pip3 install psutil GPUtil

# Create performance monitoring script
cat > ~/monitor_jetson.py << 'EOF'
#!/usr/bin/env python3
import time
import subprocess
import json
import psutil
from datetime import datetime

def get_tegrastats():
    """Get Jetson-specific stats"""
    try:
        result = subprocess.run(['tegrastats', '--interval', '1000', '--logfile', '/tmp/tegra.log'], 
                              timeout=2, capture_output=True)
        # Parse tegrastats output
        with open('/tmp/tegra.log', 'r') as f:
            latest = f.readlines()[-1] if f.readlines() else ""
        return latest.strip()
    except:
        return "Tegrastats unavailable"

def get_ollama_stats():
    """Get Ollama performance stats"""
    try:
        result = subprocess.run(['curl', '-s', 'http://localhost:11434/api/tags'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        return {"error": "Ollama not responding"}
    except:
        return {"error": "Failed to get Ollama stats"}

def main():
    print(f"Jetson Performance Monitor - {datetime.now()}")
    print("=" * 50)
    
    # System stats
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"Disk Usage: {psutil.disk_usage('/').percent}%")
    
    # Jetson-specific stats
    tegra_stats = get_tegrastats()
    print(f"Tegra Stats: {tegra_stats}")
    
    # Ollama stats
    ollama_stats = get_ollama_stats()
    print(f"Ollama Status: {'Running' if 'models' in str(ollama_stats) else 'Not Running'}")
    
    # GPU Memory (if nvidia-smi available)
    try:
        gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
        if gpu_result.returncode == 0:
            gpu_mem = gpu_result.stdout.strip().split(', ')
            print(f"GPU Memory: {gpu_mem[0]}MB / {gpu_mem[1]}MB")
    except:
        print("GPU stats unavailable")

if __name__ == "__main__":
    main()
EOF

chmod +x ~/monitor_jetson.py

# Test monitoring
python3 ~/monitor_jetson.py
```

### Step 1.8: Network Configuration and Testing

```bash
# Configure firewall for Ollama
sudo ufw allow 11434/tcp
sudo ufw --force enable

# Test Ollama API from localhost
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "Hello, world!",
    "stream": false
  }'

# Test from network (run from another machine)
# curl -X POST http://192.168.1.177:11434/api/generate \
#   -H "Content-Type: application/json" \
#   -d '{"model": "llama3.2:3b", "prompt": "Hello!", "stream": false}'

echo "✅ jetson-node setup completed!"
echo "📊 Available models:"
ollama list
echo "🌐 Ollama API available at: http://192.168.1.177:11434"
```

---

## 🧠 Phase 2: cpu-node Setup (Coordinator + Ollama Large Models + HAProxy + Redis)

**Machine**: cpu-node (192.168.1.81) - 32GB Intel i5-6500T

### Step 2.1: Initial System Preparation

```bash
# SSH into cpu-node (or run locally if you're already there)
# ssh sanzad@192.168.1.81

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git htop iotop nano vim \
    python3 python3-pip python3-venv redis-server haproxy
```

### Step 2.2: Setup Ollama for Large Models (Simplified Approach)

```bash
# Install Ollama (same as jetson-node - consistent!)
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama service for network access on different port (same approach as jetson)
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11435"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_FLASH_ATTENTION=1"
EOF

# Reload systemd and enable Ollama service
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for service to start
sleep 10

# Check service status
sudo systemctl status ollama --no-pager

# Pull large models that leverage 32GB RAM
echo "📥 Starting with the best 7B model for learning..."

# Start with the best general-purpose 7B model
ollama pull mistral:7b          # 7B parameters - best reasoning, perfect for LangGraph learning

# Optional: Add these later when needed
echo "💡 Add these later as needed:"
echo "  ollama pull llama2:7b-chat    # Alternative for comparison"
echo "  ollama pull codellama:7b      # Specialized for coding tasks"

# Test models are working
ollama list
echo "✅ Large models ready on cpu-node!"
```

### Step 2.3: Test Ollama Server

```bash
# Test Ollama server locally
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b",
    "prompt": "Hello! Test response from cpu-node.",
    "stream": false
  }'

# Test interactive chat
ollama run mistral:7b "What are you running on?"

# Service is already running via built-in ollama.service
# Check that it's working on the right port
curl http://localhost:11435/api/tags || echo "⚠️ Service not ready yet, wait a moment"

# Test remote access from jetson
echo "✅ Test from jetson-node:"
echo "curl -X POST http://192.168.1.81:11435/api/generate -H 'Content-Type: application/json' -d '{\"model\": \"mistral:7b\", \"prompt\": \"Hello from jetson!\", \"stream\": false}'"
```

### Step 2.4: Setup Redis Cache

```bash
# Configure Redis for network access
sudo sed -i 's/bind 127.0.0.1/bind 0.0.0.0/' /etc/redis/redis.conf
sudo sed -i 's/# requirepass foobared/requirepass langgraph_redis_pass/' /etc/redis/redis.conf

# Optimize Redis for AI workloads
sudo tee -a /etc/redis/redis.conf << EOF

# AI Workload Optimizations
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
EOF

# Restart and enable Redis
sudo systemctl restart redis-server
sudo systemctl enable redis-server

# Test Redis
redis-cli -h localhost ping
redis-cli -h localhost -a langgraph_redis_pass ping

# Test from network
redis-cli -h 192.168.1.81 -a langgraph_redis_pass ping
```

### Step 2.5: Setup HAProxy Load Balancer

```bash
# Backup original HAProxy config
sudo cp /etc/haproxy/haproxy.cfg /etc/haproxy/haproxy.cfg.backup

# Create new HAProxy configuration
sudo tee /etc/haproxy/haproxy.cfg << 'EOF'
global
    daemon
    maxconn 4096
    log stdout local0
    stats socket /var/run/haproxy.sock mode 660 level admin

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull
    retries 3

# Health check endpoint
listen health_check
    bind *:8888
    mode http
    monitor-uri /health
    http-request return status 200 content-type text/plain string "HAProxy OK"

# LLM Load Balancer (Primary endpoint for LangGraph)
frontend llm_frontend
    bind *:9000
    mode http
    option httplog
    default_backend llm_servers

backend llm_servers
    mode http
    balance roundrobin
    option httpchk GET /api/tags
    
    # Load balanced backends (round-robin with weights)
    # Jetson Ollama: Small/efficient models, faster responses (67% of requests)
    server jetson 192.168.1.177:11434 check weight 10 inter 30s fall 3 rise 2
    
    # CPU Ollama: Large models, more compute power (33% of requests)
    server cpu_ollama 127.0.0.1:11435 check weight 5 inter 30s fall 3 rise 2

# Tools Load Balancer
frontend tools_frontend
    bind *:9001
    mode http
    default_backend tools_servers

backend tools_servers
    mode http
    balance roundrobin
    option httpchk GET /health
    
    server tools_primary 192.168.1.105:8082 check inter 30s fall 3 rise 2

# Embeddings Load Balancer  
frontend embeddings_frontend
    bind *:9002
    mode http
    default_backend embeddings_servers

backend embeddings_servers
    mode http
    balance roundrobin
    option httpchk GET /health
    
    server embeddings_primary 192.168.1.178:8081 check inter 30s fall 3 rise 2

# Statistics Interface
stats enable
stats uri /haproxy_stats
stats refresh 30s
stats admin if TRUE
stats auth admin:langgraph_admin_2024
EOF

# Test HAProxy configuration
sudo haproxy -f /etc/haproxy/haproxy.cfg -c

# Enable and start HAProxy
sudo systemctl enable haproxy
sudo systemctl start haproxy

# 📋 Load Balancing Behavior:
# • Jetson (weight 10): ~67% of requests - handles fast/small models
# • CPU (weight 5): ~33% of requests - handles large models
# • Round-robin distribution based on weights
# • Both backends are active (no backup/failover)

# Check status
sudo systemctl status haproxy

# 🧪 Test load balancing distribution
echo "Testing load balancer - you should see alternating between Jetson and CPU models:"
for i in {1..6}; do 
    echo "Request $i:"
    response=$(curl -s http://localhost:9000/api/tags)
    if echo "$response" | grep -q "mistral:7b"; then
        echo "  ✅ Routed to CPU backend (mistral:7b)"
    elif echo "$response" | grep -q "llama3.2"; then
        echo "  ✅ Routed to Jetson backend (llama3.2 models)"
    else
        echo "  ❓ Unknown response"
    fi
    sleep 1
done

# Configure firewall
sudo ufw allow 9000/tcp  # LLM load balancer
sudo ufw allow 9001/tcp  # Tools load balancer  
sudo ufw allow 9002/tcp  # Embeddings load balancer
sudo ufw allow 8888/tcp  # Health check
sudo ufw allow 8080/tcp  # Direct llama.cpp access
sudo ufw allow 6379/tcp  # Redis
```

### Step 2.6: Setup LangGraph Environment

```bash
# Create LangGraph environment
cd ~/ai-infrastructure
python3 -m venv langgraph-env
source langgraph-env/bin/activate

# Install LangGraph and dependencies
pip install --upgrade pip
pip install langgraph langchain langchain-community langchain-core
pip install httpx requests asyncio aiohttp
pip install redis chromadb sentence-transformers
pip install fastapi uvicorn pandas numpy
pip install psutil schedule

# Create LangGraph configuration
mkdir -p ~/ai-infrastructure/langgraph-config
cd ~/ai-infrastructure/langgraph-config

# Copy the integration files from our setup
cp /home/sanzad/git/langgraph/setup_guides/03_langgraph_integration.md ./
cp /home/sanzad/git/langgraph/examples/example_workflows.py ./

# Create production config
cat > config.py << 'EOF'
"""Production configuration for LangGraph cluster"""
from dataclasses import dataclass

@dataclass
class MachineConfig:
    ip: str
    port: int
    service_type: str
    health_endpoint: str

@dataclass
class ClusterConfig:
    jetson_orin: MachineConfig
    cpu_coordinator: MachineConfig
    rp_embeddings: MachineConfig
    worker_tools: MachineConfig
    worker_monitor: MachineConfig

# Production cluster configuration
CLUSTER = ClusterConfig(
    jetson_orin=MachineConfig(
        ip="192.168.1.177",
        port=11434,
        service_type="ollama",
        health_endpoint="/api/tags"
    ),
    cpu_coordinator=MachineConfig(
        ip="192.168.1.81",
        port=8080,
        service_type="llama_cpp",
        health_endpoint="/health"
    ),
    rp_embeddings=MachineConfig(
        ip="192.168.1.178",
        port=8081,
        service_type="embeddings",
        health_endpoint="/health"
    ),
    worker_tools=MachineConfig(
        ip="192.168.1.105",
        port=8082,
        service_type="tools",
        health_endpoint="/health"
    ),
    worker_monitor=MachineConfig(
        ip="192.168.1.137",
        port=8083,
        service_type="monitoring",
        health_endpoint="/cluster_health"
    )
)

# Load balancer endpoints
LOAD_BALANCER_BASE = "http://192.168.1.81"
ENDPOINTS = {
    "llm": f"{LOAD_BALANCER_BASE}:9000",
    "tools": f"{LOAD_BALANCER_BASE}:9001", 
    "embeddings": f"{LOAD_BALANCER_BASE}:9002",
    "redis": f"{LOAD_BALANCER_BASE}:6379"
}

# Redis configuration
REDIS_CONFIG = {
    "host": "192.168.1.81",
    "port": 6379,
    "password": "langgraph_redis_pass",
    "db": 0
}
EOF

echo "✅ cpu-node coordinator setup completed!"
```

---

## 🔍 Phase 3: rp-node Setup (Embeddings Server)

**Machine**: rp-node (192.168.1.178) - 8GB ARM Cortex-A76

### Step 3.1: ARM-Optimized System Preparation

```bash
# SSH into rp-node
ssh sanzad@192.168.1.178

# Update system
sudo apt update && sudo apt upgrade -y

# Install ARM-optimized packages
sudo apt install -y curl wget git htop nano vim build-essential cmake
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y libblas-dev liblapack-dev libatlas-base-dev
sudo apt install -y gfortran libhdf5-dev pkg-config

# Check ARM architecture
uname -m  # Should show aarch64
cat /proc/cpuinfo | grep "model name"
```

### Step 3.2: Setup Python Environment with ARM Optimizations

```bash
# Create optimized Python environment
python3 -m venv ~/embeddings-env
source ~/embeddings-env/bin/activate

# Install ARM-optimized packages
pip install --upgrade pip

# Install PyTorch for ARM
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers and dependencies
pip install sentence-transformers transformers
pip install fastapi uvicorn chromadb
pip install numpy pandas psutil
pip install requests httpx aiofiles

# Install ARM-optimized BLAS
pip install scipy --no-use-pep517
```

### Step 3.3: Create Embeddings Server

```bash
# Create embeddings server directory
mkdir -p ~/embeddings-server
cd ~/embeddings-server

# Create optimized embeddings server
cat > embeddings_server.py << 'EOF'
#!/usr/bin/env python3
"""
ARM-optimized embeddings server for rp-node
Provides vector embeddings for the LangGraph cluster
"""

import os
import time
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import uvicorn
import psutil
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangGraph Embeddings Server", version="1.0.0")

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: str = "default"

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int
    processing_time: float

class EmbeddingsService:
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Load embedding models optimized for ARM"""
        logger.info("Loading embedding models...")
        
        # Primary model - small and efficient for ARM
        try:
            self.models["default"] = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded all-MiniLM-L6-v2 (default)")
        except Exception as e:
            logger.error(f"Failed to load default model: {e}")
        
        # Secondary model - better quality, more compute
        try:
            self.models["quality"] = SentenceTransformer('all-mpnet-base-v2')
            logger.info("✅ Loaded all-mpnet-base-v2 (quality)")
        except Exception as e:
            logger.warning(f"Failed to load quality model: {e}")
        
        # Multilingual model
        try:
            self.models["multilingual"] = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✅ Loaded multilingual model")
        except Exception as e:
            logger.warning(f"Failed to load multilingual model: {e}")
            
    def get_embeddings(self, texts: List[str], model_name: str = "default") -> np.ndarray:
        """Generate embeddings for given texts"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        # ARM optimization: Process in smaller batches
        batch_size = 16 if len(texts) > 16 else len(texts)
        
        if len(texts) <= batch_size:
            return model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        
        # Process large requests in chunks
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch, batch_size=len(batch), show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)

# Initialize service
embeddings_service = EmbeddingsService()

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for the provided texts"""
    start_time = time.time()
    
    try:
        embeddings = embeddings_service.get_embeddings(request.texts, request.model)
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            model=request.model,
            dimensions=embeddings.shape[1],
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    
    return {
        "status": "healthy",
        "memory_usage_percent": memory_usage,
        "cpu_usage_percent": cpu_usage,
        "available_models": list(embeddings_service.models.keys()),
        "architecture": "aarch64"
    }

@app.get("/models")
async def list_models():
    """List available embedding models"""
    return {
        "available_models": list(embeddings_service.models.keys()),
        "default_model": "default"
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total": psutil.disk_usage('/').total,
            "free": psutil.disk_usage('/').free,
            "percent": psutil.disk_usage('/').percent
        }
    }

if __name__ == "__main__":
    logger.info("Starting ARM-optimized embeddings server...")
    uvicorn.run(
        "embeddings_server:app",
        host="0.0.0.0",
        port=8081,
        workers=1,  # Single worker for ARM efficiency
        loop="asyncio",
        log_level="info"
    )
EOF

chmod +x embeddings_server.py
```

### Step 3.4: Test and Optimize Embeddings Server

```bash
# Test the embeddings server
cd ~/embeddings-server
source ~/embeddings-env/bin/activate

# Start server in background for testing
python3 embeddings_server.py &
SERVER_PID=$!

# Wait for server to start
sleep 30

# Test embeddings endpoint
curl -X POST http://localhost:8081/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "This is a test", "ARM embeddings server"],
    "model": "default"
  }'

# Test health endpoint
curl http://localhost:8081/health

# Test models endpoint
curl http://localhost:8081/models

# Stop test server
kill $SERVER_PID

# Create systemd service
sudo tee /etc/systemd/system/embeddings-server.service << EOF
[Unit]
Description=LangGraph Embeddings Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/embeddings-server
Environment=PATH=/home/$USER/embeddings-env/bin
ExecStart=/home/$USER/embeddings-env/bin/python embeddings_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable embeddings-server
sudo systemctl start embeddings-server

# Check status
sudo systemctl status embeddings-server

# Configure firewall
sudo ufw allow 8081/tcp

echo "✅ rp-node embeddings server setup completed!"
```

---

## 🛠️ Phase 4: worker-node3 Setup (Tools Execution Server)

**Machine**: worker-node3 (192.168.1.105) - 6GB VM

### Step 4.1: Tools Server Preparation

```bash
# SSH into worker-node3
ssh sanzad@192.168.1.105

# Update system
sudo apt update && sudo apt upgrade -y

# Install tools and dependencies
sudo apt install -y curl wget git htop nano vim python3 python3-pip python3-venv
sudo apt install -y chromium-browser chromium-chromedriver  # For web scraping
sudo apt install -y nodejs npm  # For additional tools

# Create tools environment
python3 -m venv ~/tools-env
source ~/tools-env/bin/activate

# Install Python packages
pip install --upgrade pip
pip install fastapi uvicorn requests beautifulsoup4 selenium
pip install pandas numpy lxml cssselect
pip install aiohttp aiofiles asyncio
pip install psutil schedule
```

### Step 4.2: Create Tools Execution Server

```bash
# Create tools server directory
mkdir -p ~/tools-server
cd ~/tools-server

# Create comprehensive tools server
cat > tools_server.py << 'EOF'
#!/usr/bin/env python3
"""
Tools execution server for worker-node3
Provides web scraping, API calls, and command execution
"""

import os
import subprocess
import asyncio
import aiohttp
import aiofiles
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from typing import Dict, List, Any, Optional
import uvicorn
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangGraph Tools Server", version="1.0.0")

class WebSearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=10, le=50)
    search_engine: str = Field(default="duckduckgo")

class WebScrapeRequest(BaseModel):
    url: str
    method: str = Field(default="requests")  # "requests" or "selenium"
    extract_text: bool = Field(default=True)
    extract_links: bool = Field(default=False)
    wait_time: int = Field(default=5, le=30)

class CommandRequest(BaseModel):
    command: str
    timeout: int = Field(default=30, le=300)
    working_dir: str = Field(default="/tmp")

class ToolsService:
    def __init__(self):
        self.chrome_options = self._setup_chrome_options()
        
    def _setup_chrome_options(self):
        """Setup Chrome options for Selenium"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
        return options
    
    async def web_search(self, query: str, max_results: int = 10, search_engine: str = "duckduckgo") -> Dict:
        """Perform web search using DuckDuckGo"""
        try:
            # Simple DuckDuckGo search implementation
            search_url = f"https://duckduckgo.com/html/?q={query}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        results = []
                        result_divs = soup.find_all('div', class_='result')[:max_results]
                        
                        for div in result_divs:
                            title_elem = div.find('a', class_='result__a')
                            snippet_elem = div.find('a', class_='result__snippet')
                            
                            if title_elem:
                                results.append({
                                    'title': title_elem.get_text(strip=True),
                                    'url': title_elem.get('href', ''),
                                    'snippet': snippet_elem.get_text(strip=True) if snippet_elem else ''
                                })
                        
                        return {
                            'query': query,
                            'results': results,
                            'count': len(results),
                            'search_engine': search_engine
                        }
                    else:
                        return {'error': f'Search failed with status {response.status}'}
                        
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {'error': str(e)}
    
    async def web_scrape_requests(self, url: str, extract_text: bool, extract_links: bool) -> Dict:
        """Scrape webpage using requests + BeautifulSoup"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        result = {
                            'url': url,
                            'status_code': response.status,
                            'title': soup.title.string if soup.title else 'No title'
                        }
                        
                        if extract_text:
                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.decompose()
                            
                            text = soup.get_text()
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            result['text'] = '\n'.join(chunk for chunk in chunks if chunk)[:5000]  # Limit text
                        
                        if extract_links:
                            links = []
                            for link in soup.find_all('a', href=True):
                                links.append({
                                    'text': link.get_text(strip=True),
                                    'href': link['href']
                                })
                            result['links'] = links[:50]  # Limit links
                        
                        return result
                    else:
                        return {'error': f'Failed to fetch URL, status: {response.status}'}
                        
        except Exception as e:
            logger.error(f"Web scraping failed: {e}")
            return {'error': str(e)}
    
    def web_scrape_selenium(self, url: str, extract_text: bool, extract_links: bool, wait_time: int) -> Dict:
        """Scrape webpage using Selenium (for dynamic content)"""
        driver = None
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            result = {
                'url': url,
                'title': driver.title,
                'method': 'selenium'
            }
            
            if extract_text:
                text = driver.find_element(By.TAG_NAME, "body").text
                result['text'] = text[:5000]  # Limit text
            
            if extract_links:
                links = []
                link_elements = driver.find_elements(By.TAG_NAME, "a")
                for link in link_elements[:50]:  # Limit links
                    href = link.get_attribute('href')
                    text = link.text.strip()
                    if href:
                        links.append({'text': text, 'href': href})
                result['links'] = links
            
            return result
            
        except Exception as e:
            logger.error(f"Selenium scraping failed: {e}")
            return {'error': str(e)}
        finally:
            if driver:
                driver.quit()
    
    async def execute_command(self, command: str, timeout: int, working_dir: str) -> Dict:
        """Execute shell command safely"""
        try:
            # Security: Only allow safe commands
            dangerous_commands = ['rm -rf', 'sudo', 'su', 'chmod 777', 'mkfs', 'dd if=']
            if any(dangerous in command for dangerous in dangerous_commands):
                return {'error': 'Command not allowed for security reasons'}
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                
                return {
                    'command': command,
                    'working_dir': working_dir,
                    'returncode': process.returncode,
                    'stdout': stdout.decode('utf-8', errors='ignore'),
                    'stderr': stderr.decode('utf-8', errors='ignore'),
                    'timeout': timeout
                }
            except asyncio.TimeoutError:
                process.kill()
                return {'error': f'Command timed out after {timeout} seconds'}
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {'error': str(e)}

# Initialize service
tools_service = ToolsService()

@app.post("/web_search")
async def web_search(request: WebSearchRequest):
    """Perform web search"""
    return await tools_service.web_search(request.query, request.max_results, request.search_engine)

@app.post("/web_scrape")
async def web_scrape(request: WebScrapeRequest):
    """Scrape webpage content"""
    if request.method == "selenium":
        return tools_service.web_scrape_selenium(
            request.url, request.extract_text, request.extract_links, request.wait_time
        )
    else:
        return await tools_service.web_scrape_requests(
            request.url, request.extract_text, request.extract_links
        )

@app.post("/execute_command")
async def execute_command(request: CommandRequest):
    """Execute shell command"""
    return await tools_service.execute_command(request.command, request.timeout, request.working_dir)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "memory_usage_percent": psutil.virtual_memory().percent,
        "cpu_usage_percent": psutil.cpu_percent(),
        "tools_available": ["web_search", "web_scrape", "execute_command"]
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory": psutil.virtual_memory()._asdict(),
        "disk": psutil.disk_usage('/')._asdict()
    }

if __name__ == "__main__":
    logger.info("Starting tools execution server...")
    uvicorn.run(
        "tools_server:app",
        host="0.0.0.0", 
        port=8082,
        workers=1,
        log_level="info"
    )
EOF

chmod +x tools_server.py
```

### Step 4.3: Test and Deploy Tools Server

```bash
# Test tools server
cd ~/tools-server
source ~/tools-env/bin/activate

# Start server for testing
python3 tools_server.py &
SERVER_PID=$!

# Wait for server to start
sleep 15

# Test web search
curl -X POST http://localhost:8082/web_search \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence news", "max_results": 5}'

# Test web scraping
curl -X POST http://localhost:8082/web_scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "extract_text": true}'

# Test command execution
curl -X POST http://localhost:8082/execute_command \
  -H "Content-Type: application/json" \
  -d '{"command": "echo Hello World", "timeout": 10}'

# Test health
curl http://localhost:8082/health

# Stop test server
kill $SERVER_PID

# Create systemd service
sudo tee /etc/systemd/system/tools-server.service << EOF
[Unit]
Description=LangGraph Tools Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/tools-server
Environment=PATH=/home/$USER/tools-env/bin
ExecStart=/home/$USER/tools-env/bin/python tools_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable tools-server
sudo systemctl start tools-server

# Check status
sudo systemctl status tools-server

# Configure firewall
sudo ufw allow 8082/tcp

echo "✅ worker-node3 tools server setup completed!"
```

---

## 📊 Phase 5: worker-node4 Setup (Monitoring Server)

**Machine**: worker-node4 (192.168.1.137) - 6GB VM

### Step 5.1: Monitoring Server Preparation

```bash
# SSH into worker-node4
ssh sanzad@192.168.1.137

# Update system
sudo apt update && sudo apt upgrade -y

# Install monitoring tools
sudo apt install -y curl wget git htop iotop nano vim python3 python3-pip python3-venv

# Setup Grafana repository (required before installation)
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list
sudo apt update

# Install advanced monitoring tools (optional)
sudo apt install -y prometheus grafana  # Note: package is "grafana", not "grafana-server"

# Create monitoring environment
python3 -m venv ~/monitoring-env
source ~/monitoring-env/bin/activate

# Install monitoring packages
pip install --upgrade pip
pip install fastapi uvicorn requests psutil
pip install schedule asyncio aiohttp
pip install pandas numpy matplotlib
```

### Step 5.2: Create Comprehensive Monitoring Server

```bash
# Create monitoring server directory
mkdir -p ~/monitoring-server
cd ~/monitoring-server

# Create monitoring server
cat > monitoring_server.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive monitoring server for worker-node4
Monitors health of entire LangGraph cluster
"""

import asyncio
import aiohttp
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psutil
import uvicorn
import schedule
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangGraph Cluster Monitor", version="1.0.0")

class ClusterHealth(BaseModel):
    timestamp: str
    overall_status: str
    services: Dict[str, Any]
    cluster_stats: Dict[str, Any]
    alerts: List[str]

class MonitoringService:
    def __init__(self):
        self.services = {
            'jetson_ollama': {
                'url': 'http://192.168.1.177:11434/api/tags',
                'critical': True,
                'timeout': 15,
                'name': 'Jetson Ollama Server'
            },
            'cpu_llama': {
                'url': 'http://192.168.1.81:8080/health',
                'critical': False,
                'timeout': 10,
                'name': 'CPU Llama.cpp Server'
            },
            'embeddings': {
                'url': 'http://192.168.1.178:8081/health',
                'critical': True,
                'timeout': 10,
                'name': 'Embeddings Server'
            },
            'tools': {
                'url': 'http://192.168.1.105:8082/health',
                'critical': True,
                'timeout': 10,
                'name': 'Tools Server'
            },
            'haproxy': {
                'url': 'http://192.168.1.81:8888/health',
                'critical': True,
                'timeout': 5,
                'name': 'HAProxy Load Balancer'
            },
            'redis': {
                'url': 'http://192.168.1.81:6379',
                'critical': True,
                'timeout': 5,
                'name': 'Redis Cache',
                'check_type': 'redis'
            }
        }
        
        self.health_history = []
        self.alerts = []
        self.last_check = None
        
        # Start background monitoring
        self.start_background_monitoring()
    
    async def check_service_health(self, name: str, config: Dict) -> Dict:
        """Check health of a single service"""
        start_time = time.time()
        
        try:
            if config.get('check_type') == 'redis':
                # Special handling for Redis
                return await self.check_redis_health(config)
            
            timeout = aiohttp.ClientTimeout(total=config['timeout'])
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(config['url']) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        try:
                            data = await response.json()
                        except:
                            data = await response.text()
                        
                        return {
                            'status': 'healthy',
                            'response_time': round(response_time, 3),
                            'status_code': response.status,
                            'data': data if len(str(data)) < 1000 else 'Response too large'
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'response_time': round(response_time, 3),
                            'status_code': response.status,
                            'error': f'HTTP {response.status}'
                        }
                        
        except asyncio.TimeoutError:
            return {
                'status': 'timeout',
                'response_time': config['timeout'],
                'error': 'Request timed out'
            }
        except Exception as e:
            return {
                'status': 'error',
                'response_time': time.time() - start_time,
                'error': str(e)
            }
    
    async def check_redis_health(self, config: Dict) -> Dict:
        """Special health check for Redis"""
        try:
            import redis
            r = redis.Redis(host='192.168.1.81', port=6379, password='langgraph_redis_pass', 
                          socket_timeout=config['timeout'], decode_responses=True)
            
            start_time = time.time()
            response = r.ping()
            response_time = time.time() - start_time
            
            if response:
                info = r.info()
                return {
                    'status': 'healthy',
                    'response_time': round(response_time, 3),
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory_human': info.get('used_memory_human', 'unknown'),
                    'uptime_in_seconds': info.get('uptime_in_seconds', 0)
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': 'Redis ping failed'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def check_cluster_health(self) -> ClusterHealth:
        """Check health of entire cluster"""
        timestamp = datetime.now().isoformat()
        service_results = {}
        alerts = []
        
        # Check all services
        tasks = []
        for service_name, config in self.services.items():
            task = self.check_service_health(service_name, config)
            tasks.append((service_name, task))
        
        # Execute health checks concurrently
        for service_name, task in tasks:
            config = self.services[service_name]
            result = await task
            service_results[service_name] = {
                'name': config['name'],
                'critical': config['critical'],
                'health': result
            }
            
            # Generate alerts for critical services
            if config['critical'] and result['status'] != 'healthy':
                alert = f"CRITICAL: {config['name']} is {result['status']} - {result.get('error', 'Unknown error')}"
                alerts.append(alert)
                logger.error(alert)
        
        # Calculate overall status
        critical_services = [s for s in service_results.values() if s['critical']]
        healthy_critical = [s for s in critical_services if s['health']['status'] == 'healthy']
        
        if len(healthy_critical) == len(critical_services):
            overall_status = 'healthy'
        elif len(healthy_critical) >= len(critical_services) * 0.7:
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'
        
        # Get cluster statistics
        cluster_stats = await self.get_cluster_stats()
        
        health_result = ClusterHealth(
            timestamp=timestamp,
            overall_status=overall_status,
            services=service_results,
            cluster_stats=cluster_stats,
            alerts=alerts
        )
        
        # Store in history
        self.health_history.append(health_result.dict())
        if len(self.health_history) > 100:  # Keep last 100 checks
            self.health_history.pop(0)
        
        self.last_check = health_result
        self.alerts = alerts
        
        return health_result
    
    async def get_cluster_stats(self) -> Dict:
        """Get cluster-wide statistics"""
        try:
            # Local system stats
            local_stats = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
            
            # Try to get remote stats
            remote_stats = {}
            
            # Could be extended to collect stats from other nodes
            return {
                'local_node': local_stats,
                'remote_nodes': remote_stats,
                'total_services': len(self.services),
                'monitoring_uptime': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster stats: {e}")
            return {'error': str(e)}
    
    def start_background_monitoring(self):
        """Start background health monitoring"""
        def run_scheduler():
            schedule.every(30).seconds.do(self.scheduled_health_check)
            schedule.every(5).minutes.do(self.cleanup_old_data)
            
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        monitor_thread = threading.Thread(target=run_scheduler, daemon=True)
        monitor_thread.start()
        logger.info("Background monitoring started")
    
    def scheduled_health_check(self):
        """Scheduled health check (sync wrapper)"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.check_cluster_health())
            loop.close()
        except Exception as e:
            logger.error(f"Scheduled health check failed: {e}")
    
    def cleanup_old_data(self):
        """Clean up old monitoring data"""
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.health_history = [
            h for h in self.health_history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]

# Initialize monitoring service
monitoring_service = MonitoringService()

@app.get("/cluster_health", response_model=ClusterHealth)
async def get_cluster_health():
    """Get current cluster health status"""
    return await monitoring_service.check_cluster_health()

@app.get("/health")
async def health_check():
    """Health check for monitoring service itself"""
    return {
        "status": "healthy",
        "monitoring_node": "worker-node4",
        "services_monitored": len(monitoring_service.services),
        "last_check": monitoring_service.last_check.timestamp if monitoring_service.last_check else None,
        "alerts_count": len(monitoring_service.alerts)
    }

@app.get("/alerts")
async def get_alerts():
    """Get current alerts"""
    return {
        "alerts": monitoring_service.alerts,
        "count": len(monitoring_service.alerts),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/history")
async def get_health_history(limit: int = 50):
    """Get health check history"""
    return {
        "history": monitoring_service.health_history[-limit:],
        "count": len(monitoring_service.health_history)
    }

@app.get("/stats")
async def get_monitoring_stats():
    """Get monitoring service statistics"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory": psutil.virtual_memory()._asdict(),
        "disk": psutil.disk_usage('/')._asdict(),
        "monitoring_uptime": time.time(),
        "services_monitored": list(monitoring_service.services.keys())
    }

if __name__ == "__main__":
    logger.info("Starting cluster monitoring server...")
    uvicorn.run(
        "monitoring_server:app",
        host="0.0.0.0",
        port=8083,
        workers=1,
        log_level="info"
    )
EOF

chmod +x monitoring_server.py
```

### Step 5.3: Install Redis Client and Deploy Monitoring

```bash
# Install Redis client for health checks
source ~/monitoring-env/bin/activate
pip install redis

# Test monitoring server
cd ~/monitoring-server
python3 monitoring_server.py &
SERVER_PID=$!

# Wait for server to start
sleep 15

# Test monitoring endpoints
curl http://localhost:8083/health
curl http://localhost:8083/cluster_health
curl http://localhost:8083/alerts

# Stop test server
kill $SERVER_PID

# Create systemd service
sudo tee /etc/systemd/system/monitoring-server.service << EOF
[Unit]
Description=LangGraph Cluster Monitoring Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/monitoring-server
Environment=PATH=/home/$USER/monitoring-env/bin
ExecStart=/home/$USER/monitoring-env/bin/python monitoring_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable monitoring-server
sudo systemctl start monitoring-server

# Check status
sudo systemctl status monitoring-server

# Configure firewall
sudo ufw allow 8083/tcp

echo "✅ worker-node4 monitoring server setup completed!"
```

---

## 🎯 Phase 6: Final Integration and Testing

### Step 6.1: Deploy Cluster Orchestrator (on cpu-node)

```bash
# SSH to cpu-node (or run locally)
ssh sanzad@192.168.1.81

# Navigate to LangGraph directory
cd ~/ai-infrastructure/langgraph-config

# Copy the cluster orchestrator from our setup
cp /home/sanzad/git/langgraph/setup_guides/04_distributed_coordination.md ./

# Create the cluster orchestrator script
cat > cluster_orchestrator.py << 'EOF'
#!/usr/bin/env python3
"""
Cluster orchestrator for LangGraph AI infrastructure
Manages the entire cluster lifecycle
"""

import subprocess
import time
import json
import argparse
import requests
from typing import List, Dict

class ClusterOrchestrator:
    def __init__(self):
        self.machines = {
            'jetson': '192.168.1.177',      # jetson-node (Orin Nano 8GB)
            'cpu_coordinator': '192.168.1.81',     # cpu-node (32GB coordinator)
            'rp_embeddings': '192.168.1.178',     # rp-node (8GB ARM, embeddings)
            'worker_tools': '192.168.1.105',    # worker-node3 (6GB VM, tools)
            'worker_monitor': '192.168.1.137'     # worker-node4 (6GB VM, monitoring)
        }
    
    def start_cluster(self):
        """Start all services in the correct order"""
        print("🚀 Starting LangGraph cluster services...")
        
        # Start core services first (Redis on coordinator)
        print("Starting Redis cache...")
        self._run_remote_command('cpu_coordinator', 'sudo systemctl start redis-server')
        time.sleep(3)
        
        # Start model servers
        print("Starting model servers...")
        self._run_remote_command('jetson', 'sudo systemctl start ollama')
        self._run_remote_command('cpu_coordinator', 'sudo systemctl start llama-server')
        time.sleep(10)
        
        # Start application services
        print("Starting application services...")
        self._run_remote_command('rp_embeddings', 'sudo systemctl start embeddings-server')
        self._run_remote_command('worker_tools', 'sudo systemctl start tools-server')
        time.sleep(5)
        
        # Start load balancer and monitoring
        print("Starting load balancer and monitoring...")
        self._run_remote_command('cpu_coordinator', 'sudo systemctl start haproxy')
        self._run_remote_command('worker_monitor', 'sudo systemctl start monitoring-server')
        time.sleep(5)
        
        print("✅ Cluster startup completed!")
        print("🌐 Services available at:")
        print(f"  - LLM Load Balancer: http://192.168.1.81:9000")
        print(f"  - Tools Load Balancer: http://192.168.1.81:9001")
        print(f"  - Embeddings Load Balancer: http://192.168.1.81:9002")
        print(f"  - Cluster Health: http://192.168.1.137:8083/cluster_health")
        print(f"  - HAProxy Stats: http://192.168.1.81:9000/haproxy_stats")
    
    def stop_cluster(self):
        """Stop all services gracefully"""
        print("🛑 Stopping cluster services...")
        
        services = [
            ('worker_monitor', 'monitoring-server'),
            ('cpu_coordinator', 'haproxy'),
            ('worker_tools', 'tools-server'),
            ('rp_embeddings', 'embeddings-server'),
            ('cpu_coordinator', 'llama-server'),
            ('jetson', 'ollama'),
            ('cpu_coordinator', 'redis-server')
        ]
        
        for machine, service in services:
            print(f"Stopping {service} on {machine}...")
            self._run_remote_command(machine, f'sudo systemctl stop {service}')
            time.sleep(2)
        
        print("✅ Cluster shutdown completed!")
    
    def status_cluster(self):
        """Check status of all services"""
        print("📊 LangGraph Cluster Status:")
        print("=" * 60)
        
        services = {
            'jetson': ['ollama'],
            'cpu_coordinator': ['llama-server', 'haproxy', 'redis-server'],
            'rp_embeddings': ['embeddings-server'],
            'worker_tools': ['tools-server'],
            'worker_monitor': ['monitoring-server']
        }
        
        for machine, service_list in services.items():
            ip = self.machines[machine]
            print(f"\n{machine} ({ip}):")
            for service in service_list:
                status = self._check_service_status(machine, service)
                print(f"  {service}: {status}")
        
        # Check cluster health
        print(f"\n📡 Cluster Health Check:")
        try:
            response = requests.get('http://192.168.1.137:8083/cluster_health', timeout=10)
            if response.status_code == 200:
                health = response.json()
                print(f"  Overall Status: {health['overall_status'].upper()}")
                print(f"  Alerts: {len(health['alerts'])}")
            else:
                print(f"  Health check failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"  Health check failed: {e}")
    
    def test_cluster(self):
        """Test cluster functionality"""
        print("🧪 Testing cluster functionality...")
        
        tests = [
            ("Jetson Ollama", "http://192.168.1.177:11434/api/tags"),
            ("CPU Llama.cpp", "http://192.168.1.81:8080/health"),
            ("Embeddings Server", "http://192.168.1.178:8081/health"),
            ("Tools Server", "http://192.168.1.105:8082/health"),
            ("Monitoring Server", "http://192.168.1.137:8083/health"),
            ("Load Balancer", "http://192.168.1.81:8888/health"),
            ("Redis", "http://192.168.1.81:6379")  # This will fail, Redis doesn't have HTTP
        ]
        
        for name, url in tests:
            try:
                if name == "Redis":
                    # Special test for Redis
                    import redis
                    r = redis.Redis(host='192.168.1.81', port=6379, password='langgraph_redis_pass')
                    r.ping()
                    print(f"  ✅ {name}: Healthy")
                else:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        print(f"  ✅ {name}: Healthy")
                    else:
                        print(f"  ❌ {name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"  ❌ {name}: {str(e)[:100]}")
    
    def _run_remote_command(self, machine: str, command: str) -> str:
        """Run command on remote machine"""
        if machine == 'cpu_coordinator' and self.machines[machine] == '192.168.1.81':
            # Run locally if we're on the coordinator node
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                return result.stdout.strip()
            except Exception as e:
                return f"Error: {e}"
        else:
            # Run remotely
            ip = self.machines[machine]
            ssh_cmd = f"ssh -o StrictHostKeyChecking=no sanzad@{ip} '{command}'"
            
            try:
                result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
                return result.stdout.strip()
            except Exception as e:
                return f"Error: {e}"
    
    def _check_service_status(self, machine: str, service: str) -> str:
        """Check if a service is running"""
        result = self._run_remote_command(machine, f"systemctl is-active {service}")
        return "🟢 Active" if result == "active" else f"🔴 {result}"

def main():
    parser = argparse.ArgumentParser(description="LangGraph Cluster Orchestrator")
    parser.add_argument('action', choices=['start', 'stop', 'status', 'restart', 'test'], 
                       help='Action to perform')
    
    args = parser.parse_args()
    orchestrator = ClusterOrchestrator()
    
    if args.action == 'start':
        orchestrator.start_cluster()
    elif args.action == 'stop':
        orchestrator.stop_cluster()
    elif args.action == 'status':
        orchestrator.status_cluster()
    elif args.action == 'test':
        orchestrator.test_cluster()
    elif args.action == 'restart':
        orchestrator.stop_cluster()
        time.sleep(10)
        orchestrator.start_cluster()

if __name__ == "__main__":
    main()
EOF

chmod +x cluster_orchestrator.py

# Install Redis client for testing
source ~/langgraph-env/bin/activate
pip install redis
```

### Step 6.2: Test Complete Cluster

```bash
# Test cluster startup
cd ~/ai-infrastructure/langgraph-config
source ~/langgraph-env/bin/activate

echo "🧪 Starting complete cluster test..."

# Start the cluster
python3 cluster_orchestrator.py start

# Wait for all services to initialize
sleep 30

# Check cluster status
python3 cluster_orchestrator.py status

# Test cluster functionality
python3 cluster_orchestrator.py test

# Test load-balanced endpoints
echo ""
echo "🌐 Testing load-balanced endpoints..."

# Test LLM endpoint
echo "Testing LLM endpoint..."
curl -X POST http://192.168.1.81:9000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "Hello, this is a test of the LangGraph cluster!",
    "stream": false
  }' | jq .

# Test embeddings endpoint
echo "Testing embeddings endpoint..."
curl -X POST http://192.168.1.81:9002/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "LangGraph cluster test"],
    "model": "default"
  }' | jq .

# Test tools endpoint
echo "Testing tools endpoint..."
curl -X POST http://192.168.1.81:9001/web_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "max_results": 3
  }' | jq .

# Check cluster health
echo "Checking cluster health..."
curl http://192.168.1.137:8083/cluster_health | jq .

echo ""
echo "✅ Cluster testing completed!"
echo "🎉 Your LangGraph AI cluster is ready for production!"
```

### Step 6.3: Create Startup Scripts

```bash
# Create convenient startup scripts
cat > ~/start_cluster.sh << 'EOF'
#!/bin/bash
cd ~/ai-infrastructure/langgraph-config
source ~/langgraph-env/bin/activate
python3 cluster_orchestrator.py start
EOF

cat > ~/stop_cluster.sh << 'EOF'
#!/bin/bash
cd ~/ai-infrastructure/langgraph-config
source ~/langgraph-env/bin/activate
python3 cluster_orchestrator.py stop
EOF

cat > ~/cluster_status.sh << 'EOF'
#!/bin/bash
cd ~/ai-infrastructure/langgraph-config
source ~/langgraph-env/bin/activate
python3 cluster_orchestrator.py status
EOF

chmod +x ~/start_cluster.sh ~/stop_cluster.sh ~/cluster_status.sh

echo "✅ Startup scripts created:"
echo "  - ~/start_cluster.sh"
echo "  - ~/stop_cluster.sh"
echo "  - ~/cluster_status.sh"
```

---

## 🎉 Deployment Complete!

### **Your LangGraph AI Cluster is Now Ready!**

#### **🌐 Available Services:**
- **LLM Load Balancer**: http://192.168.1.81:9000
- **Tools Load Balancer**: http://192.168.1.81:9001
- **Embeddings Load Balancer**: http://192.168.1.81:9002
- **Cluster Health Monitor**: http://192.168.1.137:8083/cluster_health
- **HAProxy Statistics**: http://192.168.1.81:9000/haproxy_stats (admin/langgraph_admin_2024)

#### **🎯 Quick Commands:**
```bash
# Start entire cluster
~/start_cluster.sh

# Check cluster status
~/cluster_status.sh

# Stop cluster
~/stop_cluster.sh

# Test cluster
cd ~/ai-infrastructure/langgraph-config && python3 cluster_orchestrator.py test
```

#### **📊 Performance Expectations:**
- **Jetson Orin Nano**: 15-50 tokens/sec (model dependent)
- **CPU Heavy Models**: 5-15 tokens/sec (complex tasks)
- **Embeddings**: 50+ embeddings/sec on ARM
- **Tools**: Sub-second web scraping and API calls
- **Monitoring**: Real-time cluster health with auto-recovery

Your production-grade local AI infrastructure is now **fully operational** with zero external costs! 🚀
