# Jetson Orin Nano Setup Guide

## Overview
**Machine**: jetson-node (192.168.1.177) - Jetson Orin Nano 8GB  
Your Jetson Orin Nano serves as the **primary LLM inference server**, optimized for small-to-medium models with production-grade performance and advanced optimizations.

## Prerequisites
- Jetson Orin Nano with JetPack 5.1+ installed
- SSH access configured (`ssh sanzad@192.168.1.177`)
- Internet connection for initial setup

---

## Step 1: Initial System Preparation

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

# Fix CUDA PATH if nvcc not found
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
    
    # Add CUDA to PATH
    echo "export PATH=${CUDA_PATH}/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export CUDA_HOME=${CUDA_PATH}" >> ~/.bashrc
    
    # Create symlink if needed
    if [ ! -L "/usr/local/cuda" ] && [ "$CUDA_PATH" != "/usr/local/cuda" ]; then
        sudo ln -sf $CUDA_PATH /usr/local/cuda
    fi
    
    # Reload environment
    source ~/.bashrc
    export PATH=${CUDA_PATH}/bin:$PATH
    export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:$LD_LIBRARY_PATH
fi

# Verify CUDA
nvcc --version

# If nvcc still not found, troubleshoot with these commands:
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

### 🛠️ **Common CUDA Issues and Fixes**

**Issue: `nvcc --version` returns "command not found"**

This is very common on Jetson. CUDA is installed but PATH isn't configured.

```bash
# Diagnosis commands:
ls -la /usr/local/ | grep cuda
echo $PATH
cat /etc/nv_tegra_release

# Your system has CUDA 12.6 in /usr/local/cuda-12.6
# The fix above should work, but if manual fix needed:
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

**Expected Output:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Cuda compilation tools, release 12.6, V12.6.xxx
```

**If CUDA completely missing:**
```bash
sudo apt update
sudo apt install -y cuda-toolkit-12-2
# Then add to PATH as above
```

## Step 2: Install Docker and NVIDIA Container Runtime

```bash
# SAFER Docker installation approach for Jetson
# (Research shows removing containerd can break system services)

# First, try installing Docker normally
sudo apt update

# If you get containerd.io conflicts, use this safer approach:
# Step 1: Stop conflicting services temporarily
sudo systemctl stop containerd || true
sudo systemctl stop docker || true

# Step 2: Install Docker using official script (handles dependencies better)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh

# Step 3: If still conflicts, resolve without removing containerd
if ! docker --version &> /dev/null; then
    echo "⚠️ Docker installation conflicts detected. Using alternative approach..."
    
    # Alternative 1: Use Snap (doesn't conflict with containerd)
    sudo snap install docker
    sudo groupadd docker || true
    sudo usermod -aG docker $USER
    
    # Alternative 2: If snap fails, minimal package removal (safer)
    if ! snap list | grep docker &> /dev/null; then
        echo "Snap failed, trying minimal package fix..."
        sudo apt remove -y docker.io docker-engine || true
        # Keep containerd - just remove conflicting docker packages
        sudo apt install -y docker-ce docker-ce-cli docker-compose-plugin
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

# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.4-base-ubuntu20.04 nvidia-smi
```

### 🛠️ **Common Docker Issues and Fixes**

**Issue: `containerd.io : Conflicts: containerd`**

This is very common on Jetson. **IMPORTANT**: Do NOT remove containerd - it can break system services!

```bash
# Root cause: Jetson has containerd pre-installed, conflicts with containerd.io
# SAFER approach (keeps containerd intact):

# Option 1: Use Snap (recommended - no conflicts)
sudo snap install docker
sudo groupadd docker || true
sudo usermod -aG docker $USER

# Activate Docker group membership (choose one method):
# Method 1: Logout/login (most reliable)
echo "💡 TIP: Exit and reconnect SSH to activate Docker group membership"
echo "      exit; ssh sanzad@192.168.1.177; docker --version"

# Method 2: Start fresh shell session 
sudo su - $USER  # Alternative to newgrp (avoids password prompt)

# Option 2: If snap not available, minimal package fix
sudo systemctl stop docker containerd || true
sudo apt remove -y docker.io docker-engine || true  # Remove only conflicting packages
# Keep containerd installed!
sudo apt install -y docker-ce docker-ce-cli docker-compose-plugin
sudo systemctl start containerd docker

# Option 3: Disable containerd temporarily (advanced)
sudo systemctl disable containerd
sudo systemctl stop containerd
# Then install Docker normally, but this may affect other services
```

**Issue: Permission denied accessing Docker**
```bash
# You need to logout/login after adding to docker group, or:
newgrp docker
# Or start new shell:
su - $USER
```

**Issue: Docker service won't start**
```bash
# Check what's wrong:
sudo journalctl -u docker.service
# Reset if needed:
sudo systemctl stop docker
sudo rm -rf /var/lib/docker
sudo systemctl start docker
```

**Issue: NVIDIA runtime not found**
```bash
# Verify nvidia-container-runtime installed:
dpkg -l | grep nvidia-container
# If missing:
sudo apt update && sudo apt install -y nvidia-container-runtime
# Check runtime registered:
docker info | grep nvidia
```

**Alternative: Snap installation (if above fails)**
```bash
sudo apt remove -y docker docker-engine docker.io containerd runc || true
sudo snap install docker
sudo usermod -aG docker $USER
```

**Expected Docker Output:**
```bash
$ docker --version
Docker version 24.0.7, build afdd53b

$ docker run hello-world
Hello from Docker!
This message shows that your installation appears to be working correctly.

$ docker run --rm --gpus all nvidia/cuda:12.2-runtime-ubuntu20.04 nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 540.4.0     Driver Version: 540.4.0      CUDA Version: 12.6     |
|-------------------------------+----------------------+----------------------+
|   0  Orin (nvgpu)           N/A| N/A              N/A |                  N/A |
+-------------------------------+----------------------+----------------------+
```

## Step 3: Install Ollama with Production Optimizations

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

## Step 4: Install Optimized Models

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

## Step 5: TensorRT Optimization (Advanced)

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

## Step 6: Performance Monitoring Setup

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

## Step 7: Network Configuration and Testing

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

## Model Performance Expectations

| Model | Size | RAM Usage | Tokens/sec | Use Case |
|-------|------|-----------|------------|----------|
| tinyllama:1.1b-chat-fp16 | ~700MB | ~1.5GB | 40-60 | Ultra-fast responses |
| gemma2:2b | ~1.5GB | ~2.5GB | 25-35 | Fast general purpose |
| llama3.2:3b | ~2GB | ~3GB | 15-25 | Quality responses |
| phi3:mini | ~2.3GB | ~3.5GB | 12-20 | Code generation |

## Troubleshooting

### Out of Memory Errors
```bash
# Check current memory usage
free -h
sudo dmesg | grep -i "killed process"

# If OOM, try smaller models or increase swap
ollama pull tinyllama:1.1b-chat-fp16  # Fallback to smaller model
```

### Performance Issues
```bash
# Ensure maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Check thermal throttling
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

### Network Issues
```bash
# Check if Ollama is listening
sudo netstat -tlnp | grep 11434

# Test local connection
curl http://localhost:11434/api/tags
```

## Integration Points
- **Load Balancer**: jetson-node is primary backend for HAProxy on cpu-node
- **Monitoring**: Health checks from worker-node4 (192.168.1.137:8083)
- **Coordination**: Managed by cluster orchestrator on cpu-node
- **Network**: Accessible at http://192.168.1.177:11434 for cluster integration

## Next Steps
- ✅ **Complete**: jetson-node Ollama setup with TensorRT optimization
- ⏭️ **Next**: [02_cpu_setup.md](02_cpu_setup.md) - Setup cpu-node coordinator
- 🔗 **Integration**: [03_langgraph_integration.md](03_langgraph_integration.md) - Connect to LangGraph
- 🎯 **Full Guide**: [00_complete_deployment_guide.md](00_complete_deployment_guide.md) - Complete walkthrough