# Jetson Orin Nano Setup Guide

## Overview
Your Jetson Orin Nano 8GB will serve as the primary LLM inference server, optimized for small-to-medium models with excellent performance per watt.

## Prerequisites
- Jetson Orin Nano with JetPack 5.1+ installed
- Internet connection for initial setup
- SSH access configured

## Step 1: Install Ollama on Jetson

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Verify installation
ollama --version
```

## Step 2: Configure Memory and Performance

```bash
# Maximize performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Check memory and swap
free -h
# If swap is low, increase it
sudo systemctl stop nvzramconfig
sudo fallocate -l 4G /mnt/swapfile
sudo chmod 600 /mnt/swapfile
sudo mkswap /mnt/swapfile
sudo swapon /mnt/swapfile
echo '/mnt/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Step 3: Install Recommended Models

```bash
# Install small, efficient models
ollama pull llama3.2:3b      # Best general-purpose model
ollama pull gemma2:2b        # Fast and efficient
ollama pull phi3:mini        # Great for coding tasks
ollama pull tinyllama:1.1b   # Ultra-fast responses

# Test model
ollama run llama3.2:3b "Hello, explain what you are in one sentence."
```

## Step 4: Configure Ollama for Network Access

```bash
# Edit Ollama service to bind to all interfaces
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_NUM_PARALLEL=2"
EOF

# Restart Ollama
sudo systemctl daemon-reload
sudo systemctl restart ollama

# Test remote access (replace with Jetson IP)
curl http://JETSON_IP:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Hello world!",
  "stream": false
}'
```

## Step 5: Install Open WebUI (Optional)

```bash
# Install Docker if not present
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install Open WebUI
docker run -d --network=host \
  -v ${HOME}/open-webui:/app/backend/data \
  -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# Access at http://JETSON_IP:8080
```

## Step 6: Performance Monitoring

```bash
# Install monitoring tools
sudo apt install -y htop iotop tegrastats

# Monitor while running models
# Terminal 1: Model usage
tegrastats

# Terminal 2: System resources
htop

# Terminal 3: Test inference
ollama run llama3.2:3b
```

## Model Performance Expectations

| Model | Size | RAM Usage | Tokens/sec | Use Case |
|-------|------|-----------|------------|----------|
| llama3.2:3b | ~2GB | ~3GB | 15-25 | General purpose |
| gemma2:2b | ~1.5GB | ~2.5GB | 20-30 | Fast responses |
| phi3:mini | ~2.3GB | ~3.5GB | 12-20 | Code generation |
| tinyllama:1.1b | ~700MB | ~1.5GB | 30-50 | Quick tasks |

## Troubleshooting

### Out of Memory Errors
```bash
# Check current memory usage
free -h
sudo dmesg | grep -i "killed process"

# If OOM, try smaller models or increase swap
ollama pull tinyllama:1.1b  # Fallback to smaller model
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

## Next Steps
- Proceed to CPU machine setup
- Configure LangGraph to use this Ollama instance
- Set up load balancing across your machine cluster
