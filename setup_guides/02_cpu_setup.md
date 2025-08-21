# CPU Machines Setup Guide

## Overview
Configure your CPU machines for distributed LLM inference using quantized models with llama.cpp for optimal performance.

## Machine Allocation Strategy

### 32GB Machine: Large Model Server
- **Role**: Host larger quantized models (7B-13B parameters)
- **Models**: Llama 2 13B, Code Llama 13B, Mistral 7B
- **Software**: llama.cpp with server mode

### 16GB Machines: Specialized Workers
- **Machine A**: Embeddings and vector search
- **Machine B**: Tool execution and API calls

### 8GB Machines: Support Services
- **Machine A**: Monitoring and logging
- **Machine B**: Data processing and caching

## Setup Instructions

### 32GB Machine: Large Model Server

```bash
# Install dependencies
sudo apt update
sudo apt install -y git cmake build-essential curl wget

# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=OFF  # CPU-only build
cmake --build . --config Release -j$(nproc)

# Create models directory
mkdir -p ~/models
cd ~/models

# Download quantized models (choose based on your needs)
# Llama 2 13B Chat (Q4_K_M - good balance of quality/size)
wget https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGML/resolve/main/llama-2-13b-chat.q4_K_M.bin

# Code Llama 13B (for coding tasks)
wget https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGML/resolve/main/codellama-13b-instruct.q4_K_M.bin

# Mistral 7B (efficient general purpose)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGML/resolve/main/mistral-7b-instruct-v0.1.q4_K_M.bin
```

#### Start Server Mode

```bash
# Navigate to llama.cpp build directory
cd ~/llama.cpp/build

# Start server with Llama 2 13B
./bin/llama-server \
  --model ~/models/llama-2-13b-chat.q4_K_M.bin \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096 \
  --threads $(nproc) \
  --n-gpu-layers 0

# Test the server
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'
```

#### Create Systemd Service

```bash
# Create service file
sudo tee /etc/systemd/system/llama-server.service << EOF
[Unit]
Description=Llama.cpp Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/llama.cpp/build
ExecStart=/home/$USER/llama.cpp/build/bin/llama-server --model /home/$USER/models/llama-2-13b-chat.q4_K_M.bin --host 0.0.0.0 --port 8080 --ctx-size 4096 --threads $(nproc)
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable llama-server
sudo systemctl start llama-server
sudo systemctl status llama-server
```

### 16GB Machine A: Embeddings Server

```bash
# Install Python and dependencies
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv ~/embeddings-env
source ~/embeddings-env/bin/activate

# Install requirements
pip install sentence-transformers transformers torch fastapi uvicorn chromadb

# Create embeddings server
mkdir -p ~/embeddings-server
cd ~/embeddings-server

cat > app.py << 'EOF'
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import uvicorn

app = FastAPI()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.post("/embeddings")
async def create_embeddings(texts: List[str]):
    embeddings = model.encode(texts)
    return {"embeddings": embeddings.tolist()}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
EOF

# Create systemd service
sudo tee /etc/systemd/system/embeddings-server.service << EOF
[Unit]
Description=Embeddings Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/embeddings-server
Environment=PATH=/home/$USER/embeddings-env/bin
ExecStart=/home/$USER/embeddings-env/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable embeddings-server
sudo systemctl start embeddings-server
```

### 16GB Machine B: Tool Execution Server

```bash
# Create tools environment
python3 -m venv ~/tools-env
source ~/tools-env/bin/activate

# Install tools dependencies
pip install fastapi uvicorn requests beautifulsoup4 selenium pandas

mkdir -p ~/tools-server
cd ~/tools-server

cat > app.py << 'EOF'
from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
import json
import subprocess
from typing import Dict, Any
import uvicorn

app = FastAPI()

@app.post("/web_search")
async def web_search(query: str):
    # Implement web search functionality
    # This is a placeholder - you might want to use a search API
    return {"query": query, "results": "Search results here"}

@app.post("/web_scrape")
async def web_scrape(url: str):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return {"url": url, "content": soup.get_text()[:1000]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/execute_command")
async def execute_command(command: str):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return {"command": command, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
EOF

# Create systemd service for tools
sudo tee /etc/systemd/system/tools-server.service << EOF
[Unit]
Description=Tools Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/tools-server
Environment=PATH=/home/$USER/tools-env/bin
ExecStart=/home/$USER/tools-env/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable tools-server
sudo systemctl start tools-server
```

### 8GB Machines: Support Services

#### Machine A: Monitoring

```bash
# Install monitoring stack
sudo apt install -y prometheus grafana

# Simple Python monitoring service
python3 -m venv ~/monitoring-env
source ~/monitoring-env/bin/activate
pip install fastapi uvicorn psutil requests

mkdir -p ~/monitoring
cd ~/monitoring

cat > monitor.py << 'EOF'
from fastapi import FastAPI
import psutil
import requests
import uvicorn
from datetime import datetime

app = FastAPI()

@app.get("/system_stats")
async def system_stats():
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(),
        "memory": psutil.virtual_memory()._asdict(),
        "disk": psutil.disk_usage('/')._asdict()
    }

@app.get("/cluster_health")
async def cluster_health():
    services = [
        {"name": "jetson_ollama", "url": "http://JETSON_IP:11434/api/tags"},
        {"name": "cpu_llama", "url": "http://CPU32_IP:8080/health"},
        {"name": "embeddings", "url": "http://CPU16A_IP:8081/health"},
        {"name": "tools", "url": "http://CPU16B_IP:8082/health"}
    ]
    
    health_status = {}
    for service in services:
        try:
            response = requests.get(service["url"], timeout=5)
            health_status[service["name"]] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            health_status[service["name"]] = "unhealthy"
    
    return health_status

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)
EOF
```

#### Machine B: Redis Cache

```bash
# Install Redis
sudo apt update
sudo apt install -y redis-server

# Configure Redis for network access
sudo sed -i 's/bind 127.0.0.1/bind 0.0.0.0/' /etc/redis/redis.conf
sudo systemctl restart redis-server
sudo systemctl enable redis-server

# Test Redis
redis-cli ping
```

## Performance Tuning

### Memory Optimization

```bash
# Increase shared memory limits
echo 'kernel.shmmax = 68719476736' | sudo tee -a /etc/sysctl.conf
echo 'kernel.shmall = 4294967296' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Optimize swap
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

### CPU Optimization

```bash
# Set CPU governor to performance
sudo apt install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
```

## Testing Your Setup

```bash
# Test large model server
curl http://CPU32_IP:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'

# Test embeddings server
curl -X POST http://CPU16A_IP:8081/embeddings \
  -H "Content-Type: application/json" \
  -d '["Hello world", "This is a test"]'

# Test tools server
curl -X POST http://CPU16B_IP:8082/web_scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Test monitoring
curl http://CPU8A_IP:8083/cluster_health
```

## Resource Monitoring

```bash
# Monitor model performance
htop
iotop -a
nvidia-smi  # If using any GPU acceleration

# Check memory usage by process
ps aux --sort=-%mem | head -10

# Monitor network usage
iftop -i eth0
```

## Next Steps
- Configure LangGraph to orchestrate these services
- Set up load balancing and failover
- Create example workflows using your distributed setup
