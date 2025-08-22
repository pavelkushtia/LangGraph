# CPU Node Setup Guide (Coordinator + Heavy LLM + HAProxy + Redis)

## Overview
**Machine**: cpu-node (192.168.1.81) - 32GB Intel i5-6500T  
Your cpu-node serves as the **cluster coordinator**, handling heavy LLM tasks, load balancing, caching, and LangGraph orchestration.

## Prerequisites
- 32GB RAM machine with Ubuntu/Debian
- SSH access configured (`ssh sanzad@192.168.1.81`)
- Internet connection for package downloads

---

## Step 1: Initial System Preparation

```bash
# SSH into cpu-node (or run locally if you're already there)
# ssh sanzad@192.168.1.81

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git htop iotop nano vim build-essential cmake \
    python3 python3-pip python3-venv redis-server haproxy

# Install development tools for llama.cpp
sudo apt install -y pkg-config libopenblas-dev
```

## Step 2: Setup llama.cpp for Heavy Models

```bash
# Create working directory
mkdir -p ~/ai-infrastructure
cd ~/ai-infrastructure

# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with OpenBLAS optimization for CPU
mkdir build && cd build
cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS
cmake --build . --config Release -j$(nproc)

# Create models directory
mkdir -p ~/models/llama-cpp
cd ~/models/llama-cpp

# Download quantized models for complex tasks
echo "Downloading quantized models (this may take time)..."

# Llama 2 13B Chat Q4_K_M (good balance)
wget -O llama-2-13b-chat.q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.q4_K_M.gguf"

# Code Llama 13B for coding tasks
wget -O codellama-13b-instruct.q4_K_M.gguf \
  "https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.q4_k_m.gguf"

# Mistral 7B (efficient and capable)
wget -O mistral-7b-instruct.q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.q4_k_m.gguf"

echo "✅ Models downloaded"
ls -lh ~/models/llama-cpp/
```

## Step 3: Setup llama.cpp Server

```bash
# Test llama.cpp server
cd ~/ai-infrastructure/llama.cpp/build

# Test with Mistral 7B first (smaller model)
./bin/llama-server \
  --model ~/models/llama-cpp/mistral-7b-instruct.q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096 \
  --threads $(nproc) \
  --n-gpu-layers 0 \
  --chat-template llama2 &

# Wait for server to start
sleep 10

# Test the server
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello! Test response."}],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# Stop test server
pkill llama-server

# Create systemd service for llama.cpp server
sudo tee /etc/systemd/system/llama-server.service << EOF
[Unit]
Description=Llama.cpp Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/ai-infrastructure/llama.cpp/build
ExecStart=/home/$USER/ai-infrastructure/llama.cpp/build/bin/llama-server \
  --model /home/$USER/models/llama-cpp/llama-2-13b-chat.q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096 \
  --threads $(nproc) \
  --n-gpu-layers 0 \
  --chat-template llama2
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start llama server
sudo systemctl enable llama-server
sudo systemctl start llama-server

# Check status
sudo systemctl status llama-server
```

## Step 4: Setup Redis Cache

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

## Step 5: Setup HAProxy Load Balancer

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
    
    # Primary: Jetson Ollama (fast responses, higher weight)
    server jetson 192.168.1.177:11434 check weight 10 inter 30s fall 3 rise 2
    
    # Secondary: Local llama.cpp (complex tasks, backup)
    server cpu_llm 127.0.0.1:8080 check weight 5 backup inter 30s fall 3 rise 2

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

# Check status
sudo systemctl status haproxy

# Configure firewall
sudo ufw allow 9000/tcp  # LLM load balancer
sudo ufw allow 9001/tcp  # Tools load balancer  
sudo ufw allow 9002/tcp  # Embeddings load balancer
sudo ufw allow 8888/tcp  # Health check
sudo ufw allow 8080/tcp  # Direct llama.cpp access
sudo ufw allow 6379/tcp  # Redis
```

## Step 6: Setup LangGraph Environment

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

## Step 7: Performance Tuning

```bash
# Memory Optimization
echo 'kernel.shmmax = 68719476736' | sudo tee -a /etc/sysctl.conf
echo 'kernel.shmall = 4294967296' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Optimize swap
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# CPU Optimization
sudo apt install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
```

## Step 8: Testing Setup

```bash
# Test llama.cpp server
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 50
  }'

# Test Redis
redis-cli -h 192.168.1.81 -a langgraph_redis_pass ping

# Test HAProxy health
curl http://192.168.1.81:8888/health

# Test load balancer (once other services are running)
# curl http://192.168.1.81:9000/api/generate \
#   -H "Content-Type: application/json" \
#   -d '{"model": "llama3.2:3b", "prompt": "Hello!", "stream": false}'

echo "✅ cpu-node setup verification completed!"
```

---

## Resource Allocation

| Service | Memory Usage | CPU Usage | Port | Purpose |
|---------|-------------|-----------|------|---------|
| **llama.cpp** | 8-12GB | 70-90% | 8080 | Heavy LLM inference |
| **Redis** | 4GB | 5-10% | 6379 | Caching and sessions |
| **HAProxy** | 200MB | 2-5% | 9000-9002 | Load balancing |
| **LangGraph** | 1-2GB | 10-20% | - | Workflow orchestration |
| **System** | 4-6GB | 5-10% | - | OS and utilities |

## Service Management

```bash
# Check all services
sudo systemctl status llama-server
sudo systemctl status redis-server
sudo systemctl status haproxy

# Restart services if needed
sudo systemctl restart llama-server
sudo systemctl restart redis-server
sudo systemctl restart haproxy

# View logs
sudo journalctl -u llama-server -f
sudo journalctl -u haproxy -f
```

## Troubleshooting

### Memory Issues
```bash
# Check memory usage
free -h
sudo systemctl status llama-server

# If OOM, switch to smaller model
# Edit /etc/systemd/system/llama-server.service
# Change model to mistral-7b-instruct.q4_K_M.gguf
```

### HAProxy Issues
```bash
# Check HAProxy config
sudo haproxy -f /etc/haproxy/haproxy.cfg -c

# Check backend status
curl http://192.168.1.81:9000/haproxy_stats
# Login: admin / langgraph_admin_2024
```

### Redis Issues
```bash
# Check Redis connection
redis-cli -h 192.168.1.81 -a langgraph_redis_pass ping

# Check Redis memory
redis-cli -h 192.168.1.81 -a langgraph_redis_pass info memory
```

## Integration Points
- **Primary LLM**: Routes to jetson-node (192.168.1.177:11434)
- **Secondary LLM**: Local llama.cpp server (127.0.0.1:8080)
- **Load Balancer**: HAProxy distributes requests across cluster
- **Cache**: Redis stores session data and responses
- **Monitoring**: Health checks from worker-node4

## Next Steps
- ✅ **Complete**: cpu-node coordinator with HAProxy + Redis + llama.cpp
- ⏭️ **Next**: [03_langgraph_integration.md](03_langgraph_integration.md) - LangGraph setup
- 🔗 **Workers**: [04_distributed_coordination.md](04_distributed_coordination.md) - Setup worker nodes
- 🎯 **Full Guide**: [00_complete_deployment_guide.md](00_complete_deployment_guide.md) - Complete walkthrough