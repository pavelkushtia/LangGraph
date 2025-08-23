# CPU Node Setup Guide (Coordinator + Ollama Large Models + HAProxy + Redis)

## Overview
**Machine**: cpu-node (192.168.1.81) - 32GB Intel i5-6500T  
Your cpu-node serves as the **cluster coordinator**, handling large Ollama models, load balancing, caching, and LangGraph orchestration.

**🎯 SIMPLIFIED APPROACH**: Using Ollama instead of llama.cpp for easier setup and consistent API with jetson-node!

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

# Install essential packages (simplified - no build tools needed!)
sudo apt install -y curl wget git htop iotop nano vim \
    python3 python3-pip python3-venv redis-server haproxy
```

## Step 2: Setup Ollama for Large Models (Simplified!)

```bash
# Install Ollama (same as jetson-node - consistent!)
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama for network access on different port
export OLLAMA_HOST=0.0.0.0:11435
echo 'export OLLAMA_HOST=0.0.0.0:11435' >> ~/.bashrc
source ~/.bashrc

# Start Ollama service
ollama serve &

# Wait for service to start
sleep 5

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

## Step 3: Test Ollama Server

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

# Create systemd service for Ollama (persistent service)
sudo tee /etc/systemd/system/ollama-cpu.service << EOF
[Unit]
Description=Ollama CPU Server
After=network.target

[Service]
Type=simple
User=$USER
Environment=OLLAMA_HOST=0.0.0.0:11435
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start Ollama service
sudo systemctl enable ollama-cpu
sudo systemctl start ollama-cpu

# Check status
sudo systemctl status ollama-cpu

# Test remote access from jetson
echo "✅ Test from jetson-node:"
echo "curl -X POST http://192.168.1.81:11435/api/generate -H 'Content-Type: application/json' -d '{\"model\": \"mistral:7b\", \"prompt\": \"Hello from jetson!\", \"stream\": false}'"
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
    
    # Secondary: CPU Ollama (large models, backup)  
    server cpu_ollama 127.0.0.1:11435 check weight 5 backup inter 30s fall 3 rise 2

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
sudo ufw allow 11435/tcp  # Direct Ollama access
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
        port=11435,
        service_type="ollama",
        health_endpoint="/api/tags"
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
# Test Ollama server
curl -X POST http://localhost:11435/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b",
    "prompt": "Hello! Test response from cpu-node.",
    "stream": false
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
| **Ollama** | 6-10GB | 50-70% | 11435 | Large LLM inference |
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
- **Secondary LLM**: Local Ollama server (127.0.0.1:11435)
- **Load Balancer**: HAProxy distributes requests across cluster
- **Cache**: Redis stores session data and responses
- **Monitoring**: Health checks from worker-node4

## Next Steps
- ✅ **Complete**: cpu-node coordinator with HAProxy + Redis + Ollama
- ⏭️ **Next**: [03_langgraph_integration.md](03_langgraph_integration.md) - LangGraph setup
- 🔗 **Workers**: [04_distributed_coordination.md](04_distributed_coordination.md) - Setup worker nodes
- 🎯 **Full Guide**: [00_complete_deployment_guide.md](00_complete_deployment_guide.md) - Complete walkthrough