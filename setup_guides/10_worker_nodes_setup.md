# Worker Nodes Setup Guide - Market Data Services & Technical Analysis

## 🎯 Overview

This guide sets up **worker-node3** and **worker-node4** to provide market data services, technical indicators calculation, and backtesting capabilities for the distributed trading workflow system.

## 🖥️ Node Distribution

- **worker-node3**: Market data services, technical indicators, backtesting
- **worker-node4**: Monitoring, logging, and support services

## 🚀 Phase 1: worker-node3 Setup (Market Data Services)

### 1.1 Initial Setup

```bash
# SSH into worker-node3
ssh sanzad@worker-node3

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git curl wget htop
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libssl-dev libffi-dev python3-dev
sudo apt install -y redis-server
```

### 1.2 Python Environment Setup

```bash
# Create virtual environment
python3 -m venv ~/market-data-env
source ~/market-data-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install fastapi uvicorn python-multipart
pip install pandas numpy scipy
pip install yfinance ccxt alpha_vantage quandl
pip install pandas-ta ta-lib finta
pip install backtrader zipline-reloaded
pip install redis aioredis
pip install aiohttp requests pyyaml
```

### 1.3 Create Market Data Service

```bash
# Create service directory
mkdir -p ~/market-data-services
cd ~/market-data-services

# Create the same services as in the original cpu-node2 guide
# (Copy the services from setup_guides/10_cpu_node2_setup.md)
```

## 🚀 Phase 2: worker-node4 Setup (Monitoring & Support)

### 2.1 Initial Setup

```bash
# SSH into worker-node4
ssh sanzad@worker-node4

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git curl wget htop
sudo apt install -y nginx
```

### 2.2 Python Environment Setup

```bash
# Create virtual environment
python3 -m venv ~/monitoring-env
source ~/monitoring-env/bin/activate

# Install monitoring dependencies
pip install fastapi uvicorn
pip install psutil requests
pip install prometheus_client
pip install grafana_api
```

### 2.3 Create Monitoring Service

```python
# ~/monitoring/monitoring_service.py
from fastapi import FastAPI
import psutil
import requests
from datetime import datetime

app = FastAPI(title="Trading Cluster Monitor", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }

@app.get("/cluster_status")
async def cluster_status():
    # Check all cluster nodes
    nodes = {
        "gpu-node": "192.168.1.177:8080",
        "gpu-node1": "192.168.1.178:8081", 
        "cpu-node": "192.168.1.81:8082",
        "worker-node3": "192.168.1.190:8083"
    }
    
    status = {}
    for name, endpoint in nodes.items():
        try:
            response = requests.get(f"http://{endpoint}/health", timeout=5)
            status[name] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            status[name] = "unreachable"
    
    return {
        "cluster_status": status,
        "timestamp": datetime.now()
    }
```

## 🔧 Integration with Existing Cluster

### 3.1 Update Cluster Configuration

On **cpu-node**, update the configuration:

```python
# ~/trading-orchestrator/config/config.yaml
gpu_nodes:
  gpu-node:
    host: "192.168.1.177"
    port: 8080
    services: ["fingpt", "stockgpt"]
  
  gpu-node1:
    host: "192.168.1.178"
    port: 8081
    services: ["finrl_portfolio", "finrl_risk", "finrl_trading"]

cpu_nodes:
  cpu-node:
    host: "192.168.1.81"
    port: 8082
    services: ["langgraph_orchestrator", "workflow_engine"]
  
  worker-node3:
    host: "192.168.1.190"
    port: 8085  # Changed from 8083 to avoid conflicts
    services: ["market_data_service", "technical_indicators", "backtesting_engine"]
  
  worker-node4:
    host: "192.168.1.191"
    port: 8086  # Changed from 8084 to avoid conflicts
    services: ["monitoring", "logging"]
```

### 3.2 Update Startup Script

Update the IP addresses in `scripts/start_trading_cluster.sh`:

```bash
# Configuration
GPU_NODE="192.168.1.177"
GPU_NODE1="192.168.1.178"
CPU_NODE="192.168.1.81"
WORKER_NODE3="192.168.1.190"
WORKER_NODE4="192.168.1.191"
```

## 🧪 Testing

### 4.1 Test worker-node3

```bash
# Test market data service
curl http://192.168.1.190:8083/health

# Test market data retrieval
curl -X POST "http://192.168.1.190:8083/get_market_data" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "period": "1mo"}'
```

### 4.2 Test worker-node4

```bash
# Test monitoring service
curl http://192.168.1.191:8084/health

# Test cluster status
curl http://192.168.1.191:8084/cluster_status
```

## ✅ Benefits of This Approach

1. **Reuses existing infrastructure**: No need for new machines
2. **Consistent with your setup**: Uses the same venv pattern
3. **Better resource distribution**: worker-node3/4 are perfect for data processing
4. **Easier management**: Leverages your existing worker node setup
5. **Cost-effective**: No additional hardware needed

---

Your worker nodes are now ready to support the distributed trading workflow! 🚀
