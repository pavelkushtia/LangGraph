# Distributed Setup and Load Balancing

## Overview
Configure your machines for optimal load distribution, failover, and coordination across your local cluster.

## Network Configuration

### 1. Static IP Assignment

```bash
# On each machine, set static IPs for reliability
# Example for Ubuntu/Debian

# jetson-node (Jetson Orin Nano)
sudo tee /etc/netplan/01-static.yaml << EOF
network:
  version: 2
  ethernets:
    eth0:  # or your interface name
      dhcp4: false
      addresses:
        - 192.168.1.177/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 1.1.1.1]
EOF

sudo netplan apply
```

### 2. SSH Key Setup for Coordination

```bash
# Generate SSH key on coordinator machine (32GB CPU)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/cluster_key

# Copy to all machines
ssh-copy-id -i ~/.ssh/cluster_key.pub sanzad@192.168.1.177  # jetson-node
ssh-copy-id -i ~/.ssh/cluster_key.pub sanzad@192.168.1.178  # rp-node
ssh-copy-id -i ~/.ssh/cluster_key.pub sanzad@192.168.1.190  # worker-node3
ssh-copy-id -i ~/.ssh/cluster_key.pub sanzad@192.168.1.191  # worker-node4
```

## Load Balancing and Failover

### 1. HAProxy Setup (32GB Machine)

```bash
# Install HAProxy on coordinator machine
sudo apt update
sudo apt install -y haproxy

# Configure HAProxy
sudo tee /etc/haproxy/haproxy.cfg << 'EOF'
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httpchk GET /health

# LLM Load Balancer
frontend llm_frontend
    bind *:9000
    default_backend llm_servers

backend llm_servers
    balance roundrobin
    # Primary: Jetson Ollama (fast responses)
    server jetson 192.168.1.177:11434 check weight 10
    # Secondary: Local llama.cpp (complex tasks)
    server cpu_llm 192.168.1.81:8080 check weight 5 backup

# Tools Load Balancer
frontend tools_frontend
    bind *:9001
    default_backend tools_servers

backend tools_servers
    balance roundrobin
    server tools_primary 192.168.1.190:8082 check

# Embeddings Load Balancer
frontend embeddings_frontend
    bind *:9002
    default_backend embeddings_servers

backend embeddings_servers
    balance roundrobin
    server embeddings_primary 192.168.1.178:8081 check

# Statistics
stats enable
stats uri /haproxy_stats
stats refresh 30s
stats admin if TRUE
EOF

# Enable and start HAProxy
sudo systemctl enable haproxy
sudo systemctl start haproxy

# Check status
sudo systemctl status haproxy
```

### 2. Health Monitoring Service

```bash
# Create health monitoring on 8GB Machine A
python3 -m venv ~/health-monitor
source ~/health-monitor/bin/activate
pip install requests psutil schedule

mkdir -p ~/health-monitor
cd ~/health-monitor

cat > health_monitor.py << 'EOF'
import requests
import time
import json
import logging
from datetime import datetime
import schedule
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SERVICES = {
    'jetson_ollama': {
        'url': 'http://192.168.1.177:11434/api/tags',
        'critical': True
    },
    'cpu_llama': {
        'url': 'http://192.168.1.81:8080/health',
        'critical': False
    },
    'embeddings': {
        'url': 'http://192.168.1.178:8081/health',
        'critical': True
    },
    'tools': {
        'url': 'http://192.168.1.190:8082/health',
        'critical': True
    },
    'haproxy': {
        'url': 'http://192.168.1.81:9000',
        'critical': True
    }
}

class HealthMonitor:
    def __init__(self):
        self.health_status = {}
        self.alerts = []
    
    def check_service(self, name, config):
        try:
            response = requests.get(config['url'], timeout=10)
            healthy = response.status_code == 200
            
            self.health_status[name] = {
                'healthy': healthy,
                'last_check': datetime.now().isoformat(),
                'response_time': response.elapsed.total_seconds(),
                'status_code': response.status_code
            }
            
            if not healthy and config['critical']:
                alert = f"CRITICAL: {name} is unhealthy (status: {response.status_code})"
                self.alerts.append(alert)
                logger.error(alert)
            
            return healthy
            
        except Exception as e:
            self.health_status[name] = {
                'healthy': False,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }
            
            if config['critical']:
                alert = f"CRITICAL: {name} is unreachable - {str(e)}"
                self.alerts.append(alert)
                logger.error(alert)
            
            return False
    
    def check_all_services(self):
        logger.info("Running health checks...")
        healthy_count = 0
        
        for service_name, config in SERVICES.items():
            if self.check_service(service_name, config):
                healthy_count += 1
        
        # System metrics
        self.health_status['system'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'last_check': datetime.now().isoformat()
        }
        
        logger.info(f"Health check complete: {healthy_count}/{len(SERVICES)} services healthy")
        
        # Save status to file
        with open('/tmp/cluster_health.json', 'w') as f:
            json.dump(self.health_status, f, indent=2)
        
        return self.health_status
    
    def restart_failed_service(self, service_name):
        """Attempt to restart failed services"""
        restart_commands = {
            'jetson_ollama': 'ssh -i ~/.ssh/cluster_key sanzad@192.168.1.177 "sudo systemctl restart ollama"',
            'cpu_llama': 'ssh -i ~/.ssh/cluster_key sanzad@192.168.1.81 "sudo systemctl restart llama-server"',
            'embeddings': 'ssh -i ~/.ssh/cluster_key sanzad@192.168.1.178 "sudo systemctl restart embeddings-server"',
            'tools': 'ssh -i ~/.ssh/cluster_key sanzad@192.168.1.190 "sudo systemctl restart tools-server"'
        }
        
        if service_name in restart_commands:
            try:
                import subprocess
                result = subprocess.run(restart_commands[service_name], shell=True, capture_output=True)
                logger.info(f"Attempted restart of {service_name}: {result.returncode}")
                return result.returncode == 0
            except Exception as e:
                logger.error(f"Failed to restart {service_name}: {e}")
                return False
        return False

# Initialize monitor
monitor = HealthMonitor()

def run_health_check():
    monitor.check_all_services()

def run_recovery():
    """Recovery routine for failed services"""
    for service_name, status in monitor.health_status.items():
        if service_name != 'system' and not status.get('healthy', True):
            logger.info(f"Attempting recovery for {service_name}")
            monitor.restart_failed_service(service_name)

# Schedule health checks
schedule.every(30).seconds.do(run_health_check)
schedule.every(5).minutes.do(run_recovery)

if __name__ == "__main__":
    logger.info("Starting health monitor...")
    
    # Initial check
    run_health_check()
    
    # Main loop
    while True:
        schedule.run_pending()
        time.sleep(1)
EOF

# Create systemd service
sudo tee /etc/systemd/system/health-monitor.service << EOF
[Unit]
Description=Cluster Health Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/health-monitor
Environment=PATH=/home/$USER/health-monitor/bin
ExecStart=/home/$USER/health-monitor/bin/python health_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable health-monitor
sudo systemctl start health-monitor
```

## Auto-scaling Configuration

### 1. Dynamic Model Loading

```python
# dynamic_scaling.py
import requests
import psutil
import time
from typing import Dict, List

class ModelScaler:
    def __init__(self):
        self.load_thresholds = {
            'low': 30,    # CPU/Memory below 30%
            'medium': 70, # CPU/Memory below 70%
            'high': 90    # CPU/Memory above 90%
        }
        
        self.model_configs = {
            'jetson': {
                'low_load': ['tinyllama:1.1b'],
                'medium_load': ['llama3.2:3b', 'gemma2:2b'],
                'high_load': ['phi3:mini']  # Smaller model when overloaded
            }
        }
    
    def get_system_load(self, machine_ip: str) -> Dict:
        """Get system load from a machine"""
        try:
            # Use worker-node4 for monitoring
            response = requests.get(f"http://192.168.1.191:8083/system_stats", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {'cpu_percent': 50, 'memory': {'percent': 50}}
    
    def scale_jetson_models(self):
        """Scale Jetson models based on load"""
        load_info = self.get_system_load('192.168.1.177')
        cpu_load = load_info['cpu_percent']
        memory_load = load_info['memory']['percent']
        
        avg_load = (cpu_load + memory_load) / 2
        
        if avg_load < self.load_thresholds['low']:
            # Load more models for better responses
            target_models = self.model_configs['jetson']['medium_load']
        elif avg_load < self.load_thresholds['medium']:
            # Standard load
            target_models = self.model_configs['jetson']['medium_load']
        else:
            # High load - use smaller models
            target_models = self.model_configs['jetson']['high_load']
        
        # Update model configuration (simplified)
        print(f"Scaling Jetson to models: {target_models} (load: {avg_load:.1f}%)")
        
        return target_models

# Auto-scaling daemon
if __name__ == "__main__":
    scaler = ModelScaler()
    
    while True:
        scaler.scale_jetson_models()
        time.sleep(60)  # Check every minute
```

## Backup and Recovery

### 1. Model Synchronization

```bash
# sync_models.sh - Keep models synchronized across machines
#!/bin/bash

JETSON_IP="192.168.1.100"
CPU_IP="192.168.1.101"
MODELS_DIR="/home/$USER/models"

# Sync Ollama models from Jetson to backup location
rsync -avz --progress $USER@$JETSON_IP:~/.ollama/models/ $MODELS_DIR/ollama_backup/

# Sync llama.cpp models
rsync -avz --progress $USER@$CPU_IP:~/models/ $MODELS_DIR/llamacpp_backup/

echo "Model sync completed at $(date)"
```

### 2. Configuration Backup

```bash
# backup_configs.sh
#!/bin/bash

BACKUP_DIR="/home/$USER/cluster_backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup service configurations
for IP in 192.168.1.100 192.168.1.101 192.168.1.102 192.168.1.103; do
    echo "Backing up $IP..."
    ssh -i ~/.ssh/cluster_key $USER@$IP "sudo tar czf /tmp/service_configs.tar.gz /etc/systemd/system/*.service" 2>/dev/null
    scp -i ~/.ssh/cluster_key $USER@$IP:/tmp/service_configs.tar.gz "$BACKUP_DIR/configs_$IP.tar.gz" 2>/dev/null
done

echo "Configuration backup completed in $BACKUP_DIR"
```

## Cluster Orchestration Script

```python
# cluster_orchestrator.py
import subprocess
import time
import json
import argparse
from typing import List, Dict

class ClusterOrchestrator:
    def __init__(self):
        self.machines = {
            'jetson': '192.168.1.177',      # jetson-node (Orin Nano 8GB)
            'cpu_32gb': '192.168.1.81',     # cpu-node (32GB coordinator)
            'rp_node': '192.168.1.178',     # rp-node (8GB ARM, embeddings)
            'worker_3': '192.168.1.190',    # worker-node3 (6GB VM, tools)
            'worker_4': '192.168.1.191'     # worker-node4 (6GB VM, monitoring)
        }
    
    def start_cluster(self):
        """Start all services in the correct order"""
        print("🚀 Starting cluster services...")
        
        # Start core services first (Redis on main coordinator)
        self._run_remote_command('cpu_32gb', 'sudo systemctl start redis-server')
        time.sleep(2)
        
        # Start model servers
        self._run_remote_command('jetson', 'sudo systemctl start ollama')
        self._run_remote_command('cpu_32gb', 'sudo systemctl start llama-server')
        time.sleep(5)
        
        # Start application services
        self._run_remote_command('rp_node', 'sudo systemctl start embeddings-server')
        self._run_remote_command('worker_3', 'sudo systemctl start tools-server')
        time.sleep(3)
        
        # Start load balancer and monitoring
        self._run_remote_command('cpu_32gb', 'sudo systemctl start haproxy')
        self._run_remote_command('worker_4', 'sudo systemctl start health-monitor')
        
        print("✅ Cluster startup completed!")
    
    def stop_cluster(self):
        """Stop all services gracefully"""
        print("🛑 Stopping cluster services...")
        
        services = [
            ('worker_4', 'health-monitor'),
            ('cpu_32gb', 'haproxy'),
            ('worker_3', 'tools-server'),
            ('rp_node', 'embeddings-server'),
            ('cpu_32gb', 'llama-server'),
            ('jetson', 'ollama'),
            ('cpu_32gb', 'redis-server')
        ]
        
        for machine, service in services:
            self._run_remote_command(machine, f'sudo systemctl stop {service}')
            time.sleep(1)
        
        print("✅ Cluster shutdown completed!")
    
    def status_cluster(self):
        """Check status of all services"""
        print("📊 Cluster Status:")
        
        services = {
            'jetson': ['ollama'],
            'cpu_32gb': ['llama-server', 'haproxy', 'redis-server'],
            'rp_node': ['embeddings-server'],
            'worker_3': ['tools-server'],
            'worker_4': ['health-monitor']
        }
        
        for machine, service_list in services.items():
            print(f"\n{machine} ({self.machines[machine]}):")
            for service in service_list:
                status = self._check_service_status(machine, service)
                print(f"  {service}: {status}")
    
    def _run_remote_command(self, machine: str, command: str) -> str:
        """Run command on remote machine"""
        ip = self.machines[machine]
        ssh_cmd = f"ssh -i ~/.ssh/cluster_key -o StrictHostKeyChecking=no $USER@{ip} '{command}'"
        
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
    parser = argparse.ArgumentParser(description="Cluster Orchestrator")
    parser.add_argument('action', choices=['start', 'stop', 'status', 'restart'], help='Action to perform')
    
    args = parser.parse_args()
    orchestrator = ClusterOrchestrator()
    
    if args.action == 'start':
        orchestrator.start_cluster()
    elif args.action == 'stop':
        orchestrator.stop_cluster()
    elif args.action == 'status':
        orchestrator.status_cluster()
    elif args.action == 'restart':
        orchestrator.stop_cluster()
        time.sleep(5)
        orchestrator.start_cluster()

if __name__ == "__main__":
    main()
```

## Usage Commands

```bash
# Start the entire cluster
python cluster_orchestrator.py start

# Check cluster status
python cluster_orchestrator.py status

# Stop the cluster
python cluster_orchestrator.py stop

# View HAProxy statistics
curl http://192.168.1.101:9000/haproxy_stats

# Check health status
curl http://192.168.1.104:8083/cluster_health

# Test load-balanced LLM endpoint
curl http://192.168.1.101:9000/api/generate \
  -d '{"model": "llama3.2:3b", "prompt": "Hello!", "stream": false}'
```

## Performance Monitoring Dashboard

Create a simple dashboard to monitor your cluster:

```bash
# Install lightweight monitoring
pip install flask plotly pandas

# Create dashboard.py (simplified version)
# This will show real-time metrics from your cluster
```

This distributed setup gives you a robust, scalable local AI infrastructure with automatic failover, load balancing, and health monitoring - all without any external API costs!
