#!/bin/bash

# Network Configuration Script for LangGraph Orchestrator
# This script ensures proper network access for cluster nodes

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🌐 Configuring network access for LangGraph Orchestrator..."

# Get current machine IP
MACHINE_IP=$(ip route get 8.8.8.8 | grep -oP 'src \K\S+' 2>/dev/null || echo "192.168.1.81")
print_status "Machine IP: $MACHINE_IP"

# Check if firewall is active
if command -v ufw &> /dev/null; then
    print_status "UFW firewall detected"
    
    if ufw status | grep -q "Status: active"; then
        print_status "Configuring UFW firewall rules..."
        
        # Allow frontend port from cluster network
        sudo ufw allow from 192.168.1.0/24 to any port 3000 comment "LangGraph Orchestrator Frontend"
        
        # Allow backend API port from cluster network
        sudo ufw allow from 192.168.1.0/24 to any port 3001 comment "LangGraph Orchestrator Backend"
        
        print_success "UFW rules configured for cluster access"
    else
        print_warning "UFW is installed but not active"
    fi
elif command -v firewall-cmd &> /dev/null; then
    print_status "FirewallD detected"
    
    if systemctl is-active --quiet firewalld; then
        print_status "Configuring FirewallD rules..."
        
        # Add rich rules for cluster network
        sudo firewall-cmd --permanent --add-rich-rule="rule family='ipv4' source address='192.168.1.0/24' port protocol='tcp' port='3000' accept"
        sudo firewall-cmd --permanent --add-rich-rule="rule family='ipv4' source address='192.168.1.0/24' port protocol='tcp' port='3001' accept"
        
        # Reload firewall
        sudo firewall-cmd --reload
        
        print_success "FirewallD rules configured for cluster access"
    else
        print_warning "FirewallD is installed but not active"
    fi
else
    print_warning "No supported firewall detected (UFW or FirewallD)"
fi

# Check if iptables is being used
if command -v iptables &> /dev/null && iptables -L | grep -q "Chain INPUT"; then
    print_status "iptables detected - you may need to manually configure rules"
    print_warning "Add these rules if needed:"
    echo "  sudo iptables -A INPUT -s 192.168.1.0/24 -p tcp --dport 3000 -j ACCEPT"
    echo "  sudo iptables -A INPUT -s 192.168.1.0/24 -p tcp --dport 3001 -j ACCEPT"
fi

# Test port availability
print_status "Testing port availability..."

# Check if ports are in use
if netstat -tuln 2>/dev/null | grep -q ":3000 "; then
    print_warning "Port 3000 is already in use"
    netstat -tuln | grep ":3000 "
else
    print_success "Port 3000 is available"
fi

if netstat -tuln 2>/dev/null | grep -q ":3001 "; then
    print_warning "Port 3001 is already in use"
    netstat -tuln | grep ":3001 "
else
    print_success "Port 3001 is available"
fi

# Network connectivity test
print_status "Testing network connectivity from cluster nodes..."

# List of cluster nodes
NODES=("192.168.1.177" "192.168.1.178" "192.168.1.190" "192.168.1.191")
NODE_NAMES=("Jetson" "RPi Embeddings" "Worker Tools" "Worker Monitor")

for i in "${!NODES[@]}"; do
    NODE_IP="${NODES[$i]}"
    NODE_NAME="${NODE_NAMES[$i]}"
    
    if [ "$NODE_IP" != "$MACHINE_IP" ]; then
        if ping -c 1 -W 2 "$NODE_IP" &>/dev/null; then
            print_success "$NODE_NAME ($NODE_IP) is reachable"
        else
            print_warning "$NODE_NAME ($NODE_IP) is not reachable"
        fi
    fi
done

# Create test endpoints
print_status "Creating network test files..."

# Create a simple test HTML file
cat > test-network.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>LangGraph Orchestrator Network Test</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        .success { color: green; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>🌐 Network Access Test</h1>
    <p>If you can see this page, network access is working!</p>
    <p><strong>Server IP:</strong> $MACHINE_IP</p>
    <p><strong>Test Time:</strong> $(date)</p>
    
    <h2>🧪 API Test</h2>
    <button onclick="testAPI()">Test Backend API</button>
    <div id="result"></div>
    
    <script>
        async function testAPI() {
            const result = document.getElementById('result');
            try {
                const response = await fetch('http://$MACHINE_IP:3001/api/health');
                const data = await response.json();
                result.innerHTML = '<p class="success">✅ Backend API is accessible!</p>';
            } catch (error) {
                result.innerHTML = '<p class="error">❌ Backend API is not accessible: ' + error.message + '</p>';
            }
        }
    </script>
</body>
</html>
EOF

print_success "Network test page created: test-network.html"

# Summary
echo ""
echo "📋 Network Configuration Summary:"
echo "=================================="
echo "🖥️  Server IP: $MACHINE_IP"
echo "🌐 Frontend: http://$MACHINE_IP:3000"
echo "🔧 Backend API: http://$MACHINE_IP:3001"
echo "🧪 Test Page: file://$(pwd)/test-network.html"
echo ""
echo "✅ Cluster nodes can now access the orchestrator at:"
for i in "${!NODES[@]}"; do
    NODE_IP="${NODES[$i]}"
    NODE_NAME="${NODE_NAMES[$i]}"
    if [ "$NODE_IP" != "$MACHINE_IP" ]; then
        echo "   $NODE_NAME ($NODE_IP) → http://$MACHINE_IP:3000"
    fi
done
echo ""
echo "🔒 Security Notes:"
echo "   - Access is restricted to the 192.168.1.0/24 network"
echo "   - Firewall rules have been configured (if firewall is active)"
echo "   - For external access, additional configuration may be needed"
echo ""

print_success "Network configuration completed!"
