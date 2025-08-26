#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}STATUS: $1${NC}"
}

print_success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

echo -e "${BLUE}🔑 Setting up passwordless SSH access to cluster nodes${NC}"
echo ""

# Define cluster nodes
NODES=(
    "192.168.1.177"  # Jetson
    "192.168.1.178"  # RPi Embeddings
    "192.168.1.105"  # Worker Tools
    "192.168.1.137"  # Worker Monitor
)

NODE_NAMES=(
    "Jetson"
    "RPi Embeddings"
    "Worker Tools"
    "Worker Monitor"
)

# Check if SSH key exists
if [ ! -f ~/.ssh/id_ed25519 ]; then
    print_status "Generating SSH key..."
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "langgraph-orchestrator@cpu-node"
    print_success "SSH key generated"
fi

print_status "SSH public key:"
cat ~/.ssh/id_ed25519.pub
echo ""

# Copy key to each node
for i in "${!NODES[@]}"; do
    NODE_IP="${NODES[$i]}"
    NODE_NAME="${NODE_NAMES[$i]}"
    
    print_status "Setting up passwordless access to $NODE_NAME ($NODE_IP)..."
    
    # Test if passwordless access already works
    if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no sanzad@$NODE_IP 'echo "test"' >/dev/null 2>&1; then
        print_success "Passwordless access to $NODE_NAME already configured"
        continue
    fi
    
    print_warning "Need to copy SSH key to $NODE_NAME ($NODE_IP)"
    echo "You'll be prompted for the password for sanzad@$NODE_IP"
    
    # Use ssh-copy-id with explicit options
    if ssh-copy-id -i ~/.ssh/id_ed25519.pub -o ConnectTimeout=10 sanzad@$NODE_IP; then
        print_success "SSH key copied to $NODE_NAME"
        
        # Test the connection
        if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no sanzad@$NODE_IP 'echo "Connection test successful"'; then
            print_success "Passwordless access to $NODE_NAME verified"
        else
            print_error "Failed to verify passwordless access to $NODE_NAME"
        fi
    else
        print_error "Failed to copy SSH key to $NODE_NAME"
        print_warning "You may need to manually copy the key or check network connectivity"
    fi
    
    echo ""
done

echo ""
print_status "Testing all connections..."

ALL_GOOD=true
for i in "${!NODES[@]}"; do
    NODE_IP="${NODES[$i]}"
    NODE_NAME="${NODE_NAMES[$i]}"
    
    if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no sanzad@$NODE_IP 'echo "✓"' >/dev/null 2>&1; then
        print_success "$NODE_NAME ($NODE_IP): ✓ Passwordless access working"
    else
        print_error "$NODE_NAME ($NODE_IP): ✗ Still requires password"
        ALL_GOOD=false
    fi
done

echo ""
if $ALL_GOOD; then
    print_success "🎉 All cluster nodes now have passwordless SSH access!"
    print_status "The orchestrator will no longer prompt for passwords during cluster monitoring."
else
    print_warning "Some nodes still require passwords. The orchestrator may still prompt for passwords."
    print_status "You can run this script again or manually set up SSH keys for the failing nodes."
fi

echo ""
echo "📝 Next steps:"
echo "   1. Restart the orchestrator to use passwordless SSH"
echo "   2. Run: ./start-dev.sh"
echo ""
