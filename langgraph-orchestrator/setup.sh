#!/bin/bash

# LangGraph Orchestrator Setup Script
# This script sets up the development environment for the LangGraph Orchestrator

set -e

echo "🚀 Setting up LangGraph Orchestrator..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check prerequisites
print_status "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    print_error "Node.js version 18+ is required. Current version: $(node --version)"
    exit 1
fi

print_success "Node.js $(node --version) is installed"

# Check npm
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. Please install npm."
    exit 1
fi

print_success "npm $(npm --version) is installed"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    print_error "Python 3.8+ is required. Current version: $(python3 --version)"
    exit 1
fi

print_success "Python $(python3 --version) is installed"

# Check cluster orchestrator
CLUSTER_PATH="/home/sanzad/ai-infrastructure/langgraph-config"
if [ ! -f "$CLUSTER_PATH/cluster_orchestrator.py" ]; then
    print_warning "Cluster orchestrator not found at $CLUSTER_PATH"
    print_warning "Please ensure your LangGraph cluster is set up and the path is correct"
fi

# Create data directory
print_status "Creating data directories..."
mkdir -p data
mkdir -p logs
print_success "Data directories created"

# Install root dependencies
print_status "Installing root dependencies..."
npm install

# Install backend dependencies
print_status "Installing backend dependencies..."
cd backend
npm install
cd ..

# Install frontend dependencies  
print_status "Installing frontend dependencies..."
cd frontend
npm install
cd ..

print_success "All dependencies installed successfully"

# Create environment files
print_status "Creating environment configuration files..."

# Detect current machine IP in the 192.168.1.x network
MACHINE_IP=$(ip route get 8.8.8.8 | grep -oP 'src \K\S+' 2>/dev/null || echo "192.168.1.81")
print_status "Detected machine IP: $MACHINE_IP"

# Backend .env
cat > backend/.env << EOF
NODE_ENV=development
PORT=3001
HOST=0.0.0.0
DATABASE_PATH=../data/orchestrator.db
CLUSTER_ORCHESTRATOR_PATH=/home/sanzad/ai-infrastructure/langgraph-config
DEBUG=langgraph:*
EOF

# Frontend .env
cat > frontend/.env << EOF
VITE_API_URL=http://$MACHINE_IP:3001
VITE_WS_URL=ws://$MACHINE_IP:3001
VITE_NODE_ENV=development
EOF

print_success "Environment files created"

# Build backend
print_status "Building backend..."
cd backend
npm run build
cd ..

print_success "Backend built successfully"

# Test cluster connection
print_status "Testing cluster connection..."
if [ -f "$CLUSTER_PATH/cluster_orchestrator.py" ]; then
    if cd "$CLUSTER_PATH" && python3 cluster_orchestrator.py status &>/dev/null; then
        print_success "Cluster connection test passed"
    else
        print_warning "Cluster connection test failed - cluster may be offline"
        print_warning "You can still run the orchestrator, but cluster features may not work"
    fi
else
    print_warning "Skipping cluster test - orchestrator not found"
fi

# Create startup scripts
print_status "Creating startup scripts..."

# Development startup script
cat > start-dev.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting LangGraph Orchestrator in development mode..."

# Start backend and frontend in parallel
trap 'kill $(jobs -p)' EXIT

cd backend && npm run dev &
cd frontend && npm run dev &

wait
EOF
chmod +x start-dev.sh

# Production startup script
cat > start-prod.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting LangGraph Orchestrator in production mode..."

# Build frontend
cd frontend && npm run build && cd ..

# Start backend
cd backend && npm start
EOF
chmod +x start-prod.sh

print_success "Startup scripts created"

# Final instructions
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Ensure your LangGraph cluster is running"
echo "   2. Start the development server:"
echo "      ${GREEN}./start-dev.sh${NC}"
echo "   3. Open http://localhost:3000 in your browser"
echo ""
echo "📚 Available commands:"
echo "   ${BLUE}npm run dev${NC}        - Start both frontend and backend in development mode"
echo "   ${BLUE}./start-dev.sh${NC}     - Start development servers"
echo "   ${BLUE}./start-prod.sh${NC}    - Start production server"
echo "   ${BLUE}npm run build${NC}      - Build for production"
echo ""
echo "🔧 Configuration:"
echo "   - Backend runs on: http://$MACHINE_IP:3001 (accessible from all cluster nodes)"
echo "   - Frontend runs on: http://$MACHINE_IP:3000 (accessible from all cluster nodes)"
echo "   - Local access: http://localhost:3000"
echo "   - Database: data/orchestrator.db"
echo "   - Cluster path: $CLUSTER_PATH"
echo ""
echo "🌐 Network Access:"
echo "   From any cluster node, access the UI at:"
echo "   - Jetson (192.168.1.177): http://$MACHINE_IP:3000"
echo "   - RPi (192.168.1.178): http://$MACHINE_IP:3000"
echo "   - Worker Tools (192.168.1.105): http://$MACHINE_IP:3000"
echo "   - Worker Monitor (192.168.1.137): http://$MACHINE_IP:3000"
echo ""
echo "📖 For more information, see README.md"
echo ""

print_success "LangGraph Orchestrator is ready to use!"
