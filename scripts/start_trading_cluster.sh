#!/bin/bash

# Distributed Trading Cluster Startup Script
# This script starts all services across the 4-node cluster

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GPU_NODE="192.168.1.177"
GPU_NODE1="192.168.1.178"
CPU_NODE="192.168.1.81"
WORKER_NODE3="192.168.1.190"
WORKER_NODE4="192.168.1.191"

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

# Function to check if a service is running
check_service() {
    local host=$1
    local port=$2
    local service_name=$3
    
    if curl -s "http://$host:$port/health" > /dev/null 2>&1; then
        print_success "$service_name is running on $host:$port"
        return 0
    else
        print_error "$service_name is not responding on $host:$port"
        return 1
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready on $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if check_service "$host" "$port" "$service_name" > /dev/null 2>&1; then
            print_success "$service_name is ready on $host:$port"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts - waiting 10 seconds..."
        sleep 10
        ((attempt++))
    done
    
    print_error "$service_name failed to start on $host:$port after $max_attempts attempts"
    return 1
}

# Function to start services on a remote node
start_remote_services() {
    local host=$1
    local node_name=$2
    local services=("${@:3}")
    
    print_status "Starting services on $node_name ($host)..."
    
    for service in "${services[@]}"; do
        print_status "Starting $service on $node_name..."
        
        case $service in
            "fingpt"|"stockgpt")
                ssh -o ConnectTimeout=10 "sanzad@$host" "sudo systemctl start trading-llm.service"
                ;;
            "finrl_portfolio"|"finrl_risk"|"finrl_trading")
                ssh -o ConnectTimeout=10 "sanzad@$host" "sudo systemctl start finrl-trading.service"
                ;;
            "langgraph_orchestrator"|"workflow_engine")
                ssh -o ConnectTimeout=10 "sanzad@$host" "sudo systemctl start trading-orchestrator.service"
                ;;
            "market_data_service"|"technical_indicators"|"backtesting_engine")
                ssh -o ConnectTimeout=10 "sanzad@$host" "sudo systemctl start market-data-services.service"
                ;;
            *)
                print_warning "Unknown service: $service"
                ;;
        esac
        
        # Wait a moment for service to start
        sleep 5
    done
}

# Function to check cluster health
check_cluster_health() {
    print_status "Checking cluster health..."
    
    local all_healthy=true
    
    # Check GPU nodes
    if ! check_service "$GPU_NODE" "8080" "FinGPT/StockGPT Service"; then
        all_healthy=false
    fi
    
    if ! check_service "$GPU_NODE1" "8081" "FinRL Trading Service"; then
        all_healthy=false
    fi
    
    # Check CPU nodes
    if ! check_service "$CPU_NODE" "8084" "Trading Orchestrator"; then
        all_healthy=false
    fi
    
    if ! check_service "$WORKER_NODE3" "8085" "Market Data Services"; then
        all_healthy=false
    fi
    
    if [ "$all_healthy" = true ]; then
        print_success "All cluster services are healthy!"
        return 0
    else
        print_error "Some cluster services are not healthy"
        return 1
    fi
}

# Function to start the entire cluster
start_cluster() {
    print_status "🚀 Starting Distributed Trading Cluster..."
    
    # Start GPU nodes
    print_status "Starting GPU nodes..."
    start_remote_services "$GPU_NODE" "gpu-node" "fingpt" "stockgpt"
    start_remote_services "$GPU_NODE1" "gpu-node1" "finrl_portfolio" "finrl_risk" "finrl_trading"
    
    # Start CPU nodes
    print_status "Starting CPU nodes..."
    start_remote_services "$CPU_NODE" "cpu-node" "langgraph_orchestrator" "workflow_engine"
    start_remote_services "$WORKER_NODE3" "worker-node3" "market_data_service" "technical_indicators" "backtesting_engine"
    
    # Wait for all services to be ready
    print_status "Waiting for all services to be ready..."
    
    wait_for_service "$GPU_NODE" "8080" "FinGPT/StockGPT Service"
    wait_for_service "$GPU_NODE1" "8081" "FinRL Trading Service"
    wait_for_service "$CPU_NODE" "8084" "Trading Orchestrator"
    wait_for_service "$WORKER_NODE3" "8085" "Market Data Services"
    
    # Check final cluster health
    if check_cluster_health; then
        print_success "✅ Trading cluster started successfully!"
        print_status "Dashboard: http://$CPU_NODE1:8082"
        print_status "API Endpoints:"
        print_status "  - FinGPT/StockGPT: http://$GPU_NODE:8080"
        print_status "  - FinRL Trading: http://$GPU_NODE1:8081"
        print_status "  - Trading Orchestrator: http://$CPU_NODE1:8082"
        print_status "  - Market Data Services: http://$CPU_NODE2:8083"
    else
        print_error "❌ Trading cluster startup failed"
        exit 1
    fi
}

# Function to stop the entire cluster
stop_cluster() {
    print_status "🛑 Stopping Distributed Trading Cluster..."
    
    # Stop GPU nodes
    print_status "Stopping GPU nodes..."
    ssh -o ConnectTimeout=10 "sanzad@$GPU_NODE" "sudo systemctl stop trading-llm.service" || true
    ssh -o ConnectTimeout=10 "sanzad@$GPU_NODE1" "sudo systemctl stop finrl-trading.service" || true
    
    # Stop CPU nodes
    print_status "Stopping CPU nodes..."
    ssh -o ConnectTimeout=10 "sanzad@$CPU_NODE" "sudo systemctl stop trading-orchestrator.service" || true
    ssh -o ConnectTimeout=10 "sanzad@$WORKER_NODE3" "sudo systemctl stop market-data-services.service" || true
    
    print_success "✅ Trading cluster stopped successfully!"
}

# Function to restart the entire cluster
restart_cluster() {
    print_status "🔄 Restarting Distributed Trading Cluster..."
    stop_cluster
    sleep 10
    start_cluster
}

# Function to check cluster status
check_status() {
    print_status "📊 Checking cluster status..."
    
    # Check GPU nodes
    print_status "GPU Nodes:"
    check_service "$GPU_NODE" "8080" "FinGPT/StockGPT Service" || true
    check_service "$GPU_NODE1" "8081" "FinRL Trading Service" || true
    
    # Check CPU nodes
    print_status "CPU Nodes:"
    check_service "$CPU_NODE" "8084" "Trading Orchestrator" || true
    check_service "$WORKER_NODE3" "8085" "Market Data Services" || true
    
    # Check overall health
    if check_cluster_health; then
        print_success "Cluster is healthy and operational"
    else
        print_warning "Cluster has some issues"
    fi
}

# Function to test the trading workflow
test_workflow() {
    print_status "🧪 Testing trading workflow..."
    
    # Test a simple workflow
    local test_symbols='["AAPL", "GOOGL", "MSFT"]'
    
    print_status "Starting test trading analysis workflow..."
    
    local response=$(curl -s -X POST "http://$CPU_NODE:8084/start_trading_analysis" \
        -H "Content-Type: application/json" \
        -d "{\"symbols\": $test_symbols, \"workflow_id\": \"test_$(date +%s)\"}")
    
    if [ $? -eq 0 ]; then
        local workflow_id=$(echo "$response" | grep -o '"workflow_id":"[^"]*"' | cut -d'"' -f4)
        if [ -n "$workflow_id" ]; then
            print_success "Test workflow started with ID: $workflow_id"
            print_status "You can monitor progress at: http://$CPU_NODE:8084/workflow/$workflow_id/status"
        else
            print_error "Failed to extract workflow ID from response"
        fi
    else
        print_error "Failed to start test workflow"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 {start|stop|restart|status|test}"
    echo ""
    echo "Commands:"
    echo "  start   - Start the entire distributed trading cluster"
    echo "  stop    - Stop the entire distributed trading cluster"
    echo "  restart - Restart the entire distributed trading cluster"
    echo "  status  - Check the status of all cluster services"
    echo "  test    - Test the trading workflow with sample data"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start the cluster"
    echo "  $0 status   # Check cluster health"
    echo "  $0 test     # Test trading workflow"
}

# Main script logic
case "${1:-}" in
    start)
        start_cluster
        ;;
    stop)
        stop_cluster
        ;;
    restart)
        restart_cluster
        ;;
    status)
        check_status
        ;;
    test)
        test_workflow
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

exit 0
