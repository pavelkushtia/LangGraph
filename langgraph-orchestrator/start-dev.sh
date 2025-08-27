#!/bin/bash
echo "🚀 Starting LangGraph Orchestrator in development mode..."

# Kill any existing processes on these ports
echo "🧹 Cleaning up existing processes..."
pkill -f "node.*3001" 2>/dev/null || true
pkill -f "node.*3000" 2>/dev/null || true
sleep 2

# Start backend and frontend in parallel
trap 'kill $(jobs -p) 2>/dev/null; exit' EXIT INT TERM

echo "🔧 Starting backend server..."
(cd backend && npm run dev) &
BACKEND_PID=$!

echo "🎨 Starting frontend server..."
(cd frontend && npm run dev) &
FRONTEND_PID=$!

echo "✅ Services starting up..."
echo "   - Backend: http://192.168.1.81:3001"
echo "   - Frontend: http://192.168.1.81:3000"
echo "   - Accessible from all cluster nodes"
echo ""
echo "📱 Access the UI from any cluster node:"
echo "   - Jetson (192.168.1.177): http://192.168.1.81:3000"
echo "   - RPi (192.168.1.178): http://192.168.1.81:3000"
echo "   - Worker Tools (192.168.1.190): http://192.168.1.81:3000"
echo "   - Worker Monitor (192.168.1.191): http://192.168.1.81:3000"
echo ""
echo "⏹️  Press Ctrl+C to stop all services"

# Wait for both processes
wait
