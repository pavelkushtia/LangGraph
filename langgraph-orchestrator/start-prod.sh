#!/bin/bash
echo "🚀 Starting LangGraph Orchestrator in production mode..."

# Kill any existing processes on these ports
echo "🧹 Cleaning up existing processes..."
pkill -f "node.*3001" 2>/dev/null || true
pkill -f "node.*3000" 2>/dev/null || true
sleep 2

# Build frontend
echo "🏗️ Building frontend..."
cd frontend && npm run build && cd ..

# Copy built frontend to backend's public directory
echo "📦 Preparing production assets..."
mkdir -p backend/public
cp -r frontend/dist/* backend/public/

# Start backend in production mode
echo "🚀 Starting production server..."
cd backend && npm start
