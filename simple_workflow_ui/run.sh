#!/bin/bash

# LangGraph Research Workflow UI Setup & Launch Script

set -e  # Exit on any error

echo "🚀 Starting LangGraph Research Workflow UI..."

# Check if we're in the correct directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Please run this script from the simple_workflow_ui directory"
    echo "   Expected: simple_workflow_ui/run.sh"
    exit 1
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip to latest version
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies with proper version resolution
echo "📚 Installing Python dependencies..."
echo "   - Resolving FastAPI compatibility issues..."
pip install "fastapi>=0.111.0" "uvicorn>=0.30.0" --upgrade

echo "   - Installing LLM formatting libraries..."
pip install instructor rich marvin --upgrade

echo "   - Installing remaining dependencies..."
pip install -r backend/requirements.txt

echo "✅ Dependencies installed successfully!"

# Start the FastAPI server
echo "🖥️  Starting FastAPI server..."
echo ""
echo "   🌐 Server will be available at:"
echo "   🔗 Access the UI at: http://192.168.1.81:8000"
echo "   📖 API docs at: http://192.168.1.81:8000/docs"
echo ""
echo "   💡 Test with: curl http://192.168.1.81:8000/health"
echo "   📊 Cluster status: curl http://192.168.1.81:8000/api/cluster/status"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

cd backend && python main.py
