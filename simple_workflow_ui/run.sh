#!/bin/bash

echo "🚀 Starting LangGraph Research Workflow UI..."

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Please run this script from the simple_workflow_ui directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r backend/requirements.txt

# Start the server
echo "🌐 Starting FastAPI server..."
echo "   🔗 Access the UI at: http://192.168.1.81:8000"
echo "   📚 API docs at: http://192.168.1.81:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"

cd backend && python main.py
