# LangGraph Research Workflow UI

A simple, clean web interface for executing research workflows on your distributed LangGraph cluster.

## 🚀 Features

- **LangGraph Integration**: Real workflow orchestration using StateGraph
- **Distributed Processing**: Uses your actual cluster nodes (Jetson, CPU, RPi, Workers)
- **Beautiful UI**: Clean, responsive interface with progress tracking
- **Markdown Formatting**: Properly formatted LLM responses with syntax highlighting
- **Real-time Status**: Live cluster health monitoring
- **FastAPI Backend**: Modern, fast Python API

## 🏗️ Architecture

```
simple_workflow_ui/
├── backend/          # FastAPI server
│   ├── main.py      # API endpoints
│   ├── workflow.py  # LangGraph workflow logic
│   └── requirements.txt
├── frontend/         # Static web UI
│   ├── index.html   # Main interface
│   ├── app.js       # JavaScript logic
│   └── style.css    # Custom styles
└── README.md
```

## 🎯 Workflow Steps

1. **Planning** (Jetson Orin Nano) - Fast planning with llama3.2:3b
2. **Search** (Worker Tools) - Web search for information
3. **Analysis** (CPU Coordinator) - Deep analysis with powerful model
4. **Finalization** - Combine results into formatted output

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd simple_workflow_ui/backend
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python main.py
```

The UI will be available at: **http://192.168.1.81:8000**

### 3. Access from Any Cluster Node

- **Jetson**: http://192.168.1.81:8000
- **RPi**: http://192.168.1.81:8000  
- **Worker Tools**: http://192.168.1.81:8000
- **Worker Monitor**: http://192.168.1.81:8000

## 📊 API Endpoints

- `GET /` - Main UI interface
- `POST /api/research` - Execute research workflow
- `GET /api/cluster/status` - Check cluster health
- `GET /health` - API health check

## 🎨 UI Features

### Research Interface
- **Query Input**: Natural language research queries
- **Example Queries**: One-click example prompts
- **Progress Tracking**: Real-time workflow progress
- **Results Display**: Formatted markdown with syntax highlighting

### Cluster Monitoring
- **Live Status**: Real-time cluster health indicator
- **Service Details**: Individual service status
- **Visual Indicators**: Color-coded health status

## 🔧 Technologies Used

**Backend:**
- FastAPI - Modern Python web framework
- LangGraph - Workflow orchestration
- Markdown - Response formatting
- Requests - HTTP client for cluster communication

**Frontend:**
- Tailwind CSS - Utility-first styling
- Marked.js - Markdown rendering
- Prism.js - Syntax highlighting
- Vanilla JavaScript - No heavy frameworks

## 📝 Example Queries

- "Latest breakthroughs in artificial intelligence and machine learning in 2024"
- "Climate change impacts and renewable energy solutions"
- "Current trends in cryptocurrency and blockchain technology"
- "Recent developments in quantum computing applications"

## 🛠️ Customization

### Add New Workflow Steps
Edit `backend/workflow.py` to add new nodes to the StateGraph.

### Modify UI
- `frontend/index.html` - Structure
- `frontend/style.css` - Styling  
- `frontend/app.js` - Behavior

### Configure Cluster Endpoints
Update the service URLs in `backend/main.py` and `backend/workflow.py`.

## 🔍 Troubleshooting

### Cluster Services Not Responding
Check that all services are running:
```bash
# Check status via cluster orchestrator
cd /home/sanzad/ai-infrastructure/langgraph-config
python cluster_orchestrator.py status
```

### API Errors
Check the FastAPI logs and ensure all cluster endpoints are accessible.

### UI Not Loading
Verify the frontend files are in the correct directory and the server is running on port 8000.

## 🎯 Next Steps

- Add workflow templates for different research types
- Implement workflow history and caching
- Add user authentication and saved queries
- Extend with more LangGraph nodes (embeddings, document processing)
- Add real-time streaming results
