#!/usr/bin/env python3
"""
FastAPI server for LangGraph Research Workflow UI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import asyncio
import os
from typing import Dict, Any

from workflow import ResearchWorkflow

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph Research Workflow",
    description="Simple web UI for executing research workflows on your LangGraph cluster",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow
workflow_engine = ResearchWorkflow()

# Request/Response models
class ResearchRequest(BaseModel):
    query: str

class ResearchResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    formatted_html: str = ""
    error: str = ""

# Serve static files (frontend)
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/style.css")
async def serve_css():
    """Serve the CSS file"""
    css_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "style.css")
    return FileResponse(css_path, media_type="text/css")

@app.get("/app.js")
async def serve_js():
    """Serve the JavaScript file"""
    js_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "app.js")
    return FileResponse(js_path, media_type="application/javascript")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page"""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return HTMLResponse("""
    <html>
        <body>
            <h1>LangGraph Research Workflow</h1>
            <p>Frontend not found. Please ensure frontend files are in the correct directory.</p>
            <p>API is available at <a href="/docs">/docs</a></p>
        </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "LangGraph Research Workflow API"}

@app.post("/api/research", response_model=ResearchResponse)
async def execute_research(request: ResearchRequest):
    """Execute a research workflow"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Execute the workflow
        result = await workflow_engine.execute(request.query)
        
        return ResearchResponse(
            success=result["success"],
            result=result.get("result", {}),
            formatted_html=result.get("formatted_html", ""),
            error=result.get("error", "")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@app.get("/api/cluster/status")
async def cluster_status():
    """Check cluster status"""
    import requests
    
    services = {
        "jetson_ollama": "http://192.168.1.177:11434/api/tags",
        "cpu_ollama": "http://192.168.1.81:11435/api/tags", 
        "tools_server": "http://192.168.1.190:8082/",
        "embeddings_server": "http://192.168.1.178:8081/",
        "monitoring_server": "http://192.168.1.191:8083/"
    }
    
    status = {}
    for service, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            status[service] = {
                "status": "healthy" if response.status_code < 500 else "degraded",
                "response_time": "< 5s"
            }
        except Exception as e:
            status[service] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    overall = "healthy" if all(s["status"] == "healthy" for s in status.values()) else "degraded"
    
    return {
        "overall": overall,
        "services": status,
        "timestamp": "now"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
