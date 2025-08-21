# LangGraph Integration with Local Models

## Overview
This guide shows how to integrate LangGraph with your distributed local model infrastructure, creating powerful AI workflows without external API costs.

## Installation and Setup

### Install LangGraph and Dependencies

```bash
# Create main environment for LangGraph orchestrator
python3 -m venv ~/langgraph-env
source ~/langgraph-env/bin/activate

# Install LangGraph and related packages
pip install langgraph langchain langchain-community langchain-core
pip install httpx requests asyncio aiohttp
pip install chromadb sentence-transformers
pip install fastapi uvicorn

# Optional: For advanced workflows
pip install langchain-experimental pandas numpy
```

## Core Integration Components

### 1. Local Model Providers

Create custom LangChain LLM providers for your local infrastructure:

```python
# local_models.py
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any
import requests
import json

class JetsonOllamaLLM(LLM):
    """Jetson Orin Nano Ollama LLM"""
    
    jetson_url: str = "http://192.168.1.177:11434"
    model_name: str = "llama3.2:3b"
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "jetson_ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response = requests.post(
            f"{self.jetson_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "stop": stop or []
                }
            }
        )
        return response.json()["response"]

class CPULlamaLLM(LLM):
    """CPU llama.cpp server LLM - Optional, can use Jetson-only setup"""
    
    cpu_url: str = "http://192.168.1.81:8080"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    @property
    def _llm_type(self) -> str:
        return "cpu_llama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response = requests.post(
            f"{self.cpu_url}/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",  # llama.cpp server expects this
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stop": stop
            }
        )
        return response.json()["choices"][0]["message"]["content"]

class LocalEmbeddings:
    """Local embeddings from 16GB machine A"""
    
    def __init__(self, embeddings_url: str = "http://192.168.1.178:8081"):
        self.embeddings_url = embeddings_url
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            f"{self.embeddings_url}/embeddings",
            json=texts
        )
        return response.json()["embeddings"]
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
```

### 2. Tool Integration

```python
# local_tools.py
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import requests

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query")

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for information"
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str) -> str:
        response = requests.post(
            "http://192.168.1.105:8082/web_search",
            json={"query": query}
        )
        return response.json()["results"]

class WebScrapeTool(BaseTool):
    name = "web_scrape"
    description = "Scrape content from a webpage"
    
    def _run(self, url: str) -> str:
        response = requests.post(
            "http://192.168.1.105:8082/web_scrape",
            json={"url": url}
        )
        return response.json()["content"]

class CommandExecuteTool(BaseTool):
    name = "execute_command"
    description = "Execute a shell command safely"
    
    def _run(self, command: str) -> str:
        response = requests.post(
            "http://192.168.1.105:8082/execute_command",
            json={"command": command}
        )
        result = response.json()
        return f"STDOUT: {result['stdout']}\nSTDERR: {result['stderr']}"
```

### 3. LangGraph Workflow Definition

```python
# langgraph_workflow.py
from langgraph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
import operator
from local_models import JetsonOllamaLLM, CPULlamaLLM
from local_tools import WebSearchTool, WebScrapeTool, CommandExecuteTool

# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_model: str
    task_complexity: str
    tools_used: list

# Initialize models and tools
jetson_llm = JetsonOllamaLLM()
cpu_llm = CPULlamaLLM()
web_search = WebSearchTool()
web_scrape = WebScrapeTool()
command_exec = CommandExecuteTool()

def route_to_model(state: AgentState) -> str:
    """Route to appropriate model based on task complexity"""
    last_message = state["messages"][-1].content
    
    # Simple heuristics for routing
    if len(last_message) < 100 and "code" not in last_message.lower():
        return "jetson_fast"
    elif "complex" in last_message.lower() or len(last_message) > 500:
        return "cpu_heavy"
    else:
        return "jetson_standard"

def jetson_fast_node(state: AgentState) -> AgentState:
    """Quick responses using Jetson with small model"""
    jetson_llm.model_name = "tinyllama:1.1b"
    response = jetson_llm.invoke(state["messages"][-1].content)
    
    return {
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_model": "jetson_tinyllama",
        "task_complexity": "simple",
        "tools_used": state.get("tools_used", [])
    }

def jetson_standard_node(state: AgentState) -> AgentState:
    """Standard responses using Jetson with medium model"""
    jetson_llm.model_name = "llama3.2:3b"
    response = jetson_llm.invoke(state["messages"][-1].content)
    
    return {
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_model": "jetson_llama3.2",
        "task_complexity": "medium",
        "tools_used": state.get("tools_used", [])
    }

def cpu_heavy_node(state: AgentState) -> AgentState:
    """Complex tasks using CPU large model"""
    response = cpu_llm.invoke(state["messages"][-1].content)
    
    return {
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_model": "cpu_llama13b",
        "task_complexity": "complex",
        "tools_used": state.get("tools_used", [])
    }

def tool_execution_node(state: AgentState) -> AgentState:
    """Execute tools when needed"""
    last_message = state["messages"][-1].content.lower()
    tools_used = state.get("tools_used", [])
    
    if "search" in last_message:
        # Extract search query (simplified)
        query = last_message.split("search")[-1].strip()
        result = web_search.run(query)
        tools_used.append("web_search")
        response = f"Search results: {result}"
    elif "scrape" in last_message and "http" in last_message:
        # Extract URL (simplified)
        url = [word for word in last_message.split() if word.startswith("http")][0]
        result = web_scrape.run(url)
        tools_used.append("web_scrape")
        response = f"Scraped content: {result}"
    else:
        response = "No tools needed for this request."
    
    return {
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_model": state["current_model"],
        "task_complexity": state["task_complexity"],
        "tools_used": tools_used
    }

def should_use_tools(state: AgentState) -> str:
    """Decide if tools are needed"""
    last_message = state["messages"][-1].content.lower()
    
    if any(keyword in last_message for keyword in ["search", "scrape", "url", "command"]):
        return "use_tools"
    else:
        return "no_tools"

# Build the workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("jetson_fast", jetson_fast_node)
workflow.add_node("jetson_standard", jetson_standard_node)
workflow.add_node("cpu_heavy", cpu_heavy_node)
workflow.add_node("tool_execution", tool_execution_node)

# Add edges
workflow.set_conditional_entry_point(
    route_to_model,
    {
        "jetson_fast": "jetson_fast",
        "jetson_standard": "jetson_standard",
        "cpu_heavy": "cpu_heavy"
    }
)

# Add conditional tool usage
workflow.add_conditional_edges(
    "jetson_fast",
    should_use_tools,
    {
        "use_tools": "tool_execution",
        "no_tools": END
    }
)

workflow.add_conditional_edges(
    "jetson_standard",
    should_use_tools,
    {
        "use_tools": "tool_execution",
        "no_tools": END
    }
)

workflow.add_conditional_edges(
    "cpu_heavy",
    should_use_tools,
    {
        "use_tools": "tool_execution",
        "no_tools": END
    }
)

workflow.add_edge("tool_execution", END)

# Compile the workflow
app = workflow.compile()
```

### 4. Configuration Manager

```python
# config.py
import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class MachineConfig:
    ip: str
    port: int
    service_type: str
    health_endpoint: str

@dataclass
class ClusterConfig:
    jetson_orin: MachineConfig
    cpu_32gb: MachineConfig
    cpu_16gb_a: MachineConfig
    cpu_16gb_b: MachineConfig
    cpu_8gb_a: MachineConfig
    cpu_8gb_b: MachineConfig

# Actual available nodes in your cluster
CLUSTER = ClusterConfig(
    jetson_orin=MachineConfig(
        ip="192.168.1.177",  # jetson-node (Orin Nano 8GB)
        port=11434,
        service_type="ollama",
        health_endpoint="/api/tags"
    ),
    cpu_32gb=MachineConfig(
        ip="192.168.1.81",   # cpu-node (32GB RAM coordinator)
        port=8080,
        service_type="llama_cpp",
        health_endpoint="/health"
    ),
    cpu_16gb_a=MachineConfig(
        ip="192.168.1.178",  # rp-node (8GB ARM, embeddings)
        port=8081,
        service_type="embeddings",
        health_endpoint="/health"
    ),
    cpu_16gb_b=MachineConfig(
        ip="192.168.1.105",  # worker-node3 (6GB VM, tools)
        port=8082,
        service_type="tools",
        health_endpoint="/health"
    ),
    cpu_8gb_a=MachineConfig(
        ip="192.168.1.137",  # worker-node4 (6GB VM, monitoring)
        port=8083,
        service_type="monitoring",
        health_endpoint="/cluster_health"
    ),
    cpu_8gb_b=MachineConfig(
        ip="192.168.1.81",   # cpu-node also handles redis
        port=6379,
        service_type="redis",
        health_endpoint="/ping"
    )
)

def get_service_url(service: str) -> str:
    """Get the full URL for a service"""
    machine_map = {
        "jetson": CLUSTER.jetson_orin,
        "cpu_llm": CLUSTER.cpu_32gb,
        "embeddings": CLUSTER.cpu_16gb_a,
        "tools": CLUSTER.cpu_16gb_b,
        "monitoring": CLUSTER.cpu_8gb_a,
        "redis": CLUSTER.cpu_8gb_b
    }
    
    machine = machine_map[service]
    return f"http://{machine.ip}:{machine.port}"
```

### 5. Main Application

```python
# main.py
from langgraph_workflow import app
from langchain.schema import HumanMessage
import asyncio
from config import get_service_url

async def run_workflow(user_input: str):
    """Run the LangGraph workflow with user input"""
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "current_model": "",
        "task_complexity": "",
        "tools_used": []
    }
    
    result = await app.ainvoke(initial_state)
    return result

def main():
    print("🚀 Local LangGraph Setup Ready!")
    print("Available services:")
    print(f"  - Jetson Ollama: {get_service_url('jetson')}")
    print(f"  - CPU LLM: {get_service_url('cpu_llm')}")
    print(f"  - Embeddings: {get_service_url('embeddings')}")
    print(f"  - Tools: {get_service_url('tools')}")
    print(f"  - Monitoring: {get_service_url('monitoring')}")
    
    while True:
        user_input = input("\n💬 Enter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        try:
            result = asyncio.run(run_workflow(user_input))
            print(f"\n🤖 Response: {result['messages'][-1].content}")
            print(f"📊 Model used: {result['current_model']}")
            print(f"🔧 Tools used: {result['tools_used']}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

## Running Your Setup

1. **Start all services** (follow previous setup guides)
2. **Update IPs** in `config.py`
3. **Run the main application**:

```bash
source ~/langgraph-env/bin/activate
cd ~/langgraph
python main.py
```

## Example Workflows

### Simple Query (Routes to Jetson)
```
User: "What is Python?"
→ Routes to jetson_fast (TinyLlama)
→ Quick response about Python
```

### Complex Analysis (Routes to CPU)
```
User: "Analyze the pros and cons of microservices architecture in detail"
→ Routes to cpu_heavy (Llama 13B)
→ Comprehensive analysis
```

### Tool Usage
```
User: "Search for latest AI news"
→ Routes to jetson_standard
→ Triggers web_search tool
→ Returns search results
```

## Monitoring and Health Checks

```python
# health_check.py
import requests
from config import CLUSTER

def check_cluster_health():
    """Check health of all services"""
    for name, machine in CLUSTER.__dict__.items():
        try:
            response = requests.get(
                f"http://{machine.ip}:{machine.port}{machine.health_endpoint}",
                timeout=5
            )
            status = "✅ Healthy" if response.status_code == 200 else "❌ Unhealthy"
        except:
            status = "❌ Unreachable"
        
        print(f"{name}: {status}")

if __name__ == "__main__":
    check_cluster_health()
```

## Next Steps
- Test the complete workflow
- Add more sophisticated routing logic
- Implement caching strategies
- Add monitoring and alerting
- Create domain-specific workflows
