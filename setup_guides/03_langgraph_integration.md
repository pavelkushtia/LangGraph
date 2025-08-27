# LangGraph Integration with Local Models

## Overview
This guide integrates LangGraph with your distributed local model infrastructure, creating powerful AI workflows that route intelligently across your cluster without external API costs.

## Prerequisites
- **jetson-node** (192.168.1.177) - Ollama server running
- **cpu-node** (192.168.1.81) - llama.cpp + HAProxy + Redis running
- **rp-node** (192.168.1.178) - Embeddings server (next step)
- **worker-node3** (192.168.1.190) - Tools server (next step) 
- **worker-node4** (192.168.1.191) - Monitoring server (next step)

---

## Step 1: Install LangGraph Environment

```bash
# SSH to cpu-node (coordinator)
ssh sanzad@192.168.1.81

# Navigate to LangGraph directory
cd ~/ai-infrastructure
source langgraph-env/bin/activate

# Verify LangGraph installation
python -c "import langgraph; print('LangGraph version:', langgraph.__version__)"
```

## Step 2: Create Local Model Providers

```bash
# Create local models integration
cd ~/ai-infrastructure/langgraph-config

cat > local_models.py << 'EOF'
"""
Local LLM providers for LangGraph cluster
Integrates with jetson-node Ollama and cpu-node llama.cpp
"""

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings.base import Embeddings
from typing import Optional, List, Any
import requests
import json
import time

class JetsonOllamaLLM(LLM):
    """Jetson Orin Nano Ollama LLM - Fast responses"""
    
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
        try:
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
                },
                timeout=30
            )
            return response.json()["response"]
        except Exception as e:
            raise Exception(f"Jetson Ollama error: {e}")

class CPULlamaLLM(LLM):
    """CPU llama.cpp server LLM - Complex tasks"""
    
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
        try:
            response = requests.post(
                f"{self.cpu_url}/v1/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",  # llama.cpp server expects this
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stop": stop
                },
                timeout=60
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"CPU llama.cpp error: {e}")

class LoadBalancedLLM(LLM):
    """Load-balanced LLM using HAProxy"""
    
    haproxy_url: str = "http://192.168.1.81:9000"
    model_name: str = "llama3.2:3b"
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "load_balanced"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            # HAProxy routes to best available backend
            response = requests.post(
                f"{self.haproxy_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "stop": stop or []
                    }
                },
                timeout=45
            )
            return response.json()["response"]
        except Exception as e:
            raise Exception(f"Load balancer error: {e}")

class LocalEmbeddings(Embeddings):
    """Local embeddings from rp-node"""
    
    def __init__(self, embeddings_url: str = "http://192.168.1.178:8081"):
        self.embeddings_url = embeddings_url
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = requests.post(
                f"{self.embeddings_url}/embeddings",
                json={"texts": texts, "model": "default"},
                timeout=30
            )
            return response.json()["embeddings"]
        except Exception as e:
            raise Exception(f"Embeddings error: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
EOF
```

## Step 3: Create Tool Integration

```bash
cat > local_tools.py << 'EOF'
"""
Local tools integration for LangGraph workflows
Connects to worker-node3 tools server
"""

from langchain.tools import BaseTool
from typing import Type, Dict, Any
from pydantic import BaseModel, Field
import requests

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum number of results")

class WebScrapeInput(BaseModel):
    url: str = Field(description="URL to scrape")
    extract_text: bool = Field(default=True, description="Extract text content")

class CommandInput(BaseModel):
    command: str = Field(description="Shell command to execute")
    timeout: int = Field(default=30, description="Command timeout in seconds")

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for information using DuckDuckGo"
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            response = requests.post(
                "http://192.168.1.190:8082/web_search",
                json={"query": query, "max_results": max_results},
                timeout=30
            )
            result = response.json()
            
            if "results" in result:
                formatted_results = []
                for i, item in enumerate(result["results"][:max_results], 1):
                    formatted_results.append(
                        f"{i}. **{item.get('title', 'No title')}**\n"
                        f"   URL: {item.get('url', 'No URL')}\n"
                        f"   Summary: {item.get('snippet', 'No snippet')}"
                    )
                return "\n\n".join(formatted_results)
            else:
                return f"Search failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Web search error: {e}"

class WebScrapeTool(BaseTool):
    name = "web_scrape"
    description = "Scrape content from a specific webpage"
    args_schema: Type[BaseModel] = WebScrapeInput
    
    def _run(self, url: str, extract_text: bool = True) -> str:
        try:
            response = requests.post(
                "http://192.168.1.190:8082/web_scrape",
                json={"url": url, "extract_text": extract_text},
                timeout=30
            )
            result = response.json()
            
            if "text" in result:
                return f"**Title:** {result.get('title', 'No title')}\n\n**Content:** {result['text'][:2000]}..."
            else:
                return f"Scraping failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Web scraping error: {e}"

class CommandExecuteTool(BaseTool):
    name = "execute_command"
    description = "Execute a safe shell command on the tools server"
    args_schema: Type[BaseModel] = CommandInput
    
    def _run(self, command: str, timeout: int = 30) -> str:
        try:
            response = requests.post(
                "http://192.168.1.190:8082/execute_command",
                json={"command": command, "timeout": timeout},
                timeout=timeout + 5
            )
            result = response.json()
            
            if "stdout" in result:
                output = f"**Return Code:** {result.get('returncode', 'unknown')}\n"
                if result.get('stdout'):
                    output += f"**Output:**\n{result['stdout']}\n"
                if result.get('stderr'):
                    output += f"**Errors:**\n{result['stderr']}\n"
                return output
            else:
                return f"Command failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Command execution error: {e}"
EOF
```

## Step 4: Create LangGraph Workflow

```bash
cat > langgraph_workflow.py << 'EOF'
"""
Main LangGraph workflow with intelligent routing
Routes tasks based on complexity and requirements
"""

from langgraph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence, Dict, Any
import operator
import time
from local_models import JetsonOllamaLLM, CPULlamaLLM, LoadBalancedLLM, LocalEmbeddings
from local_tools import WebSearchTool, WebScrapeTool, CommandExecuteTool

# Define the workflow state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_model: str
    task_complexity: str
    tools_used: List[str]
    context: str
    results: Dict[str, Any]

# Initialize models and tools
jetson_fast = JetsonOllamaLLM(model_name="tinyllama:1.1b-chat-fp16")
jetson_standard = JetsonOllamaLLM(model_name="llama3.2:3b")
cpu_heavy = CPULlamaLLM()
load_balanced = LoadBalancedLLM()
embeddings = LocalEmbeddings()

# Initialize tools
web_search = WebSearchTool()
web_scrape = WebScrapeTool()
command_exec = CommandExecuteTool()

def analyze_request(state: AgentState) -> str:
    """Analyze user request and route to appropriate model"""
    last_message = state["messages"][-1].content.lower()
    message_length = len(state["messages"][-1].content)
    
    # Check for tool requirements
    if any(keyword in last_message for keyword in ["search", "find", "look up", "web"]):
        return "needs_search"
    elif any(keyword in last_message for keyword in ["scrape", "extract", "url", "website"]):
        return "needs_scraping"
    elif any(keyword in last_message for keyword in ["run", "execute", "command", "check"]):
        return "needs_command"
    
    # Route based on complexity
    elif message_length < 50 and any(word in last_message for word in ["hello", "hi", "what", "simple"]):
        return "simple_task"
    elif message_length > 200 or any(word in last_message for word in ["complex", "analyze", "detailed", "comprehensive"]):
        return "complex_task"
    else:
        return "standard_task"

def simple_response_node(state: AgentState) -> AgentState:
    """Handle simple queries with fast Jetson model"""
    user_message = state["messages"][-1].content
    
    response = jetson_fast.invoke(user_message)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_model": "jetson_tinyllama",
        "task_complexity": "simple",
        "results": {**state.get("results", {}), "response": response}
    }

def standard_response_node(state: AgentState) -> AgentState:
    """Handle standard queries with balanced Jetson model"""
    user_message = state["messages"][-1].content
    
    response = jetson_standard.invoke(user_message)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_model": "jetson_llama3.2",
        "task_complexity": "standard",
        "results": {**state.get("results", {}), "response": response}
    }

def complex_response_node(state: AgentState) -> AgentState:
    """Handle complex queries with CPU heavy model"""
    user_message = state["messages"][-1].content
    
    # Add context if available
    context = state.get("context", "")
    if context:
        enhanced_prompt = f"Context: {context}\n\nUser Question: {user_message}"
    else:
        enhanced_prompt = user_message
    
    response = cpu_heavy.invoke(enhanced_prompt)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_model": "cpu_llama13b",
        "task_complexity": "complex",
        "results": {**state.get("results", {}), "response": response}
    }

def search_and_respond_node(state: AgentState) -> AgentState:
    """Search web and provide informed response"""
    user_message = state["messages"][-1].content
    
    # Extract search query (simplified)
    search_query = user_message.replace("search for", "").replace("find", "").strip()
    if not search_query:
        search_query = user_message
    
    # Perform search
    search_results = web_search.run(search_query)
    
    # Generate response with search context
    enhanced_prompt = f"""
    User Question: {user_message}
    
    Search Results:
    {search_results}
    
    Please provide a comprehensive answer based on the search results above.
    """
    
    response = jetson_standard.invoke(enhanced_prompt)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_model": "jetson_with_search",
        "task_complexity": "standard",
        "tools_used": state.get("tools_used", []) + ["web_search"],
        "context": search_results,
        "results": {**state.get("results", {}), "search_results": search_results, "response": response}
    }

def scrape_and_respond_node(state: AgentState) -> AgentState:
    """Scrape URL and provide analysis"""
    user_message = state["messages"][-1].content
    
    # Extract URL (simplified)
    url = None
    for word in user_message.split():
        if word.startswith("http"):
            url = word
            break
    
    if not url:
        response = "I need a URL to scrape. Please provide a valid URL."
    else:
        # Perform scraping
        scraped_content = web_scrape.run(url)
        
        # Generate response with scraped content
        enhanced_prompt = f"""
        User Request: {user_message}
        
        Scraped Content:
        {scraped_content}
        
        Please analyze and summarize the content based on the user's request.
        """
        
        response = jetson_standard.invoke(enhanced_prompt)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_model": "jetson_with_scraping",
        "task_complexity": "standard",
        "tools_used": state.get("tools_used", []) + ["web_scrape"],
        "results": {**state.get("results", {}), "scraped_content": scraped_content if url else None, "response": response}
    }

def command_and_respond_node(state: AgentState) -> AgentState:
    """Execute command and provide response"""
    user_message = state["messages"][-1].content
    
    # Extract command (simplified - in production, use better parsing)
    command = user_message.replace("run", "").replace("execute", "").strip()
    if command.startswith("command"):
        command = command[7:].strip()
    
    # Execute command
    command_result = command_exec.run(command)
    
    # Generate response with command output
    enhanced_prompt = f"""
    User Request: {user_message}
    
    Command Executed: {command}
    Command Output:
    {command_result}
    
    Please explain the command output and any relevant information.
    """
    
    response = jetson_standard.invoke(enhanced_prompt)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "current_model": "jetson_with_command",
        "task_complexity": "standard",
        "tools_used": state.get("tools_used", []) + ["execute_command"],
        "results": {**state.get("results", {}), "command_result": command_result, "response": response}
    }

# Build the workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("simple_response", simple_response_node)
workflow.add_node("standard_response", standard_response_node)
workflow.add_node("complex_response", complex_response_node)
workflow.add_node("search_and_respond", search_and_respond_node)
workflow.add_node("scrape_and_respond", scrape_and_respond_node)
workflow.add_node("command_and_respond", command_and_respond_node)

# Set conditional entry point
workflow.set_conditional_entry_point(
    analyze_request,
    {
        "simple_task": "simple_response",
        "standard_task": "standard_response",
        "complex_task": "complex_response",
        "needs_search": "search_and_respond",
        "needs_scraping": "scrape_and_respond",
        "needs_command": "command_and_respond"
    }
)

# All nodes lead to END
workflow.add_edge("simple_response", END)
workflow.add_edge("standard_response", END)
workflow.add_edge("complex_response", END)
workflow.add_edge("search_and_respond", END)
workflow.add_edge("scrape_and_respond", END)
workflow.add_edge("command_and_respond", END)

# Compile the workflow
app = workflow.compile()
EOF
```

## Step 5: Create Main Application

```bash
cat > main_app.py << 'EOF'
"""
Main LangGraph application
Demonstrates the complete local AI cluster in action
"""

import asyncio
from langchain.schema import HumanMessage
from langgraph_workflow import app
from config import CLUSTER, ENDPOINTS
import requests

async def run_workflow(user_input: str):
    """Run the LangGraph workflow with user input"""
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "current_model": "",
        "task_complexity": "",
        "tools_used": [],
        "context": "",
        "results": {}
    }
    
    try:
        result = await app.ainvoke(initial_state)
        return result
    except Exception as e:
        return {"error": str(e)}

def check_cluster_health():
    """Check if all cluster services are available"""
    print("🔍 Checking cluster health...")
    
    services = {
        "Jetson Ollama": "http://192.168.1.177:11434/api/tags",
        "CPU llama.cpp": "http://192.168.1.81:8080/health", 
        "HAProxy Load Balancer": "http://192.168.1.81:8888/health",
        "Redis Cache": "redis://192.168.1.81:6379"
    }
    
    all_healthy = True
    
    for name, url in services.items():
        try:
            if url.startswith("redis://"):
                import redis
                r = redis.Redis(host="192.168.1.81", port=6379, password="langgraph_redis_pass")
                r.ping()
                status = "✅ Healthy"
            else:
                response = requests.get(url, timeout=5)
                status = "✅ Healthy" if response.status_code == 200 else f"❌ HTTP {response.status_code}"
        except Exception as e:
            status = f"❌ {str(e)[:50]}"
            all_healthy = False
        
        print(f"  {name}: {status}")
    
    return all_healthy

def main():
    """Main application loop"""
    print("🚀 LangGraph Local AI Cluster")
    print("=" * 50)
    
    # Check cluster health
    if not check_cluster_health():
        print("\n⚠️  Some services are not available. Please check your setup.")
        print("   Run the setup guides to ensure all services are running.")
        return
    
    print(f"\n🌐 Available Endpoints:")
    for name, url in ENDPOINTS.items():
        print(f"  {name}: {url}")
    
    print(f"\n💬 Chat with your local AI cluster!")
    print("Examples:")
    print("  - 'Hello, how are you?' (simple - routes to fast Jetson)")
    print("  - 'Explain quantum computing in detail' (complex - routes to CPU)")
    print("  - 'Search for latest AI news' (tools - uses web search)")
    print("  - 'Scrape https://example.com' (tools - web scraping)")
    print("  - 'Run command: ls -la' (tools - command execution)")
    print()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        print("🤔 Processing...")
        try:
            result = asyncio.run(run_workflow(user_input))
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"\n🤖 Assistant: {result['messages'][-1].content}")
                print(f"📊 Model: {result.get('current_model', 'unknown')}")
                print(f"🔧 Tools: {', '.join(result.get('tools_used', [])) or 'none'}")
                print(f"⚡ Complexity: {result.get('task_complexity', 'unknown')}")
                print()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
EOF

chmod +x main_app.py
```

## Step 6: Test Integration

```bash
# Test individual components first
echo "🧪 Testing Local Models..."

# Test Jetson Ollama
python3 -c "
from local_models import JetsonOllamaLLM
llm = JetsonOllamaLLM()
print('Jetson Test:', llm.invoke('Hello!'))
"

# Test CPU llama.cpp
python3 -c "
from local_models import CPULlamaLLM
llm = CPULlamaLLM()
print('CPU Test:', llm.invoke('Hello!'))
"

# Test Load Balancer (if HAProxy is configured)
# python3 -c "
# from local_models import LoadBalancedLLM
# llm = LoadBalancedLLM()
# print('Load Balanced Test:', llm.invoke('Hello!'))
# "

echo "✅ Local models tested successfully!"
```

## Configuration Management

The cluster configuration is centralized in `config.py`:

```python
# Current cluster endpoints (configured for your machines)
ENDPOINTS = {
    "llm": "http://192.168.1.81:9000",           # HAProxy LLM load balancer
    "tools": "http://192.168.1.81:9001",         # HAProxy tools load balancer  
    "embeddings": "http://192.168.1.81:9002",    # HAProxy embeddings load balancer
    "redis": "http://192.168.1.81:6379"          # Redis cache
}

# Direct service endpoints
CLUSTER = {
    "jetson_orin": "192.168.1.177:11434",        # Primary LLM
    "cpu_coordinator": "192.168.1.81:8080",      # Secondary LLM
    "rp_embeddings": "192.168.1.178:8081",       # Embeddings
    "worker_tools": "192.168.1.190:8082",        # Tools
    "worker_monitor": "192.168.1.191:8083"       # Monitoring
}
```

## Integration Points
- **Load Balancing**: HAProxy automatically routes requests to best available LLM
- **Caching**: Redis stores embeddings and session data
- **Monitoring**: Health checks ensure robust operation
- **Tools**: Web search, scraping, and command execution
- **Embeddings**: Semantic search and routing capabilities

## Next Steps
- ✅ **Complete**: LangGraph integration with local models and tools
- ⏭️ **Next**: [04_distributed_coordination.md](04_distributed_coordination.md) - Setup remaining worker nodes
- 🎯 **Test**: Run `python3 main_app.py` to test the complete workflow
- 🔗 **Monitor**: Check cluster health at http://192.168.1.191:8083/cluster_health

## Advanced Features
- **Intelligent Routing**: Automatically selects best model for each task
- **Tool Integration**: Seamless web search, scraping, and command execution
- **Fault Tolerance**: Automatic failover between models
- **Scalability**: Easy to add more models and workers
- **Privacy**: All processing happens locally on your network