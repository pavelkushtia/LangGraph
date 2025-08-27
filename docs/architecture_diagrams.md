# LangGraph Local AI Architecture Diagrams

## Main Cluster Architecture

This diagram shows how your hardware is organized for optimal AI workloads:

```mermaid
graph TB
    subgraph "LangGraph Control Layer"
        LG["cpu-node (192.168.1.81)<br/>LangGraph Orchestrator + HAProxy"]
    end
    
    subgraph "Model Inference Layer"
        JO["jetson-node (192.168.1.177)<br/>Ollama + Small Models<br/>llama3.2:3b, llama3.2:1b"]
        CPU32["cpu-node (192.168.1.81)<br/>Ollama + Large Models<br/>mistral:7b"]
    end
    
    subgraph "Worker Nodes"
        RPI["rp-node (192.168.1.178)<br/>Embeddings + Vector Search"]
        W3["worker-node3 (192.168.1.190)<br/>Tool Execution"]
        W4["worker-node4 (192.168.1.191)<br/>Monitoring + Logging"]
    end
    
    subgraph "Storage Layer"
        VDB["Vector Database<br/>ChromaDB on rp-node"]
        CACHE["Redis Cache<br/>On cpu-node"]
    end
    
    LG --> JO
    LG --> CPU32
    LG --> RPI
    LG --> W3
    LG --> W4
    
    JO --> VDB
    CPU32 --> CACHE
    RPI --> VDB
    
    style JO fill:#e1f5fe
    style CPU32 fill:#f3e5f5
    style LG fill:#e8f5e8
    style RPI fill:#fff3e0
```

## Network Architecture

Shows how machines communicate:

```mermaid
graph LR
    subgraph "Local Network 192.168.1.x"
        J["jetson-node<br/>192.168.1.177:11434<br/>Ollama API"]
        C1["cpu-node<br/>192.168.1.81:11435<br/>Ollama + HAProxy:9000"]
        C2["rp-node<br/>192.168.1.178:8081<br/>Embeddings API"]
        C3["worker-node3<br/>192.168.1.190:8082<br/>Tools API"]
        C4["worker-node4<br/>192.168.1.191:8083<br/>Monitoring"]
        C5["cpu-node<br/>192.168.1.81:6379<br/>Redis Cache"]
    end
    
    subgraph "Load Balancer (HAProxy)"
        LB[":9000 - LLM Requests<br/>:9001 - Tool Requests<br/>:9002 - Embedding Requests"]
    end
    
    C1 --> LB
    LB -->|Route Fast| J
    LB -->|Route Heavy| C1
    LB --> C2
    LB --> C3
    
    style J fill:#81c784
    style C1 fill:#64b5f6
    style LB fill:#ffb74d
```

## LangGraph Workflow Flow

Shows how tasks flow through your system:

```mermaid
flowchart TD
    START([User Query]) --> ROUTE{Route Task}
    
    ROUTE -->|Simple/Fast| JETSON[Jetson Orin<br/>TinyLlama 1.1B<br/>30-50 tok/s]
    ROUTE -->|Standard| JETSON2[Jetson Orin<br/>Llama 3.2 3B<br/>15-25 tok/s]
    ROUTE -->|Complex| CPU[CPU 32GB<br/>Llama 13B Q4<br/>5-10 tok/s]
    
    JETSON --> TOOLS{Need Tools?}
    JETSON2 --> TOOLS
    CPU --> TOOLS
    
    TOOLS -->|Yes| EXECUTE[Tool Execution<br/>16GB Machine B]
    TOOLS -->|No| RESPONSE[Generate Response]
    
    EXECUTE --> SEARCH[Web Search]
    EXECUTE --> SCRAPE[Web Scraping]
    EXECUTE --> CMD[Command Execution]
    
    SEARCH --> EMBED[Create Embeddings<br/>16GB Machine A]
    SCRAPE --> EMBED
    CMD --> RESPONSE
    
    EMBED --> VDB[(Vector DB<br/>8GB Machine B)]
    VDB --> RESPONSE
    
    RESPONSE --> END([Return to User])
    
    style JETSON fill:#e8f5e8
    style JETSON2 fill:#e8f5e8
    style CPU fill:#f3e5f5
    style EXECUTE fill:#fff3e0
```

## Resource Allocation

Shows how your 64GB total RAM is distributed:

```mermaid
pie title RAM Allocation Across Cluster (56GB Total Available)
    "jetson-node: Jetson Orin Nano" : 8
    "cpu-node: Orchestration + Large Models" : 32
    "rp-node: ARM Embeddings Server" : 8
    "worker-node3: Tools (VM)" : 6
    "worker-node4: Monitoring (VM)" : 6
```

## Power Consumption Comparison

Why we skip RTX cards:

```mermaid
graph LR
    subgraph "Recommended Setup"
        J[Jetson Orin Nano<br/>10-25W<br/>Efficient AI]
        C[CPU Machines<br/>50-100W each<br/>Standard computing]
    end
    
    subgraph "Alternative (Not Recommended)"
        R1[RTX 2070<br/>175W<br/>High power]
        R2[RTX 2060 Super<br/>175W<br/>High power]
    end
    
    subgraph "Total Power"
        REC[Recommended: ~300W<br/>💰 Low electricity cost<br/>🌱 Cool running]
        ALT[With RTX Cards: ~650W<br/>💸 High electricity cost<br/>🔥 Hot + noisy]
    end
    
    J --> REC
    C --> REC
    R1 --> ALT
    R2 --> ALT
    
    style REC fill:#c8e6c9
    style ALT fill:#ffcdd2
```

## Development Workflow

Your learning journey with this setup:

### 📅 Learning Timeline

**🚀 Week 1: Setup Phase**
- **Day 1-2**: Configure Jetson Orin
  - Install Ollama + models (llama3.2:3b, llama3.2:1b)
  - Test basic inference
- **Day 3-4**: Setup CPU machines  
  - Configure Ollama + HAProxy + Redis
  - Test load balancing
- **Day 5-7**: LangGraph integration
  - Set up intelligent routing
  - Test basic workflows

**🧠 Week 2-4: Learning Phase** 
- **Week 2**: Basic LangGraph patterns
  - Simple agent workflows
  - State management
- **Week 3**: Complex multi-agent systems
  - Tool integration (web search, scraping)
  - Multi-step workflows
- **Week 4**: Custom workflows  
  - Performance optimization
  - Advanced routing strategies

**🎯 Month 2+: Advanced Phase**
- Custom model experiments
- Distributed optimization
- Production-ready deployments

### 📊 Progress Tracking

Monitor your progress with:
- HAProxy stats: `http://cpu-node:9000/haproxy_stats`  
- System monitoring: `htop` on each machine
- LangGraph workflow logs
- Model performance metrics

## Monitoring Dashboard Layout

What your monitoring will show:

```mermaid
graph TB
    subgraph "Monitoring Dashboard (CPU 8GB A)"
        CPU_USAGE[CPU Usage per Machine]
        MEM_USAGE[Memory Usage per Machine]
        MODEL_PERF[Model Performance Metrics]
        NETWORK[Network Traffic]
        HEALTH[Service Health Status]
        ALERTS[Alert Management]
    end
    
    subgraph "Data Sources"
        J_STATS[Jetson Stats<br/>tegrastats]
        CPU_STATS[CPU Stats<br/>htop/psutil]
        OLLAMA_STATS[Ollama Metrics<br/>API stats]
        HAPROXY_STATS[HAProxy Stats<br/>Load balancer metrics]
    end
    
    J_STATS --> CPU_USAGE
    J_STATS --> MEM_USAGE
    CPU_STATS --> CPU_USAGE
    CPU_STATS --> MEM_USAGE
    OLLAMA_STATS --> MODEL_PERF
    HAPROXY_STATS --> NETWORK
    
    style CPU_USAGE fill:#e3f2fd
    style MODEL_PERF fill:#f3e5f5
    style HEALTH fill:#e8f5e8
```

---

## Quick Reference

| Component | Hostname | IP Address | Port | Purpose |
|-----------|----------|------------|------|---------|
| Jetson Orin Nano | jetson-node | 192.168.1.177 | 11434 | Primary LLM (Ollama) |
| CPU 32GB Coordinator | cpu-node | 192.168.1.81 | 8080, 9000, 6379 | Heavy LLM + Load Balancer + Redis |
| ARM 8GB | rp-node | 192.168.1.178 | 8081 | Embeddings Server |
| VM 6GB Tools | worker-node3 | 192.168.1.190 | 8082 | Tools Server |
| VM 6GB Monitor | worker-node4 | 192.168.1.191 | 8083 | Monitoring |

**Load Balanced Endpoints:**
- LLM: `http://192.168.1.81:9000`
- Tools: `http://192.168.1.81:9001`  
- Embeddings: `http://192.168.1.81:9002`

**Available Nodes Status:**
- ✅ jetson-node (192.168.1.177) - Jetson Orin Nano 8GB
- ✅ cpu-node (192.168.1.81) - 32GB RAM Intel i5-6500T
- ✅ rp-node (192.168.1.178) - 8GB ARM Cortex-A76
- ✅ worker-node3 (192.168.1.190) - 6GB VM
- ✅ worker-node4 (192.168.1.191) - 6GB VM
- ❌ cpu-node1, cpu-node2 - Currently unavailable
- ❌ gpu-node, gpu-node1 - Currently unavailable
