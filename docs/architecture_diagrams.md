# LangGraph Local AI Architecture Diagrams

## Main Cluster Architecture

This diagram shows how your hardware is organized for optimal AI workloads:

```mermaid
graph TB
    subgraph "LangGraph Control Layer"
        LG[LangGraph Orchestrator<br/>CPU 32GB Machine]
    end
    
    subgraph "Model Inference Layer"
        JO[Jetson Orin Nano 8GB<br/>Primary LLM Server<br/>Ollama + Small Models<br/>Llama 3.2 3B, Gemma 2B]
        CPU32[CPU 32GB Machine<br/>Large Model Server<br/>llama.cpp + Quantized<br/>Llama 13B Q4_K_M]
    end
    
    subgraph "Worker Nodes"
        CPU16A[CPU 16GB Machine A<br/>Embeddings + Vector Search<br/>sentence-transformers]
        CPU16B[CPU 16GB Machine B<br/>Tool Execution<br/>Web scraping, APIs]
        CPU8A[CPU 8GB Machine A<br/>Monitoring + Logging]
        CPU8B[CPU 8GB Machine B<br/>Data Processing]
    end
    
    subgraph "Storage Layer"
        VDB[Vector Database<br/>ChromaDB/Qdrant]
        CACHE[Model Cache<br/>Redis/Local Files]
    end
    
    LG -->|Task Distribution| JO
    LG -->|Heavy Tasks| CPU32
    LG -->|Parallel Processing| CPU16A
    LG -->|Tool Calls| CPU16B
    LG -->|Coordination| CPU8A
    LG -->|Data Ops| CPU8B
    
    JO -->|Embeddings| VDB
    CPU32 -->|Results| CACHE
    CPU16A -->|Vector Search| VDB
    
    style JO fill:#e1f5fe
    style CPU32 fill:#f3e5f5
    style LG fill:#e8f5e8
```

## Network Architecture

Shows how machines communicate:

```mermaid
graph LR
    subgraph "Your Local Network 192.168.1.x"
        J[Jetson Orin<br/>192.168.1.100:11434<br/>Ollama API]
        C1[CPU 32GB<br/>192.168.1.101:8080<br/>llama.cpp + HAProxy]
        C2[CPU 16GB A<br/>192.168.1.102:8081<br/>Embeddings API]
        C3[CPU 16GB B<br/>192.168.1.103:8082<br/>Tools API]
        C4[CPU 8GB A<br/>192.168.1.104:8083<br/>Monitoring]
        C5[CPU 8GB B<br/>192.168.1.105:6379<br/>Redis Cache]
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
pie title RAM Allocation Across Cluster (64GB Total)
    "Jetson: Models + OS" : 8
    "CPU 32GB: Large Models + Orchestration" : 32
    "CPU 16GB A: Embeddings + Vector DB" : 16
    "CPU 16GB B: Tools + Services" : 16
    "CPU 8GB A: Monitoring" : 8
    "CPU 8GB B: Cache + Processing" : 8
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

```mermaid
timeline
    title Learning Path with Local AI Setup
    
    section Week 1 : Setup Phase
        Day 1-2    : Configure Jetson Orin
                   : Install Ollama + models
        Day 3-4    : Setup CPU machines
                   : Configure llama.cpp
        Day 5-7    : LangGraph integration
                   : Test basic workflows
    
    section Week 2-4 : Learning Phase
        Week 2     : Basic LangGraph patterns
                   : Simple agent workflows
        Week 3     : Complex multi-agent systems
                   : Tool integration
        Week 4     : Custom workflows
                   : Performance optimization
    
    section Month 2+ : Advanced
        Advanced   : Custom model fine-tuning
                   : Distributed optimization
                   : Production deployment
```

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

| Component | IP Address | Port | Purpose |
|-----------|------------|------|---------|
| Jetson Orin | 192.168.1.100 | 11434 | Primary LLM (Ollama) |
| CPU 32GB | 192.168.1.101 | 8080, 9000 | Heavy LLM + Load Balancer |
| CPU 16GB A | 192.168.1.102 | 8081 | Embeddings Server |
| CPU 16GB B | 192.168.1.103 | 8082 | Tools Server |
| CPU 8GB A | 192.168.1.104 | 8083 | Monitoring |
| CPU 8GB B | 192.168.1.105 | 6379 | Redis Cache |

**Load Balanced Endpoints:**
- LLM: `http://192.168.1.101:9000`
- Tools: `http://192.168.1.101:9001`  
- Embeddings: `http://192.168.1.101:9002`
