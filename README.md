# Local LangGraph AI Cluster 🚀

A complete guide to setting up a distributed LangGraph infrastructure using your local hardware - **zero external API costs!**

## 🏗️ Architecture Overview

Your optimal setup uses your **available hardware** efficiently:

- **jetson-node (Orin Nano 8GB)**: Primary LLM server (Ollama + small models)
- **cpu-node (32GB Intel)**: Coordinator + heavy LLM tasks (llama.cpp + large models)
- **rp-node (8GB ARM)**: Embeddings server (efficient ARM processing)
- **worker-node3 (6GB VM)**: Tools execution server
- **worker-node4 (6GB VM)**: Monitoring and health checks

## 🚀 Quick Start

### 1. Set Up Your Machines (Modular Approach)

Follow the setup guides in order - each guide is self-contained and updated from the comprehensive SOT:

```bash
# 1. Setup Jetson Orin Nano (Primary LLM Server)
# Follow: setup_guides/01_jetson_setup.md
# Sets up Ollama + TensorRT optimizations on jetson-node (192.168.1.177)

# 2. Setup CPU Coordinator (Heavy LLM + Load Balancer + Cache)  
# Follow: setup_guides/02_cpu_setup.md
# Sets up llama.cpp + HAProxy + Redis on cpu-node (192.168.1.81)

# 3. Setup LangGraph Integration (Workflows + Routing)
# Follow: setup_guides/03_langgraph_integration.md  
# Creates intelligent routing and tool integration

# 4. Setup Worker Nodes (Embeddings + Tools + Monitoring)
# Follow: setup_guides/04_distributed_coordination.md
# Sets up rp-node, worker-node3, worker-node4 + orchestration
```

### 2. Start Your Cluster

```bash
# All IPs are pre-configured for your actual nodes!
cd ~/ai-infrastructure/langgraph-config
source ~/langgraph-env/bin/activate

# Start entire cluster
python3 cluster_orchestrator.py start

# Check cluster status  
python3 cluster_orchestrator.py status

# Test all services
python3 cluster_orchestrator.py test
```

### 3. Test Your Setup

```bash
# Test LangGraph workflows
cd ~/ai-infrastructure/langgraph-config
python3 main_app.py

# Run example workflows
cd /home/sanzad/git/langgraph/examples/
python3 example_workflows.py
```

### 🎯 **Single Source of Truth**
- **Complete Guide**: [00_complete_deployment_guide.md](setup_guides/00_complete_deployment_guide.md) - Full walkthrough with all commands
- **Modular Guides**: 01-04 are extracted and synchronized from the complete guide

## 📊 Performance Expectations

| Machine | Model | RAM Usage | Speed | Use Case |
|---------|-------|-----------|-------|----------|
| jetson-node | Llama 3.2 3B | ~3GB | 15-25 tok/s | General chat |
| jetson-node | TinyLlama 1.1B | ~1.5GB | 30-50 tok/s | Quick responses |
| cpu-node | Llama 13B Q4 | ~8GB | 5-10 tok/s | Complex analysis |
| rp-node | Embeddings | ~2GB | 50+ emb/s | Vector search (ARM) |

## 📚 Documentation

- **[LangGraph Concepts & Workflows](docs/langgraph_concepts_and_workflows.md)** - How LangGraph orchestrates your distributed setup ⭐
- **[Architecture Diagrams](docs/architecture_diagrams.md)** - System diagrams and visual flow charts
- **[Example Workflows](examples/example_workflows.py)** - Complete LangGraph workflow implementations

## 🎯 Example Workflows

### Research Assistant
```python
# Automatically searches web, synthesizes findings
result = await research_workflow.invoke("What are the latest AI trends?")
```

### Coding Assistant
```python
# Routes simple/complex coding questions appropriately
result = await coding_workflow.invoke("Build a FastAPI app with auth")
```

### Data Analysis
```python
# Scrapes data and provides analysis
result = await data_workflow.invoke("Analyze this dataset: https://...")
```

## 🔧 Key Features

- **🆓 Zero Cost**: No LLM API costs ever (local inference only)
- **⚡ Smart Routing**: Auto-routes tasks to optimal hardware
- **🔄 Load Balancing**: HAProxy distributes load automatically
- **📊 Monitoring**: Real-time health checks + optional Langfuse/Helicone
- **🛡️ Fault Tolerance**: Automatic failover and restart
- **📈 Auto-scaling**: Dynamic model loading based on load

## 🌟 Why This Setup?

**For your hardware specifically:**
- **Jetson advantages**: ARM efficiency, unified memory, low power
- **Skip RTX cards**: They're overkill for learning and consume 175W+ each
- **Distributed approach**: Maximizes utilization of all machines
- **Local-first**: Complete privacy and control

## 🔍 Monitoring

```bash
# Check cluster health
curl http://192.168.1.137:8083/cluster_health

# View load balancer stats
open http://192.168.1.81:9000/haproxy_stats

# Monitor real-time performance
htop  # On each machine
```

## 🛠️ Customization

### Add New Models
```bash
# On Jetson
ollama pull codellama:7b

# On CPU
wget https://huggingface.co/.../model.bin
```

### Scale Up/Down
```bash
# Add more workers by modifying config.py
# Adjust model sizes based on available RAM
```

### Custom Workflows
```python
# Create new workflows in examples/
# Follow the existing patterns for routing and tools
```

### Add Advanced Monitoring (Optional)
```bash
# Option 1: Langfuse (LangSmith alternative) - Advanced tracing & analytics
# Follow: setup_guides/05_langfuse_setup.md

# Option 2: Helicone (Alternative monitoring) - Real-time monitoring & debugging  
# Follow: setup_guides/06_helicone_setup.md

# Both are completely free and self-hosted!
```

## 📁 **Setup Guide Structure**

| Guide | Purpose | Machine(s) | Status |
|-------|---------|------------|--------|
| **00_complete_deployment_guide.md** | 🎯 **Master SOT** - Complete walkthrough | All machines | ✅ Production ready |
| **01_jetson_setup.md** | Jetson Orin Nano setup | jetson-node | ✅ Synced from SOT |
| **02_cpu_setup.md** | CPU coordinator setup | cpu-node | ✅ Synced from SOT |
| **03_langgraph_integration.md** | LangGraph workflows | cpu-node | ✅ Synced from SOT |
| **04_distributed_coordination.md** | Worker nodes + orchestration | All workers | ✅ Synced from SOT |
| **05_langfuse_setup.md** | Optional: Advanced monitoring | cpu-node | ✅ Optional feature |
| **06_helicone_setup.md** | Optional: Alternative monitoring | cpu-node | ✅ Optional feature |

**Benefits of this structure:**
- ✅ **Modular**: Focus on one machine/service at a time
- ✅ **Updated**: All guides synced from the comprehensive SOT
- ✅ **Flexible**: Use individual guides or complete guide
- ✅ **Maintained**: Single source of truth prevents sync issues

## 🚨 Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Switch to smaller models
ollama pull tinyllama:1.1b
```

**Service Not Starting**
```bash
# Check logs
sudo journalctl -u ollama -f
sudo systemctl restart ollama
```

**Network Issues**
```bash
# Test connectivity
curl http://MACHINE_IP:PORT/health
```

## 📚 Learning Resources

This setup is perfect for learning:
- LangGraph workflow patterns
- Distributed AI systems
- Local model deployment
- Resource optimization
- MLOps practices

## 🎉 What You've Built

- **Production-ready** local AI infrastructure
- **Cost-effective** learning environment
- **Scalable** architecture that grows with you
- **Privacy-focused** - your data never leaves your network

## 🔗 Next Steps

1. **Experiment** with the example workflows
2. **Create** your own domain-specific flows
3. **Scale up** by adding more models or machines
4. **Optimize** based on your specific use cases
5. **Share** your workflows with the community!

---

**Happy Learning!** 🎓 You now have a professional-grade local AI infrastructure that rivals cloud solutions - for free!