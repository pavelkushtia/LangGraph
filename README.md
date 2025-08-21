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

### 1. Set Up Your Machines

Follow the setup guides in order:

```bash
# 1. Configure Jetson Orin Nano
./setup_guides/01_jetson_setup.md

# 2. Configure CPU machines
./setup_guides/02_cpu_setup.md

# 3. Integrate with LangGraph
./setup_guides/03_langgraph_integration.md

# 4. Set up distributed coordination
./setup_guides/04_distributed_coordination.md
```

### 2. Start Your Cluster

```bash
# Update IPs in config files first!
python cluster_orchestrator.py start

# Check everything is running
python cluster_orchestrator.py status
```

### 3. Test Your Setup

```bash
# Run example workflows
cd examples/
python example_workflows.py
```

## 📊 Performance Expectations

| Machine | Model | RAM Usage | Speed | Use Case |
|---------|-------|-----------|-------|----------|
| jetson-node | Llama 3.2 3B | ~3GB | 15-25 tok/s | General chat |
| jetson-node | TinyLlama 1.1B | ~1.5GB | 30-50 tok/s | Quick responses |
| cpu-node | Llama 13B Q4 | ~8GB | 5-10 tok/s | Complex analysis |
| rp-node | Embeddings | ~2GB | 50+ emb/s | Vector search (ARM) |

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
# Option 1: Langfuse (LangSmith alternative)
./setup_guides/05_langfuse_setup.md

# Option 2: Helicone (Alternative monitoring)
./setup_guides/06_helicone_setup.md

# Both are completely free and self-hosted!
```

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