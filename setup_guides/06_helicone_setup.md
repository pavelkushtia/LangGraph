# Helicone Setup - Free Open Source LLM Monitoring

## Overview
Helicone is another **completely free, open-source** LLM monitoring platform:
- ✅ Real-time LLM monitoring and debugging
- ✅ Session tracking and workflow tracing  
- ✅ Prompt management and optimization
- ✅ Cost tracking and analytics
- ✅ Self-hosted with Docker
- ✅ **Zero cost forever**

## Quick Setup on cpu-node

### Step 1: Install Helicone

```bash
# SSH to cpu-node
ssh sanzad@192.168.1.81

# Create helicone directory
mkdir -p ~/ai-infrastructure/helicone
cd ~/ai-infrastructure/helicone

# Clone Helicone (open source)
git clone https://github.com/Helicone/helicone.git
cd helicone

# Setup with Docker
cat > docker-compose.override.yml << 'EOF'
version: '3.8'
services:
  web:
    ports:
      - "3001:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@supabase-db:5432/postgres
      - NEXTAUTH_SECRET=your-secret-key
      - NEXTAUTH_URL=http://192.168.1.81:3001
EOF

# Start Helicone
docker-compose up -d

# Wait for startup
sleep 45

echo "✅ Helicone available at: http://192.168.1.81:3001"
```

### Step 2: Integration Example

```bash
# Install Helicone Python client
cd ~/ai-infrastructure/langgraph-config
source ../langgraph-env/bin/activate

pip install helicone

# Create Helicone integration
cat > helicone_integration.py << 'EOF'
from helicone import Helicone
import requests
import time

# Initialize Helicone (self-hosted)
helicone = Helicone(
    api_key="your-api-key",  # Generate in Helicone UI
    base_url="http://192.168.1.81:3001"
)

def tracked_llm_call(prompt: str, model: str = "llama3.2:3b"):
    """LLM call with Helicone tracking"""
    start_time = time.time()
    
    # Make the actual LLM call
    response = requests.post(
        "http://192.168.1.177:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    
    end_time = time.time()
    result = response.json()["response"]
    
    # Log to Helicone
    helicone.log_request(
        request={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "provider": "ollama-jetson"
        },
        response={
            "choices": [{"message": {"content": result}}],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(result.split()),
                "total_tokens": len(prompt.split()) + len(result.split())
            }
        },
        latency_ms=(end_time - start_time) * 1000,
        metadata={"endpoint": "jetson-node"}
    )
    
    return result
EOF
```

### Step 3: Configure Firewall

```bash
sudo ufw allow 3001/tcp
```

## Comparison: Langfuse vs Helicone

| Aspect | Langfuse | Helicone |
|--------|----------|----------|
| **Setup Complexity** | Simpler (2 containers) | More complex (full stack) |
| **Resource Usage** | Lighter | Heavier |
| **UI/UX** | Clean, focused | Feature-rich |
| **Integration** | LangChain native | Provider-agnostic |
| **Community** | Growing fast | Established |
| **Documentation** | Excellent | Good |

## Recommendation: Start with Langfuse

For your setup, I'd recommend **Langfuse** because:
1. **Lighter resource footprint** (important for your VMs)
2. **LangChain/LangGraph native integration**
3. **Simpler setup** (5 minutes)
4. **Better documentation**
