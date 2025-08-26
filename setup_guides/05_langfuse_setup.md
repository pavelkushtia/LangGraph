# Langfuse Setup - Open Source LangSmith Alternative

## Overview
Langfuse is a **completely free, open-source** alternative to LangSmith that provides:
- ✅ LLM observability and tracing
- ✅ Prompt management and versioning  
- ✅ Performance monitoring and analytics
- ✅ Dataset management and evaluation
- ✅ Self-hosted with Docker
- ✅ **Zero cost forever**

## Quick Setup on cpu-node

### Step 1: Install Langfuse

```bash
# SSH to cpu-node (or run locally)
ssh sanzad@192.168.1.81

# Create langfuse directory
mkdir -p ~/ai-infrastructure/langfuse
cd ~/ai-infrastructure/langfuse

# Create docker-compose for Langfuse
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  langfuse-server:
    image: langfuse/langfuse:latest
    container_name: langfuse-server
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://langfuse:langfuse@langfuse-db:5432/langfuse
      - NEXTAUTH_SECRET=your-secret-key-change-this
      - NEXTAUTH_URL=http://192.168.1.81:3000
      - SALT=your-salt-change-this
    depends_on:
      - langfuse-db
    restart: unless-stopped

  langfuse-db:
    image: postgres:15
    container_name: langfuse-db
    environment:
      - POSTGRES_DB=langfuse
      - POSTGRES_USER=langfuse
      - POSTGRES_PASSWORD=langfuse
    volumes:
      - langfuse_db_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  langfuse_db_data:
EOF

# Start Langfuse
docker-compose up -d

# Wait for startup
sleep 30

# Check status
docker-compose ps

echo "✅ Langfuse available at: http://192.168.1.81:3000"
```

### Step 2: Configure Firewall

```bash
# Allow Langfuse port
sudo ufw allow 3000/tcp

# Test access
curl http://192.168.1.81:3000/api/public/health
```

### Step 3: Integration with LangGraph

```bash
# Install Langfuse Python SDK
cd ~/ai-infrastructure/langgraph-config
source ../langgraph-env/bin/activate

pip install langfuse

# Create Langfuse integration
cat > langfuse_integration.py << 'EOF'
from langfuse import Langfuse
from langfuse.decorators import observe
from langfuse.openai import openai  # For LLM call tracking

# Initialize Langfuse (self-hosted)
langfuse = Langfuse(
    host="http://192.168.1.81:3000",
    public_key="pk-lf-your-key",  # Get from Langfuse UI
    secret_key="sk-lf-your-key"   # Get from Langfuse UI
)

@observe()
def jetson_llm_call(prompt: str, model: str = "llama3.2:3b"):
    """Tracked LLM call to Jetson"""
    import requests
    
    response = requests.post(
        "http://192.168.1.177:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    
    result = response.json()["response"]
    
    # Log to Langfuse
    langfuse.generation(
        name="jetson-ollama",
        model=model,
        input=prompt,
        output=result,
        metadata={"endpoint": "jetson-node", "ip": "192.168.1.177"}
    )
    
    return result

@observe()
def embedding_call(texts: list, model: str = "default"):
    """Tracked embedding call to rp-node"""
    import requests
    
    response = requests.post(
        "http://192.168.1.178:8081/embeddings",
        json={"texts": texts, "model": model}
    )
    
    result = response.json()
    
    # Log to Langfuse
    langfuse.generation(
        name="embeddings",
        model=f"embeddings-{model}",
        input=str(texts),
        output=f"Generated {len(result['embeddings'])} embeddings",
        metadata={"endpoint": "rp-node", "dimensions": result["dimensions"]}
    )
    
    return result
EOF
```

## Benefits Over Our Current Monitoring

| Feature | Our Monitoring | Langfuse |
|---------|----------------|----------|
| **LLM Tracing** | ❌ Basic health checks | ✅ Full conversation flows |
| **Prompt Tracking** | ❌ None | ✅ Version control + history |
| **Performance Analytics** | ❌ Basic metrics | ✅ Detailed token usage, latency |
| **Cost Analysis** | ❌ None (we're local) | ✅ Virtual cost tracking |
| **A/B Testing** | ❌ Manual | ✅ Built-in prompt testing |
| **Dataset Management** | ❌ None | ✅ Evaluation datasets |
| **Web Dashboard** | ❌ Basic health page | ✅ Professional analytics UI |

## Setup Time: 5 minutes vs LangSmith: Impossible (not open source)
