# GPU Node Setup Guide - FinGPT & StockGPT Trading LLMs

## 🎯 Overview

This guide sets up **gpu-node** to run FinGPT and StockGPT models for financial analysis and trading recommendations. This node will be the primary AI inference engine for market sentiment analysis and technical pattern recognition.

## 🖥️ Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070, RTX 3080, or similar)
- **RAM**: 16GB+ system RAM
- **Storage**: 100GB+ free space for models
- **Network**: Gigabit Ethernet connection

## 🚀 Phase 1: System Preparation

### 1.1 Initial Setup

```bash
# SSH into gpu-node
ssh sanzad@gpu-node

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git curl wget htop
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libssl-dev libffi-dev python3-dev
```

### 1.2 CUDA Installation

```bash
# Check GPU
nvidia-smi

# Install CUDA toolkit (adjust version based on your GPU)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
```

### 1.3 Python Environment Setup

```bash
# Create virtual environment
python3 -m venv ~/trading-env
source ~/trading-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch CUDA support
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

## 🧠 Phase 2: FinGPT & StockGPT Installation

### 2.1 Install Core Dependencies

```bash
# Activate environment
source ~/trading-env/bin/activate

# Install transformers and related packages
pip install transformers accelerate bitsandbytes
pip install sentencepiece protobuf

# Install financial data packages
pip install yfinance pandas pandas-ta
pip install requests beautifulsoup4 lxml

# Install web framework for API
pip install fastapi uvicorn python-multipart
```

### 2.2 FinGPT Setup

```bash
# Clone FinGPT repository
cd ~
git clone https://github.com/AI4Finance-Foundation/FinGPT.git
cd FinGPT

# Install FinGPT dependencies
pip install -r requirements.txt

# Install additional FinGPT packages
pip install finrl-meta
pip install finrl
```

### 2.3 StockGPT Setup

```bash
# Clone StockGPT repository (if available)
cd ~
git clone https://github.com/your-org/StockGPT.git
cd StockGPT

# Install StockGPT dependencies
pip install -r requirements.txt

# Alternative: Install from PyPI if available
# pip install stock-gpt
```

## 🔧 Phase 3: Model Configuration

### 3.1 Create Model Configuration

```bash
# Create configuration directory
mkdir -p ~/trading-config
cd ~/trading-config

# Create model configuration file
cat > models_config.yaml << 'EOF'
models:
  fingpt:
    name: "microsoft/DialoGPT-medium"
    type: "causal_lm"
    max_length: 512
    temperature: 0.7
    top_p: 0.9
    
  stockgpt:
    name: "gpt2-medium"
    type: "causal_lm"
    max_length: 256
    temperature: 0.6
    top_p: 0.85

gpu:
  device: "cuda:0"
  memory_fraction: 0.8
  precision: "float16"

api:
  host: "0.0.0.0"
  port: 8080
  max_concurrent_requests: 10
EOF
```

### 3.2 Create Model Loader Scripts

```python
# ~/trading-config/model_loader.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, config_path: str = "models_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device(self.config['gpu']['device'])
        
    def load_fingpt(self):
        """Load FinGPT model"""
        try:
            model_name = self.config['models']['fingpt']['name']
            logger.info(f"Loading FinGPT model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            self.models['fingpt'] = model
            self.tokenizers['fingpt'] = tokenizer
            logger.info("FinGPT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load FinGPT: {e}")
            raise
    
    def load_stockgpt(self):
        """Load StockGPT model"""
        try:
            model_name = self.config['models']['stockgpt']['name']
            logger.info(f"Loading StockGPT model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            self.models['stockgpt'] = model
            self.tokenizers['stockgpt'] = tokenizer
            logger.info("StockGPT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load StockGPT: {e}")
            raise
    
    def load_all_models(self):
        """Load all configured models"""
        self.load_fingpt()
        self.load_stockgpt()
        logger.info("All models loaded successfully")
    
    def get_model(self, model_type: str):
        """Get loaded model by type"""
        return self.models.get(model_type)
    
    def get_tokenizer(self, model_type: str):
        """Get loaded tokenizer by type"""
        return self.tokenizers.get(model_type)
```

## 🚀 Phase 4: API Service Implementation

### 4.1 Create FastAPI Service

```python
# ~/trading-config/api_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import logging
from model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trading LLM API", version="1.0.0")

# Initialize model loader
model_loader = None

class SentimentRequest(BaseModel):
    symbol: str
    news_text: str
    market_context: str = ""

class TechnicalRequest(BaseModel):
    symbol: str
    price_data: List[float]
    indicators: Dict[str, float]

class SentimentResponse(BaseModel):
    symbol: str
    sentiment_score: float
    sentiment_text: str
    confidence: float

class TechnicalResponse(BaseModel):
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    analysis: str
    risk_level: str

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global model_loader
    try:
        model_loader = ModelLoader()
        model_loader.load_all_models()
        logger.info("API service started successfully")
    except Exception as e:
        logger.error(f"Failed to start API service: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(model_loader.models.keys()) if model_loader else [],
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/analyze_sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze market sentiment using FinGPT"""
    try:
        model = model_loader.get_model('fingpt')
        tokenizer = model_loader.get_tokenizer('fingpt')
        
        # Create prompt for sentiment analysis
        prompt = f"""
        Analyze the market sentiment for {request.symbol}.
        
        News: {request.news_text}
        Market Context: {request.market_context}
        
        Provide a sentiment analysis with:
        1. Sentiment score (-1 to 1, where -1 is very bearish, 1 is very bullish)
        2. Detailed analysis
        3. Confidence level (0 to 1)
        """
        
        # Generate response
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract sentiment score (simplified parsing)
        sentiment_score = 0.0  # In real implementation, parse from response
        confidence = 0.8       # In real implementation, extract from response
        
        return SentimentResponse(
            symbol=request.symbol,
            sentiment_score=sentiment_score,
            sentiment_text=response_text,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_technical", response_model=TechnicalResponse)
async def analyze_technical(request: TechnicalRequest):
    """Analyze technical indicators using StockGPT"""
    try:
        model = model_loader.get_model('stockgpt')
        tokenizer = model_loader.get_tokenizer('stockgpt')
        
        # Create prompt for technical analysis
        prompt = f"""
        Analyze technical indicators for {request.symbol}.
        
        Price Data: {request.price_data[-5:]}  # Last 5 prices
        Indicators: {request.indicators}
        
        Provide technical analysis with:
        1. Trading signal (BUY/SELL/HOLD)
        2. Confidence level (0 to 1)
        3. Detailed analysis
        4. Risk level (LOW/MEDIUM/HIGH)
        """
        
        # Generate response
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=150,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract signal (simplified parsing)
        signal = "HOLD"        # In real implementation, parse from response
        confidence = 0.7       # In real implementation, extract from response
        risk_level = "MEDIUM"  # In real implementation, extract from response
        
        return TechnicalResponse(
            symbol=request.symbol,
            signal=signal,
            confidence=confidence,
            analysis=response_text,
            risk_level=risk_level
        )
        
    except Exception as e:
        logger.error(f"Technical analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
async def get_models_status():
    """Get status of all loaded models"""
    if not model_loader:
        return {"error": "Models not loaded"}
    
    status = {}
    for model_name, model in model_loader.models.items():
        status[model_name] = {
            "loaded": True,
            "device": str(next(model.parameters()).device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### 4.2 Create Systemd Service

```bash
# Create systemd service file
sudo tee /etc/systemd/system/trading-llm.service > /dev/null << 'EOF'
[Unit]
Description=Trading LLM API Service
After=network.target

[Service]
Type=simple
User=sanzad
WorkingDirectory=/home/sanzad/trading-config
Environment=PATH=/home/sanzad/trading-env/bin
ExecStart=/home/sanzad/trading-env/bin/python api_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable trading-llm.service
sudo systemctl start trading-llm.service

# Check service status
sudo systemctl status trading-llm.service
```

## 🧪 Phase 5: Testing & Validation

### 5.1 Test Model Loading

```bash
# Test model loading
cd ~/trading-config
source ~/trading-env/bin/activate
python3 -c "
from model_loader import ModelLoader
loader = ModelLoader()
loader.load_all_models()
print('Models loaded successfully!')
print('Available models:', list(loader.models.keys()))
"
```

### 5.2 Test API Endpoints

```bash
# Start API service
cd ~/trading-config
source ~/trading-env/bin/activate
python3 api_service.py &

# Test health endpoint
curl http://localhost:8080/health

# Test sentiment analysis
curl -X POST "http://localhost:8080/analyze_sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "news_text": "Apple reports strong Q4 earnings",
    "market_context": "Tech sector rally"
  }'

# Test technical analysis
curl -X POST "http://localhost:8080/analyze_technical" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "price_data": [150.0, 151.0, 152.0, 153.0, 154.0],
    "indicators": {"rsi": 65.5, "macd": 0.5}
  }'
```

### 5.3 Performance Testing

```bash
# Test GPU utilization
nvidia-smi

# Test memory usage
htop

# Test API response times
time curl -X POST "http://localhost:8080/analyze_sentiment" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "TSLA", "news_text": "Tesla announces new model", "market_context": "EV market growth"}'
```

## 🔧 Phase 6: Integration with LangGraph Cluster

### 6.1 Update Cluster Configuration

```bash
# On cpu-node1, update the cluster configuration
ssh cpu-node1

# Edit cluster config to include gpu-node
vim ~/ai-infrastructure/langgraph-config/config.py
```

Add to the configuration:
```python
GPU_NODES = {
    'gpu-node': {
        'host': '192.168.1.177',  # Adjust IP as needed
        'port': 8080,
        'services': ['fingpt', 'stockgpt'],
        'type': 'gpu'
    }
}
```

### 6.2 Test Cluster Integration

```bash
# Test connectivity from coordinator
curl http://192.168.1.177:8080/health

# Test model inference from coordinator
curl -X POST "http://192.168.1.177:8080/analyze_sentiment" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "GOOGL", "news_text": "Google AI breakthrough", "market_context": "AI sector growth"}'
```

## 📊 Monitoring & Maintenance

### 6.1 Log Monitoring

```bash
# View service logs
sudo journalctl -u trading-llm.service -f

# View GPU utilization
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

### 6.2 Performance Optimization

```bash
# Check model loading times
time python3 -c "
from model_loader import ModelLoader
loader = ModelLoader()
loader.load_all_models()
"

# Monitor memory usage during inference
python3 -c "
import psutil
import torch
from model_loader import ModelLoader

print(f'Initial memory: {psutil.virtual_memory().percent}%')
loader = ModelLoader()
loader.load_all_models()
print(f'After loading: {psutil.virtual_memory().percent}%')
print(f'GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
"
```

## 🚨 Troubleshooting

### Common Issues

**Out of GPU Memory**
```bash
# Reduce model precision
# In models_config.yaml, set precision to "int8" or "float16"

# Reduce batch size
# In api_service.py, limit concurrent requests
```

**Model Loading Failures**
```bash
# Check available disk space
df -h

# Check CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA support
python3 -c "import torch; print(torch.cuda.is_available())"
```

**API Service Not Starting**
```bash
# Check service status
sudo systemctl status trading-llm.service

# Check logs
sudo journalctl -u trading-llm.service -n 50

# Check port availability
netstat -tlnp | grep 8080
```

## ✅ Verification Checklist

- [ ] CUDA toolkit installed and working
- [ ] PyTorch with CUDA support installed
- [ ] FinGPT and StockGPT dependencies installed
- [ ] Models load successfully without errors
- [ ] API service starts and responds to requests
- [ ] GPU utilization during inference
- [ ] Integration with LangGraph cluster working
- [ ] Performance meets requirements (<100ms response time)

## 🔄 Next Steps

1. **Proceed to gpu-node1 setup** for FinRL models
2. **Test distributed workflow** with both GPU nodes
3. **Implement advanced features** like model fine-tuning
4. **Add monitoring and alerting** for production use

---

Your gpu-node is now ready to provide FinGPT and StockGPT inference services for the distributed trading workflow! 🚀
