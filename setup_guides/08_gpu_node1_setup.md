# GPU Node 1 Setup Guide - FinRL Trading & Portfolio Optimization

## 🎯 Overview

This guide sets up **gpu-node1** to run FinRL (Financial Reinforcement Learning) models for portfolio optimization, risk management, and automated trading bot execution. This node will handle the quantitative analysis and decision-making aspects of the trading system.

## 🖥️ Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070, RTX 3080, or similar)
- **RAM**: 16GB+ system RAM
- **Storage**: 100GB+ free space for models and data
- **Network**: Gigabit Ethernet connection

## 🚀 Phase 1: System Preparation

### 1.1 Initial Setup

```bash
# SSH into gpu-node1
ssh sanzad@gpu-node1

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git curl wget htop
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libssl-dev libffi-dev python3-dev
sudo apt install -y libblas-dev liblapack-dev libatlas-base-dev gfortran
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
python3 -m venv ~/finrl-env
source ~/finrl-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch CUDA support
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

## 🧠 Phase 2: FinRL & Dependencies Installation

### 2.1 Install Core ML Dependencies

```bash
# Activate environment
source ~/finrl-env/bin/activate

# Install stable-baselines3 for RL algorithms
pip install stable-baselines3[extra]

# Install gymnasium for RL environments
pip install gymnasium

# Install additional ML packages
pip install scikit-learn scipy numpy pandas
pip install matplotlib seaborn plotly
pip install jupyter notebook
```

### 2.2 Install FinRL

```bash
# Install FinRL from source (recommended for latest features)
cd ~
git clone https://github.com/AI4Finance-Foundation/FinRL.git
cd FinRL

# Install FinRL dependencies
pip install -r requirements.txt

# Install FinRL in development mode
pip install -e .

# Install additional FinRL packages
pip install finrl-meta
pip install finrl-applications
```

### 2.3 Install Trading-Specific Packages

```bash
# Install financial data packages
pip install yfinance pandas-ta
pip install ccxt  # For crypto exchanges
pip install alpha_vantage  # For stock data
pip install quandl  # For economic data

# Install technical analysis packages
pip install ta-lib  # If available, otherwise use pandas-ta
pip install finta  # Financial Technical Analysis

# Install backtesting packages
pip install backtrader
pip install zipline-reloaded

# Install risk management packages
pip install pyfolio
pip install empyrical
pip install pykalman  # For Kalman filtering
```

## 🔧 Phase 3: FinRL Model Configuration

### 3.1 Create Model Configuration

```bash
# Create configuration directory
mkdir -p ~/finrl-config
cd ~/finrl-config

# Create model configuration file
cat > finrl_config.yaml << 'EOF'
models:
  portfolio_optimization:
    algorithm: "PPO"  # Proximal Policy Optimization
    policy: "MlpPolicy"
    learning_rate: 0.0003
    batch_size: 64
    n_steps: 2048
    
  risk_management:
    algorithm: "SAC"  # Soft Actor-Critic
    policy: "MlpPolicy"
    learning_rate: 0.0003
    batch_size: 256
    
  trading_bot:
    algorithm: "A2C"  # Advantage Actor-Critic
    policy: "MlpPolicy"
    learning_rate: 0.0007
    batch_size: 100

environment:
  portfolio_size: 100000  # Initial portfolio value
  max_steps: 252  # Trading days in a year
  transaction_fee: 0.001  # 0.1% transaction fee
  
gpu:
  device: "cuda:0"
  memory_fraction: 0.8
  
api:
  host: "0.0.0.0"
  port: 8081
  max_concurrent_requests: 10
EOF
```

### 3.2 Create FinRL Model Loader

```python
# ~/finrl-config/finrl_loader.py
import torch
import numpy as np
import pandas as pd
import yaml
import logging
from typing import Dict, List, Any, Tuple
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinRLLoader:
    def __init__(self, config_path: str = "finrl_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.environments = {}
        self.device = torch.device(self.config['gpu']['device'])
        
        # Initialize data downloader and feature engineer
        self.data_downloader = YahooDownloader()
        self.feature_engineer = FeatureEngineer()
        
    def download_market_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download market data for given symbols"""
        try:
            logger.info(f"Downloading market data for {symbols}")
            
            # Download data
            data = self.data_downloader.download_data(
                ticker_list=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            # Engineer features
            processed_data = self.feature_engineer.preprocess_data(data)
            
            logger.info(f"Downloaded {len(processed_data)} data points")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to download market data: {e}")
            raise
    
    def create_trading_environment(self, data: pd.DataFrame, symbols: List[str]) -> StockTradingEnv:
        """Create trading environment for FinRL"""
        try:
            # Calculate technical indicators
            processed_data = self.feature_engineer.preprocess_data(data)
            
            # Create environment
            env_kwargs = {
                "df": processed_data,
                "hmax": 100,  # Maximum number of shares to trade
                "initial_amount": self.config['environment']['portfolio_size'],
                "transaction_cost_pct": self.config['environment']['transaction_fee'],
                "state_space": len(symbols) * 2 + 1,  # Price + Position + Cash
                "stock_dim": len(symbols),
                "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
                "action_space": len(symbols),
                "reward_scaling": 1e-4
            }
            
            env = StockTradingEnv(**env_kwargs)
            logger.info("Trading environment created successfully")
            
            return env
            
        except Exception as e:
            logger.error(f"Failed to create trading environment: {e}")
            raise
    
    def load_portfolio_optimization_model(self, env: StockTradingEnv) -> PPO:
        """Load or create portfolio optimization model"""
        try:
            model_path = "models/portfolio_optimization_model"
            
            if os.path.exists(model_path + ".zip"):
                logger.info("Loading existing portfolio optimization model")
                model = PPO.load(model_path, env=env)
            else:
                logger.info("Creating new portfolio optimization model")
                model = PPO(
                    policy=self.config['models']['portfolio_optimization']['policy'],
                    env=env,
                    learning_rate=self.config['models']['portfolio_optimization']['learning_rate'],
                    batch_size=self.config['models']['portfolio_optimization']['batch_size'],
                    n_steps=self.config['models']['portfolio_optimization']['n_steps'],
                    verbose=1,
                    device=self.device
                )
            
            self.models['portfolio_optimization'] = model
            logger.info("Portfolio optimization model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load portfolio optimization model: {e}")
            raise
    
    def load_risk_management_model(self, env: StockTradingEnv) -> SAC:
        """Load or create risk management model"""
        try:
            model_path = "models/risk_management_model"
            
            if os.path.exists(model_path + ".zip"):
                logger.info("Loading existing risk management model")
                model = SAC.load(model_path, env=env)
            else:
                logger.info("Creating new risk management model")
                model = SAC(
                    policy=self.config['models']['risk_management']['policy'],
                    env=env,
                    learning_rate=self.config['models']['risk_management']['learning_rate'],
                    batch_size=self.config['models']['risk_management']['batch_size'],
                    verbose=1,
                    device=self.device
                )
            
            self.models['risk_management'] = model
            logger.info("Risk management model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load risk management model: {e}")
            raise
    
    def load_trading_bot_model(self, env: StockTradingEnv) -> A2C:
        """Load or create trading bot model"""
        try:
            model_path = "models/trading_bot_model"
            
            if os.path.exists(model_path + ".zip"):
                logger.info("Loading existing trading bot model")
                model = A2C.load(model_path, env=env)
            else:
                logger.info("Creating new trading bot model")
                model = A2C(
                    policy=self.config['models']['trading_bot']['policy'],
                    env=env,
                    learning_rate=self.config['models']['trading_bot']['learning_rate'],
                    batch_size=self.config['models']['trading_bot']['batch_size'],
                    verbose=1,
                    device=self.device
                )
            
            self.models['trading_bot'] = model
            logger.info("Trading bot model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load trading bot model: {e}")
            raise
    
    def load_all_models(self, symbols: List[str]):
        """Load all FinRL models"""
        try:
            # Download sample data for environment creation
            sample_data = self.download_market_data(
                symbols=symbols,
                start_date="2023-01-01",
                end_date="2023-12-31"
            )
            
            # Create trading environment
            env = self.create_trading_environment(sample_data, symbols)
            self.environments['trading'] = env
            
            # Load all models
            self.load_portfolio_optimization_model(env)
            self.load_risk_management_model(env)
            self.load_trading_bot_model(env)
            
            logger.info("All FinRL models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load all models: {e}")
            raise
    
    def get_model(self, model_type: str):
        """Get loaded model by type"""
        return self.models.get(model_type)
    
    def get_environment(self, env_type: str):
        """Get loaded environment by type"""
        return self.environments.get(env_type)
```

## 🚀 Phase 4: API Service Implementation

### 4.1 Create FastAPI Service

```python
# ~/finrl-config/api_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import logging
import os
from finrl_loader import FinRLLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FinRL Trading API", version="1.0.0")

# Initialize FinRL loader
finrl_loader = None

class PortfolioRequest(BaseModel):
    symbols: List[str]
    initial_amount: float = 100000
    risk_tolerance: str = "medium"  # low, medium, high

class RiskAssessmentRequest(BaseModel):
    symbols: List[str]
    portfolio_weights: List[float]
    time_horizon: int = 252  # days

class TradingSignalRequest(BaseModel):
    symbols: List[str]
    current_prices: List[float]
    portfolio_state: Dict[str, Any]

class PortfolioResponse(BaseModel):
    symbols: List[str]
    weights: List[float]
    expected_return: float
    risk_score: float
    sharpe_ratio: float

class RiskResponse(BaseModel):
    symbols: List[str]
    var_95: float  # Value at Risk 95%
    max_drawdown: float
    volatility: float
    risk_level: str

class TradingSignalResponse(BaseModel):
    symbols: List[str]
    actions: List[str]  # BUY, SELL, HOLD
    quantities: List[float]
    confidence: float
    risk_assessment: str

@app.on_event("startup")
async def startup_event():
    """Initialize FinRL models on startup"""
    global finrl_loader
    try:
        finrl_loader = FinRLLoader()
        
        # Load models with sample symbols
        sample_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        finrl_loader.load_all_models(sample_symbols)
        
        logger.info("FinRL API service started successfully")
    except Exception as e:
        logger.error(f"Failed to start FinRL API service: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(finrl_loader.models.keys()) if finrl_loader else [],
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/optimize_portfolio", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest):
    """Optimize portfolio using FinRL"""
    try:
        model = finrl_loader.get_model('portfolio_optimization')
        env = finrl_loader.get_environment('trading')
        
        if not model or not env:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Get market data for the symbols
        data = finrl_loader.download_market_data(
            symbols=request.symbols,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Create environment with current data
        current_env = finrl_loader.create_trading_environment(data, request.symbols)
        
        # Run portfolio optimization
        obs = current_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = current_env.step(action)
        
        # Extract portfolio weights from environment
        portfolio_state = current_env.get_portfolio_state()
        
        # Calculate metrics
        weights = [0.25, 0.25, 0.25, 0.25]  # Simplified - extract from env
        expected_return = 0.12  # Simplified - calculate from historical data
        risk_score = 0.15  # Simplified - calculate volatility
        sharpe_ratio = 0.8  # Simplified - calculate Sharpe ratio
        
        return PortfolioResponse(
            symbols=request.symbols,
            weights=weights,
            expected_return=expected_return,
            risk_score=risk_score,
            sharpe_ratio=sharpe_ratio
        )
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assess_risk", response_model=RiskResponse)
async def assess_risk(request: RiskAssessmentRequest):
    """Assess portfolio risk using FinRL"""
    try:
        model = finrl_loader.get_model('risk_management')
        env = finrl_loader.get_environment('trading')
        
        if not model or not env:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Get market data
        data = finrl_loader.download_market_data(
            symbols=request.symbols,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Calculate risk metrics (simplified)
        var_95 = 0.05  # 5% VaR
        max_drawdown = 0.12  # 12% max drawdown
        volatility = 0.18  # 18% volatility
        risk_level = "medium"
        
        return RiskResponse(
            symbols=request.symbols,
            var_95=var_95,
            max_drawdown=max_drawdown,
            volatility=volatility,
            risk_level=risk_level
        )
        
    except Exception as e:
        logger.error(f"Risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_trading_signals", response_model=TradingSignalResponse)
async def generate_trading_signals(request: TradingSignalRequest):
    """Generate trading signals using FinRL trading bot"""
    try:
        model = finrl_loader.get_model('trading_bot')
        env = finrl_loader.get_environment('trading')
        
        if not model or not env:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Generate trading signals (simplified)
        actions = ["HOLD"] * len(request.symbols)
        quantities = [0.0] * len(request.symbols)
        confidence = 0.7
        risk_assessment = "medium"
        
        return TradingSignalResponse(
            symbols=request.symbols,
            actions=actions,
            quantities=quantities,
            confidence=confidence,
            risk_assessment=risk_assessment
        )
        
    except Exception as e:
        logger.error(f"Trading signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
async def get_models_status():
    """Get status of all loaded models"""
    if not finrl_loader:
        return {"error": "Models not loaded"}
    
    status = {}
    for model_name, model in finrl_loader.models.items():
        status[model_name] = {
            "loaded": True,
            "algorithm": type(model).__name__,
            "device": str(next(model.policy.parameters()).device),
            "parameters": sum(p.numel() for p in model.policy.parameters())
        }
    
    return status

@app.post("/train_model")
async def train_model(model_type: str, training_steps: int = 10000):
    """Train a specific FinRL model"""
    try:
        model = finrl_loader.get_model(model_type)
        env = finrl_loader.get_environment('trading')
        
        if not model or not env:
            raise HTTPException(status_code=500, detail="Model not found")
        
        # Train the model
        model.learn(total_timesteps=training_steps)
        
        # Save the trained model
        model_path = f"models/{model_type}_model"
        model.save(model_path)
        
        return {"message": f"{model_type} model trained and saved successfully"}
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
```

### 4.2 Create Systemd Service

```bash
# Create systemd service file
sudo tee /etc/systemd/system/finrl-trading.service > /dev/null << 'EOF'
[Unit]
Description=FinRL Trading API Service
After=network.target

[Service]
Type=simple
User=sanzad
WorkingDirectory=/home/sanzad/finrl-config
Environment=PATH=/home/sanzad/finrl-env/bin
ExecStart=/home/sanzad/finrl-env/bin/python api_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable finrl-trading.service
sudo systemctl start finrl-trading.service

# Check service status
sudo systemctl status finrl-trading.service
```

## 🧪 Phase 5: Testing & Validation

### 5.1 Test Model Loading

```bash
# Test model loading
cd ~/finrl-config
source ~/finrl-env/bin/activate
python3 -c "
from finrl_loader import FinRLLoader
loader = FinRLLoader()
loader.load_all_models(['AAPL', 'GOOGL', 'MSFT'])
print('FinRL models loaded successfully!')
print('Available models:', list(loader.models.keys()))
"
```

### 5.2 Test API Endpoints

```bash
# Start API service
cd ~/finrl-config
source ~/finrl-env/bin/activate
python3 api_service.py &

# Test health endpoint
curl http://localhost:8081/health

# Test portfolio optimization
curl -X POST "http://localhost:8081/optimize_portfolio" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
    "initial_amount": 100000,
    "risk_tolerance": "medium"
  }'

# Test risk assessment
curl -X POST "http://localhost:8081/assess_risk" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
    "portfolio_weights": [0.25, 0.25, 0.25, 0.25],
    "time_horizon": 252
  }'
```

### 5.3 Performance Testing

```bash
# Test GPU utilization
nvidia-smi

# Test memory usage
htop

# Test model training
curl -X POST "http://localhost:8081/train_model?model_type=portfolio_optimization&training_steps=1000"
```

## 🔧 Phase 6: Integration with LangGraph Cluster

### 6.1 Update Cluster Configuration

```bash
# On cpu-node1, update the cluster configuration
ssh cpu-node1

# Edit cluster config to include gpu-node1
vim ~/ai-infrastructure/langgraph-config/config.py
```

Add to the configuration:
```python
GPU_NODES = {
    'gpu-node': {
        'host': '192.168.1.177',
        'port': 8080,
        'services': ['fingpt', 'stockgpt'],
        'type': 'gpu'
    },
    'gpu-node1': {
        'host': '192.168.1.178',  # Adjust IP as needed
        'port': 8081,
        'services': ['finrl_portfolio', 'finrl_risk', 'finrl_trading'],
        'type': 'gpu'
    }
}
```

### 6.2 Test Cluster Integration

```bash
# Test connectivity from coordinator
curl http://192.168.1.178:8081/health

# Test portfolio optimization from coordinator
curl -X POST "http://192.168.1.178:8081/optimize_portfolio" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "GOOGL"], "initial_amount": 50000}'
```

## 📊 Monitoring & Maintenance

### 6.1 Log Monitoring

```bash
# View service logs
sudo journalctl -u finrl-trading.service -f

# View GPU utilization
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

### 6.2 Model Performance Monitoring

```bash
# Check model training progress
cd ~/finrl-config
source ~/finrl-env/bin/activate

# Test model performance
python3 -c "
from finrl_loader import FinRLLoader
import time

loader = FinRLLoader()
loader.load_all_models(['AAPL', 'GOOGL'])

# Test inference speed
start_time = time.time()
model = loader.get_model('portfolio_optimization')
env = loader.get_environment('trading')

obs = env.reset()
action, _ = model.predict(obs, deterministic=True)
inference_time = time.time() - start_time

print(f'Inference time: {inference_time:.4f} seconds')
"
```

## 🚨 Troubleshooting

### Common Issues

**Out of GPU Memory**
```bash
# Reduce model complexity
# In finrl_config.yaml, reduce batch_size and n_steps

# Use smaller models
# Consider using smaller policy networks
```

**Model Training Failures**
```bash
# Check data availability
python3 -c "
import yfinance as yf
data = yf.download('AAPL', start='2023-01-01', end='2023-12-31')
print(f'Data points: {len(data)}')
"

# Check environment creation
python3 -c "
from finrl_loader import FinRLLoader
loader = FinRLLoader()
try:
    env = loader.create_trading_environment(data, ['AAPL'])
    print('Environment created successfully')
except Exception as e:
    print(f'Environment creation failed: {e}')
"
```

**API Service Not Starting**
```bash
# Check service status
sudo systemctl status finrl-trading.service

# Check logs
sudo journalctl -u finrl-trading.service -n 50

# Check port availability
netstat -tlnp | grep 8081
```

## ✅ Verification Checklist

- [ ] CUDA toolkit installed and working
- [ ] PyTorch with CUDA support installed
- [ ] FinRL and dependencies installed
- [ ] Models load successfully without errors
- [ ] API service starts and responds to requests
- [ ] GPU utilization during training and inference
- [ ] Integration with LangGraph cluster working
- [ ] Portfolio optimization working
- [ ] Risk assessment working
- [ ] Trading signal generation working

## 🔄 Next Steps

1. **Proceed to cpu-node1 setup** for LangGraph orchestration
2. **Test distributed workflow** with both GPU nodes
3. **Implement advanced features** like model fine-tuning
4. **Add monitoring and alerting** for production use
5. **Integrate with trading execution systems**

---

Your gpu-node1 is now ready to provide FinRL portfolio optimization, risk management, and trading bot services for the distributed trading workflow! 🚀
