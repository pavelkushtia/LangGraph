# CPU Node Setup Guide - LangGraph Trading Workflow Orchestrator

## 🎯 Overview

This guide sets up **cpu-node** as the central coordinator for the distributed trading workflow system. This node will orchestrate LangGraph workflows, coordinate between GPU nodes, and manage the overall trading decision-making process.

## 🖥️ Hardware Requirements

- **CPU**: Multi-core processor (Intel i5/i7 or AMD Ryzen 5/7)
- **RAM**: 16GB+ system RAM (32GB recommended)
- **Storage**: 100GB+ free space for workflows and data
- **Network**: Gigabit Ethernet connection

## 🚀 Phase 1: System Preparation

### 1.1 Initial Setup

```bash
# SSH into cpu-node
ssh sanzad@cpu-node

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git curl wget htop
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libssl-dev libffi-dev python3-dev
sudo apt install -y redis-server haproxy
```

### 1.2 Python Environment Setup

```bash
# Create virtual environment
python3 -m venv ~/trading-orchestrator-env
source ~/trading-orchestrator-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install langgraph langchain langchain-community
pip install fastapi uvicorn python-multipart
pip install redis aioredis
pip install pandas numpy scipy
pip install requests aiohttp
pip install pyyaml jinja2
```

## 🧠 Phase 2: LangGraph Trading Workflow Implementation

### 2.1 Create Trading Workflow Structure

```bash
# Create workflow directory structure
mkdir -p ~/trading-orchestrator
cd ~/trading-orchestrator

mkdir -p workflows
mkdir -p models
mkdir -p services
mkdir -p config
mkdir -p utils
mkdir -p tests
```

### 2.2 Create Trading Workflow State

```python
# ~/trading-orchestrator/models/trading_state.py
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain.schema import BaseMessage
from langgraph import add_messages
from datetime import datetime
from enum import Enum

class AnalysisType(str, Enum):
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    PATTERN = "pattern"
    RISK = "risk"
    PORTFOLIO = "portfolio"

class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class TradingWorkflowState(TypedDict):
    # Core workflow state
    messages: Annotated[List[BaseMessage], add_messages]
    current_step: str
    workflow_id: str
    timestamp: datetime
    
    # Trading analysis state
    symbols: List[str]
    analysis_type: AnalysisType
    market_data: Dict[str, Any]
    
    # Sentiment analysis results
    sentiment_scores: Dict[str, float]
    sentiment_texts: Dict[str, str]
    sentiment_confidence: Dict[str, float]
    
    # Technical analysis results
    technical_signals: Dict[str, SignalType]
    technical_indicators: Dict[str, Dict[str, float]]
    technical_confidence: Dict[str, float]
    
    # Pattern recognition results
    pattern_signals: Dict[str, str]
    pattern_confidence: Dict[str, float]
    
    # Risk assessment results
    risk_scores: Dict[str, float]
    risk_levels: Dict[str, RiskLevel]
    var_95: Dict[str, float]
    max_drawdown: Dict[str, float]
    
    # Portfolio optimization results
    portfolio_weights: Dict[str, float]
    expected_returns: Dict[str, float]
    sharpe_ratios: Dict[str, float]
    
    # Final trading decisions
    trade_recommendations: List[Dict[str, Any]]
    execution_status: str
    portfolio_impact: Dict[str, Any]
    
    # Performance tracking
    historical_performance: List[Dict[str, Any]]
    model_feedback: Dict[str, Any]
    
    # Error handling
    errors: List[str]
    warnings: List[str]
```

### 2.3 Create Trading Workflow Nodes

```python
# ~/trading-orchestrator/workflows/trading_nodes.py
from typing import Dict, Any, List
from langchain.schema import HumanMessage, AIMessage
import asyncio
import aiohttp
import logging
from models.trading_state import TradingWorkflowState, AnalysisType, SignalType, RiskLevel

logger = logging.getLogger(__name__)

class TradingWorkflowNodes:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gpu_nodes = config.get('gpu_nodes', {})
        self.cpu_nodes = config.get('cpu_nodes', {})
        
    async def analyze_sentiment(self, state: TradingWorkflowState) -> TradingWorkflowState:
        """Analyze market sentiment using FinGPT on gpu-node"""
        try:
            symbols = state["symbols"]
            sentiment_data = {}
            sentiment_texts = {}
            sentiment_confidence = {}
            
            # Get FinGPT service endpoint
            fingpt_endpoint = f"http://{self.gpu_nodes['gpu-node']['host']}:{self.gpu_nodes['gpu-node']['port']}/analyze_sentiment"
            
            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    try:
                        # Create sentiment analysis request
                        request_data = {
                            "symbol": symbol,
                            "news_text": f"Market analysis for {symbol}",
                            "market_context": "General market conditions"
                        }
                        
                        async with session.post(fingpt_endpoint, json=request_data) as response:
                            if response.status == 200:
                                result = await response.json()
                                sentiment_data[symbol] = result.get('sentiment_score', 0.0)
                                sentiment_texts[symbol] = result.get('sentiment_text', '')
                                sentiment_confidence[symbol] = result.get('confidence', 0.0)
                            else:
                                logger.warning(f"Sentiment analysis failed for {symbol}")
                                sentiment_data[symbol] = 0.0
                                sentiment_texts[symbol] = 'Analysis failed'
                                sentiment_confidence[symbol] = 0.0
                                
                    except Exception as e:
                        logger.error(f"Error analyzing sentiment for {symbol}: {e}")
                        sentiment_data[symbol] = 0.0
                        sentiment_texts[symbol] = f'Error: {str(e)}'
                        sentiment_confidence[symbol] = 0.0
            
            # Update state
            state.update({
                "sentiment_scores": sentiment_data,
                "sentiment_texts": sentiment_texts,
                "sentiment_confidence": sentiment_confidence,
                "current_step": "sentiment_analysis_completed"
            })
            
            # Add analysis message
            state["messages"].append(AIMessage(
                content=f"Sentiment analysis completed for {len(symbols)} symbols. "
                        f"Average sentiment: {sum(sentiment_data.values()) / len(sentiment_data):.3f}"
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            state["errors"].append(f"Sentiment analysis error: {str(e)}")
            return state
    
    async def analyze_technical(self, state: TradingWorkflowState) -> TradingWorkflowState:
        """Analyze technical indicators using StockGPT on gpu-node"""
        try:
            symbols = state["symbols"]
            technical_signals = {}
            technical_indicators = {}
            technical_confidence = {}
            
            # Get StockGPT service endpoint
            stockgpt_endpoint = f"http://{self.gpu_nodes['gpu-node']['host']}:{self.gpu_nodes['gpu-node']['port']}/analyze_technical"
            
            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    try:
                        # Create technical analysis request
                        request_data = {
                            "symbol": symbol,
                            "price_data": [100.0, 101.0, 102.0, 103.0, 104.0],  # Sample data
                            "indicators": {"rsi": 65.5, "macd": 0.5}  # Sample indicators
                        }
                        
                        async with session.post(stockgpt_endpoint, json=request_data) as response:
                            if response.status == 200:
                                result = await response.json()
                                technical_signals[symbol] = SignalType(result.get('signal', 'HOLD'))
                                technical_indicators[symbol] = {"rsi": 65.5, "macd": 0.5}  # Sample
                                technical_confidence[symbol] = result.get('confidence', 0.0)
                            else:
                                logger.warning(f"Technical analysis failed for {symbol}")
                                technical_signals[symbol] = SignalType.HOLD
                                technical_indicators[symbol] = {}
                                technical_confidence[symbol] = 0.0
                                
                    except Exception as e:
                        logger.error(f"Error analyzing technical for {symbol}: {e}")
                        technical_signals[symbol] = SignalType.HOLD
                        technical_indicators[symbol] = {}
                        technical_confidence[symbol] = 0.0
            
            # Update state
            state.update({
                "technical_signals": technical_signals,
                "technical_indicators": technical_indicators,
                "technical_confidence": technical_confidence,
                "current_step": "technical_analysis_completed"
            })
            
            # Add analysis message
            buy_signals = sum(1 for signal in technical_signals.values() if signal == SignalType.BUY)
            sell_signals = sum(1 for signal in technical_signals.values() if signal == SignalType.SELL)
            hold_signals = sum(1 for signal in technical_signals.values() if signal == SignalType.HOLD)
            
            state["messages"].append(AIMessage(
                content=f"Technical analysis completed. Signals: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD"
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            state["errors"].append(f"Technical analysis error: {str(e)}")
            return state
    
    async def assess_risk(self, state: TradingWorkflowState) -> TradingWorkflowState:
        """Assess portfolio risk using FinRL on gpu-node1"""
        try:
            symbols = state["symbols"]
            risk_scores = {}
            risk_levels = {}
            var_95 = {}
            max_drawdown = {}
            
            # Get FinRL service endpoint
            finrl_endpoint = f"http://{self.gpu_nodes['gpu-node1']['host']}:{self.gpu_nodes['gpu-node1']['port']}/assess_risk"
            
            async with aiohttp.ClientSession() as session:
                try:
                    # Create risk assessment request
                    request_data = {
                        "symbols": symbols,
                        "portfolio_weights": [1.0 / len(symbols)] * len(symbols),  # Equal weights
                        "time_horizon": 252  # Trading days
                    }
                    
                    async with session.post(finrl_endpoint, json=request_data) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            # Extract risk metrics for each symbol
                            for i, symbol in enumerate(symbols):
                                risk_scores[symbol] = result.get('volatility', 0.15)
                                risk_levels[symbol] = RiskLevel(result.get('risk_level', 'MEDIUM'))
                                var_95[symbol] = result.get('var_95', 0.05)
                                max_drawdown[symbol] = result.get('max_drawdown', 0.12)
                        else:
                            logger.warning("Risk assessment failed")
                            # Set default risk values
                            for symbol in symbols:
                                risk_scores[symbol] = 0.15
                                risk_levels[symbol] = RiskLevel.MEDIUM
                                var_95[symbol] = 0.05
                                max_drawdown[symbol] = 0.12
                                
                except Exception as e:
                    logger.error(f"Error in risk assessment: {e}")
                    # Set default risk values
                    for symbol in symbols:
                        risk_scores[symbol] = 0.15
                        risk_levels[symbol] = RiskLevel.MEDIUM
                        var_95[symbol] = 0.05
                        max_drawdown[symbol] = 0.12
            
            # Update state
            state.update({
                "risk_scores": risk_scores,
                "risk_levels": risk_levels,
                "var_95": var_95,
                "max_drawdown": max_drawdown,
                "current_step": "risk_assessment_completed"
            })
            
            # Add analysis message
            avg_risk = sum(risk_scores.values()) / len(risk_scores)
            state["messages"].append(AIMessage(
                content=f"Risk assessment completed. Average risk score: {avg_risk:.3f}"
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            state["errors"].append(f"Risk assessment error: {str(e)}")
            return state
    
    async def optimize_portfolio(self, state: TradingWorkflowState) -> TradingWorkflowState:
        """Optimize portfolio using FinRL on gpu-node1"""
        try:
            symbols = state["symbols"]
            portfolio_weights = {}
            expected_returns = {}
            sharpe_ratios = {}
            
            # Get FinRL service endpoint
            finrl_endpoint = f"http://{self.gpu_nodes['gpu-node1']['host']}:{self.gpu_nodes['gpu-node1']['port']}/optimize_portfolio"
            
            async with aiohttp.ClientSession() as session:
                try:
                    # Create portfolio optimization request
                    request_data = {
                        "symbols": symbols,
                        "initial_amount": 100000,
                        "risk_tolerance": "medium"
                    }
                    
                    async with session.post(finrl_endpoint, json=request_data) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            # Extract portfolio metrics
                            for i, symbol in enumerate(symbols):
                                portfolio_weights[symbol] = result.get('weights', [0.25, 0.25, 0.25, 0.25])[i]
                                expected_returns[symbol] = result.get('expected_return', 0.12)
                                sharpe_ratios[symbol] = result.get('sharpe_ratio', 0.8)
                        else:
                            logger.warning("Portfolio optimization failed")
                            # Set default portfolio values
                            for i, symbol in enumerate(symbols):
                                portfolio_weights[symbol] = 1.0 / len(symbols)
                                expected_returns[symbol] = 0.12
                                sharpe_ratios[symbol] = 0.8
                                
                except Exception as e:
                    logger.error(f"Error in portfolio optimization: {e}")
                    # Set default portfolio values
                    for i, symbol in enumerate(symbols):
                        portfolio_weights[symbol] = 1.0 / len(symbols)
                        expected_returns[symbol] = 0.12
                        sharpe_ratios[symbol] = 0.8
            
            # Update state
            state.update({
                "portfolio_weights": portfolio_weights,
                "expected_returns": expected_returns,
                "sharpe_ratios": sharpe_ratios,
                "current_step": "portfolio_optimization_completed"
            })
            
            # Add analysis message
            avg_return = sum(expected_returns.values()) / len(expected_returns)
            avg_sharpe = sum(sharpe_ratios.values()) / len(sharpe_ratios)
            
            state["messages"].append(AIMessage(
                content=f"Portfolio optimization completed. Expected return: {avg_return:.3f}, Sharpe ratio: {avg_sharpe:.3f}"
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            state["errors"].append(f"Portfolio optimization error: {str(e)}")
            return state
    
    async def generate_trading_signals(self, state: TradingWorkflowState) -> TradingWorkflowState:
        """Generate final trading signals using FinRL on gpu-node1"""
        try:
            symbols = state["symbols"]
            
            # Get FinRL service endpoint
            finrl_endpoint = f"http://{self.gpu_nodes['gpu-node1']['host']}:{self.gpu_nodes['gpu-node1']['port']}/generate_trading_signals"
            
            async with aiohttp.ClientSession() as session:
                try:
                    # Create trading signal request
                    request_data = {
                        "symbols": symbols,
                        "current_prices": [100.0] * len(symbols),  # Sample prices
                        "portfolio_state": {
                            "weights": list(state["portfolio_weights"].values()),
                            "risk_scores": list(state["risk_scores"].values())
                        }
                    }
                    
                    async with session.post(finrl_endpoint, json=request_data) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            # Create trade recommendations
                            trade_recommendations = []
                            for i, symbol in enumerate(symbols):
                                recommendation = {
                                    "symbol": symbol,
                                    "action": result.get('actions', ['HOLD'] * len(symbols))[i],
                                    "quantity": result.get('quantities', [0.0] * len(symbols))[i],
                                    "confidence": result.get('confidence', 0.7),
                                    "sentiment_score": state["sentiment_scores"].get(symbol, 0.0),
                                    "technical_signal": state["technical_signals"].get(symbol, SignalType.HOLD),
                                    "risk_level": state["risk_levels"].get(symbol, RiskLevel.MEDIUM),
                                    "expected_return": state["expected_returns"].get(symbol, 0.0),
                                    "timestamp": state["timestamp"]
                                }
                                trade_recommendations.append(recommendation)
                        else:
                            logger.warning("Trading signal generation failed")
                            # Create default recommendations
                            trade_recommendations = []
                            for symbol in symbols:
                                recommendation = {
                                    "symbol": symbol,
                                    "action": "HOLD",
                                    "quantity": 0.0,
                                    "confidence": 0.5,
                                    "sentiment_score": state["sentiment_scores"].get(symbol, 0.0),
                                    "technical_signal": state["technical_signals"].get(symbol, SignalType.HOLD),
                                    "risk_level": state["risk_levels"].get(symbol, RiskLevel.MEDIUM),
                                    "expected_return": state["expected_returns"].get(symbol, 0.0),
                                    "timestamp": state["timestamp"]
                                }
                                trade_recommendations.append(recommendation)
                                
                except Exception as e:
                    logger.error(f"Error in trading signal generation: {e}")
                    # Create default recommendations
                    trade_recommendations = []
                    for symbol in symbols:
                        recommendation = {
                            "symbol": symbol,
                            "action": "HOLD",
                            "quantity": 0.0,
                            "confidence": 0.5,
                            "sentiment_score": state["sentiment_scores"].get(symbol, 0.0),
                            "technical_signal": state["technical_signals"].get(symbol, SignalType.HOLD),
                            "risk_level": state["risk_levels"].get(symbol, RiskLevel.MEDIUM),
                            "expected_return": state["expected_returns"].get(symbol, 0.0),
                            "timestamp": state["timestamp"]
                        }
                        trade_recommendations.append(recommendation)
            
            # Update state
            state.update({
                "trade_recommendations": trade_recommendations,
                "execution_status": "PENDING",
                "current_step": "trading_signals_generated"
            })
            
            # Add analysis message
            buy_count = sum(1 for rec in trade_recommendations if rec['action'] == 'BUY')
            sell_count = sum(1 for rec in trade_recommendations if rec['action'] == 'SELL')
            hold_count = sum(1 for rec in trade_recommendations if rec['action'] == 'HOLD')
            
            state["messages"].append(AIMessage(
                content=f"Trading signals generated. {buy_count} BUY, {sell_count} SELL, {hold_count} HOLD recommendations"
            ))
            
            return state
            
        except Exception as e:
            logger.error(f"Trading signal generation failed: {e}")
            state["errors"].append(f"Trading signal generation error: {str(e)}")
            return state
```

### 2.4 Create Main Trading Workflow

```python
# ~/trading-orchestrator/workflows/trading_workflow.py
from langgraph import StateGraph, END
from typing import Dict, Any, List
import logging
from models.trading_state import TradingWorkflowState
from workflows.trading_nodes import TradingWorkflowNodes

logger = logging.getLogger(__name__)

class TradingWorkflow:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes = TradingWorkflowNodes(config)
        
    def create_workflow(self):
        """Create the main trading analysis workflow"""
        
        # Create workflow graph
        workflow = StateGraph(TradingWorkflowState)
        
        # Add nodes
        workflow.add_node("sentiment_analysis", self.nodes.analyze_sentiment)
        workflow.add_node("technical_analysis", self.nodes.analyze_technical)
        workflow.add_node("risk_assessment", self.nodes.assess_risk)
        workflow.add_node("portfolio_optimization", self.nodes.optimize_portfolio)
        workflow.add_node("generate_signals", self.nodes.generate_trading_signals)
        
        # Define edges (sequential workflow)
        workflow.add_edge("sentiment_analysis", "technical_analysis")
        workflow.add_node("technical_analysis", "risk_assessment")
        workflow.add_edge("risk_assessment", "portfolio_optimization")
        workflow.add_edge("portfolio_optimization", "generate_signals")
        workflow.add_edge("generate_signals", END)
        
        return workflow.compile()
    
    async def run_workflow(self, symbols: List[str], workflow_id: str = None) -> TradingWorkflowState:
        """Run the trading workflow for given symbols"""
        try:
            if workflow_id is None:
                workflow_id = f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create initial state
            initial_state = TradingWorkflowState(
                messages=[],
                current_step="started",
                workflow_id=workflow_id,
                timestamp=datetime.now(),
                symbols=symbols,
                analysis_type=AnalysisType.SENTIMENT,
                market_data={},
                sentiment_scores={},
                sentiment_texts={},
                sentiment_confidence={},
                technical_signals={},
                technical_indicators={},
                technical_confidence={},
                pattern_signals={},
                pattern_confidence={},
                risk_scores={},
                risk_levels={},
                var_95={},
                max_drawdown={},
                portfolio_weights={},
                expected_returns={},
                sharpe_ratios={},
                trade_recommendations=[],
                execution_status="PENDING",
                portfolio_impact={},
                historical_performance=[],
                model_feedback={},
                errors=[],
                warnings=[]
            )
            
            # Add initial message
            initial_state["messages"].append(HumanMessage(
                content=f"Start trading analysis for symbols: {', '.join(symbols)}"
            ))
            
            # Run workflow
            workflow = self.create_workflow()
            final_state = await workflow.ainvoke(initial_state)
            
            logger.info(f"Trading workflow completed for {workflow_id}")
            return final_state
            
        except Exception as e:
            logger.error(f"Trading workflow failed: {e}")
            raise
```

## 🚀 Phase 3: API Service Implementation

### 3.1 Create FastAPI Service

```python
# ~/trading-orchestrator/api_service.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import logging
import yaml
from datetime import datetime
from workflows.trading_workflow import TradingWorkflow
from models.trading_state import TradingWorkflowState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Workflow Orchestrator", version="1.0.0")

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize trading workflow
trading_workflow = TradingWorkflow(config)

# Store running workflows
running_workflows: Dict[str, TradingWorkflowState] = {}

class TradingRequest(BaseModel):
    symbols: List[str]
    workflow_id: Optional[str] = None

class WorkflowStatus(BaseModel):
    workflow_id: str
    status: str
    current_step: str
    progress: float
    messages: List[str]
    errors: List[str]

class WorkflowResult(BaseModel):
    workflow_id: str
    symbols: List[str]
    trade_recommendations: List[Dict[str, Any]]
    execution_status: str
    timestamp: datetime

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Trading Workflow Orchestrator",
        "active_workflows": len(running_workflows),
        "timestamp": datetime.now()
    }

@app.post("/start_trading_analysis", response_model=Dict[str, str])
async def start_trading_analysis(request: TradingRequest, background_tasks: BackgroundTasks):
    """Start a new trading analysis workflow"""
    try:
        workflow_id = request.workflow_id or f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if workflow already exists
        if workflow_id in running_workflows:
            raise HTTPException(status_code=400, detail=f"Workflow {workflow_id} already exists")
        
        # Validate symbols
        if not request.symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required")
        
        if len(request.symbols) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed")
        
        # Create initial state
        initial_state = TradingWorkflowState(
            messages=[],
            current_step="started",
            workflow_id=workflow_id,
            timestamp=datetime.now(),
            symbols=request.symbols,
            # ... other fields initialized
        )
        
        # Store workflow
        running_workflows[workflow_id] = initial_state
        
        # Start workflow in background
        background_tasks.add_task(run_workflow_background, workflow_id, request.symbols)
        
        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": f"Trading analysis started for {len(request.symbols)} symbols"
        }
        
    except Exception as e:
        logger.error(f"Failed to start trading analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_workflow_background(workflow_id: str, symbols: List[str]):
    """Run workflow in background"""
    try:
        logger.info(f"Starting background workflow {workflow_id}")
        
        # Run the workflow
        final_state = await trading_workflow.run_workflow(symbols, workflow_id)
        
        # Update stored workflow
        running_workflows[workflow_id] = final_state
        
        logger.info(f"Background workflow {workflow_id} completed")
        
    except Exception as e:
        logger.error(f"Background workflow {workflow_id} failed: {e}")
        # Update workflow with error
        if workflow_id in running_workflows:
            running_workflows[workflow_id]["errors"].append(str(e))
            running_workflows[workflow_id]["current_step"] = "failed"

@app.get("/workflow/{workflow_id}/status", response_model=WorkflowStatus)
async def get_workflow_status(workflow_id: str):
    """Get status of a specific workflow"""
    if workflow_id not in running_workflows:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    workflow = running_workflows[workflow_id]
    
    # Calculate progress based on current step
    step_progress = {
        "started": 0.0,
        "sentiment_analysis_completed": 0.2,
        "technical_analysis_completed": 0.4,
        "risk_assessment_completed": 0.6,
        "portfolio_optimization_completed": 0.8,
        "trading_signals_generated": 1.0,
        "failed": 0.0
    }
    
    progress = step_progress.get(workflow["current_step"], 0.0)
    
    return WorkflowStatus(
        workflow_id=workflow_id,
        status="running" if workflow["current_step"] != "failed" else "failed",
        current_step=workflow["current_step"],
        progress=progress,
        messages=[msg.content for msg in workflow["messages"]],
        errors=workflow["errors"]
    )

@app.get("/workflow/{workflow_id}/result", response_model=WorkflowResult)
async def get_workflow_result(workflow_id: str):
    """Get final result of a completed workflow"""
    if workflow_id not in running_workflows:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    workflow = running_workflows[workflow_id]
    
    if workflow["current_step"] != "trading_signals_generated":
        raise HTTPException(status_code=400, detail=f"Workflow {workflow_id} not completed")
    
    return WorkflowResult(
        workflow_id=workflow_id,
        symbols=workflow["symbols"],
        trade_recommendations=workflow["trade_recommendations"],
        execution_status=workflow["execution_status"],
        timestamp=workflow["timestamp"]
    )

@app.get("/workflows", response_model=List[Dict[str, Any]])
async def list_workflows():
    """List all workflows"""
    workflows = []
    for workflow_id, workflow in running_workflows.items():
        workflows.append({
            "workflow_id": workflow_id,
            "symbols": workflow["symbols"],
            "current_step": workflow["current_step"],
            "timestamp": workflow["timestamp"],
            "status": "running" if workflow["current_step"] != "failed" else "failed"
        })
    
    return workflows

@app.delete("/workflow/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow"""
    if workflow_id not in running_workflows:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    del running_workflows[workflow_id]
    return {"message": f"Workflow {workflow_id} deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)
```

### 3.2 Create Configuration File

```yaml
# ~/trading-orchestrator/config/config.yaml
gpu_nodes:
  gpu-node:
    host: "192.168.1.177"
    port: 8080
    services:
      - fingpt
      - stockgpt
    type: "gpu"
  
  gpu-node1:
    host: "192.168.1.178"
    port: 8081
    services:
      - finrl_portfolio
      - finrl_risk
      - finrl_trading
    type: "gpu"

cpu_nodes:
  cpu-node1:
    host: "192.168.1.81"
    port: 8084  # Changed from 8082 to avoid conflict with existing services
    services:
      - langgraph_orchestrator
      - workflow_engine
    type: "coordinator"
  
  cpu-node2:
    host: "192.168.1.82"
    port: 8083
    services:
      - market_data_service
      - technical_indicators
      - backtesting_engine
    type: "data_processor"

workflow:
  max_concurrent_workflows: 10
  timeout_seconds: 300
  retry_attempts: 3
  
  analysis:
    default_symbols:
      - "AAPL"
      - "GOOGL"
      - "MSFT"
      - "TSLA"
    sentiment_weight: 0.3
    technical_weight: 0.4
    risk_weight: 0.3

api:
  host: "0.0.0.0"
  port: 8084  # Changed from 8082 to avoid conflict with existing services
  max_concurrent_requests: 20
  request_timeout: 30

logging:
  level: "INFO"
  file: "logs/trading_orchestrator.log"
  max_size: "100MB"
  backup_count: 5
```

### 3.3 Create Systemd Service

```bash
# Create systemd service file
sudo tee /etc/systemd/system/trading-orchestrator.service > /dev/null << 'EOF'
[Unit]
Description=Trading Workflow Orchestrator
After=network.target

[Service]
Type=simple
User=sanzad
WorkingDirectory=/home/sanzad/trading-orchestrator
Environment=PATH=/home/sanzad/trading-orchestrator-env/bin
ExecStart=/home/sanzad/trading-orchestrator-env/bin/python api_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable trading-orchestrator.service
sudo systemctl start trading-orchestrator.service

# Check service status
sudo systemctl status trading-orchestrator.service
```

## 🧪 Phase 4: Testing & Validation

### 4.1 Test Workflow Components

```bash
# Test workflow components
cd ~/trading-orchestrator
source ~/trading-orchestrator-env/bin/activate

# Test state model
python3 -c "
from models.trading_state import TradingWorkflowState, AnalysisType, SignalType, RiskLevel
from datetime import datetime

state = TradingWorkflowState(
    messages=[],
    current_step='started',
    workflow_id='test_123',
    timestamp=datetime.now(),
    symbols=['AAPL', 'GOOGL'],
    analysis_type=AnalysisType.SENTIMENT,
    market_data={},
    sentiment_scores={},
    sentiment_texts={},
    sentiment_confidence={},
    technical_signals={},
    technical_indicators={},
    technical_confidence={},
    pattern_signals={},
    pattern_confidence={},
    risk_scores={},
    risk_levels={},
    var_95={},
    max_drawdown={},
    portfolio_weights={},
    expected_returns={},
    sharpe_ratios={},
    trade_recommendations=[],
    execution_status='PENDING',
    portfolio_impact={},
    historical_performance=[],
    model_feedback={},
    errors=[],
    warnings=[]
)

print('TradingWorkflowState created successfully!')
print(f'Workflow ID: {state[\"workflow_id\"]}')
print(f'Symbols: {state[\"symbols\"]}')
"
```

### 4.2 Test API Endpoints

```bash
# Start API service
cd ~/trading-orchestrator
source ~/trading-orchestrator-env/bin/activate
python3 api_service.py &

# Test health endpoint
curl http://localhost:8082/health

# Test workflow start
curl -X POST "http://localhost:8082/start_trading_analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "workflow_id": "test_workflow_001"
  }'

# Test workflow status
curl http://localhost:8082/workflow/test_workflow_001/status

# List all workflows
curl http://localhost:8082/workflows
```

### 4.3 Test Integration with GPU Nodes

```bash
# Test connectivity to gpu-node
curl http://192.168.1.177:8080/health

# Test connectivity to gpu-node1
curl http://192.168.1.178:8081/health

# Test sentiment analysis from gpu-node
curl -X POST "http://192.168.1.177:8080/analyze_sentiment" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "news_text": "Apple reports strong earnings", "market_context": "Tech rally"}'

# Test portfolio optimization from gpu-node1
curl -X POST "http://192.168.1.178:8081/optimize_portfolio" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "GOOGL"], "initial_amount": 50000}'
```

## 🔧 Phase 5: Integration with LangGraph Cluster

### 5.1 Update Cluster Configuration

```bash
# Update the main cluster configuration
cd ~/ai-infrastructure/langgraph-config

# Edit config.py to include the new trading workflow
vim config.py
```

Add to the configuration:
```python
TRADING_WORKFLOW = {
    'host': '192.168.1.81',
    'port': 8082,
    'service': 'trading_orchestrator',
    'type': 'workflow_coordinator'
}

# Update GPU_NODES if not already present
GPU_NODES = {
    'gpu-node': {
        'host': '192.168.1.177',
        'port': 8080,
        'services': ['fingpt', 'stockgpt'],
        'type': 'gpu'
    },
    'gpu-node1': {
        'host': '192.168.1.178',
        'port': 8081,
        'services': ['finrl_portfolio', 'finrl_risk', 'finrl_trading'],
        'type': 'gpu'
    }
}
```

### 5.2 Test Cluster Integration

```bash
# Test from coordinator
curl http://192.168.1.81:8082/health

# Test workflow start from coordinator
curl -X POST "http://192.168.1.81:8082/start_trading_analysis" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["TSLA", "NVDA"], "workflow_id": "cluster_test_001"}'
```

## 📊 Monitoring & Maintenance

### 5.1 Log Monitoring

```bash
# View service logs
sudo journalctl -u trading-orchestrator.service -f

# Monitor system resources
htop

# Check workflow status
curl http://localhost:8082/workflows
```

### 5.2 Performance Monitoring

```bash
# Test workflow execution time
cd ~/trading-orchestrator
source ~/trading-orchestrator-env/bin/activate

python3 -c "
import asyncio
import time
from workflows.trading_workflow import TradingWorkflow
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Test workflow
workflow = TradingWorkflow(config)
symbols = ['AAPL', 'GOOGL']

start_time = time.time()
try:
    result = asyncio.run(workflow.run_workflow(symbols, 'perf_test'))
    execution_time = time.time() - start_time
    print(f'Workflow completed in {execution_time:.2f} seconds')
    print(f'Final step: {result[\"current_step\"]}')
    print(f'Recommendations: {len(result[\"trade_recommendations\"])}')
except Exception as e:
    print(f'Workflow failed: {e}')
"
```

## 🚨 Troubleshooting

### Common Issues

**Workflow Not Starting**
```bash
# Check service status
sudo systemctl status trading-orchestrator.service

# Check logs
sudo journalctl -u trading-orchestrator.service -n 50

# Check port availability
netstat -tlnp | grep 8082
```

**GPU Node Connection Issues**
```bash
# Test network connectivity
ping 192.168.1.177
ping 192.168.1.178

# Test service endpoints
curl http://192.168.1.177:8080/health
curl http://192.168.1.178:8081/health

# Check firewall settings
sudo ufw status
```

**Workflow Execution Failures**
```bash
# Check workflow logs
cd ~/trading-orchestrator
tail -f logs/trading_orchestrator.log

# Test individual workflow components
python3 -c "
from workflows.trading_nodes import TradingWorkflowNodes
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

nodes = TradingWorkflowNodes(config)
print('TradingWorkflowNodes created successfully')
"
```

## ✅ Verification Checklist

- [ ] Python environment with LangGraph dependencies installed
- [ ] Trading workflow state model working
- [ ] Trading workflow nodes implemented
- [ ] Main trading workflow created
- [ ] API service running and responding
- [ ] Configuration file properly set up
- [ ] Systemd service configured and running
- [ ] Integration with GPU nodes working
- [ ] Workflow execution completing successfully
- [ ] API endpoints responding correctly

## 🔄 Next Steps

1. **Proceed to cpu-node2 setup** for market data services
2. **Test complete distributed workflow** with all nodes
3. **Implement advanced features** like workflow scheduling
4. **Add monitoring and alerting** for production use
5. **Integrate with trading execution systems**

---

Your cpu-node1 is now ready to orchestrate distributed trading workflows using LangGraph! 🚀
