# CPU Node 2 Setup Guide - Market Data Services & Technical Analysis

## 🎯 Overview

This guide sets up **cpu-node2** to provide market data services, technical indicators calculation, and backtesting capabilities for the distributed trading workflow system. This node will handle real-time data feeds, technical analysis, and strategy validation.

## 🖥️ Hardware Requirements

- **CPU**: Multi-core processor (Intel i5/i7 or AMD Ryzen 5/7)
- **RAM**: 16GB+ system RAM (32GB recommended)
- **Storage**: 200GB+ free space for market data and historical records
- **Network**: Gigabit Ethernet connection

## 🚀 Phase 1: System Preparation

### 1.1 Initial Setup

```bash
# SSH into cpu-node2
ssh sanzad@cpu-node2

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git curl wget htop
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libssl-dev libffi-dev python3-dev
sudo apt install -y postgresql postgresql-contrib
sudo apt install -y redis-server
sudo apt install -y nginx
```

### 1.2 Python Environment Setup

```bash
# Create virtual environment
python3 -m venv ~/market-data-env
source ~/market-data-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install fastapi uvicorn python-multipart
pip install pandas numpy scipy
pip install yfinance ccxt alpha_vantage quandl
pip install pandas-ta ta-lib finta
pip install backtrader zipline-reloaded
pip install psycopg2-binary redis aioredis
pip install aiohttp requests
pip install pyyaml jinja2
pip install matplotlib seaborn plotly
pip install schedule croniter
```

## 🧠 Phase 2: Market Data Services Implementation

### 2.1 Create Market Data Service Structure

```bash
# Create service directory structure
mkdir -p ~/market-data-services
cd ~/market-data-services

mkdir -p services
mkdir -p data
mkdir -p analysis
mkdir -p backtesting
mkdir -p config
mkdir -p utils
mkdir -p logs
mkdir -p cache
```

### 2.2 Create Market Data Service

```python
# ~/market-data-services/services/market_data_service.py
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import yfinance as yf
import ccxt
from alpha_vantage.timeseries import TimeSeries
import redis
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.get('redis', {}).get('host', 'localhost'),
            port=config.get('redis', {}).get('port', 6379),
            db=0
        )
        
        # Initialize data sources
        self.yf_tickers = {}
        self.exchanges = {}
        self.alpha_vantage = None
        
        # Initialize Alpha Vantage if API key provided
        if config.get('alpha_vantage', {}).get('api_key'):
            self.alpha_vantage = TimeSeries(
                key=config['alpha_vantage']['api_key'],
                output_format='pandas'
            )
        
        # Initialize crypto exchanges
        self._init_exchanges()
        
    def _init_exchanges(self):
        """Initialize cryptocurrency exchanges"""
        try:
            self.exchanges = {
                'binance': ccxt.binance({
                    'apiKey': self.config.get('binance', {}).get('api_key', ''),
                    'secret': self.config.get('binance', {}).get('secret', ''),
                    'sandbox': self.config.get('binance', {}).get('sandbox', True)
                }),
                'coinbase': ccxt.coinbasepro({
                    'apiKey': self.config.get('coinbase', {}).get('api_key', ''),
                    'secret': self.config.get('coinbase', {}).get('secret', ''),
                    'password': self.config.get('coinbase', {}).get('password', '')
                })
            }
            logger.info("Cryptocurrency exchanges initialized")
        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
    
    async def get_stock_data(self, symbols: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Get stock market data using yfinance"""
        try:
            data = {}
            cache_key = f"stock_data:{','.join(symbols)}:{period}:{interval}"
            
            # Check cache first
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved stock data from cache for {len(symbols)} symbols")
                return json.loads(cached_data)
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period, interval=interval)
                    
                    if not df.empty:
                        # Clean and standardize data
                        df = self._clean_stock_data(df)
                        data[symbol] = df
                        logger.info(f"Retrieved data for {symbol}: {len(df)} records")
                    else:
                        logger.warning(f"No data retrieved for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error retrieving data for {symbol}: {e}")
                    continue
            
            # Cache the data for 5 minutes
            if data:
                self.redis_client.setex(cache_key, 300, json.dumps(data))
            
            return data
            
        except Exception as e:
            logger.error(f"Stock data retrieval failed: {e}")
            return {}
    
    async def get_crypto_data(self, symbols: List[str], limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """Get cryptocurrency data using ccxt"""
        try:
            data = {}
            cache_key = f"crypto_data:{','.join(symbols)}:{limit}"
            
            # Check cache first
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved crypto data from cache for {len(symbols)} symbols")
                return json.loads(cached_data)
            
            for symbol in symbols:
                try:
                    # Try Binance first, fallback to Coinbase
                    exchange = self.exchanges.get('binance') or self.exchanges.get('coinbase')
                    if not exchange:
                        logger.warning("No exchange available for crypto data")
                        continue
                    
                    # Get OHLCV data
                    ohlcv = await exchange.fetch_ohlcv(symbol, '1d', limit=limit)
                    
                    if ohlcv:
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Clean and standardize data
                        df = self._clean_crypto_data(df)
                        data[symbol] = df
                        logger.info(f"Retrieved crypto data for {symbol}: {len(df)} records")
                    else:
                        logger.warning(f"No crypto data retrieved for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error retrieving crypto data for {symbol}: {e}")
                    continue
            
            # Cache the data for 2 minutes (crypto data changes more frequently)
            if data:
                self.redis_client.setex(cache_key, 120, json.dumps(data))
            
            return data
            
        except Exception as e:
            logger.error(f"Crypto data retrieval failed: {e}")
            return {}
    
    async def get_real_time_price(self, symbols: List[str]) -> Dict[str, float]:
        """Get real-time prices for symbols"""
        try:
            prices = {}
            
            for symbol in symbols:
                try:
                    # Check if it's a crypto symbol
                    if '/' in symbol:
                        # Crypto symbol
                        exchange = self.exchanges.get('binance') or self.exchanges.get('coinbase')
                        if exchange:
                            ticker = await exchange.fetch_ticker(symbol)
                            prices[symbol] = ticker['last']
                    else:
                        # Stock symbol
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        prices[symbol] = info.get('regularMarketPrice', 0.0)
                        
                except Exception as e:
                    logger.error(f"Error getting real-time price for {symbol}: {e}")
                    prices[symbol] = 0.0
            
            return prices
            
        except Exception as e:
            logger.error(f"Real-time price retrieval failed: {e}")
            return {}
    
    def _clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize stock data"""
        # Remove rows with missing values
        df = df.dropna()
        
        # Ensure all columns are numeric
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid data
        df = df[df['Close'] > 0]
        
        return df
    
    def _clean_crypto_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize crypto data"""
        # Remove rows with missing values
        df = df.dropna()
        
        # Ensure all columns are numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid data
        df = df[df['close'] > 0]
        
        return df
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get market summary and indices"""
        try:
            indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']  # S&P 500, Dow, NASDAQ, VIX
            summary = {}
            
            for index in indices:
                try:
                    ticker = yf.Ticker(index)
                    info = ticker.info
                    
                    summary[index] = {
                        'name': info.get('longName', index),
                        'price': info.get('regularMarketPrice', 0),
                        'change': info.get('regularMarketChange', 0),
                        'change_percent': info.get('regularMarketChangePercent', 0),
                        'volume': info.get('volume', 0)
                    }
                    
                except Exception as e:
                    logger.error(f"Error getting market summary for {index}: {e}")
                    continue
            
            return summary
            
        except Exception as e:
            logger.error(f"Market summary retrieval failed: {e}")
            return {}
```

### 2.3 Create Technical Indicators Service

```python
# ~/market-data-services/services/technical_indicators.py
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicatorsService:
    def __init__(self):
        self.indicators = {}
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            rsi = ta.rsi(prices, length=period)
            return rsi
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return pd.Series(index=prices.index)
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            macd = ta.macd(prices, fast=fast, slow=slow, signal=signal)
            return {
                'macd': macd[f'MACD_{fast}_{slow}_{signal}'],
                'signal': macd[f'MACDs_{fast}_{slow}_{signal}'],
                'histogram': macd[f'MACDh_{fast}_{slow}_{signal}']
            }
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return {}
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            bb = ta.bbands(prices, length=period, std=std_dev)
            return {
                'upper': bb[f'BBU_{period}_{std_dev}'],
                'middle': bb[f'BBM_{period}_{std_dev}'],
                'lower': bb[f'BBL_{period}_{std_dev}']
            }
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            return {}
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            stoch = ta.stoch(high, low, close, k=k_period, d=d_period)
            return {
                'k': stoch[f'STOCHk_{k_period}_{d_period}'],
                'd': stoch[f'STOCHd_{k_period}_{d_period}']
            }
        except Exception as e:
            logger.error(f"Stochastic calculation failed: {e}")
            return {}
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            atr = ta.atr(high, low, close, length=period)
            return atr
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            return pd.Series(index=close.index)
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a dataframe"""
        try:
            result_df = df.copy()
            
            # Ensure we have required columns
            required_columns = ['Close', 'High', 'Low', 'Volume']
            if not all(col in result_df.columns for col in required_columns):
                logger.error("DataFrame missing required columns for technical analysis")
                return result_df
            
            # Calculate RSI
            result_df['RSI'] = self.calculate_rsi(result_df['Close'])
            
            # Calculate MACD
            macd_data = self.calculate_macd(result_df['Close'])
            if macd_data:
                result_df['MACD'] = macd_data['macd']
                result_df['MACD_Signal'] = macd_data['signal']
                result_df['MACD_Histogram'] = macd_data['histogram']
            
            # Calculate Bollinger Bands
            bb_data = self.calculate_bollinger_bands(result_df['Close'])
            if bb_data:
                result_df['BB_Upper'] = bb_data['upper']
                result_df['BB_Middle'] = bb_data['middle']
                result_df['BB_Lower'] = bb_data['lower']
            
            # Calculate Stochastic
            stoch_data = self.calculate_stochastic(result_df['High'], result_df['Low'], result_df['Close'])
            if stoch_data:
                result_df['Stoch_K'] = stoch_data['k']
                result_df['Stoch_D'] = stoch_data['d']
            
            # Calculate ATR
            result_df['ATR'] = self.calculate_atr(result_df['High'], result_df['Low'], result_df['Close'])
            
            # Calculate moving averages
            result_df['SMA_20'] = ta.sma(result_df['Close'], length=20)
            result_df['SMA_50'] = ta.sma(result_df['Close'], length=50)
            result_df['EMA_12'] = ta.ema(result_df['Close'], length=12)
            result_df['EMA_26'] = ta.ema(result_df['Close'], length=26)
            
            # Calculate volume indicators
            result_df['Volume_SMA'] = ta.sma(result_df['Volume'], length=20)
            result_df['OBV'] = ta.obv(result_df['Close'], result_df['Volume'])
            
            logger.info(f"Calculated technical indicators for {len(result_df)} data points")
            return result_df
            
        except Exception as e:
            logger.error(f"Technical indicators calculation failed: {e}")
            return df
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate trading signals based on technical indicators"""
        try:
            signals = {}
            
            for symbol in df.columns:
                if symbol == 'Close':
                    continue
                
                # Simple signal generation logic (can be enhanced)
                if 'RSI' in df.columns and not df['RSI'].isna().all():
                    rsi = df['RSI'].iloc[-1]
                    if rsi < 30:
                        signals[symbol] = 'BUY'
                    elif rsi > 70:
                        signals[symbol] = 'SELL'
                    else:
                        signals[symbol] = 'HOLD'
                else:
                    signals[symbol] = 'HOLD'
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {}
```

### 2.4 Create Backtesting Service

```python
# ~/market-data-services/services/backtesting_service.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime, timedelta
import backtrader as bt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestingService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
    
    def run_backtest(self, strategy_class: type, data: pd.DataFrame, 
                    initial_cash: float = 100000, commission: float = 0.001) -> Dict[str, Any]:
        """Run backtest using Backtrader"""
        try:
            # Create Backtrader engine
            cerebro = bt.Cerebro()
            
            # Add data feed
            data_feed = self._create_data_feed(data)
            cerebro.adddata(data_feed)
            
            # Add strategy
            cerebro.addstrategy(strategy_class)
            
            # Set initial cash and commission
            cerebro.broker.setcash(initial_cash)
            cerebro.broker.setcommission(commission=commission)
            
            # Run backtest
            logger.info(f"Starting backtest with {initial_cash} initial cash")
            results = cerebro.run()
            
            # Extract results
            portfolio_value = cerebro.broker.getvalue()
            total_return = (portfolio_value - initial_cash) / initial_cash
            
            # Get trade statistics
            trades = self._extract_trades(results[0])
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(data, trades, initial_cash, portfolio_value)
            
            result = {
                'initial_cash': initial_cash,
                'final_value': portfolio_value,
                'total_return': total_return,
                'trades': trades,
                'performance': performance,
                'strategy': strategy_class.__name__
            }
            
            logger.info(f"Backtest completed. Final value: {portfolio_value:.2f}, Return: {total_return:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {}
    
    def _create_data_feed(self, data: pd.DataFrame) -> bt.feeds.PandasData:
        """Create Backtrader data feed from pandas DataFrame"""
        try:
            # Ensure data has required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Rename columns if needed
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data[new_col] = data[old_col]
            
            # Create data feed
            data_feed = bt.feeds.PandasData(
                dataname=data,
                datetime=None,  # Use index as datetime
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume='Volume',
                openinterest=None
            )
            
            return data_feed
            
        except Exception as e:
            logger.error(f"Data feed creation failed: {e}")
            raise
    
    def _extract_trades(self, strategy) -> List[Dict[str, Any]]:
        """Extract trade information from strategy results"""
        try:
            trades = []
            
            if hasattr(strategy, 'trades'):
                for trade in strategy.trades:
                    trade_info = {
                        'entry_date': trade.dtopen,
                        'exit_date': trade.dtclose,
                        'entry_price': trade.price,
                        'exit_price': trade.pclose,
                        'size': trade.size,
                        'pnl': trade.pnl,
                        'pnlcomm': trade.pnlcomm,
                        'status': 'closed' if trade.isclosed else 'open'
                    }
                    trades.append(trade_info)
            
            return trades
            
        except Exception as e:
            logger.error(f"Trade extraction failed: {e}")
            return []
    
    def _calculate_performance_metrics(self, data: pd.DataFrame, trades: List[Dict], 
                                     initial_cash: float, final_value: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            # Basic metrics
            total_return = (final_value - initial_cash) / initial_cash
            
            # Calculate daily returns
            if 'Close' in data.columns:
                daily_returns = data['Close'].pct_change().dropna()
                
                # Volatility
                volatility = daily_returns.std() * np.sqrt(252)  # Annualized
                
                # Sharpe ratio (assuming risk-free rate of 0.02)
                risk_free_rate = 0.02
                excess_returns = daily_returns - risk_free_rate/252
                sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(252)
                
                # Maximum drawdown
                cumulative_returns = (1 + daily_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # Win rate
                if trades:
                    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
                    win_rate = winning_trades / len(trades)
                else:
                    win_rate = 0.0
                
                performance = {
                    'total_return': total_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'total_trades': len(trades)
                }
                
                return performance
            
            return {}
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {}
    
    def compare_strategies(self, strategies: List[type], data: pd.DataFrame, 
                          initial_cash: float = 100000) -> Dict[str, Dict[str, Any]]:
        """Compare multiple strategies"""
        try:
            results = {}
            
            for strategy in strategies:
                logger.info(f"Testing strategy: {strategy.__name__}")
                result = self.run_backtest(strategy, data, initial_cash)
                results[strategy.__name__] = result
            
            return results
            
        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            return {}
```

## 🚀 Phase 3: API Service Implementation

### 3.1 Create FastAPI Service

```python
# ~/market-data-services/api_service.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import logging
import yaml
from datetime import datetime, timedelta
import pandas as pd

from services.market_data_service import MarketDataService
from services.technical_indicators import TechnicalIndicatorsService
from services.backtesting_service import BacktestingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Market Data Services", version="1.0.0")

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize services
market_data_service = MarketDataService(config)
technical_indicators_service = TechnicalIndicatorsService()
backtesting_service = BacktestingService(config)

class DataRequest(BaseModel):
    symbols: List[str]
    period: str = "1y"
    interval: str = "1d"

class TechnicalAnalysisRequest(BaseModel):
    symbols: List[str]
    indicators: List[str]
    period: str = "1y"

class BacktestRequest(BaseModel):
    strategy: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_cash: float = 100000

class MarketDataResponse(BaseModel):
    symbols: List[str]
    data: Dict[str, Dict[str, Any]]
    timestamp: datetime

class TechnicalAnalysisResponse(BaseModel):
    symbols: List[str]
    indicators: Dict[str, Dict[str, Any]]
    signals: Dict[str, str]
    timestamp: datetime

class BacktestResponse(BaseModel):
    strategy: str
    initial_cash: float
    final_value: float
    total_return: float
    performance: Dict[str, float]
    trades: List[Dict[str, Any]]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Market Data Services",
        "timestamp": datetime.now()
    }

@app.post("/get_market_data", response_model=MarketDataResponse)
async def get_market_data(request: DataRequest):
    """Get market data for specified symbols"""
    try:
        # Get stock data
        stock_data = await market_data_service.get_stock_data(request.symbols, request.period, request.interval)
        
        # Get crypto data for symbols that look like crypto
        crypto_symbols = [s for s in request.symbols if '/' in s]
        crypto_data = {}
        if crypto_symbols:
            crypto_data = await market_data_service.get_crypto_data(crypto_symbols)
        
        # Combine data
        all_data = {**stock_data, **crypto_data}
        
        # Convert to serializable format
        serializable_data = {}
        for symbol, df in all_data.items():
            serializable_data[symbol] = {
                'data': df.to_dict('records'),
                'columns': list(df.columns),
                'index': df.index.strftime('%Y-%m-%d').tolist()
            }
        
        return MarketDataResponse(
            symbols=request.symbols,
            data=serializable_data,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Market data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_technical_analysis", response_model=TechnicalAnalysisResponse)
async def get_technical_analysis(request: TechnicalAnalysisRequest):
    """Get technical analysis for specified symbols"""
    try:
        # Get market data first
        data = await market_data_service.get_stock_data(request.symbols, request.period)
        
        indicators_data = {}
        signals_data = {}
        
        for symbol, df in data.items():
            try:
                # Calculate technical indicators
                df_with_indicators = technical_indicators_service.calculate_all_indicators(df)
                
                # Extract indicator values
                indicators_data[symbol] = {}
                for col in df_with_indicators.columns:
                    if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        indicators_data[symbol][col] = df_with_indicators[col].iloc[-1] if not df_with_indicators[col].isna().all() else None
                
                # Generate signals
                signals = technical_indicators_service.generate_signals(df_with_indicators)
                signals_data[symbol] = signals.get(symbol, 'HOLD')
                
            except Exception as e:
                logger.error(f"Technical analysis failed for {symbol}: {e}")
                indicators_data[symbol] = {}
                signals_data[symbol] = 'HOLD'
        
        return TechnicalAnalysisResponse(
            symbols=request.symbols,
            indicators=indicators_data,
            signals=signals_data,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Technical analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run backtest for specified strategy and symbols"""
    try:
        # Get market data for backtesting period
        data = await market_data_service.get_stock_data(request.symbols, "max")
        
        # Filter data by date range
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        filtered_data = {}
        for symbol, df in data.items():
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_df = df.loc[mask]
            if not filtered_df.empty:
                filtered_data[symbol] = filtered_df
        
        if not filtered_data:
            raise HTTPException(status_code=400, detail="No data available for specified date range")
        
        # For now, use a simple moving average strategy
        # In production, you would load the actual strategy class
        class SimpleMAStrategy(bt.Strategy):
            params = (('fast', 10), ('slow', 30))
            
            def __init__(self):
                self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast)
                self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow)
                self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
            
            def next(self):
                if self.crossover > 0:  # Golden cross
                    self.buy()
                elif self.crossover < 0:  # Death cross
                    self.sell()
        
        # Run backtest
        result = backtesting_service.run_backtest(
            SimpleMAStrategy, 
            list(filtered_data.values())[0],  # Use first symbol for now
            request.initial_cash
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Backtest failed")
        
        return BacktestResponse(
            strategy=request.strategy,
            initial_cash=result['initial_cash'],
            final_value=result['final_value'],
            total_return=result['total_return'],
            performance=result['performance'],
            trades=result['trades']
        )
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_real_time_prices")
async def get_real_time_prices(symbols: str):
    """Get real-time prices for symbols (comma-separated)"""
    try:
        symbol_list = [s.strip() for s in symbols.split(',')]
        prices = await market_data_service.get_real_time_price(symbol_list)
        
        return {
            "prices": prices,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Real-time price retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_market_summary")
async def get_market_summary():
    """Get market summary and indices"""
    try:
        summary = await market_data_service.get_market_summary()
        
        return {
            "summary": summary,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Market summary retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
```

### 3.2 Create Configuration File

```yaml
# ~/market-data-services/config/config.yaml
redis:
  host: "localhost"
  port: 6379
  db: 0

alpha_vantage:
  api_key: ""  # Add your Alpha Vantage API key here
  rate_limit: 5  # requests per minute

binance:
  api_key: ""  # Add your Binance API key here
  secret: ""   # Add your Binance secret here
  sandbox: true

coinbase:
  api_key: ""  # Add your Coinbase API key here
  secret: ""   # Add your Coinbase secret here
  password: "" # Add your Coinbase password here

data_sources:
  default_period: "1y"
  default_interval: "1d"
  cache_duration:
    stock: 300  # 5 minutes
    crypto: 120  # 2 minutes
  
  symbols:
    default_stocks:
      - "AAPL"
      - "GOOGL"
      - "MSFT"
      - "TSLA"
      - "NVDA"
      - "AMZN"
      - "META"
      - "NFLX"
    
    default_crypto:
      - "BTC/USDT"
      - "ETH/USDT"
      - "BNB/USDT"
      - "ADA/USDT"
      - "SOL/USDT"

technical_analysis:
  default_indicators:
    - "RSI"
    - "MACD"
    - "Bollinger_Bands"
    - "Stochastic"
    - "ATR"
    - "Moving_Averages"
  
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bollinger_period: 20
  bollinger_std: 2.0

backtesting:
  default_initial_cash: 100000
  default_commission: 0.001
  max_lookback: 2520  # 10 years of daily data

api:
  host: "0.0.0.0"
  port: 8083
  max_concurrent_requests: 50
  request_timeout: 60

logging:
  level: "INFO"
  file: "logs/market_data_services.log"
  max_size: "100MB"
  backup_count: 5
```

### 3.3 Create Systemd Service

```bash
# Create systemd service file
sudo tee /etc/systemd/system/market-data-services.service > /dev/null << 'EOF'
[Unit]
Description=Market Data Services
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=sanzad
WorkingDirectory=/home/sanzad/market-data-services
Environment=PATH=/home/sanzad/market-data-env/bin
ExecStart=/home/sanzad/market-data-env/bin/python api_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable market-data-services.service
sudo systemctl start market-data-services.service

# Check service status
sudo systemctl status market-data-services.service
```

## 🧪 Phase 4: Testing & Validation

### 4.1 Test Market Data Service

```bash
# Test market data service
cd ~/market-data-services
source ~/market-data-env/bin/activate

python3 -c "
from services.market_data_service import MarketDataService
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Test service
service = MarketDataService(config)
print('MarketDataService created successfully!')

# Test stock data retrieval
import asyncio
async def test():
    data = await service.get_stock_data(['AAPL', 'GOOGL'], '1mo', '1d')
    print(f'Retrieved data for {len(data)} symbols')
    for symbol, df in data.items():
        print(f'{symbol}: {len(df)} records')

asyncio.run(test())
"
```

### 4.2 Test Technical Indicators

```bash
# Test technical indicators
python3 -c "
from services.technical_indicators import TechnicalIndicatorsService
import pandas as pd
import yfinance as yf

# Get sample data
ticker = yf.Ticker('AAPL')
data = ticker.history(period='1y')

# Test indicators
service = TechnicalIndicatorsService()
df_with_indicators = service.calculate_all_indicators(data)

print('Technical indicators calculated successfully!')
print(f'Columns: {list(df_with_indicators.columns)}')
print(f'Data points: {len(df_with_indicators)}')

# Test signal generation
signals = service.generate_signals(df_with_indicators)
print(f'Signals: {signals}')
"
```

### 4.3 Test API Endpoints

```bash
# Start API service
cd ~/market-data-services
source ~/market-data-env/bin/activate
python3 api_service.py &

# Test health endpoint
curl http://localhost:8083/health

# Test market data endpoint
curl -X POST "http://localhost:8083/get_market_data" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "GOOGL"],
    "period": "1mo",
    "interval": "1d"
  }'

# Test technical analysis endpoint
curl -X POST "http://localhost:8083/get_technical_analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "GOOGL"],
    "indicators": ["RSI", "MACD"],
    "period": "1mo"
  }'

# Test real-time prices
curl "http://localhost:8083/get_real_time_prices?symbols=AAPL,GOOGL,TSLA"

# Test market summary
curl http://localhost:8083/get_market_summary
```

## 🔧 Phase 5: Integration with LangGraph Cluster

### 5.1 Update Cluster Configuration

```bash
# On cpu-node1, update the cluster configuration
ssh cpu-node1

# Edit cluster config to include cpu-node2
vim ~/ai-infrastructure/langgraph-config/config.py
```

Add to the configuration:
```python
CPU_NODES = {
    'cpu-node1': {
        'host': '192.168.1.81',
        'port': 8082,
        'services': ['langgraph_orchestrator', 'workflow_engine'],
        'type': 'coordinator'
    },
    'cpu-node2': {
        'host': '192.168.1.82',  # Adjust IP as needed
        'port': 8083,
        'services': ['market_data_service', 'technical_indicators', 'backtesting_engine'],
        'type': 'data_processor'
    }
}
```

### 5.2 Test Cluster Integration

```bash
# Test connectivity from coordinator
curl http://192.168.1.82:8083/health

# Test market data from coordinator
curl -X POST "http://192.168.1.82:8083/get_market_data" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "period": "1mo"}'
```

## 📊 Monitoring & Maintenance

### 5.1 Log Monitoring

```bash
# View service logs
sudo journalctl -u market-data-services.service -f

# Monitor system resources
htop

# Check Redis status
redis-cli ping
redis-cli info memory
```

### 5.2 Performance Monitoring

```bash
# Test data retrieval performance
cd ~/market-data-services
source ~/market-data-env/bin/activate

python3 -c "
import asyncio
import time
from services.market_data_service import MarketDataService
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Test performance
service = MarketDataService(config)
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

async def test_performance():
    start_time = time.time()
    data = await service.get_stock_data(symbols, '1y', '1d')
    end_time = time.time()
    
    print(f'Retrieved data for {len(symbols)} symbols in {end_time - start_time:.2f} seconds')
    for symbol, df in data.items():
        print(f'{symbol}: {len(df)} records')

asyncio.run(test_performance())
"
```

## 🚨 Troubleshooting

### Common Issues

**Service Not Starting**
```bash
# Check service status
sudo systemctl status market-data-services.service

# Check logs
sudo journalctl -u market-data-services.service -n 50

# Check port availability
netstat -tlnp | grep 8083
```

**Data Retrieval Failures**
```bash
# Test yfinance
python3 -c "
import yfinance as yf
ticker = yf.Ticker('AAPL')
data = ticker.history(period='1mo')
print(f'AAPL data: {len(data)} records')
"

# Test ccxt
python3 -c "
import ccxt
exchange = ccxt.binance()
ticker = exchange.fetch_ticker('BTC/USDT')
print(f'BTC price: {ticker[\"last\"]}')
"
```

**Redis Connection Issues**
```bash
# Check Redis status
sudo systemctl status redis-server

# Test Redis connection
redis-cli ping

# Check Redis logs
sudo journalctl -u redis-server -f
```

## ✅ Verification Checklist

- [ ] Python environment with all dependencies installed
- [ ] Market data service working and retrieving data
- [ ] Technical indicators service calculating indicators correctly
- [ ] Backtesting service running backtests
- [ ] API service running and responding to requests
- [ ] Configuration file properly set up
- [ ] Systemd service configured and running
- [ ] Redis server running and accessible
- [ ] Integration with LangGraph cluster working
- [ ] All API endpoints responding correctly

## 🔄 Next Steps

1. **Test complete distributed workflow** with all nodes
2. **Implement advanced features** like real-time data streaming
3. **Add monitoring and alerting** for production use
4. **Integrate with trading execution systems**
5. **Implement data persistence and historical analysis**

---

Your cpu-node2 is now ready to provide market data services, technical analysis, and backtesting capabilities for the distributed trading workflow! 🚀
