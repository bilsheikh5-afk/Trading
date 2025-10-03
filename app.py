from flask import Flask
from flask_cors import CORS  # If using CORS

app = Flask(__trading advisor__)
CORS(app)  # Enable CORS if needed

@app.route('/')
def home():
    return {'message': 'Trading Advisor API is live!'}

@app.route('/health')
def health():
    return {'status': 'healthy'}

if __trading advisor__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # For local testing
from flask import send_from_directory
import os

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import ccxt
import pandas as pd
import numpy as np
import ta
import asyncio
import threading
import time
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for caching
market_data_cache = {}
analysis_cache = {}
cache_expiry = {}

class LiveMarketData:
    def __init__(self):
        self.exchanges = {
            'binance': ccxt.binance(),
            'kucoin': ccxt.kucoin(),
            'bybit': ccxt.bybit(),
            'okx': ccxt.okx()
        }
        self.symbol_categories = {
            'Crypto': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 
                      'MATIC/USDT', 'LTC/USDT', 'XRP/USDT', 'DOGE/USDT', 'AVAX/USDT'],
            'Forex': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 
                     'USD/CHF', 'NZD/USD', 'EUR/GBP'],
            'Indices': ['US30/USD', 'SPX/USD', 'NAS100/USD', 'DJI/USD', 'IXIC/USD'],
            'Commodities': ['XAU/USD', 'XAG/USD', 'OIL/USD', 'XPT/USD', 'XPD/USD']
        }
        
    def get_available_symbols(self, exchange_id='binance'):
        """Get all available symbols from exchange"""
        try:
            exchange = self.exchanges.get(exchange_id, self.exchanges['binance'])
            markets = exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            logger.error(f"Error fetching symbols from {exchange_id}: {e}")
            return []
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 200, exchange_id: str = 'binance'):
        """Fetch OHLCV data from exchange"""
        cache_key = f"{exchange_id}_{symbol}_{timeframe}_{limit}"
        
        # Check cache (5-minute expiry)
        if cache_key in market_data_cache:
            if time.time() - cache_expiry.get(cache_key, 0) < 300:
                return market_data_cache[cache_key]
        
        try:
            exchange = self.exchanges.get(exchange_id, self.exchanges['binance'])
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.dropna()
            
            # Cache the data
            market_data_cache[cache_key] = df
            cache_expiry[cache_key] = time.time()
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame):
        """Calculate technical indicators"""
        if df is None or len(df) < 50:
            return None
            
        df = df.copy()
        
        # EMAs
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        
        return df

class TradingAnalyzer:
    def __init__(self):
        self.market_data = LiveMarketData()
    
    def analyze_symbol(self, symbol: str, exchange_id: str = 'binance', capital: float = 10000):
        """Analyze a single symbol for trading signals"""
        try:
            # Fetch data for multiple timeframes
            df_1h = self.market_data.fetch_ohlcv(symbol, '1h', 200, exchange_id)
            df_4h = self.market_data.fetch_ohlcv(symbol, '4h', 100, exchange_id)
            
            if df_1h is None or df_4h is None:
                return None
            
            # Calculate indicators
            df_1h = self.market_data.calculate_indicators(df_1h)
            df_4h = self.market_data.calculate_indicators(df_4h)
            
            if df_1h is None or len(df_1h) < 50:
                return None
            
            # Get current values
            current_price = df_1h['close'].iloc[-1]
            ema_fast = df_1h['ema_12'].iloc[-1]
            ema_slow = df_1h['ema_26'].iloc[-1]
            rsi = df_1h['rsi'].iloc[-1]
            atr = df_1h['atr'].iloc[-1]
            
            # HTF trend
            htf_ema_fast = df_4h['ema_12'].iloc[-1] if df_4h is not None else ema_fast
            htf_ema_slow = df_4h['ema_26'].iloc[-1] if df_4h is not None else ema_slow
            htf_trend_up = htf_ema_fast > htf_ema_slow
            
            # Generate signals
            signal = self.generate_signal(df_1h, df_4h, htf_trend_up)
            
            if signal['direction'] == 'HOLD':
                return None
            
            # Calculate position sizing
            position_info = self.calculate_position(
                signal['direction'], current_price, signal['sl_price'], 
                atr, capital, symbol
            )
            
            # Calculate score
            score = self.calculate_score(df_1h, df_4h, signal['direction'])
            
            return {
                'symbol': symbol,
                'area': self.get_symbol_area(symbol),
                'direction': signal['direction'],
                'score': round(score, 3),
                'price': round(current_price, 6),
                'sl': round(signal['sl_price'], 6),
                'tp': round(signal['tp_price'], 6),
                'rr': round(signal['rr_ratio'], 2),
                'atr': round(atr, 6),
                'rsi': round(rsi, 2),
                'ema_fast': round(ema_fast, 6),
                'ema_slow': round(ema_slow, 6),
                'htf_trend_ok': signal['htf_trend_ok'],
                'alloc_usd': round(position_info['alloc_usd'], 2),
                'qty': round(position_info['qty'], 6),
                'timeframe_lt': '1h',
                'timeframe_ht': '4h',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def generate_signal(self, df_lt, df_ht, htf_trend_up):
        """Generate trading signal based on indicators"""
        current_price = df_lt['close'].iloc[-1]
        ema_fast = df_lt['ema_12'].iloc[-1]
        ema_slow = df_lt['ema_26'].iloc[-1]
        rsi = df_lt['rsi'].iloc[-1]
        atr = df_lt['atr'].iloc[-1]
        
        # Check for EMA crossover
        ema_fast_prev = df_lt['ema_12'].iloc[-2]
        ema_slow_prev = df_lt['ema_26'].iloc[-2]
        
        bull_cross = ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow
        bear_cross = ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow
        
        direction = 'HOLD'
        sl_price = tp_price = rr_ratio = 0
        htf_trend_ok = False
        
        if bull_cross and rsi < 65 and htf_trend_up:
            direction = 'BUY'
            sl_price = current_price - (atr * 2)
            tp_price = current_price + (abs(current_price - sl_price) * 2)
            rr_ratio = 2.0
            htf_trend_ok = True
        elif bear_cross and rsi > 35 and not htf_trend_up:
            direction = 'SELL'
            sl_price = current_price + (atr * 2)
            tp_price = current_price - (abs(current_price - sl_price) * 2)
            rr_ratio = 2.0
            htf_trend_ok = True
        
        return {
            'direction': direction,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'rr_ratio': rr_ratio,
            'htf_trend_ok': htf_trend_ok
        }
    
    def calculate_position(self, direction, price, sl, atr, capital, symbol):
        """Calculate position size based on risk management"""
        if direction == 'HOLD':
            return {'alloc_usd': 0, 'qty': 0}
        
        # Risk per trade (1% of capital)
        risk_amount = capital * 0.01
        
        # Stop distance
        stop_distance = abs(price - sl)
        if stop_distance == 0:
            return {'alloc_usd': 0, 'qty': 0}
        
        # Calculate quantity
        qty = risk_amount / stop_distance
        
        # Calculate allocation
        alloc_usd = qty * price
        
        # Ensure allocation doesn't exceed 10% of capital
        max_alloc = capital * 0.1
        if alloc_usd > max_alloc:
            alloc_usd = max_alloc
            qty = alloc_usd / price
        
        return {
            'alloc_usd': alloc_usd,
            'qty': qty
        }
    
    def calculate_score(self, df_lt, df_ht, direction):
        """Calculate signal quality score (0-1)"""
        if direction == 'HOLD':
            return 0
        
        score = 0.5  # Base score
        
        # RSI factor
        rsi = df_lt['rsi'].iloc[-1]
        if direction == 'BUY':
            rsi_score = max(0, 1 - (rsi / 70))
        else:
            rsi_score = max(0, rsi / 30 - 1)
        score += rsi_score * 0.2
        
        # Volume factor (if available)
        if 'volume' in df_lt.columns:
            volume_avg = df_lt['volume'].tail(20).mean()
            current_volume = df_lt['volume'].iloc[-1]
            if volume_avg > 0:
                volume_ratio = current_volume / volume_avg
                volume_score = min(1, volume_ratio / 2)
                score += volume_score * 0.1
        
        # ATR factor (volatility)
        atr_pct = df_lt['atr'].iloc[-1] / df_lt['close'].iloc[-1]
        volatility_score = max(0, 1 - abs(atr_pct - 0.02) / 0.02)
        score += volatility_score * 0.2
        
        return min(1.0, score)
    
    def get_symbol_area(self, symbol):
        """Categorize symbol into market area"""
        symbol_upper = symbol.upper()
        for area, symbols in self.market_data.symbol_categories.items():
            if any(s in symbol_upper for s in symbols):
                return area
        return 'Crypto' if '/USDT' in symbol_upper else 'Other'

# Initialize analyzer
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
analyzer = TradingAnalyzer()

# API Routes
@app.route('/')
def home():
    return jsonify({
        "message": "Trading Advisor API",
        "version": "1.0",
        "endpoints": {
            "/api/analyze": "POST - Analyze trading opportunities",
            "/api/symbols": "GET - Get available symbols",
            "/api/ohlcv": "GET - Get OHLCV data",
            "/api/health": "GET - Health check"
        }
    })

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/symbols')
def get_symbols():
    exchange = request.args.get('exchange', 'binance')
    category = request.args.get('category', 'all')
    
    if category == 'all':
        symbols = []
        for area_symbols in analyzer.market_data.symbol_categories.values():
            symbols.extend(area_symbols)
    else:
        symbols = analyzer.market_data.symbol_categories.get(category, [])
    
    return jsonify({
        "exchange": exchange,
        "category": category,
        "symbols": symbols,
        "count": len(symbols)
    })

@app.route('/api/ohlcv')
def get_ohlcv():
    symbol = request.args.get('symbol', 'BTC/USDT')
    timeframe = request.args.get('timeframe', '1h')
    limit = int(request.args.get('limit', '100'))
    exchange = request.args.get('exchange', 'binance')
    
    df = analyzer.market_data.fetch_ohlcv(symbol, timeframe, limit, exchange)
    
    if df is None:
        return jsonify({"error": "Failed to fetch data"}), 400
    
    # Convert to list of dicts for JSON
    ohlcv_data = []
    for _, row in df.tail(limit).iterrows():
        ohlcv_data.append({
            'timestamp': row['timestamp'].isoformat(),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']) if 'volume' in row else 0
        })
    
    return jsonify({
        "symbol": symbol,
        "timeframe": timeframe,
        "data": ohlcv_data
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_markets():
    try:
        data = request.get_json()
        
        # Get parameters
        categories = data.get('categories', ['Crypto'])
        exchange = data.get('exchange', 'binance')
        top_n = data.get('top_n', 5)
        capital = data.get('capital', 10000)
        
        # Get symbols to analyze
        symbols_to_analyze = []
        for category in categories:
            symbols_to_analyze.extend(
                analyzer.market_data.symbol_categories.get(category, [])
            )
        
        # Analyze symbols
        recommendations = []
        for symbol in symbols_to_analyze:
            result = analyzer.analyze_symbol(symbol, exchange, capital)
            if result:
                recommendations.append(result)
        
        # Sort by score and get top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        top_recommendations = recommendations[:top_n]
        
        return jsonify({
            "success": True,
            "analysis_time": datetime.now().isoformat(),
            "parameters": {
                "categories": categories,
                "exchange": exchange,
                "top_n": top_n,
                "capital": capital
            },
            "recommendations": top_recommendations,
            "total_analyzed": len(symbols_to_analyze),
            "signals_found": len(top_recommendations)
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/analyze-symbol', methods=['POST'])
def analyze_single_symbol():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC/USDT')
        exchange = data.get('exchange', 'binance')
        capital = data.get('capital', 10000)
        
        result = analyzer.analyze_symbol(symbol, exchange, capital)
        
        if result:
            return jsonify({
                "success": True,
                "recommendation": result
            })
        else:
            return jsonify({
                "success": False,
                "message": "No trading signal found"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Trading Advisor API Server...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/symbols - Get available symbols")
    print("  GET  /api/ohlcv - Get market data")
    print("  POST /api/analyze - Analyze multiple symbols")
    print("  POST /api/analyze-symbol - Analyze single symbol")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
