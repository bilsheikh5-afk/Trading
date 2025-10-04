# path: main.py
"""
FastAPI backend for Live Trading Advisor.
- Crypto: Binance public API (spot), 1h klines
- Forex & Commodities: Yahoo Finance via yfinance (1h candles)
- Indicators: RSI(14), EMA(20/50), ATR(14)
- CORS enabled for browser frontends
"""

from __future__ import annotations

import os
import math
import time
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# WHY: yfinance avoids API keys for forex/commodities; python-binance gives real crypto tick data.
import yfinance as yf
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException

# ---------- Config ----------

BINANCE_REQUEST_LIMIT = int(os.getenv("BINANCE_REQUEST_LIMIT", "60"))  # max symbols to scan for crypto
YF_REQUEST_LIMIT = int(os.getenv("YF_REQUEST_LIMIT", "40"))           # max symbols for forex/commodities
DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "5"))
DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", "10000"))
RISK_REWARD = float(os.getenv("RISK_REWARD", "2.0"))

# Static maps for non-crypto universes (extend as needed)
FOREX_TICKERS: Dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "EURJPY": "EURJPY=X",
}

COMMODITY_TICKERS: Dict[str, str] = {
    "GOLD": "GC=F",         # Gold Futures
    "SILVER": "SI=F",       # Silver Futures
    "CRUDE_OIL": "CL=F",    # WTI Crude
    "BRENT_OIL": "BZ=F",    # Brent Crude
    "NATGAS": "NG=F",       # Natural Gas
    "COPPER": "HG=F",       # Copper
    "CORN": "ZC=F",
    "SOYBEAN": "ZS=F",
}

# ---------- App ----------

app = FastAPI(title="Live Trading Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # WHY: frontend may be served from file:// or any host
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize Binance public client (no keys required for public endpoints)
BINANCE = BinanceClient()


# ---------- Models ----------

class AnalyzeRequest(BaseModel):
    categories: List[str]
    exchange: str = "binance"
    top_n: int = DEFAULT_TOP_N
    capital: float = DEFAULT_CAPITAL


# ---------- TA helpers ----------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI(14), EMA(20/50), ATR(14)."""
    close = df["close"]
    high, low = df["high"], df["low"]

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll = 14
    avg_gain = up.ewm(alpha=1/roll, min_periods=roll, adjust=False).mean()
    avg_loss = down.ewm(alpha=1/roll, min_periods=roll, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi.fillna(method="bfill")

    # EMA
    df["ema_fast"] = close.ewm(span=20, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=50, adjust=False).mean()

    # ATR
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    return df


def generate_signal(row: pd.Series) -> Optional[str]:
    """Return BUY/SELL/None based on EMA crossover + RSI guardrails."""
    if pd.isna(row["ema_fast"]) or pd.isna(row["ema_slow"]) or pd.isna(row["rsi"]):
        return None
    if row["ema_fast"] > row["ema_slow"] and row["rsi"] < 70:
        return "BUY"
    if row["ema_fast"] < row["ema_slow"] and row["rsi"] > 30:
        return "SELL"
    return None


def score_signal(row: pd.Series) -> float:
    """Magnitude of EMA separation with RSI & trend bonuses."""
    if pd.isna(row["ema_slow"]) or row["ema_slow"] == 0:
        return 0.0
    base = abs(row["ema_fast"] - row["ema_slow"]) / abs(row["ema_slow"]) * 100
    rsi_boost = (70 - row["rsi"]) / 10 if row["ema_fast"] > row["ema_slow"] else (row["rsi"] - 30) / 10
    trend_ok = 3.0 if row["ema_fast"] > row["ema_slow"] else 0.0
    return round(float(base + max(rsi_boost, 0) + trend_ok), 2)


def build_reco(symbol: str, area: str, row: pd.Series, capital: float) -> Dict[str, Any]:
    price = float(row["close"])
    direction = generate_signal(row)
    if not direction:
        return {}

    # WHY: simple RR based SL/TP for demo; backend users can refine later.
    sl = price * (0.98 if direction == "BUY" else 1.02)
    tp = price * (1.04 if direction == "BUY" else 0.96)

    alloc = min(1000.0, capital * 0.05)  # cap per-trade allocation
    qty = 0.0 if price <= 0 else alloc / price
    return {
        "symbol": symbol,
        "area": area,
        "direction": direction,
        "price": round(price, 6),
        "sl": round(sl, 6),
        "tp": round(tp, 6),
        "rr": RISK_REWARD,
        "rsi": round(float(row["rsi"]), 2),
        "atr": round(float(row["atr"]), 6),
        "ema_fast": round(float(row["ema_fast"]), 6),
        "ema_slow": round(float(row["ema_slow"]), 6),
        "htf_trend_ok": bool(row["ema_fast"] > row["ema_slow"]),
        "alloc_usd": round(alloc, 2),
        "qty": round(qty, 6),
        "score": score_signal(row),
        "timeframe_lt": "1h",
        "timeframe_ht": "4h",
    }


# ---------- Data sources ----------

def fetch_crypto_symbols_usdt(limit: int) -> List[str]:
    try:
        info = BINANCE.get_exchange_info()
        syms = [s["symbol"] for s in info["symbols"] if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"]
        return syms[:limit]
    except Exception as e:
        print("Binance exchange_info error:", e)
        return []


def fetch_binance_1h_df(symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
    try:
        kl = BINANCE.get_klines(symbol=symbol, interval=BinanceClient.KLINE_INTERVAL_1HOUR, limit=limit)
        if not kl:
            return None
        df = pd.DataFrame(kl, columns=[
            "open_time","open","high","low","close","volume","close_time","qav",
            "num_trades","taker_base","taker_quote","ignore"
        ])
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        return df[["open","high","low","close"]].reset_index(drop=True)
    except BinanceAPIException as e:
        print(f"Binance API error for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"Binance error for {symbol}: {e}")
        return None


def yf_download_1h(ticker: str, period: str = "60d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(tickers=ticker, interval="60m", period=period, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)[["open","high","low","close"]]
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"yfinance error for {ticker}: {e}")
        return None


# ---------- Endpoints ----------

@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/symbols")
def get_symbols(category: str = Query("Crypto")) -> Dict[str, List[str]]:
    c = category.strip().lower()
    if c == "crypto":
        return {"symbols": fetch_crypto_symbols_usdt(limit=500)}
    if c == "forex":
        return {"symbols": list(FOREX_TICKERS.keys())}
    if c == "commodities":
        return {"symbols": list(COMMODITY_TICKERS.keys())}
    return {"symbols": []}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    start_ts = time.time()
    categories = [c.strip().lower() for c in req.categories or []]
    if not categories:
        raise HTTPException(status_code=400, detail="At least one category required")

    recommendations: List[Dict[str, Any]] = []
    total_analyzed = 0

    # ---- Crypto (Binance) ----
    if "crypto" in categories:
        crypto_symbols = fetch_crypto_symbols_usdt(limit=BINANCE_REQUEST_LIMIT)
        for sym in crypto_symbols:
            df = fetch_binance_1h_df(sym, limit=120)
            if df is None or df.empty:
                continue
            total_analyzed += 1
            compute_indicators(df)
            last = df.iloc[-1]
            reco = build_reco(sym, "Crypto", last, req.capital)
            if reco:
                recommendations.append(reco)

    # ---- Forex (Yahoo Finance) ----
    if "forex" in categories:
        for name, yf_sym in list(FOREX_TICKERS.items())[:YF_REQUEST_LIMIT]:
            df = yf_download_1h(yf_sym)
            if df is None or df.empty:
                continue
            total_analyzed += 1
            compute_indicators(df)
            last = df.iloc[-1]
            reco = build_reco(name, "Forex", last, req.capital)
            if reco:
                recommendations.append(reco)

    # ---- Commodities (Yahoo Finance) ----
    if "commodities" in categories:
        for name, yf_sym in list(COMMODITY_TICKERS.items())[:YF_REQUEST_LIMIT]:
            df = yf_download_1h(yf_sym)
            if df is None or df.empty:
                continue
            total_analyzed += 1
            compute_indicators(df)
            last = df.iloc[-1]
            reco = build_reco(name, "Commodities", last, req.capital)
            if reco:
                recommendations.append(reco)

    # sort + trim
    recommendations.sort(key=lambda r: r["score"], reverse=True)
    top = recommendations[: max(1, req.top_n)]

    return {
        "success": True,
        "recommendations": top,
        "signals_found": len(recommendations),
        "total_analyzed": total_analyzed,
        "analysis_time": pd.Timestamp.utcnow().isoformat(),
        "elapsed_sec": round(time.time() - start_ts, 3),
    }


# ---------- Local run ----------

if __name__ == "__main__":
    # Run: python main.py
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")), reload=True)
