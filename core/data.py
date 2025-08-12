# core/data.py
import pandas as pd
import yfinance as yf
from typing import List, Tuple

def read_tickers_from_csv(file) -> List[str]:
    df = pd.read_csv(file, header=None)
    syms = [str(x).strip() for x in df.iloc[:, 0].tolist() if isinstance(x, (str, int, float))]
    syms = [s for s in syms if s and s.endswith(".IS")]
    return sorted(list(dict.fromkeys(syms)))

def _download_raw(tickers: List[str], start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not tickers:
        return (pd.DataFrame(),)*4
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    def pick(name):
        s = data[name] if name in data else pd.DataFrame()
        if isinstance(s, pd.Series):
            s = s.to_frame(tickers[0])
        return s
    close = pick("Close"); high = pick("High"); low = pick("Low"); vol = pick("Volume")
    for df in (close, high, low, vol):
        df.sort_index(inplace=True); df.ffill(inplace=True); df.dropna(axis=1, how="all", inplace=True)
    return close, high, low, vol

def fetch_ohlcv(tickers: List[str], start: str, end: str) -> dict:
    close, high, low, vol = _download_raw(tickers, start, end)
    return {"close": close, "high": high, "low": low, "volume": vol}

def tl_turnover(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    common = [c for c in close.columns if c in volume.columns]
    return close[common] * volume[common]  # ~ TL ciro (Adj Close * Volume)

def liquidity_filter(turnover: pd.DataFrame, lookback: int, tl_threshold: float) -> pd.Series:
    if turnover.empty:
        return pd.Series(dtype=bool)
    med = turnover.tail(lookback).median(axis=0, numeric_only=True)
    return med >= tl_threshold

