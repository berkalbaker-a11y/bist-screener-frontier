# core/indicators.py
from typing import Dict
import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator, StochasticOscillator, MFIIndicator
from ta.trend import ADXIndicator, SMAIndicator, MACD
from ta.volatility import AverageTrueRange

def _safe_last(series: pd.Series, default=np.nan):
    try:
        return float(pd.Series(series).dropna().iloc[-1])
    except Exception:
        return default

def _safe_prev(series: pd.Series, default=np.nan):
    try:
        return float(pd.Series(series).dropna().iloc[-2])
    except Exception:
        return default

def compute_all_indicators(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series,
    params: Dict
) -> Dict:
    df = pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume}).dropna()
    if len(df) < 60:
        return {}

    rsi_len   = params.get("rsi_len", 14)
    macd_fast = params.get("macd_fast", 12)
    macd_slow = params.get("macd_slow", 26)
    macd_sig  = params.get("macd_sig", 9)
    stoch_k   = params.get("stoch_k", 14)
    stoch_sm  = params.get("stoch_sm", 3)
    adx_len   = params.get("adx_len", 14)
    mfi_len   = params.get("mfi_len", 14)
    sma_fast  = params.get("sma_fast", 50)
    sma_slow  = params.get("sma_slow", 200)
    atr_len   = params.get("atr_len", 14)

    # Momentum / Trend
    rsi = RSIIndicator(df["close"], window=rsi_len).rsi()

    macd_ind = MACD(df["close"], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_sig)
    macd_hist = macd_ind.macd_diff()

    stoch_ind = StochasticOscillator(df["high"], df["low"], df["close"], window=stoch_k, smooth_window=stoch_sm)
    stoch_k_series = stoch_ind.stoch()
    stoch_d_series = stoch_ind.stoch_signal()

    adx_ind = ADXIndicator(df["high"], df["low"], df["close"], window=adx_len)
    adx = adx_ind.adx()
    di_plus = adx_ind.adx_pos()
    di_minus = adx_ind.adx_neg()

    mfi = MFIIndicator(df["high"], df["low"], df["close"], df["volume"].fillna(0), window=mfi_len).money_flow_index()
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=atr_len).average_true_range()

    sma_f = SMAIndicator(df["close"], window=sma_fast).sma_indicator()
    sma_s = SMAIndicator(df["close"], window=sma_slow).sma_indicator()

    high_52w = df["close"].rolling(252).max()
    low_52w  = df["close"].rolling(252).min()

    ret_3m = df["close"].pct_change(63)
    ret_6m = df["close"].pct_change(126)
    ret_12m= df["close"].pct_change(252)

    last = {
        "close": _safe_last(df["close"]),
        "rsi": _safe_last(rsi),
        "macd_hist": _safe_last(macd_hist),
        "stoch_k": _safe_last(stoch_k_series),
        "stoch_d": _safe_last(stoch_d_series),
        "stoch_k_prev": _safe_prev(stoch_k_series),
        "stoch_d_prev": _safe_prev(stoch_d_series),
        "di_plus": _safe_last(di_plus),
        "di_minus": _safe_last(di_minus),
        "adx": _safe_last(adx),
        "mfi": _safe_last(mfi),
        "sma_fast": _safe_last(sma_f),
        "sma_slow": _safe_last(sma_s),
        "high_52w": _safe_last(high_52w),
        "low_52w": _safe_last(low_52w),
        "atr": _safe_last(atr),
        "ret_3m": _safe_last(ret_3m),
        "ret_6m": _safe_last(ret_6m),
        "ret_12m": _safe_last(ret_12m),
    }

    last["near_52w_high"] = last["close"] / last["high_52w"] if last["high_52w"] else np.nan
    last["atr_pct"] = (last["atr"] / last["close"]) if last["close"] else np.nan
    last["di_bias_bull"] = (last["di_plus"] > last["di_minus"]) if not np.isnan(last["di_plus"]) else False
    last["stoch_cross_up"] = (last["stoch_k_prev"] <= last["stoch_d_prev"]) and (last["stoch_k"] > last["stoch_d"])
    last["stoch_cross_down"] = (last["stoch_k_prev"] >= last["stoch_d_prev"]) and (last["stoch_k"] < last["stoch_d"])

    return last

