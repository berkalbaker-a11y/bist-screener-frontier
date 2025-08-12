# core/portfolio.py
from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, List, Tuple
from math import sqrt
try:
    from sklearn.covariance import LedoitWolf
    _HAS_SK = True
except Exception:
    _HAS_SK = False
from scipy.optimize import minimize

EPS = 1e-12
TRADING_DAYS = 252

def _to_np(x): return np.asarray(x, dtype=float)
def _annualize(mu_d, cov_d): return mu_d * TRADING_DAYS, cov_d * TRADING_DAYS

def _safe_cov(rets: pd.DataFrame) -> np.ndarray:
    X = rets.dropna().values
    if X.shape[0] < 2: return np.eye(X.shape[1])
    if _HAS_SK:
        try: return LedoitWolf().fit(X).covariance_
        except Exception: pass
    return np.cov(X, rowvar=False)

def _make_bounds(n: int, cap: float) -> List[Tuple[float, float]]:
    cap = float(cap); return [(0.0, cap if cap > 0 else 1.0) for _ in range(n)]

def _project_simplex(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 0.0); s = w.sum()
    return (w / s) if s > 0 else np.ones_like(w)/len(w)

def mu_cov(prices: pd.DataFrame, lookback: int):
    px = prices.dropna(how="all").ffill().tail(lookback)
    rets = px.pct_change().dropna()
    if rets.empty: raise ValueError("Lookback için yeterli veri yok.")
    mu_d = rets.mean(); cov_d = pd.DataFrame(_safe_cov(rets), index=rets.columns, columns=rets.columns)
    mu, cov = _annualize(mu_d.values, cov_d.values)
    return pd.Series(mu, index=rets.columns), pd.DataFrame(cov, index=rets.columns, columns=rets.columns)

def _risk(w, cov): return float(np.maximum(w @ cov @ w, EPS))
def _ret(w, mu):   return float(mu @ w)

def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float=0.0, cap: float=0.15):
    n = len(mu); mu_v = _to_np(mu); cov_m = _to_np(cov)
    def obj(w): return -((_ret(w, mu_v) - rf) / (sqrt(_risk(w, cov_m)) + EPS))
    cons = [{"type":"eq","fun":lambda w: np.sum(w)-1.0}]
    bounds = _make_bounds(n, cap); w0 = np.ones(n)/n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter":500})
    w = _project_simplex(res.x if res.success else w0)
    return pd.Series(w, index=mu.index)

def min_var(cov: pd.DataFrame, cap: float=0.15):
    n = len(cov); cov_m = _to_np(cov)
    def obj(w): return _risk(w, cov_m)
    cons = [{"type":"eq","fun":lambda w: np.sum(w)-1.0}]
    bounds = _make_bounds(n, cap); w0 = np.ones(n)/n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter":500})
    w = _project_simplex(res.x if res.success else w0)
    return pd.Series(w, index=cov.index)

def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, points: int=20, cap: float=0.15):
    mu_v = _to_np(mu); cov_m = _to_np(cov); n = len(mu_v)
    t_min, t_max = float(mu_v.min()), float(mu_v.max())
    targets = np.linspace(t_min, t_max, points)
    bounds = _make_bounds(n, cap); cons_base=[{"type":"eq","fun":lambda w: np.sum(w)-1.0}]
    rows=[]
    for t in targets:
        def obj(w): return _risk(w, cov_m)
        cons = cons_base + [{"type":"eq","fun":lambda w, t=t: _ret(w, mu_v)-t}]
        w0 = np.ones(n)/n
        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter":500})
        if not res.success: continue
        w = _project_simplex(res.x)
        rows.append({"return": _ret(w, mu_v), "vol": sqrt(_risk(w, cov_m))})
    return pd.DataFrame(rows).sort_values("vol")

def backtest_walk_forward(prices: pd.DataFrame, rf_annual: float=0.0, lookback: int=252,
                          rebalance: str="M", cap: float=0.15, fee_bps: float=2.0, slippage_bps: float=5.0):
    px = prices.dropna(how="all").ffill()
    daily = px.pct_change().fillna(0.0)
    rule = {"W":"W-FRI","M":"M","Q":"Q"}[rebalance]
    dates = pd.Series(px.index[lookback:]).resample(rule).last().dropna()
    rebal_dates = list(dates)
    if len(rebal_dates) < 2: raise ValueError("Backtest için yeterli rebalans tarihi yok.")

    eq = []; eq_dates=[]
    w_prev=None; turnover=0.0; cost_rate=(fee_bps+slippage_bps)/10000.0
    rf_daily = rf_annual/TRADING_DAYS

    start_idx = px.index.get_loc(rebal_dates[0])
    eq_val=1.0; eq.append(eq_val); eq_dates.append(px.index[start_idx-1])

    for i in range(len(rebal_dates)-1):
        d0, d1 = rebal_dates[i], rebal_dates[i+1]
        hist = px.loc[:d0].tail(lookback)
        mu, cov = mu_cov(hist, lookback)
        w = max_sharpe(mu, cov, rf=rf_annual, cap=cap).reindex(px.columns).fillna(0.0).values
        if w_prev is not None:
            tw = float(np.sum(np.abs(w - w_prev))); turnover += tw
            eq_val *= (1 - cost_rate*tw)
        w_prev = w.copy()

        seg = daily.loc[(daily.index > d0) & (daily.index <= d1)]
        for r in seg.values @ w:
            eq_val *= (1 + r + 0.0*rf_daily)
            eq.append(eq_val)
        eq_dates += list(seg.index)

    ser = pd.Series(eq, index=eq_dates)
    ret = ser.pct_change().dropna()
    cagr = ser.iloc[-1]**(TRADING_DAYS/len(ret)) - 1 if len(ret)>0 else np.nan
    vol = ret.std()*sqrt(TRADING_DAYS) if len(ret)>0 else np.nan
    sharpe = ((ret.mean()*TRADING_DAYS) - rf_annual) / (vol + EPS) if vol==vol else np.nan
    roll_max = ser.cummax(); mdd = (ser/roll_max - 1.0).min() if len(ser)>0 else np.nan
    calmar = (cagr/abs(mdd)) if (mdd is not None and mdd<0) else np.nan
    avg_turn = turnover / max(1, len(rebal_dates)-1)

    return {"equity": ser, "metrics": {"CAGR": float(cagr), "Vol": float(vol),
            "Sharpe": float(sharpe), "MaxDD": float(mdd), "Calmar": float(calmar),
            "AvgTurnover": float(avg_turn)}}
# ---- Kardinalite kontrollü (K adet) portföy ----
def _sharpe_of(w, mu_v, cov_m, rf):
    r = float(mu_v @ w) - rf
    v = float(np.sqrt(max(w @ cov_m @ w, EPS)))
    return r / (v + EPS)

def max_sharpe_k(mu: pd.Series, cov: pd.DataFrame, k: int, rf: float=0.0, cap: float=0.15) -> pd.Series:
    """Greedy forward selection: tam K hisse seç, sonra bu altkümede Max-Sharpe çöz."""
    tickers = list(mu.index)
    n = len(tickers)
    k = max(1, min(k, n))
    if k == n:  # zaten hepsi
        return max_sharpe(mu, cov, rf=rf, cap=cap)

    mu_v = np.asarray(mu.values, float)
    cov_m = np.asarray(cov.values, float)
    selected = []
    remaining = list(range(n))

    # 1) İlk hisse: tek başına en yüksek tekil Sharpe (cap sınırı yok sayılmaz; yine de sonradan altkümede çözeceğiz)
    best_j = None; best_score = -1e9
    for j in remaining:
        w = np.zeros(n); w[j] = 1.0
        s = _sharpe_of(w, mu_v, cov_m, rf)
        if s > best_score:
            best_score, best_j = s, j
    selected.append(best_j); remaining.remove(best_j)

    # 2) Greedy ekleme
    while len(selected) < k:
        best_j = None; best_score = -1e9
        for j in remaining:
            idx = selected + [j]
            mu_sub = pd.Series(mu_v[idx], index=[tickers[t] for t in idx])
            cov_sub = pd.DataFrame(cov_m[np.ix_(idx, idx)], index=mu_sub.index, columns=mu_sub.index)
            w_sub = max_sharpe(mu_sub, cov_sub, rf=rf, cap=cap)
            s = _sharpe_of(w_sub.values, mu_sub.values, cov_sub.values, rf)
            if s > best_score:
                best_score, best_j = s, j
        selected.append(best_j); remaining.remove(best_j)

    # 3) Final çözüm: seçilen altkümede tekrar Max-Sharpe
    idx = selected
    mu_sub = pd.Series(mu_v[idx], index=[tickers[t] for t in idx])
    cov_sub = pd.DataFrame(cov_m[np.ix_(idx, idx)], index=mu_sub.index, columns=mu_sub.index)
    w_sub = max_sharpe(mu_sub, cov_sub, rf=rf, cap=cap)

    # 4) Full vektöre yerleştir
    w_full = pd.Series(0.0, index=tickers)
    w_full.loc[w_sub.index] = w_sub.values
    return w_full

# Backtest'i K ile kullanmak için küçük bir sargı
def backtest_walk_forward_k(prices: pd.DataFrame, k: int, rf_annual: float=0.0,
                            lookback: int=252, rebalance: str="M", cap: float=0.15,
                            fee_bps: float=2.0, slippage_bps: float=5.0):
    px = prices.dropna(how="all").ffill()
    daily = px.pct_change().fillna(0.0)
    rule = {"W":"W-FRI","M":"M","Q":"Q"}[rebalance]
    dates = pd.Series(px.index[lookback:]).resample(rule).last().dropna()
    rebal_dates = list(dates)
    if len(rebal_dates) < 2: raise ValueError("Backtest için yeterli rebalans tarihi yok.")

    eq = []; eq_dates=[]; w_prev=None; turnover=0.0
    cost_rate = (fee_bps+slippage_bps)/10000.0; rf_daily = rf_annual/252.0

    start_idx = px.index.get_loc(rebal_dates[0])
    eq_val=1.0; eq.append(eq_val); eq_dates.append(px.index[start_idx-1])

    for i in range(len(rebal_dates)-1):
        d0, d1 = rebal_dates[i], rebal_dates[i+1]
        hist = px.loc[:d0].tail(lookback)
        mu, cov = mu_cov(hist, lookback)
        w_series = max_sharpe_k(mu, cov, k=k, rf=rf_annual, cap=cap).reindex(px.columns).fillna(0.0)
        w = w_series.values
        if w_prev is not None:
            tw = float(np.sum(np.abs(w - w_prev))); turnover += tw
            eq_val *= (1 - cost_rate*tw)
        w_prev = w.copy()

        seg = daily.loc[(daily.index > d0) & (daily.index <= d1)]
        for r in seg.values @ w:
            eq_val *= (1 + r + 0.0*rf_daily)
            eq.append(eq_val); eq_dates.append(seg.index[seg.index.get_loc(seg.index[0]) + (len(eq_dates) - (len(eq)-1))] if len(seg)>0 else d1)

    ser = pd.Series(eq, index=eq_dates)
    ret = ser.pct_change().dropna()
    cagr = ser.iloc[-1]**(252/len(ret)) - 1 if len(ret)>0 else np.nan
    vol = ret.std()*np.sqrt(252) if len(ret)>0 else np.nan
    sharpe = ((ret.mean()*252) - rf_annual) / (vol + EPS) if vol==vol else np.nan
    mdd = (ser/ser.cummax() - 1.0).min() if len(ser)>0 else np.nan
    calmar = (cagr/abs(mdd)) if (mdd is not None and mdd<0) else np.nan
    avg_turn = turnover / max(1, len(rebal_dates)-1)
    return {"equity": ser, "metrics": {"CAGR": float(cagr), "Vol": float(vol),
            "Sharpe": float(sharpe), "MaxDD": float(mdd), "Calmar": float(calmar),
            "AvgTurnover": float(avg_turn)}}
