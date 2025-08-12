# core/portfolio.py
from __future__ import annotations
import numpy as np
import pandas as pd
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

# ---------- Yardımcılar ----------
def _to_np(x: pd.Series | pd.DataFrame) -> np.ndarray:
    return np.asarray(x, dtype=float)

def _annualize(mu_daily: np.ndarray, cov_daily: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return mu_daily * TRADING_DAYS, cov_daily * TRADING_DAYS

def _safe_cov(rets: pd.DataFrame) -> np.ndarray:
    X = rets.dropna().values
    if X.shape[0] < 2:
        return np.eye(X.shape[1])
    if _HAS_SK:
        try:
            lw = LedoitWolf().fit(X)
            return lw.covariance_
        except Exception:
            pass
    # fallback: sample covariance
    return np.cov(X, rowvar=False)

def _make_bounds(n: int, cap: float) -> List[Tuple[float, float]]:
    cap = float(cap)
    return [(0.0, cap if cap > 0 else 1.0) for _ in range(n)]

def _project_simplex(w: np.ndarray) -> np.ndarray:
    # L1-simplex projeksiyonu (sum=1, w>=0)
    w = np.maximum(w, 0.0)
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(w) / len(w)

# ---------- μ/Σ ----------
def mu_cov(prices: pd.DataFrame, lookback: int) -> Tuple[pd.Series, pd.DataFrame]:
    px = prices.dropna(how="all").ffill().tail(lookback)
    rets = px.pct_change().dropna()
    if rets.empty:
        raise ValueError("Lookback penceresi için yeterli veri yok.")
    mu_d = rets.mean()
    cov_d = pd.DataFrame(_safe_cov(rets), index=rets.columns, columns=rets.columns)
    mu, cov = _annualize(mu_d.values, cov_d.values)
    return pd.Series(mu, index=rets.columns), pd.DataFrame(cov, index=rets.columns, columns=rets.columns)

# ---------- Optimizasyonlar (SLSQP) ----------
def _risk(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.maximum(w @ cov @ w, EPS))

def _ret(w: np.ndarray, mu: np.ndarray) -> float:
    return float(mu @ w)

def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0, cap: float = 0.15) -> pd.Series:
    tickers = list(mu.index)
    n = len(tickers)
    mu_v = _to_np(mu)
    cov_m = _to_np(cov)
    # objective: -Sharpe
    def obj(w):
        r = _ret(w, mu_v) - rf
        v = sqrt(_risk(w, cov_m))
        return -r / (v + EPS)
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = _make_bounds(n, cap)
    w0 = np.ones(n) / n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    w = _project_simplex(res.x if res.success else w0)
    return pd.Series(w, index=tickers)

def min_var(cov: pd.DataFrame, cap: float = 0.15) -> pd.Series:
    tickers = list(cov.index)
    n = len(tickers)
    cov_m = _to_np(cov)
    def obj(w): return _risk(w, cov_m)
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = _make_bounds(n, cap)
    w0 = np.ones(n) / n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    w = _project_simplex(res.x if res.success else w0)
    return pd.Series(w, index=tickers)

def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, points: int = 20, cap: float = 0.15) -> pd.DataFrame:
    mu_v = _to_np(mu); cov_m = _to_np(cov); n = len(mu_v)
    t_min, t_max = float(mu_v.min()), float(mu_v.max())
    targets = np.linspace(t_min, t_max, points)
    bounds = _make_bounds(n, cap)
    cons_base = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]
    rows = []
    for t in targets:
        def obj(w): return _risk(w, cov_m)
        cons = cons_base + [{"type":"eq", "fun": lambda w, t=t: _ret(w, mu_v) - t}]
        w0 = np.ones(n)/n
        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
        if not res.success:
            continue
        w = _project_simplex(res.x)
        r = _ret(w, mu_v); v = sqrt(_risk(w, cov_m))
        rows.append({"return": r, "vol": v})
    return pd.DataFrame(rows).sort_values("vol")

# ---------- Backtest (walk-forward, rebalans) ----------
def backtest_walk_forward(prices: pd.DataFrame,
                          rf_annual: float = 0.0,
                          lookback: int = 252,
                          rebalance: str = "M",      # 'W','M','Q'
                          cap: float = 0.15,
                          fee_bps: float = 2.0,      # binde 2 = 20 bps değil! burada bps = yüzde puanın 1/100'ü; 2.0 bps = %0.02
                          slippage_bps: float = 5.0  # 5 bps = %0.05
                          ) -> Dict:
    px = prices.dropna(how="all").ffill()
    daily_ret = px.pct_change().fillna(0.0)
    # rebalans tarihleri: dönem sonları, lookback sonrası
    resample_rule = {"W":"W-FRI","M":"M","Q":"Q"}[rebalance]
    rebal_dates = px.index[lookback:]
    rebal_dates = pd.Index(sorted(set(pd.Series(rebal_dates).resample(resample_rule).last().dropna().tolist())))
    if len(rebal_dates) < 2:
        raise ValueError("Backtest için yeterli rebalans tarihi yok.")
    equity = [1.0]
    dates = []
    weights_prev = None
    turnover_sum = 0.0
    fee = (fee_bps + slippage_bps) / 10000.0  # toplam maliyet oranı

    start_idx = px.index.get_loc(rebal_dates[0])
    all_dates = px.index[start_idx:]
    rf_daily = rf_annual / TRADING_DAYS

    w_records = []

    for i, d0 in enumerate(rebal_dates[:-1]):
        d1 = rebal_dates[i+1]
        hist = px.loc[:d0].tail(lookback)
        mu, cov = mu_cov(hist, lookback)
        W = max_sharpe(mu, cov, rf=rf_annual, cap=cap).reindex(px.columns).fillna(0.0).values
        # turnover & maliyet
        if weights_prev is None:
            tw = np.sum(np.abs(W))  # ilk kurulum için "yatırım" saymayız; maliyet uygulamayalım
            weights_prev = W.copy()
        else:
            tw = float(np.sum(np.abs(W - weights_prev)))
            turnover_sum += tw
            equity[-1] *= (1 - fee * tw)  # maliyeti bir defa uygula
            weights_prev = W.copy()

        # dönem içi getiriler
        seg = daily_ret.loc[(daily_ret.index > d0) & (daily_ret.index <= d1)]
        if seg.empty: 
            continue
        # portföy günlük getirisi
        port_r = (seg.values @ W).astype(float)
        # rf günlük ekleme (opsiyonel)
        port_r = port_r + rf_daily * 0.0
        for r in port_r:
            equity.append(equity[-1] * (1.0 + r))
            dates.append(seg.index[0] if len(dates)==0 else seg.index[min(len(dates), len(seg.index)-1)])
        w_records.append((d0, {c: float(W[j]) for j, c in enumerate(px.columns)}))

    eq = pd.Series(equity, index=[px.index[start_idx-1]] + list(seg.index if len(seg)>0 else px.index[start_idx:][:len(equity)-1]))
    ret = eq.pct_change().dropna()
    cagr = eq.iloc[-1]**(TRADING_DAYS/len(ret)) - 1 if len(ret)>0 else np.nan
    vol = ret.std()*sqrt(TRADING_DAYS) if len(ret)>0 else np.nan
    sharpe = ((ret.mean()*TRADING_DAYS) - rf_annual) / (vol + EPS) if vol==vol else np.nan
    roll_max = eq.cummax()
    dd = (eq/roll_max - 1.0).min() if len(eq)>0 else np.nan
    calmar = (cagr / abs(dd)) if (dd is not None and dd<0) else np.nan
    avg_turnover = turnover_sum / max(1, len(rebal_dates)-1)

    return {
        "equity": eq,
        "metrics": {
            "CAGR": float(cagr),
            "Vol": float(vol),
            "Sharpe": float(sharpe),
            "MaxDD": float(dd),
            "Calmar": float(calmar),
            "AvgTurnover": float(avg_turnover)
        },
        "weights": w_records
    }
