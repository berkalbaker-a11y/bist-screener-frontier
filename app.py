# app.py — Sprint-2: Screening + Rule Builder + Frontier & Backtest
import json
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.portfolio import mu_cov, max_sharpe, min_var, efficient_frontier, backtest_walk_forward, max_sharpe_k, backtest_walk_forward_k


from core.data import read_tickers_from_csv, fetch_ohlcv, tl_turnover, liquidity_filter
from core.indicators import compute_all_indicators
from core.rules import preset_json, eval_ruleset
from core.portfolio import mu_cov, max_sharpe, min_var, efficient_frontier, backtest_walk_forward

st.set_page_config(page_title="BIST Screener • Sprint-2", layout="wide")
st.title("📈 BIST Screener — Screening + Rule Builder + Frontier")

# -------- Sidebar: Universe & Data --------
st.sidebar.header("Evren & Veri")
preset_file = st.sidebar.file_uploader("Evren (CSV, 1 sütun – EREGL.IS gibi)", type=["csv"])
if preset_file is not None:
    tickers = read_tickers_from_csv(preset_file)
else:
    try:
        with open("presets/bist30.csv","r",encoding="utf-8") as f:
            tickers = [x.strip() for x in f if x.strip()]
    except Exception:
        tickers = []

tickers_text = st.sidebar.text_area("Semboller (virgülle)", value=",".join(tickers), height=80)
tickers = [s.strip() for s in tickers_text.split(",") if s.strip()]

c1,c2 = st.sidebar.columns(2)
start = c1.date_input("Başlangıç", value=pd.to_datetime("2019-01-01").date())
end   = c2.date_input("Bitiş", value=date.today())

st.sidebar.subheader("Likidite Filtresi (TL ciro)")
liq_lookback = st.sidebar.number_input("Lookback (gün)", 20, 504, 126, step=7)
liq_thresh   = st.sidebar.number_input("Medyan TL ciro ≥", 0.0, 1e10, 5_000_000.0, step=500_000.0)

# -------- Sidebar: Indicator Params --------
with st.sidebar.expander("İndikatör Parametreleri", expanded=False):
    rsi_len   = st.number_input("RSI len", 5, 50, 14)
    macd_fast = st.number_input("MACD fast", 2, 50, 12)
    macd_slow = st.number_input("MACD slow", 5, 100, 26)
    macd_sig  = st.number_input("MACD signal", 2, 30, 9)
    stoch_k   = st.number_input("Stoch %K", 5, 50, 14)
    stoch_sm  = st.number_input("Stoch smooth K", 1, 20, 3)
    adx_len   = st.number_input("ADX len", 5, 50, 14)
    mfi_len   = st.number_input("MFI len", 5, 50, 14)
    sma_fast  = st.number_input("SMA fast", 5, 250, 50)
    sma_slow  = st.number_input("SMA slow", 10, 400, 200)
    atr_len   = st.number_input("ATR len", 5, 50, 14)

ind_params = dict(rsi_len=rsi_len, macd_fast=macd_fast, macd_slow=macd_slow, macd_sig=macd_sig,
                  stoch_k=stoch_k, stoch_sm=stoch_sm, adx_len=adx_len, mfi_len=mfi_len,
                  sma_fast=sma_fast, sma_slow=sma_slow, atr_len=atr_len)

st.divider()

# -------- Rule Builder (JSON) --------
st.subheader("🧩 Rule Builder (JSON ile özelleştir)")
preset_choice = st.selectbox("Preset seç", ["short","mid","long"], index=1)
rules_json_text = st.text_area("Rules JSON", value=preset_json(preset_choice), height=380)
u = st.file_uploader("JSON yükle", type=["json"])
if u is not None: rules_json_text = u.read().decode("utf-8")

try:
    RULESET = json.loads(rules_json_text); rules_ok=True
except Exception as e:
    rules_ok=False; st.error(f"Rules JSON hatalı: {e}")

run = st.button("🚀 Taramayı Başlat", type="primary")

# -------- Main --------
if run and rules_ok:
    if not tickers:
        st.warning("Lütfen en az 1 sembol girin."); st.stop()

    with st.spinner("Veri indiriliyor..."):
        ohlcv = fetch_ohlcv(tickers, start=str(start), end=str(end))
        close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]

    if close.empty:
        st.error("Veri alınamadı. Tarih/evreni kontrol edin."); st.stop()

    # Likidite filtresi
    tlt = tl_turnover(close, volume)
    liq_pass = liquidity_filter(tlt, int(liq_lookback), float(liq_thresh))
    liquid_universe = [c for c, ok in liq_pass.items() if bool(ok)]
    st.success(f"Likidite filtresini geçen: {len(liquid_universe)} / {len(tickers)}")
    st.dataframe(pd.DataFrame({"Medyan TL Ciro ≥ Eşik": liq_pass}).T)

    if not liquid_universe:
        st.warning("Likidite filtresini geçen sembol yok. Eşiği düşürün."); st.stop()

    # İndikatörler + Kurallar
    rows=[]
    with st.spinner("İndikatörler hesaplanıyor..."):
        for sym in liquid_universe:
            met = compute_all_indicators(close[sym], high[sym], low[sym], volume[sym], ind_params)
            if not met: continue
            rs = eval_ruleset(RULESET, met)
            rows.append({
                "ticker": sym, "pass": rs["pass"], "score": rs["score"], "pass_count": rs["pass_count"],
                "close": met.get("close"), "rsi": met.get("rsi"), "adx": met.get("adx"),
                "di_bull": met.get("di_bias_bull"), "macd_hist": met.get("macd_hist"),
                "stoch_k": met.get("stoch_k"), "stoch_d": met.get("stoch_d"), "mfi": met.get("mfi"),
                "sma_fast": met.get("sma_fast"), "sma_slow": met.get("sma_slow"),
                "near_52w": met.get("near_52w_high"), "atr_pct": met.get("atr_pct"),
                "ret_3m": met.get("ret_3m"), "ret_6m": met.get("ret_6m"), "ret_12m": met.get("ret_12m"),
            })

    if not rows:
        st.warning("Hesaplanabilir veri çıkmadı."); st.stop()

    df = pd.DataFrame(rows).sort_values(["pass","score","ret_3m"], ascending=[False,False,False])
    st.subheader("📋 Screening Sonuçları")
    st.dataframe(df, use_container_width=True)
    st.download_button("CSV indir (sonuçlar)", df.to_csv(index=False).encode("utf-8"),
                       file_name="screening_results.csv", mime="text/csv")

    st.divider()
    st.subheader("🎯 Seçim")
    topn = st.slider("Top-N (skora göre)", 1, min(20, len(df)), min(10, len(df)))
    picked = st.multiselect("Portföye aday seç (manuel düzenlenebilir)",
                            options=df["ticker"].tolist(),
                            default=df.head(topn)["ticker"].tolist())
    st.write("Seçilenler:", picked)

    # -------- Portfolio & Backtest --------
    st.sidebar.header("Portföy Parametreleri")
    pf_lookback = st.sidebar.slider("μ/Σ Lookback (gün)", 60, 756, 252, step=21)
    rf = st.sidebar.number_input("Risksiz faiz (yıllık, %)", value=0.0, step=0.5) / 100.0
    cap = st.sidebar.slider("Tek hisse tavanı (%)", 5, 50, 15) / 100.0
    rebalance = st.sidebar.selectbox("Rebalans", ["W","M","Q"], index=1)
    fee_bps = st.sidebar.number_input("Komisyon (bps)", value=2.0, step=1.0, help="1 bps = %0.01")
    slip_bps = st.sidebar.number_input("Slippage (bps)", value=5.0, step=1.0)

    if picked:
        sub_close = close[picked].dropna(how="all").ffill()
        if len(sub_close.columns) >= 2 and len(sub_close) > pf_lookback + 20:
            st.subheader("🟩 Efficient Frontier")
            try:
                mu, cov = mu_cov(sub_close, pf_lookback)
                w_ms = max_sharpe(mu, cov, rf=rf, cap=cap)
                w_mv = min_var(cov, cap=cap)

                fr = efficient_frontier(mu, cov, points=25, cap=cap)
                fig = go.Figure()
                if not fr.empty:
                    fig.add_trace(go.Scatter(x=fr["vol"], y=fr["return"], mode="lines", name="Frontier"))
                ms_v = float(np.sqrt(np.maximum(w_ms.values @ cov.values @ w_ms.values, 1e-12)))
                ms_r = float(mu @ w_ms)
                mv_v = float(np.sqrt(np.maximum(w_mv.values @ cov.values @ w_mv.values, 1e-12)))
                mv_r = float(mu @ w_mv)
                fig.add_trace(go.Scatter(x=[ms_v], y=[ms_r], mode="markers", name="Max-Sharpe", marker=dict(size=10)))
                fig.add_trace(go.Scatter(x=[mv_v], y=[mv_r], mode="markers", name="Min-Var", marker=dict(size=10)))
                fig.update_layout(xaxis_title="Volatilite (yıllık)", yaxis_title="Getiri (yıllık)", height=420)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1: st.write("**Max-Sharpe Ağırlıklar**"); st.dataframe(w_ms.sort_values(ascending=False).to_frame("weight"))
                with c2: st.write("**Min-Var Ağırlıklar**");  st.dataframe(w_mv.sort_values(ascending=False).to_frame("weight"))

                st.subheader("📈 Walk-Forward Backtest (Max-Sharpe)")
                bt = backtest_walk_forward(sub_close, rf_annual=rf, lookback=pf_lookback,
                                           rebalance=rebalance, cap=cap, fee_bps=fee_bps, slippage_bps=slip_bps)
                eq = bt["equity"]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity"))
                fig2.update_layout(height=400, xaxis_title="Tarih", yaxis_title="Portföy Değeri")
                st.plotly_chart(fig2, use_container_width=True)
                st.write("**Metrikler**"); st.json(bt["metrics"])
            except Exception as e:
                st.error(f"Portföy/Backtest hatası: {e}")
        else:
            st.warning("Portföy için en az 2 sembol ve yeterli tarih penceresi gerekli.")
