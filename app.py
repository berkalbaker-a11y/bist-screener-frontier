# app.py — Sprint-1: Screening + Rule Builder (ta kütüphanesiyle)
import json, io
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np

from core.data import read_tickers_from_csv, fetch_ohlcv, tl_turnover, liquidity_filter
from core.indicators import compute_all_indicators
from core.rules import DEFAULT_PRESETS, preset_json, eval_ruleset

st.set_page_config(page_title="BIST Screener • Sprint-1", layout="wide")
st.title("📈 BIST Screener — Sprint-1 (Tarama + Rule Builder)")

# ---------------- Sidebar: Universe & Data ----------------
st.sidebar.header("Evren & Veri")
preset_file = st.sidebar.file_uploader("Evren (CSV, 1 sütun – EREGL.IS gibi)", type=["csv"])
if preset_file is not None:
    tickers = read_tickers_from_csv(preset_file)
else:
    try:
        with open("presets/bist30.csv", "r", encoding="utf-8") as f:
            tickers = [x.strip() for x in f if x.strip()]
    except Exception:
        tickers = []

tickers_text = st.sidebar.text_area("Semboller (virgülle)", value=",".join(tickers), height=80)
tickers = [s.strip() for s in tickers_text.split(",") if s.strip()]

colA, colB = st.sidebar.columns(2)
start = colA.date_input("Başlangıç", value=pd.to_datetime("2019-01-01").date())
end   = colB.date_input("Bitiş", value=date.today())

st.sidebar.subheader("Likidite Filtresi (TL ciro)")
liq_lookback = st.sidebar.number_input("Lookback (gün)", min_value=20, max_value=504, value=126, step=7)
liq_thresh   = st.sidebar.number_input("Medyan TL ciro ≥", min_value=0.0, value=5_000_000.0, step=500_000.0)

# ---------------- Sidebar: Indicator Params --------------
with st.sidebar.expander("İndikatör Parametreleri", expanded=False):
    rsi_len   = st.number_input("RSI len", 5, 50, 14)
    macd_fast = st.number_input("MACD fast", 2, 50, 12)
    macd_slow = st.number_input("MACD slow", 5, 100, 26)
    macd_sig  = st.number_input("MACD signal", 2, 30, 9)
    stoch_k   = st.number_input("Stoch %K", 5, 50, 14)
    stoch_sm  = st.number_input("Stoch smooth K (signal)", 1, 20, 3)
    adx_len   = st.number_input("ADX len", 5, 50, 14)
    mfi_len   = st.number_input("MFI len", 5, 50, 14)
    sma_fast  = st.number_input("SMA fast", 5, 250, 50)
    sma_slow  = st.number_input("SMA slow", 10, 400, 200)
    atr_len   = st.number_input("ATR len", 5, 50, 14)

ind_params = dict(
    rsi_len=rsi_len, macd_fast=macd_fast, macd_slow=macd_slow, macd_sig=macd_sig,
    stoch_k=stoch_k, stoch_sm=stoch_sm, adx_len=adx_len, mfi_len=mfi_len,
    sma_fast=sma_fast, sma_slow=sma_slow, atr_len=atr_len
)

st.divider()

# ---------------- Rule Builder (JSON) ----------------
st.subheader("🧩 Rule Builder (JSON ile özelleştir)")
preset_choice = st.selectbox("Preset seç", ["short", "mid", "long"], index=1)
default_rules_json = preset_json(preset_choice)

col1, col2 = st.columns([3,1])
with col1:
    rules_json_text = st.text_area("Rules JSON", value=default_rules_json, height=380)
with col2:
    st.caption("Preset kaydet/indir")
    st.download_button("JSON indir", data=default_rules_json.encode("utf-8"),
                       file_name=f"rules_{preset_choice}.json", mime="application/json")
    uploaded_rules = st.file_uploader("JSON yükle", type=["json"])
if uploaded_rules is not None:
    rules_json_text = uploaded_rules.read().decode("utf-8")

try:
    RULESET = json.loads(rules_json_text)
    rules_ok = True
except Exception as e:
    rules_ok = False
    st.error(f"Rules JSON hatalı: {e}")

run = st.button("🚀 Taramayı Başlat", type="primary")

# ---------------- Main Process ----------------
if run and rules_ok:
    if not tickers:
        st.warning("Lütfen en az 1 sembol girin.")
        st.stop()

    with st.spinner("Veri indiriliyor..."):
        ohlcv = fetch_ohlcv(tickers, start=str(start), end=str(end))
        close = ohlcv["close"]; high = ohlcv["high"]; low = ohlcv["low"]; volume = ohlcv["volume"]

    if close.empty:
        st.error("Veri alınamadı. Tarih/evreni kontrol edin.")
        st.stop()

    # Likidite
    tlt = tl_turnover(close, volume)
    liq_pass = liquidity_filter(tlt, int(liq_lookback), float(liq_thresh))
    liquid_universe = [c for c, ok in liq_pass.items() if bool(ok)]

    st.success(f"Likidite filtresini geçen: {len(liquid_universe)} / {len(tickers)}")
    st.dataframe(pd.DataFrame({"Medyan TL Ciro ≥ Eşik": liq_pass}).T)

    if not liquid_universe:
        st.warning("Likidite filtresini geçen sembol yok. Eşiği düşürmeyi deneyin.")
        st.stop()

    # İndikatör & kurallar
    rows = []
    with st.spinner("İndikatörler hesaplanıyor..."):
        for sym in liquid_universe:
            met = compute_all_indicators(close[sym], high[sym], low[sym], volume[sym], ind_params)
            if not met:
                continue
            rs = eval_ruleset(RULESET, met)
            rows.append({
                "ticker": sym,
                "pass": rs["pass"],
                "score": rs["score"],
                "pass_count": rs["pass_count"],
                # çekirdek metrikler
                "close": met.get("close"),
                "rsi": met.get("rsi"),
                "adx": met.get("adx"),
                "di_bull": met.get("di_bias_bull"),
                "macd_hist": met.get("macd_hist"),
                "stoch_k": met.get("stoch_k"),
                "stoch_d": met.get("stoch_d"),
                "mfi": met.get("mfi"),
                "sma_fast": met.get("sma_fast"),
                "sma_slow": met.get("sma_slow"),
                "near_52w": met.get("near_52w_high"),
                "atr_pct": met.get("atr_pct"),
                "ret_3m": met.get("ret_3m"),
                "ret_6m": met.get("ret_6m"),
                "ret_12m": met.get("ret_12m"),
            })

    if not rows:
        st.warning("Hesaplanabilir veri çıkmadı.")
        st.stop()

    df = pd.DataFrame(rows).sort_values(["pass","score","ret_3m"], ascending=[False, False, False])
    st.subheader("📋 Screening Sonuçları")
    st.dataframe(df, use_container_width=True)
    st.download_button("CSV indir (sonuçlar)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="screening_results.csv", mime="text/csv")

    st.divider()
    st.subheader("🎯 Seçim")
    topn = st.slider("Top-N (skora göre)", 1, min(20, len(df)), min(10, len(df)))
    suggested = df.head(topn)["ticker"].tolist()
    picked = st.multiselect("Portföye aday seç (manuel düzenlenebilir)",
                            options=df["ticker"].tolist(), default=suggested)
    st.write("Seçilenler:", picked)

    st.info("✅ Sprint-1 bitti. Sprint-2’de: Efficient Frontier + Max-Sharpe / Min-Var + rebalans & metrikler.")
