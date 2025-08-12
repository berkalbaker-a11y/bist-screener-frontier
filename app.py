# app.py — Setup Sanity Check (minimal)
import streamlit as st

st.set_page_config(page_title="BIST Screener & Frontier • Setup", layout="wide")
st.title("✅ Kurulum Tamam: BIST Screener & Frontier (Setup)")

st.markdown("""
Bu sayfa sadece **ortamın hazır olduğunu** doğrulamak içindir.
Deploy başarılıysa, bir sonraki adımda **Sprint-1** kodunu ekleyeceğiz.
""")

ok = []
warn = []

def try_import(name, alias=None):
    try:
        mod = __import__(name) if alias is None else __import__(name, fromlist=[alias])
        return True, getattr(mod, "__version__", "ok")
    except Exception as e:
        return False, str(e)
        
checks = {
    "streamlit": try_import("streamlit"),
    "pandas": try_import("pandas"),
    "numpy": try_import("numpy"),
    "yfinance": try_import("yfinance"),
    "ta": try_import("ta"),              # <-- burası değişti
    "plotly": try_import("plotly"),
    "pyyaml": try_import("yaml"),
}

st.subheader("Paket Kontrolü")
for name, (ok_flag, ver) in checks.items():
    if ok_flag:
        st.success(f"{name}: {ver}")
    else:
        st.warning(f"{name}: **yüklenemedi** → {ver}")

st.divider()
st.info("✅ Bu sayfayı görüyorsan deploy başarılı demektir. Devam için bana **“Sprint-1 kodla”** yaz.")
