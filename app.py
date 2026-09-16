"""
app.py
Streamlit web UI for Darukaa.Earth Biodiversity Intelligence Chatbot.
Run: streamlit run app.py
"""

import streamlit as st
from chatbot import DarukaaChatbot

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Darukaa.Earth",
    page_icon="🌿",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f5f7f5; }
    h1 { color: #2e7d32 !important; }
    .metric-box {
        background-color: #e8f5e9;
        border: 1px solid #2e7d32;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 13px;
        color: #1b5e20;
    }
    .tip-box {
        background-color: #e8f5e9;
        border-left: 3px solid #2e7d32;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 12px;
        color: #1b5e20;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("# 🌿 Darukaa.Earth")
st.markdown("**AI Biodiversity Intelligence** — Ask about your land, get science-backed recommendations.")
st.divider()

# ── Session state ──────────────────────────────────────────────────────────
if "bot" not in st.session_state:
    st.session_state.bot = DarukaaChatbot()
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧪 Quick Data Input")
    st.markdown("Fill known values and click **Load Data**")

    soc   = st.number_input("Soil Organic Carbon (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    ph    = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rain  = st.number_input("Rainfall (mm/year)", min_value=0, max_value=5000, value=0, step=10)
    temp  = st.number_input("Temperature (°C)", min_value=-20, max_value=60, value=0, step=1)
    luse  = st.selectbox("Land Use", ["", "monoculture", "agroforestry", "intercropping",
                                       "intensive_grazing", "native_grassland", "degraded_forest", "urban"])

    if st.button("⚡ Load Data", use_container_width=True):
        data = {}
        if soc > 0:   data["soil_organic_carbon_pct"] = soc
        if ph > 0:    data["soil_ph"] = ph
        if rain > 0:  data["rainfall_mm_year"] = rain
        if temp > 0:  data["temperature_c"] = temp
        if luse:      data["land_use"] = luse

        if data:
            import json
            msg = json.dumps(data)
            st.session_state.messages.append({"role": "user", "content": msg})
            with st.spinner("Analysing..."):
                try:
                    response = st.session_state.bot.chat(msg)
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                        response = "⚠️ Gemini free tier daily limit reached (20 requests/day). Please wait until tomorrow or upgrade your plan."
                    else:
                        response = f"⚠️ Error: {err}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        else:
            st.warning("Fill at least one field.")

    st.divider()

    # Show accumulated data
    if st.session_state.bot._pending_data:
        st.markdown("### 📊 Loaded Metrics")
        for k, v in st.session_state.bot._pending_data.items():
            st.markdown(f'<div class="metric-box">**{k}**: {v}</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.bot.reset()
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div class="tip-box">💡 Tip: You can also paste JSON directly in the chat:<br><code>{"soil_organic_carbon_pct": 0.3, "rainfall_mm_year": 280, "land_use": "monoculture"}</code></div>', unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🌿" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your land or describe a biodiversity problem..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("Darukaa is thinking..."):
            try:
                response = st.session_state.bot.chat(prompt)
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                    response = (
                        "⚠️ **Gemini free tier daily limit reached (20 requests/day).**\n\n"
                        "Options:\n"
                        "- Wait until tomorrow for the quota to reset\n"
                        "- Or upgrade at [Google AI Studio](https://aistudio.google.com) for higher limits"
                    )
                else:
                    response = f"⚠️ An error occurred: {err}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
