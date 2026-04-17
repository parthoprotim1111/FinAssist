"""
Run: streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from app.chat_ui import render_chat_page
from app.ui_styles import inject_app_styles

st.set_page_config(
    page_title="Financial Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styles first so main padding clears the fixed top toolbar / header.
inject_app_styles()

st.markdown(
     "## FinAssist · LLM-Driven Financial Decision Support System\n"
    "*Structured dialogue, deterministic calculations, and personalized recommendations*"
)

render_chat_page(inject_styles=False)
