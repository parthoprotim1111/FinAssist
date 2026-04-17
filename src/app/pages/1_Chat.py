"""Dedicated chat page (same UI as Home)."""

from __future__ import annotations

import streamlit as st

from app.chat_ui import render_chat_page

st.set_page_config(page_title="Chat", layout="wide")
st.title("Chat")
render_chat_page()
