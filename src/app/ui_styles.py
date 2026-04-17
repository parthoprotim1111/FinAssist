"""Minimal Streamlit tweaks: default look is Streamlit’s light theme (see .streamlit/config.toml)."""

from __future__ import annotations

import streamlit as st


def inject_app_styles() -> None:
    # Only: clear the fixed top chrome so the first heading is not clipped.
    st.markdown(
        """
<style>
  .main .block-container {
    padding-top: 5.5rem;
    max-width: 1200px;
  }
</style>
        """,
        unsafe_allow_html=True,
    )
