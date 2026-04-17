from __future__ import annotations

import json

import streamlit as st

from data_collection.consent import CONSENT_TEXT, consent_granted
from data_collection.export_json import anonymize_session_export, dumps_pretty
from dialogue.state_machine import DialogueEngine
from llm.mock_backend import MockLLMBackend
from utils.config_loader import load_yaml


st.set_page_config(page_title="Data & Exports", layout="wide")
st.title("Data collection & exports")

st.markdown(CONSENT_TEXT)

consent = st.checkbox("I agree to the terms above for export purposes", value=False)

if "engine" in st.session_state and isinstance(st.session_state.engine, DialogueEngine):
    engine = st.session_state.engine
    backend_name = st.session_state.get("backend_name", "unknown")
    language = "en"
    payload = anonymize_session_export(
        engine.slots,
        backend_name=backend_name,
        language=language,
        consent=consent_granted(consent),
    )
    st.subheader("Preview (anonymized)")
    st.code(dumps_pretty(payload))
    if consent:
        st.download_button(
            "Download JSON",
            data=dumps_pretty(payload),
            file_name="session_export.json",
            mime="application/json",
        )
    else:
        st.info("Enable consent to unlock download.")
else:
    st.info("Start a session on the main page first; export uses the in-memory dialogue state.")

st.subheader("Configuration snapshot")
st.code(json.dumps(load_yaml("configs/dialogue.yaml"), indent=2))
