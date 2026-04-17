from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.charts import latency_bar_chart, score_bar_chart
from evaluation.csv_output import save_dataframe_csv
from evaluation.metrics import score_output, timed_generate
from evaluation.run_eval import load_backend
from utils.config_loader import load_yaml, project_root


st.set_page_config(page_title="Benchmarks", layout="wide")
st.title("Model & technique comparison")

st.markdown(
    "Compare **mock**, **HF primary**, and **HF alt** on the same evaluation fixtures."
)

backend_choice = st.selectbox(
    "Backend",
    ["mock", "hf_primary", "hf_alt"],
)
techniques = st.multiselect(
    "Techniques",
    ["zero_shot", "few_shot", "cot"],
    default=["few_shot"],
)
low_vram = st.checkbox("Low VRAM", value=True)

fixtures_path = project_root() / "src" / "evaluation" / "fixtures" / "example_prompts.json"
fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
eval_cfg = load_yaml("configs/evaluation.yaml")
weights = eval_cfg.get("metrics", {})

if st.button("Run batch eval"):
    rows = []
    from dialogue.state_machine import DialogueEngine
    from dialogue.slots import (
        DialogueState,
        FinancialPreferences,
        FinancialRequirements,
        PersonalInfo,
        SessionSlots,
        TaskDefinition,
    )

    def slots_from(blob):
        s = SessionSlots()
        s.task_definition = TaskDefinition(**blob["task_definition"])
        s.personal_summary = PersonalInfo(**blob["personal_summary"])
        s.financial_requirements = FinancialRequirements(**blob["financial_requirements"])
        s.financial_preferences = FinancialPreferences(**blob["financial_preferences"])
        return s

    if backend_choice == "hf_primary":
        backend = load_backend("hf", low_vram)
    elif backend_choice == "hf_alt":
        backend = load_backend("hf_alt", low_vram)
    else:
        backend = load_backend("mock", low_vram)

    try:
        for tech in techniques:
            for item in fixtures:
                slots = slots_from(item["slots"])
                engine = DialogueEngine(backend)
                engine.slots = slots
                engine.state = DialogueState.RECOMMEND

                def run_one():
                    return engine.generate_recommendations(
                        locale_prompt="Respond in clear, professional English.",
                        technique=tech,
                    )

                result, latency = timed_generate(run_one)
                m = score_output(
                    result.text,
                    latency_s=latency,
                    weights=weights,
                    slots=slots.to_context_dict(),
                )
                rows.append(
                    {
                        "prompt_id": item["id"],
                        "model": backend.name,
                        "technique": tech,
                        "latency_s": latency,
                        "schema_valid": m["schema_valid"],
                        "score": m["score"],
                    }
                )
    finally:
        backend.unload()

    df = pd.DataFrame(rows)
    out = project_root() / "reports" / "eval_results.csv"
    saved_path = save_dataframe_csv(df, out)
    if saved_path.resolve() != out.resolve():
        st.warning(
            f"Could not write `{out.name}` (it may be open in another app, e.g. Excel). "
            f"Results saved to **`{saved_path.name}`** in the same folder."
        )
    st.success(f"Saved `{saved_path}`")
    st.dataframe(df)
    f1 = latency_bar_chart(df)
    if f1:
        st.plotly_chart(f1, use_container_width=True)
    f2 = score_bar_chart(df)
    if f2:
        st.plotly_chart(f2, use_container_width=True)

st.caption("Backends use `HFLocalBackend` (primary vs alt) or `MockLLMBackend`; see `configs/models.yaml`.")
