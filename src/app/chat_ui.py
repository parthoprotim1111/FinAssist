from __future__ import annotations

import json
import textwrap

import streamlit as st

from app.ui_styles import inject_app_styles
from dialogue.guided_flow import (
    guided_reply_after_turn,
    progress_step,
    stage_label,
    welcome_message,
)
from dialogue.slot_tracking import list_missing_required_fields
from dialogue.slots import DialogueState, SessionSlots
from dialogue.state_machine import DialogueEngine
from finassist.justification import ensure_justification_fields
from finassist.schemas import recommendation_to_display
from llm.backend_base import LLMBackend
from llm.hf_local import HFLocalBackend
from llm.mock_backend import MockLLMBackend
from utils.config_loader import load_yaml, project_root


def _make_backend(name: str, low_vram: bool) -> LLMBackend:
    if name == "mock":
        return MockLLMBackend()
    if name == "hf_primary":
        return HFLocalBackend(use_primary=True, low_vram=low_vram)
    if name == "hf_alt":
        return HFLocalBackend(use_primary=False, low_vram=low_vram)
    return MockLLMBackend()


def _dialogue_cfg():
    return load_yaml(project_root() / "configs" / "dialogue.yaml")


def _locale_prompt() -> str:
    d = _dialogue_cfg()
    return str(d.get("locale_prompt", "Respond in clear, professional English."))


def _maybe_load_hf(engine: DialogueEngine) -> None:
    if isinstance(engine.backend, HFLocalBackend):
        engine.backend.load()


def _format_slot_value(value: str, *, max_len: int = 120, wrap_at: int = 56) -> str:
    v = " ".join((value or "").strip().split())
    if not v:
        return ""
    if len(v) > max_len:
        v = v[: max_len - 1].rstrip() + "…"
    return textwrap.fill(
        v,
        width=wrap_at,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _slots_summary_rows(
    slots: SessionSlots, *, include_empty: bool = False
) -> list[tuple[str, str]]:
    t = slots.task_definition
    p = slots.personal_summary
    r = slots.financial_requirements
    f = slots.financial_preferences
    pairs = [
        ("Task summary", t.summary),
        ("Goal", t.goal),
        ("Age range", p.age_range),
        ("Employment", p.employment_status),
        ("Region", p.country_region),
        ("Dependents", p.dependents),
        ("Risk tolerance", p.risk_tolerance),
        ("Budget", r.monthly_budget_hint),
        ("Horizon", r.time_horizon_months),
        ("Liquidity", r.liquidity_needs),
        ("Constraints", r.constraints),
        ("Product preferences", f.product_preferences),
        ("Ethics", f.ethical_constraints),
        ("Automation comfort", f.automation_comfort),
    ]
    rows: list[tuple[str, str]] = []
    for label, raw in pairs:
        cleaned = _format_slot_value(raw or "")
        if cleaned:
            rows.append((label, cleaned))
        elif include_empty:
            rows.append((label, "—"))
    return rows


def _render_slots_summary_panel(slots: SessionSlots) -> None:
    rows = _slots_summary_rows(slots, include_empty=False)
    if not rows:
        st.caption("No captured values yet.")
        return
    for label, value in rows:
        c1, c2 = st.columns([0.43, 0.57], gap="small")
        with c1:
            st.markdown(f"**{label}**")
        with c2:
            st.write(value)
    st.caption("Empty fields are hidden.")


def _render_progress(engine: DialogueEngine) -> None:
    step, total = progress_step(engine.state)
    if engine.state == DialogueState.DONE:
        st.success("Flow complete.")
        return
    pct = min(step / total, 1.0)
    st.progress(pct)
    st.caption(f"Stage {min(step, total)} of {total}")
    st.caption(stage_label(engine.state))


def render_chat_page(*, inject_styles: bool = True) -> None:
    if inject_styles:
        inject_app_styles()
    dcfg = _dialogue_cfg()

    with st.sidebar:
        st.markdown("### Assistant settings")
        st.session_state.setdefault("backend_name", "mock")
        st.session_state.setdefault("low_vram", False)
        st.session_state.setdefault("consent_export", False)

        backend = st.selectbox(
            "Model backend",
            options=["mock", "hf_primary", "hf_alt"],
            format_func=lambda x: {
                "mock": "Mock (rules + heuristics, no GPU)",
                "hf_primary": "Local HF primary (+ optional LoRA)",
                "hf_alt": "Local HF comparison model",
            }[x],
            key="backend_select",
        )
        st.session_state.backend_name = backend
        st.session_state.low_vram = st.checkbox(
            "Low VRAM mode", value=st.session_state.low_vram
        )
        st.session_state.consent_export = st.checkbox(
            "Consent to anonymized export (Data page)",
            value=st.session_state.consent_export,
        )
        st.caption(
            "Main path: **Transformers + optional LoRA**. Benchmarks use mock or HF primary/alt."
        )

        if st.button("Reset conversation", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k in ("messages", "engine", "_engine_backend"):
                    del st.session_state[k]
            st.rerun()

    b = _make_backend(st.session_state.backend_name, st.session_state.low_vram)
    key = f"{backend}|{st.session_state.low_vram}"
    if "engine" not in st.session_state or st.session_state.get("_engine_backend") != key:
        st.session_state.engine = DialogueEngine(b)
        st.session_state._engine_backend = key
        welcome = welcome_message()
        st.session_state.engine.last_assistant_message = welcome
        st.session_state.messages = [{"role": "assistant", "content": welcome}]

    engine: DialogueEngine = st.session_state.engine
    locale_prompt = _locale_prompt()

    hero_l, hero_r = st.columns([2, 1])
    with hero_l:
        st.markdown("### Intelligent Conversation Flow")
        st.caption(
                    "Structured data collection across four stages to generate accurate, personalized recommendations."
                    )
    with hero_r:
        with st.container():
            st.caption("Progress")
            _render_progress(engine)

    col_chat, col_sum = st.columns([1.45, 1], gap="large")

    with col_sum:
        st.markdown("#### Collected information")
        _render_slots_summary_panel(engine.slots)
        st.caption("Updates as you answer. Use **Quick edit** if needed.")
        with st.expander("Session debug (global extraction)", expanded=False):
            if engine.last_turn_updated_paths:
                st.markdown("**Updated on last turn:**")
                st.code("\n".join(engine.last_turn_updated_paths))
            else:
                st.caption("No slot paths changed on the last turn (or parse failed).")
            miss = list_missing_required_fields(engine.slots)
            if miss:
                st.markdown("**Still missing for workflow:**")
                st.code(", ".join(miss))
            else:
                st.success("All required workflow fields are present.")
        with st.expander("Raw session JSON (debug)", expanded=False):
            st.code(json.dumps(engine.slots.to_context_dict(), indent=2))
        with st.expander("Quick edit (fallback)", expanded=False):
            slots: SessionSlots = engine.slots
            td_summary = st.text_input("Task topic", value=slots.task_definition.summary)
            td_goal = st.text_input("Task goal / outcome", value=slots.task_definition.goal)
            age = st.text_input("Age range", value=slots.personal_summary.age_range)
            emp = st.text_input("Employment", value=slots.personal_summary.employment_status)
            risk = st.text_input("Risk (low/medium/high)", value=slots.personal_summary.risk_tolerance)
            country = st.text_input("Region", value=slots.personal_summary.country_region)
            dep = st.text_input("Dependents", value=slots.personal_summary.dependents)
            horizon = st.text_input(
                "Time horizon",
                value=slots.financial_requirements.time_horizon_months,
            )
            liq = st.text_input("Liquidity needs", value=slots.financial_requirements.liquidity_needs)
            budget = st.text_input(
                "Budget / savings hint",
                value=slots.financial_requirements.monthly_budget_hint,
            )
            cons = st.text_input("Constraints", value=slots.financial_requirements.constraints)
            pref = st.text_input(
                "Product preferences",
                value=slots.financial_preferences.product_preferences,
            )
            eth = st.text_input(
                "Ethical constraints",
                value=slots.financial_preferences.ethical_constraints,
            )
            auto = st.text_input(
                "Automation comfort",
                value=slots.financial_preferences.automation_comfort,
            )
            if st.button("Apply to session", use_container_width=True):
                before_snapshot = json.loads(
                    json.dumps(engine.slots.to_context_dict())
                )
                slots.task_definition.summary = td_summary
                slots.task_definition.goal = td_goal
                slots.personal_summary.age_range = age
                slots.personal_summary.employment_status = emp
                slots.personal_summary.risk_tolerance = risk
                slots.personal_summary.country_region = country
                slots.personal_summary.dependents = dep
                slots.financial_requirements.time_horizon_months = horizon
                slots.financial_requirements.liquidity_needs = liq
                slots.financial_requirements.monthly_budget_hint = budget
                slots.financial_requirements.constraints = cons
                slots.financial_preferences.product_preferences = pref
                slots.financial_preferences.ethical_constraints = eth
                slots.financial_preferences.automation_comfort = auto
                engine.slots = slots
                engine.refresh_last_turn_diff_from(before_snapshot)
                engine.advance_state_until_stable()
                follow = guided_reply_after_turn(
                    engine.state,
                    engine.slots,
                    extraction_ok=True,
                )
                engine.last_assistant_message = follow
                st.session_state.messages.append(
                    {"role": "assistant", "content": follow}
                )
                st.rerun()

    with col_chat:
        st.markdown("##### Messages")
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if engine.state == DialogueState.DONE:
            st.info("You can reset the session from the sidebar to start again.")
            return

        if engine.state == DialogueState.RECOMMEND:
            tech = st.selectbox(
                "Prompting technique",
                ["few_shot", "zero_shot", "cot"],
                key="rec_technique",
                help="Label for benchmarks only; the recommendation prompt is identical for each option.",
            )
            if st.button("Generate personalized recommendations", type="primary"):
                with st.spinner("Generating recommendations…"):
                    _maybe_load_hf(engine)
                    result = engine.generate_recommendations(
                        locale_prompt=locale_prompt,
                        technique=tech,
                    )
                    rec, issues = ensure_justification_fields(
                        result.text,
                        slots=engine.slots.to_context_dict(),
                    )
                    if rec:
                        body = recommendation_to_display(rec)
                    else:
                        raw = result.text
                        preview = raw[:3000]
                        if len(raw) > 3600:
                            preview += "\n\n...[middle omitted]...\n\n" + raw[-1000:]
                        body = f"Raw output (parse failed):\n\n```\n{preview}\n```"
                    if issues:
                        body = (
                            "**Schema checks failed:**\n\n"
                            + "\n".join(f"- {i}" for i in issues)
                            + "\n\n---\n\n"
                            + body
                        )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": body}
                    )
                    engine.mark_done()
                st.rerun()
            st.stop()

        user = st.chat_input("Type your answer…")
        if user:
            st.session_state.messages.append({"role": "user", "content": user})
            with st.spinner("Updating your plan…"):
                _maybe_load_hf(engine)
                msg, ok = engine.process_user_message(user, locale_prompt)
                limit = int(dcfg.get("extraction_retry_limit", 2))
                if not ok and engine.extraction_failures >= limit:
                    msg += "\n\n_Use **Quick edit** on the right if this keeps failing._"
                st.session_state.messages.append({"role": "assistant", "content": msg})
            st.rerun()
