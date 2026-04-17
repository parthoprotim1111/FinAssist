"""
Deterministic guided copy for each dialogue stage.

The LLM only fills slots; the assistant's visible reply always comes from here
so users see clear next-step questions instead of generic acknowledgements.
"""

from __future__ import annotations

from dialogue.slots import DialogueState, SessionSlots
from dialogue.validators import (
    LIQUIDITY_NEEDS_OPTIONAL_FOR_REQUIREMENTS_PHASE,
    personal_filled,
    preferences_filled,
    requirements_filled,
    task_filled,
)

_COPY: dict[str, str] = {
    "welcome_title": "Welcome",
    "welcome_body": (
        "I will walk you through **four short steps** so recommendations match your situation. "
        "This is an educational demo — **not** personalized financial advice."
    ),
    "step1_header": "Step 1 — Task",
    "ask_task_summary": (
        "**What financial topic** do you want help with? "
        "(One short line is enough — e.g. *emergency fund*, *debt payoff*, *retirement*.)"
    ),
    "ask_task_goal": (
        "**What outcome are you aiming for?** "
        "Include anything measurable if you can (amount, timeline, or success criteria)."
    ),
    "step2_header": "Step 2 — About you",
    "ask_personal": (
        "Please share:\n"
        "- **Age range or life stage** (e.g. 25–34)\n"
        "- **Employment status** (e.g. employed, student, self-employed)\n"
        "- **Country or region** (broad — no street address)\n"
        "- **Dependents** (number or “none”)\n"
        "- **Risk tolerance** — *low*, *medium*, or *high*\n\n"
        "Optional notes: anything else that affects decisions."
    ),
    "ask_personal_missing": "I still need: **{missing}**.",
    "step3_header": "Step 3 — Financial requirements",
    "ask_requirements": (
        "Please describe:\n"
        "- **Rough monthly budget or savings capacity** (ballpark is fine)\n"
        "- **Time horizon** in months or years\n"
        "- **Liquidity needs** (how much you need in cash vs. invested)\n"
        "- **Constraints** (e.g. no crypto, tax-advantaged only, ethical screens)"
    ),
    "ask_requirements_missing": "I still need at least: **{missing}**.",
    "step4_header": "Step 4 — Preferences",
    "ask_preferences": (
        "Almost done. Please share:\n"
        "- **Product or style preferences** (e.g. index funds, ETFs, cash savings)\n"
        "- **Ethical / values constraints** (or “none”)\n"
        "- **Comfort with automation** (e.g. auto-transfers, robo-tools — or “prefer manual”)"
    ),
    "ask_preferences_missing": (
        "Please add either **product preferences** or **ethical constraints** (or both)."
    ),
    "ready": (
        "**All four steps are complete.** Use the button below to generate your structured recommendations."
    ),
    "parse_error": (
        "I could not read structured details from that message. "
        "Try shorter sentences, or use the **Quick edit** panel on the right."
    ),
}

_LBL = {
    "lbl_age": "age range",
    "lbl_emp": "employment status",
    "lbl_risk": "risk tolerance",
    "lbl_horizon": "time horizon",
    "lbl_budget": "monthly budget or savings capacity",
    "lbl_liq": "liquidity needs",
}


def _missing_personal_labels(slots: SessionSlots) -> list[str]:
    p = slots.personal_summary
    need: list[str] = []
    if not p.age_range.strip():
        need.append(_LBL["lbl_age"])
    if not p.employment_status.strip():
        need.append(_LBL["lbl_emp"])
    if not p.risk_tolerance.strip():
        need.append(_LBL["lbl_risk"])
    return need


def _missing_requirements_labels(slots: SessionSlots) -> list[str]:
    r = slots.financial_requirements
    need: list[str] = []
    if not r.time_horizon_months.strip():
        need.append(_LBL["lbl_horizon"])
    if not r.monthly_budget_hint.strip():
        need.append(_LBL["lbl_budget"])
    if not LIQUIDITY_NEEDS_OPTIONAL_FOR_REQUIREMENTS_PHASE and not r.liquidity_needs.strip():
        need.append(_LBL["lbl_liq"])
    return need


def welcome_message() -> str:
    c = _COPY
    return (
        f"### {c['welcome_title']}\n\n{c['welcome_body']}\n\n"
        f"#### {c['step1_header']}\n\n{c['ask_task_summary']}"
    )


def guided_reply_after_turn(
    state: DialogueState,
    slots: SessionSlots,
    *,
    extraction_ok: bool,
) -> str:
    """Message shown to the user after each turn (replaces LLM acknowledgements)."""
    c = _COPY

    if not extraction_ok:
        return (
            f"{c['parse_error']}\n\n---\n\n"
            f"{guided_reply_after_turn(state, slots, extraction_ok=True)}"
        )

    if state == DialogueState.RECOMMEND:
        return f"#### {c['step4_header']}\n\n{c['ready']}"

    if state == DialogueState.DEFINE_TASK:
        if not slots.task_definition.summary.strip():
            return f"#### {c['step1_header']}\n\n{c['ask_task_summary']}"
        if not slots.task_definition.goal.strip():
            return f"#### {c['step1_header']}\n\n{c['ask_task_goal']}"
        return f"#### {c['step1_header']}\n\n{c['ask_task_goal']}"

    if state == DialogueState.COLLECT_PERSONAL:
        miss = _missing_personal_labels(slots)
        base = f"#### {c['step2_header']}\n\n{c['ask_personal']}"
        if miss:
            base += "\n\n" + c["ask_personal_missing"].format(missing=", ".join(miss))
        return base

    if state == DialogueState.REQUIREMENTS:
        miss = _missing_requirements_labels(slots)
        base = f"#### {c['step3_header']}\n\n{c['ask_requirements']}"
        if miss:
            base += "\n\n" + c["ask_requirements_missing"].format(
                missing=", ".join(miss)
            )
        return base

    if state == DialogueState.PREFERENCES:
        if not preferences_filled(slots.financial_preferences):
            return (
                f"#### {c['step4_header']}\n\n{c['ask_preferences']}\n\n"
                f"{c['ask_preferences_missing']}"
            )
        return f"#### {c['step4_header']}\n\n{c['ask_preferences']}"

    return c.get("ready", "")


def progress_step(state: DialogueState) -> tuple[int, int]:
    """Current step index (1-based) and total (4) for UI."""
    m = {
        DialogueState.DEFINE_TASK: 1,
        DialogueState.COLLECT_PERSONAL: 2,
        DialogueState.REQUIREMENTS: 3,
        DialogueState.PREFERENCES: 4,
        DialogueState.RECOMMEND: 5,
        DialogueState.DONE: 5,
    }
    step = m.get(state, 1)
    return min(step, 4), 4


def stage_label(state: DialogueState) -> str:
    c = _COPY
    keys = {
        DialogueState.DEFINE_TASK: "step1_header",
        DialogueState.COLLECT_PERSONAL: "step2_header",
        DialogueState.REQUIREMENTS: "step3_header",
        DialogueState.PREFERENCES: "step4_header",
        DialogueState.RECOMMEND: "ready",
        DialogueState.DONE: "ready",
    }
    return c.get(keys.get(state, "step1_header"), "")
