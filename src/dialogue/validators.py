from __future__ import annotations

from dialogue.slots import (
    DialogueState,
    FinancialPreferences,
    FinancialRequirements,
    PersonalInfo,
    SessionSlots,
    TaskDefinition,
)

# If True, liquidity_needs may be left empty when requirements_filled() runs (not recommended for Step 3).
LIQUIDITY_NEEDS_OPTIONAL_FOR_REQUIREMENTS_PHASE = False


def task_filled(t: TaskDefinition) -> bool:
    return bool(t.summary.strip() and t.goal.strip())


def personal_filled(p: PersonalInfo) -> bool:
    return bool(
        p.age_range.strip()
        and p.employment_status.strip()
        and p.risk_tolerance.strip()
    )


def requirements_filled(r: FinancialRequirements) -> bool:
    """
    Step 3 (REQUIREMENTS) is complete only when planning inputs are usable:
    - time horizon
    - monthly budget hint
    - liquidity needs (unless explicitly optional via module flag)
    """
    horizon = r.time_horizon_months.strip()
    budget = r.monthly_budget_hint.strip()
    liq = r.liquidity_needs.strip()
    if not (horizon and budget):
        return False
    if LIQUIDITY_NEEDS_OPTIONAL_FOR_REQUIREMENTS_PHASE:
        return True
    return bool(liq)


def preferences_filled(p: FinancialPreferences) -> bool:
    return bool(p.product_preferences.strip() or p.ethical_constraints.strip())


def slots_ready_for_recommendation(slots: SessionSlots) -> bool:
    return (
        task_filled(slots.task_definition)
        and personal_filled(slots.personal_summary)
        and requirements_filled(slots.financial_requirements)
        and preferences_filled(slots.financial_preferences)
    )


def compute_state_from_slots(slots: SessionSlots) -> tuple[DialogueState, list[str]]:
    """
    Canonical dialogue state from slot completeness (avoids stale state drift).
    Returns (state, debug reasons) — reasons list what is satisfied or blocking.
    """
    reasons: list[str] = []
    if not task_filled(slots.task_definition):
        reasons.append("blocked: task_definition incomplete (need summary and goal)")
        return DialogueState.DEFINE_TASK, reasons
    reasons.append("ok: task_definition")

    if not personal_filled(slots.personal_summary):
        reasons.append("blocked: personal_summary incomplete")
        return DialogueState.COLLECT_PERSONAL, reasons
    reasons.append("ok: personal_summary")

    if not requirements_filled(slots.financial_requirements):
        r = slots.financial_requirements
        missing: list[str] = []
        if not r.time_horizon_months.strip():
            missing.append("time_horizon_months")
        if not r.monthly_budget_hint.strip():
            missing.append("monthly_budget_hint")
        if not LIQUIDITY_NEEDS_OPTIONAL_FOR_REQUIREMENTS_PHASE and not r.liquidity_needs.strip():
            missing.append("liquidity_needs")
        reasons.append(
            "blocked: financial_requirements incomplete (" + ", ".join(missing) + ")"
        )
        return DialogueState.REQUIREMENTS, reasons
    reasons.append("ok: financial_requirements")

    if not preferences_filled(slots.financial_preferences):
        reasons.append("blocked: financial_preferences incomplete")
        return DialogueState.PREFERENCES, reasons
    reasons.append("ok: financial_preferences")

    return DialogueState.RECOMMEND, reasons


def redact_for_export(text: str, max_len: int = 400) -> str:
    t = text.strip().replace("\n", " ")
    if len(t) > max_len:
        return t[: max_len - 3] + "..."
    return t
