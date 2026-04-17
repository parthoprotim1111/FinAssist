from __future__ import annotations

from typing import Any

from finassist.calculation_echo import enrich_recommendation_with_calc, polish_recommendation_third_person
from finassist.schemas import (
    REQUIRED_RECOMMENDATION_DISCLAIMER,
    FinancialRecommendation,
    parse_recommendation_json,
)


def _compress_plan_to_four_steps(steps: list[str]) -> list[str]:
    """
    Enforce exactly 4 ordered steps.
    If more than 4 steps are provided, keep the first 3 and fold the remainder into step 4.
    """
    cleaned = [s.strip() for s in steps if isinstance(s, str) and s.strip()]
    if len(cleaned) <= 4:
        return cleaned
    merged_tail = "; ".join(cleaned[3:])
    return [cleaned[0], cleaned[1], cleaned[2], f"{cleaned[3]}. {merged_tail}"]


_DEFAULT_STEPS: tuple[str, ...] = (
    "List all debts with balances, APRs, and minimum payments.",
    "Pay at least the minimum on every debt each month; rank by APR for any extra payment.",
    "After minimums, allocate the full monthly amount budgeted for debt to the highest-APR balance until it is cleared, then roll to the next.",
    "Each month, review balances, APRs, and minimums; adjust allocations if income, rates, or feasibility change.",
)

_DEFAULT_RISK = (
    "The monthly payment may be insufficient versus the payoff goal on the simplified principal-only path.",
    "Interest and fees extend real-world repayment beyond principal-only timelines; actual duration is typically longer.",
)

_DEFAULT_ASSUMPTION = (
    "Interest and fees are not modeled in the simplified principal figures.",
    "The stated monthly budget is assumed to remain available for debt through the horizon.",
)


def _normalize_recommendation(rec: FinancialRecommendation) -> FinancialRecommendation:
    """
    Compress long plans, then pad to schema minima so UI validation does not fail on short arrays.
    """
    rec.step_by_step_plan = _compress_plan_to_four_steps(rec.step_by_step_plan)
    for i in range(len(rec.step_by_step_plan), 4):
        rec.step_by_step_plan.append(_DEFAULT_STEPS[i])

    while len(rec.risk_notes) < 2:
        rec.risk_notes.append(_DEFAULT_RISK[len(rec.risk_notes) % 2])

    while len(rec.assumptions) < 2:
        rec.assumptions.append(_DEFAULT_ASSUMPTION[len(rec.assumptions) % 2])

    if not rec.user_summary.strip():
        rec.user_summary = rec.summary.strip() or "The user is pursuing their stated financial goal."
    if not rec.summary.strip():
        rec.summary = rec.user_summary.strip() or "Planning summary based on collected profile data."
    if not rec.main_goal.strip():
        rec.main_goal = "Work toward the stated task and horizon using the agreed constraints."
    if not rec.recommended_strategy.strip():
        rec.recommended_strategy = (
            "Use the simplified feasibility numbers above and keep a consistent monthly debt allocation."
        )
    if not rec.alternative_option.strip():
        rec.alternative_option = (
            "If priorities change, revisit the horizon or monthly allocation before changing tactics."
        )
    if not rec.disclaimer.strip():
        rec.disclaimer = REQUIRED_RECOMMENDATION_DISCLAIMER

    return rec


def _structured_recommendation_complete(rec: FinancialRecommendation) -> bool:
    return bool(
        rec.user_summary.strip()
        and rec.main_goal.strip()
        and rec.recommended_strategy.strip()
        and len(rec.step_by_step_plan) == 4
        and len(rec.risk_notes) >= 2
        and len(rec.assumptions) >= 2
        and rec.alternative_option.strip()
        and rec.summary.strip()
        and rec.disclaimer.strip()
    )


def _legacy_recommendation_complete(rec: FinancialRecommendation) -> bool:
    return bool(
        rec.summary.strip()
        and rec.recommendations
        and all(r.rationale.strip() and r.caveats.strip() for r in rec.recommendations)
    )


def _collect_schema_issues(rec: FinancialRecommendation) -> list[str]:
    """Human-readable gaps when structured output is incomplete."""
    issues: list[str] = []
    if not rec.user_summary.strip() and not rec.summary.strip():
        issues.append("Missing user_summary (or summary for legacy).")
    elif not rec.user_summary.strip():
        issues.append("user_summary is empty.")
    if not rec.main_goal.strip():
        issues.append("main_goal is empty.")
    if not rec.recommended_strategy.strip():
        issues.append("recommended_strategy is empty.")
    if len(rec.step_by_step_plan) != 4:
        issues.append("step_by_step_plan must contain exactly 4 ordered steps.")
    if len(rec.risk_notes) < 2:
        issues.append("risk_notes needs at least 2 bullets.")
    if len(rec.assumptions) < 2:
        issues.append("assumptions needs at least 2 items.")
    if not rec.alternative_option.strip():
        issues.append("alternative_option is empty.")
    if not rec.summary.strip():
        issues.append("summary is empty.")
    if not rec.disclaimer.strip():
        issues.append("disclaimer is empty.")
    lower_goal = rec.main_goal.lower()
    if lower_goal.startswith("i ") or " i " in lower_goal:
        issues.append("main_goal should use third-person wording (not first person).")
    lower_strategy = rec.recommended_strategy.lower()
    if lower_strategy.startswith("i ") or " i " in lower_strategy:
        issues.append(
            "recommended_strategy should use third-person wording (not first person)."
        )
    if "increase minimum" in lower_strategy or "increase minimums" in lower_strategy:
        issues.append(
            "recommended_strategy contains incorrect avalanche wording about increasing minimums."
        )
    known_facts = {
        rec.main_goal.strip().lower(),
        rec.summary.strip().lower(),
        rec.user_summary.strip().lower(),
    }
    for a in rec.assumptions:
        al = a.strip().lower()
        if al and al in known_facts:
            issues.append("assumptions should not repeat known user facts verbatim.")
    for item in rec.recommendations:
        if not item.rationale.strip():
            issues.append(f"Missing rationale for: {item.title}")
        if not item.caveats.strip():
            issues.append(f"Missing caveats for: {item.title}")
    return issues


def ensure_justification_fields(
    text: str,
    slots: dict[str, Any] | None = None,
) -> tuple[FinancialRecommendation | None, list[str]]:
    """Validate parsed recommendation; return issues for UI before display."""
    rec = parse_recommendation_json(text)
    issues: list[str] = []
    if rec is None:
        issues.append("Output is not valid recommendation JSON.")
        return None, issues
    rec = _normalize_recommendation(rec)
    if slots is not None:
        rec = enrich_recommendation_with_calc(rec, slots)
    else:
        rec = polish_recommendation_third_person(rec)

    rec.disclaimer = REQUIRED_RECOMMENDATION_DISCLAIMER

    if _structured_recommendation_complete(rec) or _legacy_recommendation_complete(rec):
        return rec, issues

    issues.extend(_collect_schema_issues(rec))
    return rec, issues
