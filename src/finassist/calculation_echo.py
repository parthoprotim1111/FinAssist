"""
Deterministic post-processing: ensure LLM prose reflects Python debt metrics.

Uses the same payload as the recommendation prompt (build_recommendation_calc_payload).
"""

from __future__ import annotations

import math
import re
from typing import Any

from finassist.debt_calculations import (
    _parse_horizon_months,
    build_recommendation_calc_payload,
)
from finassist.schemas import FinancialRecommendation


def _numbers_from_text(s: str) -> list[float]:
    """Extract plausible numeric literals from prose (currency-style and plain)."""
    if not s:
        return []
    t = s.replace(",", "")
    nums: list[float] = []
    for m in re.finditer(r"\$?\s*([\d]+(?:\.\d+)?)\s*k\b", t, re.I):
        try:
            nums.append(float(m.group(1)) * 1000)
        except ValueError:
            continue
    for m in re.finditer(r"\$?\s*([\d]+(?:\.\d+)?)\b", t):
        try:
            nums.append(float(m.group(1)))
        except ValueError:
            continue
    return nums


def _number_mentioned(text: str, v: float | None) -> bool:
    if v is None:
        return True
    for n in _numbers_from_text(text):
        if math.isclose(n, float(v), rel_tol=0.02, abs_tol=0.51):
            return True
    return False


def _infeasibility_stated(text: str) -> bool:
    tl = text.lower()
    return any(
        p in tl
        for p in (
            "not achievable",
            "not feasible",
            "shortfall",
            "short about",
            "short by",
            "does not meet",
            "cannot meet",
            "unable to meet",
            "cannot achieve",
            "goal is not",
            "not possible",
        )
    )


def _format_money(v: float) -> str:
    if abs(v - round(v)) < 0.01:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def _dedupe_sentences(text: str, *, max_sentences: int = 6) -> str:
    """Drop repeated sentences (normalized) and cap length to reduce LLM + feasibility overlap."""
    raw = text.strip()
    if not raw:
        return raw
    parts = re.split(r"(?<=[.!?])\s+", raw)
    kept: list[str] = []
    seen: set[str] = set()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        n = re.sub(r"\s+", " ", p.lower())
        if n in seen:
            continue
        seen.add(n)
        kept.append(p)
        if len(kept) >= max_sentences:
            break
    return " ".join(kept)


_META_LINE_PREFIX = re.compile(
    r"^\s*(Task summary|Task goal|Age range|Employment status|Region|Dependents|"
    r"Risk tolerance|Additional notes|Monthly budget hint|Time horizon|Liquidity needs|Constraints|"
    r"Product preferences|Ethical constraints|Automation comfort)\s*:",
    re.I,
)


def _scrub_meta_phrases(s: str) -> str:
    """Remove profile label lines and common instruction-like leakage from model output."""
    if not s:
        return s
    lines_out: list[str] = []
    for line in s.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _META_LINE_PREFIX.match(stripped):
            continue
        lines_out.append(stripped)
    t = " ".join(lines_out)
    for phrase in (
        "Be realistic.",
        "be realistic.",
        "Remember to ",
        "You should ",
        "It is important to remember that ",
    ):
        t = t.replace(phrase, "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _compose_one_sentence_main_goal(slots: dict[str, Any], payload: dict[str, Any]) -> str:
    """Single polished sentence; paraphrases intent without copying raw task goal text."""
    total = payload.get("total_debt")
    horizon = payload.get("horizon_months")
    task = slots.get("task_definition") or {}
    ug = (task.get("goal") or "").strip()
    summary = (task.get("summary") or "").strip()
    combined = f"{ug} {summary}".strip()

    if total is not None and horizon is not None:
        return (
            f"The user seeks to retire approximately ${_format_money(total)} in principal within "
            f"{horizon} months."
        )

    inferred_mo = _parse_horizon_months(combined) if combined else None
    amounts = _numbers_from_text(combined) if combined else []
    debt_hint = max(amounts) if amounts else None

    if debt_hint is not None and inferred_mo is not None:
        return (
            f"The user aims to pay down about ${_format_money(debt_hint)} in principal within "
            f"{inferred_mo} months."
        )
    if debt_hint is not None:
        return f"The user aims to reduce debt principal by about ${_format_money(debt_hint)}."
    if inferred_mo is not None:
        return (
            f"The user targets resolving the stated debt within approximately {inferred_mo} months."
        )
    if _mentions_debt_payoff_context(combined):
        return "The user is pursuing the debt payoff objective described in the task."
    return ""


def _mentions_debt_payoff_context(text: str) -> bool:
    t = text.lower()
    return any(
        k in t
        for k in (
            "debt",
            "pay off",
            "payoff",
            "card",
            "balance",
            "loan",
            "owe",
        )
    )


def _merge_feasibility_into_strategy(strategy: str, feasibility_para: str) -> str:
    """Append feasibility prose without subsection headers (single flowing narrative)."""
    strategy = strategy.strip()
    feasibility_para = feasibility_para.strip()
    if not feasibility_para:
        return strategy
    if not strategy:
        return feasibility_para
    if strategy[-1] not in ".!?":
        strategy = strategy + "."
    return f"{strategy} {feasibility_para}".strip()


def _strengthen_main_goal(rec: FinancialRecommendation, slots: dict[str, Any], payload: dict[str, Any]) -> None:
    """Replace main_goal with one synthesized sentence (no verbatim user-goal copy)."""
    line = _compose_one_sentence_main_goal(slots, payload)
    if line:
        rec.main_goal = line


def _apply_professional_debt_steps(rec: FinancialRecommendation, payload: dict[str, Any]) -> None:
    """Clear action sequence: minimums everywhere, full budget to highest APR, monthly review."""
    if payload.get("total_debt") is None and payload.get("required_monthly_payment") is None:
        return
    bud = payload.get("monthly_budget_amount")
    if bud is not None:
        pay_line = (
            f"After minimums are met, allocate the full monthly amount available for debt—about "
            f"${_format_money(bud)}—to the highest-APR balance until it is paid off, then roll that payment to the next highest APR."
        )
    else:
        pay_line = (
            "After minimums are met, allocate the full monthly amount budgeted for debt to the highest-APR "
            "balance until it is paid off, then roll that payment to the next highest APR."
        )
    tail = (
        "Each month, review balances, APRs, and minimums; adjust how much goes to each account if rates "
        "or income change."
    )
    rec.step_by_step_plan = [
        "List all debts with balances, APRs, and minimum payments.",
        "Pay at least the minimum on every debt each month.",
        pay_line,
        tail,
    ]


def _build_feasibility_paragraph(payload: dict[str, Any]) -> str:
    """At most 2–3 concise third-person sentences; avoid redundant budget/month stacking."""
    rmp = payload.get("required_monthly_payment")
    est = payload.get("estimated_payoff_months")
    feas = payload.get("feasible_with_current_budget")
    gap = payload.get("shortfall_or_surplus")
    budget = payload.get("monthly_budget_amount")

    parts: list[str] = []

    if feas is False:
        if rmp is not None:
            parts.append(
                f"The simplified principal-only floor implies about {_format_money(rmp)} per month "
                f"toward debt to clear balances by the stated horizon."
            )
        if gap is not None and gap < 0:
            parts.append(
                f"The user is short about {_format_money(abs(gap))} per month versus that floor before "
                f"interest and minimum dynamics."
            )
        elif rmp is not None and budget is not None and gap is not None:
            parts.append(
                f"With about {_format_money(budget)} per month available toward debt, the payoff goal is "
                f"not achievable on this principal-only path without raising payments or extending the timeline."
            )
        elif not parts:
            parts.append(
                "Under this simplification, the stated budget and horizon do not meet a feasible "
                "principal-only payoff path without changes."
            )
    else:
        if rmp is not None:
            parts.append(
                f"The simplified principal-only model implies about {_format_money(rmp)} per month toward debt "
                f"for the stated horizon."
            )
        if feas is True and budget is not None and rmp is not None:
            parts.append(
                f"The user indicated about {_format_money(budget)} per month toward debt; that meets or exceeds "
                f"the principal-floor estimate (interest and minimums still add real-world pressure)."
            )
        elif est is not None and len(parts) < 2:
            parts.append(
                f"At the stated monthly payment, the principal-only timeline is roughly {_format_money(est)} "
                f"months (interest not modeled)."
            )

    return " ".join(parts[:3]).strip()


def _employment_is_variable(slots: dict[str, Any]) -> bool:
    emp = str((slots.get("personal_summary") or {}).get("employment_status") or "").lower()
    return any(
        w in emp
        for w in (
            "self",
            "freelance",
            "gig",
            "contract",
            "variable",
            "1099",
            "consult",
            "independent",
        )
    )


def _build_risk_bullets(payload: dict[str, Any], slots: dict[str, Any]) -> tuple[str, str]:
    """
    Two bullets: (1) payment vs goal / principal floor, (2) longer real-world duration due to interest.
    """
    feas = payload.get("feasible_with_current_budget")
    gap = payload.get("shortfall_or_surplus")
    rmp = payload.get("required_monthly_payment")

    if feas is False and gap is not None and gap < 0:
        r1 = (
            f"The monthly payment available is below the simplified principal-only requirement by about "
            f"{_format_money(abs(gap))} for the stated horizon—before interest and minimum dynamics."
        )
    elif feas is False:
        r1 = (
            "The stated monthly budget appears insufficient for the payoff goal on the principal-only "
            "path unless the user increases payments or extends the timeline."
        )
    elif feas is True and rmp is not None:
        r1 = (
            f"Minimum payments and interest can absorb more cash than the simplified ~{_format_money(rmp)} "
            f"principal floor alone implies, so liquidity can still be tight."
        )
    else:
        r1 = (
            "If income falls or minimum payments rise, the planned monthly amount may stop matching "
            "the payoff goal on the simplified principal path."
        )

    r2 = (
        "Interest and fees are not modeled; in practice repayment usually lasts longer than principal-only "
        "timelines suggest."
    )
    if _employment_is_variable(slots):
        r2 += " Irregular income adds further risk to keeping steady payments."

    return r1, r2


def _risk_has_stale_calc_dump(rec: FinancialRecommendation) -> bool:
    joined = " ".join(rec.risk_notes)
    return "Deterministic metrics" in joined or "Feasibility (deterministic calculation)" in joined


def _risk_notes_cover_themes(rec: FinancialRecommendation) -> bool:
    """Payment/shortfall theme plus interest/income theme across the two bullets (no duplicate calc wall)."""
    if not rec.risk_notes or len(rec.risk_notes) < 2:
        return False
    if _risk_has_stale_calc_dump(rec):
        return False
    combined = (rec.risk_notes[0] + " " + rec.risk_notes[1]).lower()
    payment_theme = any(
        k in combined
        for k in (
            "payment",
            "shortfall",
            "budget",
            "principal",
            "floor",
            "insufficient",
            "margin",
            "gap",
            "horizon",
            "below",
        )
    )
    second_theme = any(
        k in combined
        for k in (
            "interest",
            "fee",
            "modeled",
            "income",
            "variable",
            "instability",
            "accumulate",
            "balances",
        )
    )
    return payment_theme and second_theme


def _needs_strategy_block(rec: FinancialRecommendation, payload: dict[str, Any]) -> bool:
    """Strategy must carry the numbers and infeasibility wording; do not rely on risk_notes alone."""
    strategy = rec.recommended_strategy or ""
    rmp = payload.get("required_monthly_payment")
    est = payload.get("estimated_payoff_months")
    feas = payload.get("feasible_with_current_budget")
    if rmp is not None and not _number_mentioned(strategy, rmp):
        return True
    if est is not None and not _number_mentioned(strategy, est):
        return True
    if feas is False and not _infeasibility_stated(strategy):
        return True
    return False


def _patch_step_plan_for_feasibility(rec: FinancialRecommendation, payload: dict[str, Any]) -> None:
    if len(rec.step_by_step_plan) != 4:
        return
    feas = payload.get("feasible_with_current_budget")
    if feas is None:
        return
    joined = " ".join(rec.step_by_step_plan).lower()
    if feas is False:
        if any(k in joined for k in ("extend", "increase", "longer horizon", "raise", "more per month")):
            return
        rec.step_by_step_plan[3] = (
            "Ongoing review: if the simplified feasibility check shows the goal is not achievable on the "
            "current budget, the user should extend the payoff timeline or increase the monthly amount "
            "allocated to debt, then reassess."
        )
    else:
        if "review" in joined and ("quarter" in joined or "month" in joined):
            return
        rec.step_by_step_plan[3] = (
            "Ongoing review: the user should track balances monthly and adjust if income, rates, or "
            "minimum payments diverge from the simplified principal-only estimate."
        )


def _apply_third_person_prose(s: str) -> str:
    """Replace first person with third person (the user)."""
    if not s:
        return s
    t = s
    t = re.sub(r"\bI['’]m\b", "The user is", t, flags=re.I)
    t = re.sub(r"\bI['’]ve\b", "The user has", t, flags=re.I)
    t = re.sub(r"\bI['’]d\b", "The user would", t, flags=re.I)
    t = re.sub(r"\bI\b", "the user", t)
    t = re.sub(r"\bmy\b", "the user's", t, flags=re.I)
    t = re.sub(r"\bme\b", "the user", t, flags=re.I)
    t = re.sub(r"(^|[.!?]\s+)the user\b", lambda m: m.group(1) + "The user", t)
    return t


def _polish_assumptions(rec: FinancialRecommendation) -> None:
    """Prefer modeling limits + budget sustainability; drop weak duplicates of user facts."""
    defaults = [
        "Interest and fees are not modeled in the simplified principal figures; actual timelines differ.",
        "The stated monthly budget is assumed to remain available for debt through the horizon unless circumstances change.",
    ]
    if len(rec.assumptions) < 2:
        return
    known = {
        rec.user_summary.strip().lower(),
        rec.main_goal.strip().lower(),
        rec.summary.strip().lower(),
    }
    cleaned: list[str] = []
    for a in rec.assumptions:
        al = a.strip()
        if not al:
            continue
        if al.lower() in known:
            continue
        cleaned.append(al)
    if len(cleaned) < 2:
        rec.assumptions = defaults[:2]
    else:
        rec.assumptions = cleaned[:2]
        while len(rec.assumptions) < 2:
            rec.assumptions.append(defaults[len(rec.assumptions)])


def _polish_all_text_fields(rec: FinancialRecommendation) -> None:
    rec.user_summary = _apply_third_person_prose(_scrub_meta_phrases(rec.user_summary))
    rec.main_goal = _apply_third_person_prose(rec.main_goal)
    rec.recommended_strategy = _apply_third_person_prose(_scrub_meta_phrases(rec.recommended_strategy))
    rec.alternative_option = _apply_third_person_prose(_scrub_meta_phrases(rec.alternative_option))
    rec.summary = _apply_third_person_prose(_scrub_meta_phrases(rec.summary))
    rec.step_by_step_plan = [_apply_third_person_prose(s) for s in rec.step_by_step_plan]
    rec.risk_notes = [_apply_third_person_prose(s) for s in rec.risk_notes]
    rec.assumptions = [_apply_third_person_prose(s) for s in rec.assumptions]


def polish_recommendation_third_person(rec: FinancialRecommendation) -> FinancialRecommendation:
    """Apply third-person wording to all string fields (no calculation enrichment)."""
    _polish_all_text_fields(rec)
    rec.recommended_strategy = _dedupe_sentences((rec.recommended_strategy or "").strip(), max_sentences=6)
    return rec


def enrich_recommendation_with_calc(
    rec: FinancialRecommendation, slots: dict[str, Any]
) -> FinancialRecommendation:
    """
    Ensure strategy states Python metrics in flowing prose (no labeled feasibility blocks);
    set payment- and interest-focused risk bullets; tighten main_goal and concrete step actions.
    """
    payload = build_recommendation_calc_payload(slots)
    rmp = payload.get("required_monthly_payment")
    est = payload.get("estimated_payoff_months")
    feas = payload.get("feasible_with_current_budget")

    if rmp is None and est is None and feas is None:
        polish_recommendation_third_person(rec)
        return rec

    paragraph = _build_feasibility_paragraph(payload)
    if not paragraph:
        polish_recommendation_third_person(rec)
        return rec

    _strengthen_main_goal(rec, slots, payload)
    _apply_professional_debt_steps(rec, payload)

    need_strategy = _needs_strategy_block(rec, payload)
    need_risk = not _risk_notes_cover_themes(rec)

    if need_strategy:
        rec.recommended_strategy = _merge_feasibility_into_strategy(
            (rec.recommended_strategy or "").strip(),
            paragraph,
        )

    if need_risk:
        r1, r2 = _build_risk_bullets(payload, slots)
        rec.risk_notes = [r1, r2]

    _patch_step_plan_for_feasibility(rec, payload)
    rec.recommended_strategy = _dedupe_sentences(
        (rec.recommended_strategy or "").strip(),
        max_sentences=6,
    )
    _polish_assumptions(rec)
    _polish_all_text_fields(rec)
    return rec


__all__ = ["enrich_recommendation_with_calc", "polish_recommendation_third_person"]
