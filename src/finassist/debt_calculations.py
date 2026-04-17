"""
Deterministic, rule-based debt and payoff math for recommendations.

The local LLM explains; this module supplies numbers so the model does not invent them.
All figures are simplified (principal-only floor; ignores interest and fees).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class DebtPayoffMetrics:
    """Structured outputs for prompts and optional UI reuse."""

    total_debt: float | None
    monthly_budget_amount: float | None
    horizon_months: int | None
    liquidity_needs: str | None
    required_monthly_payment: float | None
    """Rough principal payment per month to clear debt by horizon (interest not modeled)."""
    estimated_payoff_months: float | None
    """Months to clear debt at the parsed monthly budget (principal-only)."""
    feasible_with_current_budget: bool | None
    """True if required_monthly_payment <= monthly_budget when both known."""
    shortfall_or_surplus: float | None
    """monthly_budget - required_monthly_payment (negative means shortfall)."""
    methodology: str


def _mentions_debt_context(text: str) -> bool:
    t = text.lower()
    return any(
        k in t
        for k in (
            "debt",
            "credit card",
            "card balance",
            "loan",
            "payoff",
            "pay off",
            "owe",
            "balance",
            "apr",
        )
    )


def _parse_currency_amount(text: str) -> float | None:
    """First plausible currency-like number in free text (USD-style and plain numbers)."""
    if not text:
        return None
    m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)\s*k\b", text, re.I)
    if m:
        try:
            return float(m.group(1).replace(",", "")) * 1000
        except ValueError:
            pass
    plain = text.replace(",", "")
    m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", plain)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _all_currency_amounts(text: str) -> list[float]:
    """Collect plausible money amounts; used to pick a conservative total-debt hint."""
    if not text:
        return []
    found: list[float] = []
    for m in re.finditer(r"\$?\s*([\d,]+(?:\.\d+)?)\s*k\b", text, re.I):
        try:
            found.append(float(m.group(1).replace(",", "")) * 1000)
        except ValueError:
            continue
    plain = text.replace(",", "")
    for m in re.finditer(r"\$?\s*([\d,]+(?:\.\d+)?)\b", plain):
        try:
            v = float(m.group(1).replace(",", ""))
            if v > 0:
                found.append(v)
        except ValueError:
            continue
    return found


def _extract_total_debt(slots: dict[str, Any]) -> float | None:
    task = slots.get("task_definition") or {}
    personal = slots.get("personal_summary") or {}
    req = slots.get("financial_requirements") or {}
    combined = " ".join(
        [
            str(task.get("summary") or ""),
            str(task.get("goal") or ""),
            str(personal.get("notes") or ""),
            str(req.get("constraints") or ""),
        ]
    )
    if not _mentions_debt_context(combined):
        return None
    amounts = _all_currency_amounts(combined)
    if not amounts:
        return None
    return max(amounts)


def _parse_monthly_budget(hint: str) -> float | None:
    return _parse_currency_amount(hint or "")


def _parse_horizon_months(text: str) -> int | None:
    if not text:
        return None
    t = text.strip().lower()
    if t.isdigit():
        n = int(t)
        return n if 1 <= n <= 360 else None
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*(?:month|months|mos?)\b", text, re.I)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= 360 and 1 <= b <= 360:
            return int(round((a + b) / 2))
    m = re.search(r"(\d+)\s*(?:month|months|mos?)\b", text, re.I)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 360 else None
    m = re.search(r"(\d+)\s*(?:year|years)\b", text, re.I)
    if m:
        n = int(m.group(1)) * 12
        return n if 1 <= n <= 360 else None
    return None


def compute_debt_payoff_metrics(slots: dict[str, Any]) -> DebtPayoffMetrics:
    """
    Compute simplified metrics from collected slot dict (session context shape).
    Missing inputs yield None fields; never guesses hidden balances.
    """
    req = slots.get("financial_requirements") or {}
    liq = (req.get("liquidity_needs") or "").strip() or None

    total = _extract_total_debt(slots)
    budget = _parse_monthly_budget(str(req.get("monthly_budget_hint") or ""))
    horizon = _parse_horizon_months(str(req.get("time_horizon_months") or ""))

    required: float | None = None
    if total is not None and horizon is not None and horizon > 0:
        required = total / float(horizon)

    est_months: float | None = None
    if total is not None and budget is not None and budget > 0:
        est_months = total / budget

    feasible: bool | None = None
    gap: float | None = None
    if required is not None and budget is not None:
        gap = budget - required
        feasible = budget + 1e-9 >= required

    methodology = (
        "Principal-only simplified math: required_monthly_payment = total_debt / horizon_months; "
        "estimated_payoff_months = total_debt / monthly_budget_amount; "
        "interest and minimums are not modeled."
    )

    return DebtPayoffMetrics(
        total_debt=total,
        monthly_budget_amount=budget,
        horizon_months=horizon,
        liquidity_needs=liq,
        required_monthly_payment=required,
        estimated_payoff_months=est_months,
        feasible_with_current_budget=feasible,
        shortfall_or_surplus=gap,
        methodology=methodology,
    )


def debt_metrics_to_prompt_dict(m: DebtPayoffMetrics) -> dict[str, Any]:
    """JSON-friendly dict with nulls for unknowns (Jinja / model consumption)."""

    def _num(x: float | None) -> float | None:
        if x is None:
            return None
        return round(x, 2)

    return {
        "total_debt": _num(m.total_debt),
        "monthly_budget_amount": _num(m.monthly_budget_amount),
        "horizon_months": m.horizon_months,
        "liquidity_needs": m.liquidity_needs,
        "required_monthly_payment": _num(m.required_monthly_payment),
        "estimated_payoff_months": _num(m.estimated_payoff_months),
        "feasible_with_current_budget": m.feasible_with_current_budget,
        "shortfall_or_surplus": _num(m.shortfall_or_surplus),
        "methodology": m.methodology,
    }


def build_recommendation_calc_payload(slots: dict[str, Any]) -> dict[str, Any]:
    """Single entry point for recommendation prompt wiring."""
    return debt_metrics_to_prompt_dict(compute_debt_payoff_metrics(slots))


def debt_metrics_as_json(slots: dict[str, Any], *, compact: bool = True) -> str:
    import json

    payload = build_recommendation_calc_payload(slots)
    if compact:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "DebtPayoffMetrics",
    "compute_debt_payoff_metrics",
    "build_recommendation_calc_payload",
    "debt_metrics_as_json",
]
