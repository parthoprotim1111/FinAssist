"""Helpers for comparing slot snapshots and listing workflow gaps."""

from __future__ import annotations

from typing import Any

from dialogue.slots import SessionSlots
from dialogue.validators import LIQUIDITY_NEEDS_OPTIONAL_FOR_REQUIREMENTS_PHASE


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; non-dict values overwrite (latest wins)."""
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def flatten_slots(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Dotted paths -> string values for diffing."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_slots(v, key))
        else:
            out[key] = "" if v is None else str(v)
    return out


def paths_changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Paths whose values differ (including new or removed keys)."""
    keys = set(before) | set(after)
    changed: list[str] = []
    for k in sorted(keys):
        if before.get(k, "") != after.get(k, ""):
            changed.append(k)
    return changed


def list_missing_required_fields(slots: SessionSlots) -> list[str]:
    """All required workflow fields that are still empty (global view, not step-local)."""
    L = {
        "summary": "task topic (summary)",
        "goal": "task goal / outcome",
        "age_range": "age range",
        "employment_status": "employment status",
        "risk_tolerance": "risk tolerance",
        "time_horizon_months": "time horizon",
        "monthly_budget_hint": "monthly budget or savings capacity",
        "liquidity_needs": "liquidity needs",
        "prefs": "product preferences or ethical constraints",
    }
    miss: list[str] = []
    t = slots.task_definition
    if not t.summary.strip():
        miss.append(L["summary"])
    if not t.goal.strip():
        miss.append(L["goal"])
    p = slots.personal_summary
    if not p.age_range.strip():
        miss.append(L["age_range"])
    if not p.employment_status.strip():
        miss.append(L["employment_status"])
    if not p.risk_tolerance.strip():
        miss.append(L["risk_tolerance"])
    r = slots.financial_requirements
    if not r.time_horizon_months.strip():
        miss.append(L["time_horizon_months"])
    if not r.monthly_budget_hint.strip():
        miss.append(L["monthly_budget_hint"])
    if not LIQUIDITY_NEEDS_OPTIONAL_FOR_REQUIREMENTS_PHASE and not r.liquidity_needs.strip():
        miss.append(L["liquidity_needs"])
    f = slots.financial_preferences
    if not (f.product_preferences.strip() or f.ethical_constraints.strip()):
        miss.append(L["prefs"])
    return miss
