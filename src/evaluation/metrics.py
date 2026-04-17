from __future__ import annotations

import time
from typing import Any

from finassist.justification import ensure_justification_fields
from finassist.schemas import parse_recommendation_json


def score_output(
    raw_text: str,
    *,
    latency_s: float,
    weights: dict[str, float],
    slots: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec, issues = ensure_justification_fields(raw_text, slots=slots)
    schema_ok = rec is not None and parse_recommendation_json(raw_text) is not None
    has_assumptions = bool(rec and rec.assumptions)
    has_caveat = bool(
        rec
        and (
            any(r.caveats for r in rec.recommendations)
            or bool(rec.risk_notes)
        )
    )
    score = 0.0
    if schema_ok:
        score += weights.get("schema_valid_weight", 1.0)
    if has_assumptions:
        score += weights.get("has_assumptions_weight", 0.5)
    if has_caveat:
        score += weights.get("has_caveat_weight", 0.5)
    return {
        "schema_valid": schema_ok,
        "has_assumptions": has_assumptions,
        "has_caveat": has_caveat,
        "score": score,
        "latency_s": latency_s,
        "issues": issues,
    }


def timed_generate(fn, *args, **kwargs) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
