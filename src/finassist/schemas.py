from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

REQUIRED_RECOMMENDATION_DISCLAIMER = (
    "Educational demonstration only — not personalized financial advice."
)


class RecommendationItem(BaseModel):
    title: str
    rationale: str
    caveats: str = ""
    suggested_next_steps: list[str] = Field(default_factory=list)


class FinancialRecommendation(BaseModel):
    """Structured recommendation output. Prefer user_summary / main_goal / plan fields."""

    model_config = ConfigDict(extra="ignore")

    summary: str = ""
    user_summary: str = ""
    main_goal: str = ""
    recommended_strategy: str = ""
    step_by_step_plan: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    alternative_option: str = ""
    assumptions: list[str] = Field(default_factory=list)
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    disclaimer: str = Field(default=REQUIRED_RECOMMENDATION_DISCLAIMER)

    @field_validator("risk_notes", mode="before")
    @classmethod
    def _risk_notes_as_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []


def _strip_markdown_json_fence(text: str) -> str:
    """Remove markdown ``` / ```json fences (outer wrappers and common inline patterns)."""
    import re

    t = text.strip()
    for _ in range(6):
        if not t.startswith("```"):
            break
        t = re.sub(r"^```(?:json)?\s*\n?", "", t, count=1, flags=re.IGNORECASE)
        t = re.sub(r"\n?```\s*$", "", t)
        t = t.strip()
    # Inline: prose then ```json ... ``` (take fenced body only)
    m = re.search(r"```(?:json)?\s*\n", t, flags=re.IGNORECASE)
    if m:
        after = t[m.end() :]
        end_fence = after.rfind("```")
        if end_fence != -1:
            t = after[:end_fence].strip()
    return t.strip()


def _outermost_brace_slice(text: str) -> str | None:
    """First `{` through last `}`; helps when the model appends prose after valid JSON."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    return text[start : end + 1]


def _first_balanced_brace_json(text: str) -> str | None:
    """Extract from first `{` to the matching `}` (strings and escapes aware). Incomplete → None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
    return None


def _validate_recommendation_dict(obj: Any) -> FinancialRecommendation | None:
    if not isinstance(obj, dict):
        return None
    try:
        return FinancialRecommendation.model_validate(obj)
    except Exception:
        return None


def _strip_json_comments(s: str) -> str:
    """
    Remove // line and /* */ block comments outside JSON strings.
    LLMs often emit JSONC-style output, which standard json.loads rejects.
    """
    out: list[str] = []
    i = 0
    n = len(s)
    in_str = False
    esc = False
    while i < n:
        c = s[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "/":
                i += 2
                while i < n and s[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                    i += 1
                if i + 1 < n:
                    i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _repair_trailing_commas(s: str) -> str:
    """Remove trailing commas before } or ] (invalid JSON but common in LLM output)."""
    prev = None
    out = s.replace("\ufeff", "")
    while prev != out:
        prev = out
        out = re.sub(r",\s*([}\]])", r"\1", out)
    return out


def _loads_and_validate_json_str(s: str) -> FinancialRecommendation | None:
    """Try json.loads on raw text, then on trailing-comma-repaired text (common LLM mistake)."""
    base = _strip_json_comments(s)
    for cand in (base, _repair_trailing_commas(base)):
        if not cand.strip():
            continue
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        rec = _validate_recommendation_dict(obj)
        if rec is not None:
            return rec
    return None


def _raw_decode_recommendation(s: str) -> FinancialRecommendation | None:
    """Parse first JSON object via JSONDecoder.raw_decode; retry with trailing-comma repair."""
    from json import JSONDecoder

    decoder = JSONDecoder()
    base = _strip_json_comments(s)
    for cand in (base, _repair_trailing_commas(base)):
        t = cand.strip()
        if not t.startswith("{"):
            continue
        try:
            obj, _ = decoder.raw_decode(t)
        except json.JSONDecodeError:
            continue
        rec = _validate_recommendation_dict(obj)
        if rec is not None:
            return rec
    return None


def parse_recommendation_json(text: str) -> FinancialRecommendation | None:
    """
    Parse a FinancialRecommendation from model output.

    Tries, in order:
    1. Strip markdown fences (``` / ```json).
    2. Slice from first `{` to last `}` (handles trailing non-JSON after the object).
    3. First balanced `{`…`}` substring + json.loads (complete object; fixes extra prose).
    4. JSONDecoder.raw_decode from start or each `{` (handles preamble).
    5. Greedy regex fallback.

    Before json.loads / raw_decode, ``//`` and ``/* */`` comments outside strings are removed
    (models often emit JSONC-like text; the standard library rejects it).
    """
    cleaned = _strip_markdown_json_fence(text.strip())

    # Fast path: outermost braces (handles trailing non-JSON prose after the closing `}`)
    outer = _outermost_brace_slice(cleaned)
    if outer:
        rec = _loads_and_validate_json_str(outer)
        if rec is not None:
            return rec

    # Balanced-brace extraction (preferred when model adds text or partial fences)
    balanced = _first_balanced_brace_json(cleaned)
    if balanced:
        rec = _loads_and_validate_json_str(balanced)
        if rec is not None:
            return rec

    # Whole string is one JSON object (allows trailing prose after first object)
    s = cleaned.strip()
    if s.startswith("{"):
        rec = _raw_decode_recommendation(s)
        if rec is not None:
            return rec

    # First valid JSON object anywhere (e.g. reasoning before `{`)
    for i, ch in enumerate(cleaned):
        if ch != "{":
            continue
        rec = _raw_decode_recommendation(cleaned[i:])
        if rec is not None:
            return rec

    # Fallback: greedy brace match
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        rec = _loads_and_validate_json_str(m.group())
        if rec is not None:
            return rec

    return None


def recommendation_to_display(rec: FinancialRecommendation) -> str:
    lines: list[str] = []

    us = rec.user_summary.strip() or rec.summary.strip()
    if us:
        lines.append("### User summary")
        lines.append(us)
        lines.append("")

    if rec.main_goal.strip():
        lines.append("### Main goal")
        lines.append(rec.main_goal.strip())
        lines.append("")

    if rec.recommended_strategy.strip():
        lines.append("### Recommended strategy")
        lines.append(rec.recommended_strategy.strip())
        lines.append("")

    if rec.step_by_step_plan:
        lines.append("### Step-by-step plan")
        for i, step in enumerate(rec.step_by_step_plan, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    if rec.risk_notes:
        lines.append("### Risk notes")
        for n in rec.risk_notes:
            lines.append(f"- {n}")
        lines.append("")

    if rec.alternative_option.strip():
        lines.append("### Alternative option")
        lines.append(rec.alternative_option.strip())
        lines.append("")

    if rec.assumptions:
        lines.append("### Assumptions")
        for a in rec.assumptions:
            lines.append(f"- {a}")
        lines.append("")

    if rec.recommendations:
        lines.append("### Additional detail")
        for i, r in enumerate(rec.recommendations, 1):
            lines.append(f"{i}. **{r.title}**")
            lines.append(f"   Why: {r.rationale}")
            if r.caveats:
                lines.append(f"   Caveats: {r.caveats}")
            if r.suggested_next_steps:
                lines.append("   Next steps: " + "; ".join(r.suggested_next_steps))
            lines.append("")

    lines.append(rec.disclaimer)
    return "\n".join(lines).strip()


def empty_stub_context() -> dict[str, Any]:
    return {
        "task_definition": {"summary": "", "goal": ""},
        "personal_summary": {},
        "financial_requirements": {},
        "financial_preferences": {},
    }
