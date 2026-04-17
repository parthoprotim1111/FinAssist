from __future__ import annotations

import json
import re
from typing import Any

from dialogue.slot_tracking import deep_merge
from finassist.schemas import REQUIRED_RECOMMENDATION_DISCLAIMER
from llm.backend_base import GenerationResult, LLMBackend


class MockLLMBackend(LLMBackend):
    """Deterministic backend: heuristic slot extraction + slot-aware mock recommendations."""

    name = "mock"

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.2,
        max_input_tokens: int | None = None,
        truncation_side: str | None = None,
    ) -> GenerationResult:
        if "User context (JSON):" in prompt or (
            "assumptions" in prompt.lower() and "recommendations" in prompt.lower()
        ):
            return GenerationResult(text=_mock_recommendation_json(prompt))
        if (
            "Current dialogue phase:" in prompt
            or "Phase:" in prompt
            or "slot_updates" in prompt.lower()
        ):
            return GenerationResult(text=_mock_slot_extraction(prompt))
        if "extract" in prompt.lower() or "json" in prompt.lower():
            return GenerationResult(text=_mock_slot_extraction(prompt))
        if "recommendation" in prompt.lower() or "recommend" in prompt.lower():
            return GenerationResult(text=_mock_recommendation_json(prompt))
        return GenerationResult(
            text="I can help with that. Please share a bit more detail."
        )


def _user_from_prompt(prompt: str) -> str:
    patterns = (
        r'User message \(extract from this[^)]*\):\s*"""(.*?)"""',
        r'User message:\s*"""(.*?)"""',
        r'User said:\s*"""(.*?)"""',
    )
    for pat in patterns:
        m = re.search(pat, prompt, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _is_greeting_or_trivial(text: str) -> bool:
    t = text.strip().lower().rstrip("!?.")
    if len(t) < 2:
        return True
    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "hii",
        "yo",
    }
    return t in greetings


def _task_section_filled(merged: dict[str, Any]) -> bool:
    td = merged.get("task_definition") or {}
    return bool(str(td.get("summary", "")).strip() and str(td.get("goal", "")).strip())


def _mock_slot_extraction(prompt: str) -> str:
    """Heuristic global extraction: infer all supported fields every turn (phase ignored)."""
    user = _user_from_prompt(prompt)
    if _is_greeting_or_trivial(user):
        return json.dumps({"slot_updates": {}})

    merged: dict[str, Any] = {}
    for part in (
        _extract_personal(user),
        _extract_requirements(user),
        _extract_preferences(user, fallback_full_text=False),
    ):
        if part:
            merged = deep_merge(merged, part)

    if not _task_section_filled(merged):
        snippet = user.strip()[:500]
        task_part = _extract_task(snippet)
        if task_part:
            merged = deep_merge(merged, task_part)

    return json.dumps({"slot_updates": merged})


def _extract_task(text: str) -> dict[str, Any]:
    t = text.strip()
    if len(t) < 8:
        return {}
    parts = re.split(r"[\n\.]+", t, maxsplit=1)
    if len(parts) == 2 and len(parts[1].strip()) > 5:
        return {
            "task_definition": {
                "summary": parts[0].strip()[:200],
                "goal": parts[1].strip()[:400],
            }
        }
    if "," in t and len(t) > 40:
        a, b = t.split(",", 1)
        if len(a.strip()) > 5 and len(b.strip()) > 5:
            return {
                "task_definition": {
                    "summary": a.strip()[:200],
                    "goal": b.strip()[:400],
                }
            }
    mid = max(len(t) // 2, 20)
    return {
        "task_definition": {
            "summary": t[:mid].strip()[:200],
            "goal": t[mid:].strip()[:400] if len(t) > mid else "",
        }
    }


def _extract_personal(text: str) -> dict[str, Any]:
    low = text.lower()
    out: dict[str, str] = {}
    if re.search(r"\d{2}\s*[-–]\s*\d{2}|age\s*\d+|in my\s+\d", low):
        m = re.search(r"(\d{2}\s*[-–]\s*\d{2}|age\s*\d+|(?:early|mid|late)\s+\d{2}s)", low)
        if m:
            out["age_range"] = m.group(1).strip()
    for word in ("employed", "student", "self-employed", "self employed", "retired", "unemployed"):
        if word in low:
            out["employment_status"] = word.replace("-", " ")
            break
    if "risk" in low and "medium" in low:
        out["risk_tolerance"] = "medium"
    elif "risk" in low and "low" in low:
        out["risk_tolerance"] = "low"
    elif "risk" in low and "high" in low:
        out["risk_tolerance"] = "high"
    elif "medium" in low:
        out["risk_tolerance"] = "medium"
    elif "low risk" in low:
        out["risk_tolerance"] = "low"
    elif "high risk" in low:
        out["risk_tolerance"] = "high"
    for country in ("india", "usa", "u.s.", "uk", "canada", "germany"):
        if country in low:
            out["country_region"] = country.title()
            break
    if "dependent" in low or "child" in low or "kids" in low:
        if "no " in low or "none" in low or "0" in low:
            out["dependents"] = "0"
        else:
            out["dependents"] = "1+"
    if len(text.strip()) > 30 and not out:
        out["notes"] = text.strip()[:400]
    elif len(text.strip()) > 30:
        out["notes"] = text.strip()[:400]
    if not out:
        return {}
    return {"personal_summary": out}


def _extract_requirements(text: str) -> dict[str, Any]:
    low = text.lower()
    out: dict[str, str] = {}
    if re.search(r"\d+\s*(month|year|yr|mo)", low):
        m = re.search(r"(\d{1,4})\s*(month|year|yr|mo)", low)
        if m:
            out["time_horizon_months"] = m.group(0)
    if "month" in low and re.search(r"\d+", text):
        m = re.search(r"\$?\d[\d,]*", text)
        if m:
            out["monthly_budget_hint"] = m.group(0)
    for liq in ("high liquidity", "need cash", "liquid", "emergency access"):
        if liq in low:
            out["liquidity_needs"] = "high"
            break
    if "liquidity_needs" not in out and ("low" in low and "liquidity" in low):
        out["liquidity_needs"] = "low"
    if "liquidity_needs" not in out and len(text) > 40:
        out["liquidity_needs"] = "medium"
    if "no crypto" in low or "no cryptocurrency" in low:
        out["constraints"] = "no crypto"
    if not out:
        return {}
    return {"financial_requirements": out}


def _extract_preferences(
    text: str, *, fallback_full_text: bool = False
) -> dict[str, Any]:
    low = text.lower()
    out: dict[str, str] = {}
    if any(x in low for x in ("index", "etf", "mutual fund", "savings", "fd ", "bond")):
        out["product_preferences"] = text.strip()[:300]
    if any(x in low for x in ("esg", "ethical", "tobacco", "fossil")):
        out["ethical_constraints"] = text.strip()[:300]
    if "auto" in low or "automate" in low:
        out["automation_comfort"] = "high"
    elif "manual" in low:
        out["automation_comfort"] = "low"
    if not out:
        if fallback_full_text:
            out["product_preferences"] = text.strip()[:300]
        else:
            return {}
    return {"financial_preferences": out}


def _parse_slots_from_recommend_prompt(prompt: str) -> dict[str, Any] | None:
    """Parse the first JSON object after the profile block (not later schema text)."""
    markers = ("User profile (JSON):", "User context (JSON):")
    i = -1
    marker = ""
    for m in markers:
        i = prompt.find(m)
        if i >= 0:
            marker = m
            break
    if i < 0:
        return None
    rest = prompt[i + len(marker) :].lstrip()
    if not rest.startswith("{"):
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(rest)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _is_debt_focus(goal: str, summary: str) -> bool:
    t = f"{goal} {summary}".lower()
    return any(
        k in t
        for k in (
            "debt",
            "credit card",
            "card",
            "loan",
            "payoff",
            "pay off",
            "balance",
            "apr",
            "interest",
            "owe",
        )
    )


def _fmt_or_missing(val: str, label: str) -> str:
    v = (val or "").strip()
    return v if v else f"({label} not provided in dialogue)"


def _first_currency_number(text: str) -> float | None:
    """Extract first plausible currency amount from free-form text."""
    if not text:
        return None
    m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)\s*k\b", text, re.I)
    if m:
        try:
            return float(m.group(1).replace(",", "")) * 1000
        except ValueError:
            pass
    m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", text.replace(",", ""))
    if not m:
        m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _first_int_months(text: str) -> int | None:
    if not text:
        return None
    t = text.strip()
    if t.isdigit():
        n = int(t)
        return n if 1 <= n <= 120 else None
    m = re.search(r"(\d+)\s*(?:month|months|mo)\b", text, re.I)
    if m:
        try:
            n = int(m.group(1))
            return n if 1 <= n <= 120 else None
        except ValueError:
            pass
    return None


def _mock_recommendation_json(prompt: str = "") -> str:
    ctx = _parse_slots_from_recommend_prompt(prompt) or {}
    task = ctx.get("task_definition", {}) or {}
    personal = ctx.get("personal_summary", {}) or {}
    req = ctx.get("financial_requirements", {}) or {}
    pref = ctx.get("financial_preferences", {}) or {}

    summary = (task.get("summary") or "").strip() or "your stated task"
    goal = (task.get("goal") or "").strip() or "your stated outcome"
    risk_raw = (personal.get("risk_tolerance") or "").strip()
    risk = risk_raw if risk_raw else _fmt_or_missing("", "Risk tolerance")
    employment = (personal.get("employment_status") or "").strip()
    dependents = (personal.get("dependents") or "").strip()
    horizon = _fmt_or_missing(req.get("time_horizon_months", ""), "Time horizon")
    liq = _fmt_or_missing(req.get("liquidity_needs", ""), "Liquidity")
    budget = _fmt_or_missing(req.get("monthly_budget_hint", ""), "Monthly budget")
    prefs = _fmt_or_missing(pref.get("product_preferences", ""), "Product preferences")
    constraints = _fmt_or_missing(req.get("constraints", ""), "Constraints")

    debt = _is_debt_focus(goal, summary)
    combined_goal = f"{goal} {summary}"
    debt_guess = _first_currency_number(combined_goal)
    horizon_m = _first_int_months(req.get("time_horizon_months") or "") or _first_int_months(
        combined_goal
    )
    budget_amt = _first_currency_number(req.get("monthly_budget_hint") or "")

    def _self_employed_note() -> str:
        el = employment.lower()
        if re.search(r"\bself[- ]?employed\b", el) or any(
            w in el
            for w in (
                "freelance",
                "gig worker",
                "gig economy",
                "1099",
                "sole proprietor",
                "business owner",
                "consultant",
                "independent contractor",
            )
        ):
            return (
                " Irregular income: keep a larger cash buffer than a salaried peer; do not exceed "
                f"your stated risk tolerance ({risk_raw or risk})."
            )
        return ""

    def _dependents_note() -> str:
        if not dependents:
            return ""
        return (
            f" With dependents ({dependents}), prioritize a realistic emergency buffer before "
            "maximum aggression on debt."
        )

    if debt:
        num_line = ""
        if debt_guess and horizon_m:
            floor = debt_guess / horizon_m
            num_line = (
                f" Rough average principal needed ≈ ${debt_guess:,.0f} ÷ {horizon_m} mo ≈ ${floor:,.0f}/mo "
                f"(interest makes the true requirement higher). "
            )
            if budget_amt is not None:
                if budget_amt + 1e-6 >= floor:
                    num_line += (
                        f"A ${budget_amt:,.0f}/mo budget toward debt can cover that principal floor "
                        "(add margin for interest and minimums on other debts). "
                    )
                else:
                    num_line += (
                        f"A ${budget_amt:,.0f}/mo budget looks tight vs ~${floor:,.0f}/mo principal—"
                        "extend the horizon, raise extra payment capacity, or revisit the goal. "
                    )
            if horizon_m and 12 <= horizon_m <= 18:
                num_line += f"A {horizon_m}-month horizon sits in the 12–18 month band; feasibility depends on sticking to the payment plan. "
        elif horizon_m:
            num_line = (
                f" With a {horizon_m}-month horizon, estimate monthly principal as (total debt ÷ {horizon_m}); "
                "compare to your monthly budget hint. "
            )

        strategy = (
            f"Debt avalanche: pay highest APR first after minimums everywhere. "
            f"{num_line}"
            f"Your stated risk tolerance is {risk_raw or risk} — avoid new investment risk until "
            f"high-APR revolving debt is controlled. Horizon context: {horizon}. "
            f"Liquidity: {liq} — keep required buffers before extra payoff.{_self_employed_note()}"
            f"{_dependents_note()}"
        )
        steps = [
            f"Month 0: list every card/loan with balance, APR, minimum; goal check: {goal}.",
            f"Months 1–3: avalanche — route the budget hint ({budget}) to highest APR after minimums; review at month 3.",
            f"Months 4–6: roll freed cash forward; quarterly checkpoint; constraints: {constraints}.",
            f"Ongoing: freeze new card spending; autopay minimums; if income shifts"
            f"{' (' + employment + ')' if employment else ''}, revise the monthly extra.",
            f"If eligible, compare balance-transfer offers (fee vs promo length) only if payoff before promo end is realistic.",
        ]
        dep_tail = ""
        if dependents:
            dep_tail = f" Dependents ({dependents}) add fixed costs—keep an emergency buffer."
        risk_notes = [
            f"Risk tolerance is {risk_raw or risk}: variable income or rate hikes can delay payoff—keep liquidity per {liq}.",
            "Balance transfers and consolidation loans carry fees; new spending undoes progress." + dep_tail,
        ]
        alt = (
            "**Snowball method:** pay smallest balances first for psychological wins. "
            "It may cost more interest than avalanche but can work better if motivation is the bottleneck."
        )
    else:
        dep_sav = ""
        if dependents:
            dep_sav = (
                f" With dependents ({dependents}), keep a larger emergency buffer before taking "
                f"investment risk; risk tolerance stays **{risk_raw or risk}**."
            )
        strategy = (
            f"Align saving and investing with **{goal}** over **{horizon}**, respecting stated risk tolerance **{risk_raw or risk}** "
            f"and **liquidity ({liq})**. Use **{prefs}** as a filter for product selection. "
            f"Budget hint **{budget}** informs monthly contribution size; constraints: **{constraints}**."
            f"{_self_employed_note()}{dep_sav}"
        )
        steps = [
            f"Quantify the target implied by: {goal}; split into monthly contributions over {horizon}.",
            f"Months 1–3: allocate cash to liquidity ({liq}), then goal funding per **{risk_raw or risk}** risk.",
            f"Month 3 checkpoint: compare actual savings rate to budget hint ({budget}); adjust automation.",
            "Automate transfers on payday where comfortable; review quarterly.",
            f"Shortlist products matching preferences ({prefs}) and verify fees/taxes in your region.",
        ]
        risk_notes = [
            f"Market and inflation risk affect long-horizon plans; **{risk}** profile should match asset mix.",
            "Rules and tax treatment depend on country and account type—verify locally.",
        ]
        alt = (
            "A more conservative path: increase cash and short-duration savings first, then invest "
            "once the goal’s minimum liquidity bar is met—useful if income is uncertain."
        )

    user_summary = (
        f"You want **{summary}**, with goal **{goal}**. "
        f"Profile: risk **{risk}**, horizon **{horizon}**, liquidity **{liq}**, "
        f"budget hint **{budget}**, preferences **{prefs}**."
    )

    payload = {
        "user_summary": user_summary,
        "main_goal": (
            f"Primary objective from the dialogue: **{goal}**, in the context of **{summary}**. "
            "All steps below should advance this goal—not a substitute goal."
        ),
        "recommended_strategy": strategy,
        "step_by_step_plan": steps,
        "risk_notes": risk_notes,
        "alternative_option": alt,
        "assumptions": [
            f"Figures and labels you provided (risk: {risk}; horizon: {horizon}) are directionally accurate.",
            "Tax rules, credit products, and protections depend on your country—verify before acting.",
        ],
        "summary": f"Plan centered on: {goal} ({risk} risk, {horizon} horizon).",
        "recommendations": [],
        "disclaimer": REQUIRED_RECOMMENDATION_DISCLAIMER,
    }
    return json.dumps(payload)
