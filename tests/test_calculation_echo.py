"""Tests for deterministic echo of Python debt metrics into recommendation prose."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from finassist.calculation_echo import enrich_recommendation_with_calc
from finassist.debt_calculations import build_recommendation_calc_payload
from finassist.justification import ensure_justification_fields
from finassist.schemas import REQUIRED_RECOMMENDATION_DISCLAIMER, FinancialRecommendation


def _debt_slots() -> dict:
    return {
        "task_definition": {"summary": "Credit cards", "goal": "Pay off $6000 in 12 months"},
        "personal_summary": {},
        "financial_requirements": {
            "monthly_budget_hint": "$200 per month",
            "time_horizon_months": "12 months",
        },
        "financial_preferences": {},
    }


def _minimal_rec(*, strategy: str, risk1: str = "Income may vary.", risk2: str = "Cash flow pressure.") -> FinancialRecommendation:
    return FinancialRecommendation(
        user_summary="Summary.",
        main_goal="Pay down debt.",
        recommended_strategy=strategy,
        step_by_step_plan=["a", "b", "c", "d"],
        risk_notes=[risk1, risk2],
        alternative_option="Alt.",
        assumptions=["Unknown future expenses.", "Unknown rate changes."],
        summary="Headline.",
        recommendations=[],
        disclaimer="Educational demonstration only — not personalized financial advice.",
    )


def test_enrich_synthesizes_main_goal_one_sentence_not_verbatim_task_goal():
    slots = _debt_slots()
    rec = _minimal_rec(strategy="Debt avalanche: pay highest APR after minimums everywhere.")
    out = enrich_recommendation_with_calc(rec, slots)
    assert out.main_goal.count(".") <= 1
    assert out.main_goal.lower().startswith("the user seeks to retire approximately")
    raw_goal = (slots["task_definition"]["goal"] or "").strip().lower()
    assert out.main_goal.strip().lower() != raw_goal


def test_ensure_justification_replaces_nonstandard_disclaimer():
    raw = """{
      "user_summary": "u",
      "main_goal": "g",
      "recommended_strategy": "s.",
      "step_by_step_plan": ["a", "b", "c", "d"],
      "risk_notes": ["r1", "r2"],
      "alternative_option": "x",
      "assumptions": ["y1", "y2"],
      "summary": "z",
      "recommendations": [],
      "disclaimer": "Consult a licensed advisor only."
    }"""
    rec, _ = ensure_justification_fields(raw, slots=None)
    assert rec is not None
    assert rec.disclaimer == REQUIRED_RECOMMENDATION_DISCLAIMER


def test_enrich_appends_block_when_strategy_omits_metrics():
    slots = _debt_slots()
    rec = _minimal_rec(strategy="Debt avalanche: pay highest APR after minimums everywhere.")
    out = enrich_recommendation_with_calc(rec, slots)
    assert "Simplified feasibility" not in out.recommended_strategy
    assert "principal-only:" not in out.recommended_strategy.lower()
    assert "simplified" in out.recommended_strategy.lower() or "500" in out.recommended_strategy.replace(
        ",", ""
    )
    assert "Deterministic metrics" not in " ".join(out.risk_notes)
    payload = build_recommendation_calc_payload(slots)
    assert payload["required_monthly_payment"] is not None
    assert str(int(payload["required_monthly_payment"])) in out.recommended_strategy.replace(",", "") or str(
        payload["required_monthly_payment"]
    ) in out.recommended_strategy
    assert len(out.risk_notes) == 2
    joined_risk = " ".join(out.risk_notes).lower()
    assert "insufficient" in joined_risk or "payment" in joined_risk
    assert "interest" in joined_risk or "income" in joined_risk


def test_enrich_no_extra_block_when_strategy_and_risk_already_cover_metrics():
    slots = _debt_slots()
    payload = build_recommendation_calc_payload(slots)
    rmp = payload["required_monthly_payment"]
    est = payload["estimated_payoff_months"]
    assert rmp is not None and est is not None
    strategy = (
        f"Avalanche after minimums. Principal floor requires {rmp:.2f} per month; "
        f"not achievable with current budget. Estimated payoff at {est:.2f} months."
    )
    risk2 = (
        f"Shortfall: payoff goal is not achievable; "
        f"metrics show ~{rmp:.0f}/mo required and ~{est:.0f} months; interest not modeled."
    )
    rec = _minimal_rec(strategy=strategy, risk2=risk2)
    before_s = rec.recommended_strategy
    before_r = list(rec.risk_notes)
    out = enrich_recommendation_with_calc(rec, slots)
    assert "Simplified feasibility (principal-only):" not in out.recommended_strategy
    assert out.recommended_strategy == before_s
    assert out.risk_notes == before_r


def test_enrich_replaces_risk_when_second_bullet_lacks_interest_theme():
    slots = _debt_slots()
    payload = build_recommendation_calc_payload(slots)
    rmp = payload["required_monthly_payment"]
    est = payload["estimated_payoff_months"]
    strategy = (
        f"Avalanche after minimums. Required principal floor {rmp:.2f}/mo; "
        f"estimated payoff {est:.2f} months; not achievable with stated budget."
    )
    rec = _minimal_rec(strategy=strategy, risk2="Dependents add pressure.")
    out = enrich_recommendation_with_calc(rec, slots)
    joined = " ".join(out.risk_notes).lower()
    assert "interest" in joined or "income" in joined
    assert "Deterministic metrics" not in joined


def test_infeasibility_phrase_appended_when_missing():
    slots = _debt_slots()
    rec = _minimal_rec(
        strategy="Use avalanche: pay minimums then target highest APR.",
        risk1="x",
        risk2="Interest accrual is a risk outside the simplified model.",
    )
    out = enrich_recommendation_with_calc(rec, slots)
    low = out.recommended_strategy.lower()
    assert (
        "not achievable" in low
        or "not feasible" in low
        or "shortfall" in low
        or "short about" in low
        or "insufficient" in low
    )


def test_ensure_justification_with_slots_runs_enrich():
    slots = _debt_slots()
    raw = """{
      "user_summary": "u",
      "main_goal": "g",
      "recommended_strategy": "Only generic avalanche advice.",
      "step_by_step_plan": ["a", "b", "c", "d"],
      "risk_notes": ["r1", "r2"],
      "alternative_option": "x",
      "assumptions": ["y1", "y2"],
      "summary": "z",
      "recommendations": [],
      "disclaimer": "Educational demonstration only — not personalized financial advice."
    }"""
    rec, issues = ensure_justification_fields(raw, slots=slots)
    assert rec is not None
    assert "principal floor" in rec.recommended_strategy.lower() or "500" in rec.recommended_strategy.replace(
        ",",
        "",
    )
    assert "Simplified feasibility" not in rec.recommended_strategy
    assert not issues


def test_recommend_jinja_includes_must_blocks_for_calc_fields():
    root = Path(__file__).resolve().parents[1] / "src" / "llm" / "prompts"
    env = Environment(loader=FileSystemLoader(str(root)))
    tmpl = env.get_template("recommend.jinja2")
    calc = {
        "required_monthly_payment": 500.0,
        "estimated_payoff_months": 30.0,
        "feasible_with_current_budget": False,
        "total_debt": 6000.0,
        "monthly_budget_amount": 200.0,
        "horizon_months": 12,
        "liquidity_needs": None,
        "shortfall_or_surplus": -300.0,
        "methodology": "test",
    }
    out = tmpl.render(
        slots={
            "task_definition": {"summary": "", "goal": ""},
            "personal_summary": {},
            "financial_requirements": {},
            "financial_preferences": {},
        },
        slots_json="{}",
        calc_json="{}",
        calc=calc,
        language="English",
        locale_prompt="",
    )
    assert "required_monthly_payment is 500.0" in out
    assert "MUST" in out
    assert "not achievable" in out.lower()
    assert "risk_notes MUST NOT repeat" in out
