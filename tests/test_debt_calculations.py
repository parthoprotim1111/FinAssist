from finassist.debt_calculations import build_recommendation_calc_payload, compute_debt_payoff_metrics


def test_debt_metrics_full_inputs():
    slots = {
        "task_definition": {
            "summary": "Credit cards",
            "goal": "Pay off total balance of $10,800",
        },
        "personal_summary": {},
        "financial_requirements": {
            "monthly_budget_hint": "$600 per month",
            "time_horizon_months": "18 months",
            "liquidity_needs": "medium",
            "constraints": "",
        },
        "financial_preferences": {},
    }
    m = compute_debt_payoff_metrics(slots)
    assert m.total_debt == 10800.0
    assert m.monthly_budget_amount == 600.0
    assert m.horizon_months == 18
    assert m.required_monthly_payment is not None and abs(m.required_monthly_payment - 600.0) < 0.01
    assert m.estimated_payoff_months is not None and abs(m.estimated_payoff_months - 18.0) < 0.01
    assert m.feasible_with_current_budget is True
    assert m.shortfall_or_surplus is not None and abs(m.shortfall_or_surplus) < 0.01


def test_debt_metrics_missing_debt_returns_nulls():
    slots = {
        "task_definition": {"summary": "Savings", "goal": "Build emergency fund"},
        "personal_summary": {},
        "financial_requirements": {
            "monthly_budget_hint": "500 per month",
            "time_horizon_months": "12 months",
            "liquidity_needs": "high",
            "constraints": "",
        },
        "financial_preferences": {},
    }
    d = build_recommendation_calc_payload(slots)
    assert d["total_debt"] is None
    assert d["required_monthly_payment"] is None


def test_payload_json_serializable():
    slots = {
        "task_definition": {"summary": "Debt", "goal": "Pay $5k on cards"},
        "personal_summary": {},
        "financial_requirements": {
            "monthly_budget_hint": "",
            "time_horizon_months": "10 months",
            "liquidity_needs": "",
            "constraints": "",
        },
        "financial_preferences": {},
    }
    d = build_recommendation_calc_payload(slots)
    assert d["total_debt"] == 5000.0
    assert d["monthly_budget_amount"] is None
    assert d["feasible_with_current_budget"] is None
