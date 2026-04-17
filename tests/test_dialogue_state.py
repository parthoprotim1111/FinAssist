from dialogue.slots import DialogueState
from dialogue.state_machine import DialogueEngine
from llm.mock_backend import MockLLMBackend
from dialogue.validators import compute_state_from_slots, requirements_filled


def test_global_extraction_advances_multiple_phases_one_turn():
    """Mock extracts all groups from one message; state should jump to RECOMMEND."""
    eng = DialogueEngine(MockLLMBackend())
    msg = (
        "Emergency fund for 3 months, age 25-34 employed India no kids, risk medium, "
        "horizon 12 months high liquidity 5000 monthly no crypto, index funds"
    )
    _, ok = eng.process_user_message(msg, "Respond in clear, professional English.")
    assert ok
    assert eng.state == DialogueState.RECOMMEND
    assert eng.last_turn_updated_paths
    assert "financial_requirements.time_horizon_months" in eng.last_turn_updated_paths


def test_dialogue_progression_mock():
    eng = DialogueEngine(MockLLMBackend())
    assert eng.state == DialogueState.DEFINE_TASK
    msg, ok = eng.process_user_message("x", "Respond in clear, professional English.")
    assert ok
    assert eng.state != DialogueState.DONE


def test_recommendation_validate():
    eng = DialogueEngine(MockLLMBackend())
    eng.state = DialogueState.RECOMMEND
    r = eng.generate_recommendations(
        "Respond in clear, professional English.", technique="zero_shot"
    )
    assert eng.validate_recommendation_output(r.text)


def test_single_sentence_global_extraction_personal_fields():
    eng = DialogueEngine(MockLLMBackend())
    msg = (
        "I am in my late 40s, self-employed in India, with two dependents, "
        "and I prefer low-risk decisions."
    )
    _, ok = eng.process_user_message(msg, "Respond in clear, professional English.")
    assert ok
    p = eng.slots.personal_summary
    assert p.age_range == "45-49"
    assert p.employment_status == "self-employed"
    assert p.country_region == "IN"
    assert p.dependents == "2"
    assert p.risk_tolerance == "low"


def test_requirements_progress_without_explicit_liquidity():
    eng = DialogueEngine(MockLLMBackend())
    msg = (
        "Need debt payoff in 18 months with 15000 INR per month budget; "
        "liquidity needs are medium."
    )
    _, ok = eng.process_user_message(msg, "Respond in clear, professional English.")
    assert ok
    assert requirements_filled(eng.slots.financial_requirements)


def test_soft_liquidity_inference_from_user_language():
    eng = DialogueEngine(MockLLMBackend())
    msg = "I am self-employed and want to keep an emergency savings buffer."
    _, ok = eng.process_user_message(msg, "Respond in clear, professional English.")
    assert ok
    assert eng.slots.financial_requirements.liquidity_needs == "high"


def test_preferences_extracted_from_single_sentence_and_advances():
    eng = DialogueEngine(MockLLMBackend())
    eng.slots.task_definition.summary = "Credit card debt payoff"
    eng.slots.task_definition.goal = "Clear debt in 18 months"
    eng.slots.personal_summary.age_range = "45-50"
    eng.slots.personal_summary.employment_status = "self-employed"
    eng.slots.personal_summary.risk_tolerance = "low"
    eng.slots.financial_requirements.time_horizon_months = "18 months"
    eng.slots.financial_requirements.monthly_budget_hint = "15000 INR per month"
    eng.slots.financial_requirements.liquidity_needs = "medium"
    eng.state = DialogueState.PREFERENCES

    msg = (
        "I prefer safe options and savings accounts, no ethical constraints, "
        "and I will review myself with no automation."
    )
    _, ok = eng.process_user_message(msg, "Respond in clear, professional English.")
    assert ok
    prefs = eng.slots.financial_preferences
    assert "safe options" in prefs.product_preferences
    assert "savings accounts" in prefs.product_preferences
    assert prefs.ethical_constraints == "none"
    assert prefs.automation_comfort == "manual"
    assert eng.state == DialogueState.RECOMMEND


def test_monthly_budget_hint_extraction_common_phrases():
    eng = DialogueEngine(MockLLMBackend())
    msg = "I can only pay $600 monthly and need to clear debt."
    _, ok = eng.process_user_message(msg, "Respond in clear, professional English.")
    assert ok
    assert eng.slots.financial_requirements.monthly_budget_hint == "$600 per month"
    assert eng.slots.financial_requirements.liquidity_needs == "medium"


def test_monthly_budget_hint_extraction_plain_phrase():
    eng = DialogueEngine(MockLLMBackend())
    msg = "around 900 per month is possible for me."
    _, ok = eng.process_user_message(msg, "Respond in clear, professional English.")
    assert ok
    assert eng.slots.financial_requirements.monthly_budget_hint == "900 per month"


def test_requirements_not_complete_without_budget_or_liquidity():
    eng = DialogueEngine(MockLLMBackend())
    eng.slots.task_definition.summary = "Debt"
    eng.slots.task_definition.goal = "Pay off cards"
    eng.slots.personal_summary.age_range = "30-40"
    eng.slots.personal_summary.employment_status = "employed"
    eng.slots.personal_summary.risk_tolerance = "medium"
    eng.slots.financial_requirements.time_horizon_months = "18 months"
    # budget and liquidity intentionally empty
    st, reasons = compute_state_from_slots(eng.slots)
    assert st == DialogueState.REQUIREMENTS
    assert any("monthly_budget_hint" in r for r in reasons)


def test_stale_state_resyncs_from_slots():
    """If state incorrectly says PREFERENCES but requirements are incomplete, sync fixes it."""
    eng = DialogueEngine(MockLLMBackend())
    eng.slots.task_definition.summary = "Debt"
    eng.slots.task_definition.goal = "Pay off cards"
    eng.slots.personal_summary.age_range = "30-40"
    eng.slots.personal_summary.employment_status = "employed"
    eng.slots.personal_summary.risk_tolerance = "medium"
    eng.slots.financial_requirements.time_horizon_months = "12 months"
    eng.state = DialogueState.PREFERENCES
    eng.sync_state_from_slots()
    assert eng.state == DialogueState.REQUIREMENTS
