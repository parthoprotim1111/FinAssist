from finassist.schemas import FinancialRecommendation, parse_recommendation_json
from finassist.justification import ensure_justification_fields


def test_parse_recommendation():
    raw = """{"summary":"x","assumptions":["a"],"recommendations":[{"title":"t","rationale":"r","caveats":"c","suggested_next_steps":["n"]}],"disclaimer":"d"}"""
    rec = parse_recommendation_json(raw)
    assert rec is not None
    assert isinstance(rec, FinancialRecommendation)


def test_parse_fenced_json_with_preamble():
    raw = """Here you go:
```json
{"user_summary":"u","main_goal":"g","recommended_strategy":"s","step_by_step_plan":["a","b"],"risk_notes":["r"],"alternative_option":"x","assumptions":["y"],"summary":"z","recommendations":[],"disclaimer":"Educational demonstration only — not personalized financial advice."}
```
"""
    rec = parse_recommendation_json(raw)
    assert rec is not None
    assert rec.user_summary == "u"


def test_parse_extracts_json_after_preamble():
    """CoT-style text before the JSON object."""
    raw = """Let me think through constraints first.

{"user_summary":"u","main_goal":"g","recommended_strategy":"s","step_by_step_plan":["a","b"],"risk_notes":["r"],"alternative_option":"x","assumptions":["y"],"summary":"z","recommendations":[],"disclaimer":"Educational demonstration only — not personalized financial advice."}"""
    rec = parse_recommendation_json(raw)
    assert rec is not None
    assert rec.user_summary == "u"


def test_parse_accepts_jsonc_line_comments_outside_strings():
    """Models often emit // labels between keys; json.loads rejects unless comments are stripped."""
    raw = r"""
{
  "user_summary": "The user has $10,000 in credit card debt.",
  "main_goal": "$10,000 credit card debt paid off within 12 months",
  "recommended_strategy": "Prioritize actions:",

  // Step-by-step Plan
  "step_by_step_plan": [
    "List debts.",
    "Pay minimums.",
    "Allocate to highest APR.",
    "Review monthly."
  ],

  // Risk Notes
  "risk_notes": [
    "Payment may be tight.",
    "Interest extends duration."
  ],

  "alternative_option": "Increase payments if possible.",
  "assumptions": [
    "Interest not modeled.",
    "Budget sustainable."
  ],
  "summary": "Summary line.",
  "recommendations": [],
  "disclaimer": "Educational demonstration only — not personalized financial advice."
}
"""
    rec = parse_recommendation_json(raw)
    assert rec is not None
    assert len(rec.step_by_step_plan) == 4
    assert rec.main_goal.strip().startswith("$10,000")


def test_parse_accepts_trailing_commas_common_llm_mistake():
    raw = """{"user_summary":"u","main_goal":"g","recommended_strategy":"s","step_by_step_plan":["a","b",],"risk_notes":["r"],"alternative_option":"x","assumptions":["y"],"summary":"z","recommendations":[],"disclaimer":"Educational demonstration only — not personalized financial advice.",}"""
    rec = parse_recommendation_json(raw)
    assert rec is not None
    assert rec.user_summary == "u"


def test_parse_with_markdown_fence_and_string_risk_notes():
    raw = '''```json
{
  "user_summary": "u",
  "main_goal": "g",
  "recommended_strategy": "s",
  "step_by_step_plan": ["a", "b"],
  "risk_notes": "single string note",
  "alternative_option": "alt",
  "assumptions": ["x"],
  "summary": "head",
  "recommendations": [],
  "disclaimer": "Educational demonstration only — not personalized financial advice."
}
```'''
    rec = parse_recommendation_json(raw)
    assert rec is not None
    assert rec.risk_notes == ["single string note"]


def test_ensure_justification_compresses_plan_to_four_steps():
    raw = """{
      "user_summary":"u",
      "main_goal":"g",
      "recommended_strategy":"s",
      "step_by_step_plan":["a","b","c","d","e"],
      "risk_notes":["r1","r2"],
      "alternative_option":"x",
      "assumptions":["y1","y2"],
      "summary":"z",
      "recommendations":[],
      "disclaimer":"Educational demonstration only — not personalized financial advice."
    }"""
    rec, issues = ensure_justification_fields(raw)
    assert rec is not None
    assert len(rec.step_by_step_plan) == 4
    assert "e" in rec.step_by_step_plan[-1]
    assert not issues
