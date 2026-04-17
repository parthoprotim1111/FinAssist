from __future__ import annotations

import json
import logging
import re
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from dialogue.slots import (
    DialogueState,
    FinancialPreferences,
    FinancialRequirements,
    PersonalInfo,
    SessionSlots,
    TaskDefinition,
)
from dialogue.validators import compute_state_from_slots
from dialogue.guided_flow import guided_reply_after_turn
from dialogue.slot_tracking import deep_merge, flatten_slots, paths_changed
from finassist.debt_calculations import build_recommendation_calc_payload, debt_metrics_as_json
from finassist.schemas import _first_balanced_brace_json, parse_recommendation_json
from llm.backend_base import LLMBackend, GenerationResult
from llm.hf_local import HFLocalBackend
from utils.config_loader import load_yaml

logger = logging.getLogger(__name__)


def _recommendation_max_new_tokens(backend: LLMBackend) -> int:
    """Longer generation for recommendation JSON (see configs/models.yaml recommendation)."""
    from llm.hf_local import HFLocalBackend

    mcfg = load_yaml("configs/models.yaml")
    rec = mcfg.get("recommendation", {})
    low = isinstance(backend, HFLocalBackend) and getattr(backend, "_low_vram", False)
    if low:
        return int(rec.get("low_vram_max_new_tokens", 512))
    return int(rec.get("max_new_tokens", 1024))


def _apply_slot_updates(slots: SessionSlots, updates: dict[str, Any]) -> None:
    def _coerce_string_values(d: dict[str, Any]) -> dict[str, Any]:
        """
        Coerce extractor outputs into string-friendly scalar values expected by slot models.
        - list/tuple/set -> ", " joined string
        - None -> ""
        - other non-dict values -> str(value)
        """
        out: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = _coerce_string_values(v)
            elif isinstance(v, (list, tuple, set)):
                out[k] = ", ".join(str(x).strip() for x in v if str(x).strip())
            elif v is None:
                out[k] = ""
            elif isinstance(v, str):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    updates = _coerce_string_values(updates)
    if "task_definition" in updates:
        slots.task_definition = TaskDefinition.model_validate(
            deep_merge(slots.task_definition.model_dump(), updates["task_definition"])
        )
    if "personal_summary" in updates:
        slots.personal_summary = PersonalInfo.model_validate(
            deep_merge(
                slots.personal_summary.model_dump(), updates["personal_summary"]
            )
        )
    if "financial_requirements" in updates:
        slots.financial_requirements = FinancialRequirements.model_validate(
            deep_merge(
                slots.financial_requirements.model_dump(),
                updates["financial_requirements"],
            )
        )
    if "financial_preferences" in updates:
        slots.financial_preferences = FinancialPreferences.model_validate(
            deep_merge(
                slots.financial_preferences.model_dump(),
                updates["financial_preferences"],
            )
        )


def _parse_extraction_json(text: str) -> dict[str, Any] | None:
    """Parse first balanced JSON object (avoids greedy-regex failures on nested braces)."""
    t = text.strip()
    slice_s = _first_balanced_brace_json(t)
    if slice_s is None:
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            return None
        slice_s = m.group()
    try:
        obj = json.loads(slice_s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _normalize_extraction_payload(parsed: dict[str, Any] | None) -> dict[str, Any]:
    """Accept both {"slot_updates": {...}} and direct group dict payloads."""
    if not isinstance(parsed, dict):
        return {}
    if isinstance(parsed.get("slot_updates"), dict):
        return parsed["slot_updates"]
    allowed = {
        "task_definition",
        "personal_summary",
        "financial_requirements",
        "financial_preferences",
    }
    direct = {k: v for k, v in parsed.items() if k in allowed and isinstance(v, dict)}
    return direct


def _fallback_extract_slots(user_message: str) -> dict[str, Any]:
    """
    Lightweight regex fallback for small-model misses.
    Only returns fields with clear evidence from the user message.
    """
    text = (user_message or "").strip()
    if not text:
        return {}
    low = text.lower()
    out: dict[str, Any] = {}

    # personal_summary
    personal: dict[str, str] = {}
    if any(
        k in low
        for k in (
            "self-employed",
            "self employed",
            "freelance",
            "gig",
            "contractor",
        )
    ):
        personal["employment_status"] = "self-employed"
    elif "student" in low:
        personal["employment_status"] = "student"
    elif any(k in low for k in ("employed", "full-time", "full time", "salaried")):
        personal["employment_status"] = "employed"

    word_to_num = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
    }

    if "no dependents" in low or "no kids" in low or "no children" in low:
        personal["dependents"] = "0"
    else:
        m_dep = re.search(r"\b(\d+)\s*(?:dependents?|kids?|children)\b", low)
        if m_dep:
            personal["dependents"] = m_dep.group(1)
        else:
            m_dep_word = re.search(
                r"\b(zero|one|two|three|four|five|six)\s*(?:dependents?|kids?|children)\b",
                low,
            )
            if m_dep_word:
                personal["dependents"] = word_to_num[m_dep_word.group(1)]

    if re.search(r"\blow[- ]?risk\b", low) or "low-risk decisions" in low:
        personal["risk_tolerance"] = "low"
    elif "conservative" in low:
        personal["risk_tolerance"] = "low"
    elif re.search(r"\bmedium[- ]?risk\b", low):
        personal["risk_tolerance"] = "medium"
    elif re.search(r"\bhigh[- ]?risk\b", low):
        personal["risk_tolerance"] = "high"

    # Age range normalization
    m_age = re.search(r"\b(\d{2})\s*[-–]\s*(\d{2})\b", low)
    if m_age:
        personal["age_range"] = f"{m_age.group(1)}-{m_age.group(2)}"
    else:
        m_decade = re.search(r"\b(?:in my\s+)?(early|mid|late)\s+(\d{2})s\b", low)
        if m_decade:
            band = m_decade.group(1)
            decade = int(m_decade.group(2))
            if band == "early":
                personal["age_range"] = f"{decade}-{decade + 4}"
            elif band == "mid":
                personal["age_range"] = f"{decade + 5}-{decade + 9}"
            else:
                personal["age_range"] = f"{decade + 5}-{decade + 9}"

    # Region from common country mentions
    region_map = {
        "india": "IN",
        "indian": "IN",
        "united states": "US",
        "usa": "US",
        "u.s.": "US",
        "uk": "UK",
        "united kingdom": "UK",
        "canada": "CA",
        "australia": "AU",
    }
    for key, value in region_map.items():
        if key in low:
            personal["country_region"] = value
            break
    if "country_region" not in personal and re.search(r"\bUS\b", text):
        personal["country_region"] = "US"
    if personal:
        out["personal_summary"] = personal

    # financial_requirements
    req: dict[str, str] = {}
    m_horizon = re.search(
        r"\b(\d+\s*(?:-\s*\d+)?\s*(?:months?|mos?|years?))\b",
        text,
        re.IGNORECASE,
    )
    if m_horizon:
        req["time_horizon_months"] = re.sub(r"\s+", " ", m_horizon.group(1)).strip()

    # Monthly budget extraction from natural phrasing.
    m_budget = re.search(
        r"(?:can\s+only\s+pay|can\s+pay|only\s+pay|pay|budget|set\s+aside|save)\s*"
        r"(?:about|around|roughly|approximately|only|up\s*to|upto)?\s*"
        r"(\$?\s*[\d,]+(?:\.\d+)?)\s*(inr|usd|eur|gbp)?\s*(?:/ ?month|per month|monthly)\b",
        text,
        re.IGNORECASE,
    )
    if not m_budget:
        m_budget = re.search(
            r"(\$?\s*[\d,]+(?:\.\d+)?)\s*(inr|usd|eur|gbp)?\s*(?:/ ?month|per month|monthly)\b",
            text,
            re.IGNORECASE,
        )
    if m_budget:
        amount = re.sub(r"\s+", " ", m_budget.group(1)).strip()
        currency = (m_budget.group(2) or "").strip().upper()
        if amount.startswith("$"):
            req["monthly_budget_hint"] = f"{amount} per month"
        elif currency:
            req["monthly_budget_hint"] = f"{amount} {currency} per month"
        else:
            req["monthly_budget_hint"] = f"{amount} per month"

    if "high liquidity" in low:
        req["liquidity_needs"] = "high"
    elif "low liquidity" in low:
        req["liquidity_needs"] = "low"
    elif "medium liquidity" in low:
        req["liquidity_needs"] = "medium"
    elif any(
        k in low
        for k in (
            "emergency savings",
            "emergency fund",
            "keep cash buffer",
            "need cash buffer",
            "cash reserve",
        )
    ):
        req["liquidity_needs"] = "high"
    elif any(
        k in low
        for k in (
            "tight budget",
            "budget is tight",
            "limited budget",
            "cash is tight",
            "money is tight",
            "can only pay",
            "only pay",
            "cash constrained",
            "cash-constrained",
            "limited cash flow",
            "cash flow is tight",
        )
    ):
        req["liquidity_needs"] = "medium"
    if req:
        out["financial_requirements"] = req

    # financial_preferences
    pref: dict[str, str] = {}

    # Product preferences: prefer explicit product/style phrases and keep them concise.
    product_terms: list[str] = []
    if "safe option" in low or "safe options" in low:
        product_terms.append("safe options")
    if "savings account" in low or "savings accounts" in low:
        product_terms.append("savings accounts")
    if "fixed deposit" in low:
        product_terms.append("fixed deposits")
    if "index fund" in low or "index funds" in low:
        product_terms.append("index funds")
    if "etf" in low or "etfs" in low:
        product_terms.append("ETFs")
    if product_terms:
        pref["product_preferences"] = ", ".join(dict.fromkeys(product_terms))

    # Ethical constraints normalization.
    if (
        "no ethical constraints" in low
        or "no ethics" in low
        or re.search(r"\bethical(?:\s+constraints?)?\s*[:\-]?\s*none\b", low)
        or re.search(r"\bnone\b", low)
        and ("ethical" in low or "ethics" in low)
    ):
        pref["ethical_constraints"] = "none"
    else:
        m_eth = re.search(
            r"\b(?:ethical(?:\s+constraints?)?|ethics|values)\s*[:\-]?\s*([^,.;]+)",
            text,
            re.IGNORECASE,
        )
        if m_eth:
            candidate = m_eth.group(1).strip()
            if candidate and candidate.lower() not in {"none", "no", "n/a", "na"}:
                pref["ethical_constraints"] = candidate

    # Automation comfort normalization.
    if any(
        k in low
        for k in (
            "prefer manual",
            "manual",
            "review myself",
            "review it myself",
            "no automation",
            "without automation",
            "don't automate",
            "dont automate",
        )
    ):
        pref["automation_comfort"] = "manual"
    elif any(
        k in low
        for k in (
            "comfortable with automation",
            "okay with automation",
            "auto-transfer",
            "auto transfer",
            "automate",
            "automation is fine",
        )
    ):
        pref["automation_comfort"] = "high"

    if pref:
        out["financial_preferences"] = pref

    # task_definition fallback only when the message clearly states a financial objective.
    if any(
        k in low
        for k in (
            "goal",
            "want to",
            "pay off",
            "debt",
            "credit card",
            "save",
            "retire",
            "emergency fund",
            "invest",
        )
    ):
        out["task_definition"] = {
            "summary": text[:200],
            "goal": text[:300],
        }

    return out


class DialogueEngine:
    """Hybrid: rules own transitions; LLM returns JSON slot_updates + assistant_message."""

    def __init__(
        self,
        backend: LLMBackend,
        dialogue_config_path: str = "configs/dialogue.yaml",
    ) -> None:
        self.backend = backend
        self.cfg = load_yaml(dialogue_config_path)
        self.state = DialogueState.DEFINE_TASK
        self.slots = SessionSlots()
        self.extraction_failures = 0
        self.last_assistant_message = ""
        self.last_recommendation_raw: str | None = None
        self.last_turn_updated_paths: list[str] = []

    def current_phase_key(self) -> str:
        return self.state.value

    def sync_state_from_slots(self) -> None:
        """
        Set dialogue state from slot completeness (single source of truth).
        Avoids stale state where the UI phase drifts ahead of actual slot data.
        """
        if self.state == DialogueState.DONE:
            return
        prev = self.state
        new_state, reasons = compute_state_from_slots(self.slots)
        self.state = new_state
        if prev != self.state:
            logger.info(
                "dialogue_state_transition: %s -> %s | %s",
                prev.value,
                self.state.value,
                " | ".join(reasons),
            )
        else:
            logger.debug(
                "dialogue_state_unchanged: %s | %s",
                self.state.value,
                " | ".join(reasons),
            )

    def advance_state_until_stable(self) -> None:
        """Backward-compatible alias: state is derived from slots in one pass."""
        self.sync_state_from_slots()

    def refresh_last_turn_diff_from(self, slots_before: dict[str, Any]) -> None:
        """Set `last_turn_updated_paths` by diffing a prior snapshot to current slots."""
        self.last_turn_updated_paths = paths_changed(
            flatten_slots(slots_before),
            flatten_slots(self.slots.to_context_dict()),
        )

    def _prompt_env(self) -> Environment:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        return Environment(
            loader=FileSystemLoader(str(root / "llm" / "prompts")),
            autoescape=select_autoescape(enabled_extensions=()),
        )

    def render_extraction_prompt(
        self,
        user_message: str,
        locale_prompt: str,
    ) -> str:
        tmpl = self._prompt_env().get_template("dialogue_collect.jinja2")
        return tmpl.render(
            state=self.state.value,
            slots_json=json.dumps(
                self.slots.to_context_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            user_message=user_message,
            locale_prompt=locale_prompt,
        )

    def process_user_message(
        self,
        user_message: str,
        locale_prompt: str,
    ) -> tuple[str, bool]:
        """Returns (assistant_message, extraction_ok)."""
        if self.state in (DialogueState.RECOMMEND, DialogueState.DONE):
            return (
                self.last_assistant_message,
                True,
            )
        before_snapshot = self.slots.to_context_dict()
        prompt = self.render_extraction_prompt(user_message, locale_prompt)
        result = self.backend.generate(prompt, temperature=0.1)
        parsed = _parse_extraction_json(result.text)
        parsed_ok = parsed is not None
        slot_updates = _normalize_extraction_payload(parsed)
        heuristic_updates = _fallback_extract_slots(user_message)
        if heuristic_updates:
            # Promote obvious, high-confidence values from raw user text.
            slot_updates = deep_merge(slot_updates, heuristic_updates)
        if not parsed_ok and not slot_updates:
            self.extraction_failures += 1
            self.last_turn_updated_paths = []
            msg = guided_reply_after_turn(
                self.state,
                self.slots,
                extraction_ok=False,
            )
            self.last_assistant_message = msg
            return msg, False
        self.extraction_failures = 0
        _apply_slot_updates(self.slots, slot_updates)
        self.last_turn_updated_paths = paths_changed(
            flatten_slots(before_snapshot),
            flatten_slots(self.slots.to_context_dict()),
        )
        self.advance_state_until_stable()
        msg = guided_reply_after_turn(
            self.state,
            self.slots,
            extraction_ok=True,
        )
        self.last_assistant_message = msg
        return msg, True

    def generate_recommendations(
        self,
        locale_prompt: str,
        technique: str = "few_shot",
    ) -> GenerationResult:
        _ = technique  # API/benchmarks compatibility; prompt stays minimal (no technique block)
        tmpl = self._prompt_env().get_template("recommend.jinja2")
        # Always use the latest merged session slots (no stale cache).
        slots_ctx = self.slots.to_context_dict()
        calc = build_recommendation_calc_payload(slots_ctx)
        calc_json = debt_metrics_as_json(slots_ctx, compact=True)
        prompt = tmpl.render(
            slots=slots_ctx,
            slots_json=json.dumps(slots_ctx, ensure_ascii=False, separators=(",", ":")),
            calc=calc,
            calc_json=calc_json,
            language="English",
            locale_prompt=locale_prompt,
        )
        _max_nt = _recommendation_max_new_tokens(self.backend)
        mcfg = load_yaml("configs/models.yaml")
        rec_cfg = mcfg.get("recommendation", {})
        low_hf = isinstance(self.backend, HFLocalBackend) and getattr(
            self.backend, "_low_vram", False
        )
        max_in = (
            rec_cfg.get("low_vram_max_input_tokens")
            if low_hf
            else rec_cfg.get("max_input_tokens")
        )
        trunc = rec_cfg.get("truncation_side")
        gen_kw: dict[str, Any] = {}
        if max_in is not None:
            gen_kw["max_input_tokens"] = int(max_in)
        if isinstance(self.backend, HFLocalBackend) and isinstance(trunc, str) and trunc:
            gen_kw["truncation_side"] = trunc
        result = self.backend.generate(
            prompt,
            temperature=0.2,
            max_new_tokens=_max_nt,
            **gen_kw,
        )
        self.last_recommendation_raw = result.text
        return result

    def validate_recommendation_output(self, text: str) -> bool:
        return parse_recommendation_json(text) is not None

    def mark_done(self) -> None:
        self.state = DialogueState.DONE
