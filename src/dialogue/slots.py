from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DialogueState(str, Enum):
    DEFINE_TASK = "define_task"
    COLLECT_PERSONAL = "collect_personal"
    REQUIREMENTS = "requirements"
    PREFERENCES = "preferences"
    RECOMMEND = "recommend"
    DONE = "done"


class TaskDefinition(BaseModel):
    summary: str = ""
    goal: str = ""


class PersonalInfo(BaseModel):
    """Synthetic-style fields only — no real account numbers."""

    age_range: str = ""
    employment_status: str = ""
    country_region: str = ""
    dependents: str = ""
    risk_tolerance: str = ""  # low/medium/high
    notes: str = ""


class FinancialRequirements(BaseModel):
    monthly_budget_hint: str = ""
    time_horizon_months: str = ""
    liquidity_needs: str = ""
    constraints: str = ""


class FinancialPreferences(BaseModel):
    ethical_constraints: str = ""
    product_preferences: str = ""
    automation_comfort: str = ""


class SessionSlots(BaseModel):
    task_definition: TaskDefinition = Field(default_factory=TaskDefinition)
    personal_summary: PersonalInfo = Field(default_factory=PersonalInfo)
    financial_requirements: FinancialRequirements = Field(
        default_factory=FinancialRequirements
    )
    financial_preferences: FinancialPreferences = Field(
        default_factory=FinancialPreferences
    )

    def to_context_dict(self) -> dict[str, Any]:
        return {
            "task_definition": self.task_definition.model_dump(),
            "personal_summary": self.personal_summary.model_dump(),
            "financial_requirements": self.financial_requirements.model_dump(),
            "financial_preferences": self.financial_preferences.model_dump(),
        }
