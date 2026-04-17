from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from dialogue.slots import SessionSlots
from dialogue.validators import redact_for_export


def anonymize_session_export(
    slots: SessionSlots,
    *,
    backend_name: str,
    language: str,
    consent: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = json.dumps(slots.to_context_dict(), sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "session_fingerprint": digest,
        "language": language,
        "backend": backend_name,
        "consent": consent,
        "slots_redacted": {
            "task_definition": redact_for_export(
                slots.task_definition.summary + " " + slots.task_definition.goal
            ),
            "risk_tolerance": redact_for_export(slots.personal_summary.risk_tolerance),
            "constraints": redact_for_export(slots.financial_requirements.constraints),
        },
    }
    if extra:
        payload["extra"] = extra
    return payload


def dumps_pretty(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
