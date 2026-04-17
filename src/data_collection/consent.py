from __future__ import annotations

CONSENT_TEXT = """
I understand this tool is an educational prototype and not financial advice.
I consent to export anonymized session metadata (no bank details) for coursework
and optional model improvement, if I use the export feature.
"""


def consent_granted(session_flag: bool) -> bool:
    return bool(session_flag)
