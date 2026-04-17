from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_from_turns(
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Build one training row in chat format for TRL SFT."""
    return {"messages": messages}
