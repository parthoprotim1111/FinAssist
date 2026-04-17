"""CSV output helpers: fallback path when the default file is locked (e.g. open in Excel on Windows)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from utils.config_loader import project_root


# #region agent log
def _agent_debug_log(location: str, hypothesis_id: str, message: str, data: dict) -> None:
    try:
        log_path = project_root() / "debug-d1ec3d.log"
        payload = {
            "sessionId": "d1ec3d",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass


# #endregion


def save_dataframe_csv(df: pd.DataFrame, primary: Path) -> Path:
    """Write DataFrame to ``primary``; on PermissionError use a timestamped sibling in the same folder."""
    primary.parent.mkdir(parents=True, exist_ok=True)
    # #region agent log
    _agent_debug_log(
        "evaluation/csv_output.py:save_dataframe_csv",
        "B1",
        "attempt_primary",
        {"path": str(primary.resolve())},
    )
    # #endregion
    try:
        df.to_csv(primary, index=False)
        # #region agent log
        _agent_debug_log(
            "evaluation/csv_output.py:save_dataframe_csv",
            "B1",
            "saved_primary",
            {"path": str(primary.resolve())},
        )
        # #endregion
        return primary
    except PermissionError as e:
        # #region agent log
        _agent_debug_log(
            "evaluation/csv_output.py:save_dataframe_csv",
            "B1",
            "permission_error_pandas",
            {
                "errno": e.errno,
                "winerror": getattr(e, "winerror", None),
                "str": str(e),
            },
        )
        # #endregion
        alt = primary.with_name(f"{primary.stem}_{int(time.time())}{primary.suffix}")
        df.to_csv(alt, index=False)
        # #region agent log
        _agent_debug_log(
            "evaluation/csv_output.py:save_dataframe_csv",
            "B2",
            "saved_fallback",
            {"path": str(alt.resolve())},
        )
        # #endregion
        return alt
