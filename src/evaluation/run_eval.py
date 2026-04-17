"""
Batch evaluation over fixtures. Primary backend: HF local (optional; use mock if unavailable).

Usage:
  python -m evaluation.run_eval
  python -m evaluation.run_eval --backend mock
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dialogue.state_machine import DialogueEngine
from dialogue.slots import (
    DialogueState,
    FinancialPreferences,
    FinancialRequirements,
    PersonalInfo,
    SessionSlots,
    TaskDefinition,
)
from evaluation.metrics import score_output, timed_generate
from llm.backend_base import LLMBackend
from llm.mock_backend import MockLLMBackend
from utils.config_loader import load_yaml, project_root


# #region agent log
def _agent_debug_log(hypothesis_id: str, message: str, data: dict) -> None:
    try:
        log_path = project_root() / "debug-d1ec3d.log"
        payload = {
            "sessionId": "d1ec3d",
            "hypothesisId": hypothesis_id,
            "location": "run_eval.py",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass


# #endregion


def load_backend(name: str, low_vram: bool) -> LLMBackend:
    if name == "mock":
        return MockLLMBackend()
    from llm.hf_local import HFLocalBackend

    if name == "hf":
        b = HFLocalBackend(low_vram=low_vram)
        b.load()
        return b
    if name == "hf_alt":
        b = HFLocalBackend(use_primary=False, low_vram=low_vram)
        b.load()
        return b
    raise ValueError(f"Unknown backend {name}")


def _slots_from_fixture(blob: dict) -> SessionSlots:
    s = SessionSlots()
    s.task_definition = TaskDefinition(**blob["task_definition"])
    s.personal_summary = PersonalInfo(**blob["personal_summary"])
    s.financial_requirements = FinancialRequirements(**blob["financial_requirements"])
    s.financial_preferences = FinancialPreferences(**blob["financial_preferences"])
    return s


def run(
    backend_name: str,
    technique: str,
    low_vram: bool,
    fixtures_path: Path,
    output_csv: Path,
) -> None:
    eval_cfg = load_yaml("configs/evaluation.yaml")
    weights = eval_cfg.get("metrics", {})
    backend = load_backend(backend_name, low_vram=low_vram)
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rp = output_csv.resolve()
    # #region agent log
    _pre = {
        "path": str(rp),
        "exists": rp.exists(),
        "parent": str(rp.parent),
        "parent_exists": rp.parent.exists(),
        "parent_w_ok": os.access(rp.parent, os.W_OK) if rp.parent.exists() else None,
    }
    if rp.exists():
        try:
            _pre["file_w_ok"] = os.access(rp, os.W_OK)
        except OSError as e:
            _pre["access_err"] = str(e)
    _agent_debug_log("H1", "pre_csv_open", _pre)
    # #endregion

    out_path = output_csv
    try:
        f = output_csv.open("w", newline="", encoding="utf-8")
        # #region agent log
        _agent_debug_log("H1", "opened_primary_csv", {"path": str(output_csv.resolve())})
        # #endregion
    except PermissionError as e:
        # #region agent log
        _agent_debug_log(
            "H1",
            "permission_error_primary_path",
            {"errno": e.errno, "winerror": getattr(e, "winerror", None), "str": str(e)},
        )
        # #endregion
        out_path = output_csv.with_name(
            f"{output_csv.stem}_{int(time.time())}{output_csv.suffix}"
        )
        print(
            f"Could not write {output_csv} (file may be open in another app). "
            f"Writing to {out_path}",
            file=sys.stderr,
        )
        f = out_path.open("w", newline="", encoding="utf-8")
        # #region agent log
        _agent_debug_log("H5", "opened_fallback_csv", {"path": str(out_path.resolve())})
        # #endregion

    fieldnames = [
        "prompt_id",
        "model",
        "technique",
        "latency_s",
        "schema_valid",
        "score",
        "output_excerpt",
    ]
    with f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for item in fixtures:
            slots = _slots_from_fixture(item["slots"])
            engine = DialogueEngine(backend)
            engine.slots = slots
            engine.state = DialogueState.RECOMMEND

            def _gen():
                return engine.generate_recommendations(
                    locale_prompt="Respond in clear, professional English.",
                    technique=technique,
                )

            result, latency = timed_generate(_gen)
            metrics = score_output(
                result.text,
                latency_s=latency,
                weights=weights,
                slots=slots.to_context_dict(),
            )
            excerpt = result.text[:500].replace("\n", " ")
            w.writerow(
                {
                    "prompt_id": item["id"],
                    "model": backend.name,
                    "technique": technique,
                    "latency_s": f"{latency:.3f}",
                    "schema_valid": metrics["schema_valid"],
                    "score": f"{metrics['score']:.3f}",
                    "output_excerpt": excerpt,
                }
            )


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=["mock", "hf", "hf_alt"])
    ap.add_argument("--technique", default="few_shot")
    ap.add_argument("--low-vram", action="store_true")
    ap.add_argument(
        "--fixtures",
        type=Path,
        default=root / "src" / "evaluation" / "fixtures" / "example_prompts.json",
    )
    ap.add_argument("--output", type=Path, default=root / "reports" / "eval_results.csv")
    args = ap.parse_args()
    run(
        args.backend,
        args.technique,
        args.low_vram,
        args.fixtures,
        args.output,
    )


if __name__ == "__main__":
    main()
