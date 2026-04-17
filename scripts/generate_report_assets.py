"""Build charts from reports/eval_results.csv for the written report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from utils.config_loader import project_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Defaults to reports/eval_results.csv",
    )
    args = ap.parse_args()
    root = project_root()
    csv_path = args.csv or (root / "reports" / "eval_results.csv")
    if not csv_path.exists():
        print("No CSV at", csv_path, "- run python -m evaluation.run_eval first")
        return
    df = pd.read_csv(csv_path)
    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "latency_s" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in df.select_dtypes(include="number").columns:
            pass
        df["latency_s"] = pd.to_numeric(df["latency_s"], errors="coerce")
        if "technique" in df.columns:
            for tech, g in df.groupby("technique"):
                ax.bar(g["prompt_id"].astype(str), g["latency_s"], label=tech)
            ax.legend()
        else:
            ax.bar(df["prompt_id"].astype(str), df["latency_s"])
        ax.set_ylabel("Latency (s)")
        ax.set_xlabel("Prompt")
        fig.tight_layout()
        p = out_dir / "latency_by_prompt.png"
        fig.savefig(p, dpi=150)
        print("Wrote", p)
        plt.close(fig)

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df["prompt_id"].astype(str), df["score"])
        ax.set_ylabel("Score")
        ax.set_xlabel("Prompt")
        fig.tight_layout()
        p = out_dir / "score_by_prompt.png"
        fig.savefig(p, dpi=150)
        print("Wrote", p)
        plt.close(fig)


if __name__ == "__main__":
    main()
