"""Download model weights from Hugging Face Hub (snapshot)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import snapshot_download
from utils.config_loader import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/models.yaml",
        help="YAML with primary.base_model_id",
    )
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    model_id = cfg["primary"]["base_model_id"]
    snapshot_download(repo_id=model_id, local_dir_use_symlinks=False)
    print("Downloaded:", model_id)


if __name__ == "__main__":
    main()
