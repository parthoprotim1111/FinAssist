"""Merge LoRA adapter into base weights (optional; large disk)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config_loader import load_yaml, project_root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training.yaml")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    root = project_root()
    base_id = cfg["base_model_id"]
    adapter = root / cfg["output_dir"]
    out = root / "models" / "merged_model"

    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    tok.save_pretrained(str(out))
    print("Merged model saved to", out)


if __name__ == "__main__":
    main()
