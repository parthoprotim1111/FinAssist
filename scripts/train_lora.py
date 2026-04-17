"""LoRA fine-tuning with TRL SFTTrainer (GPU recommended)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from utils.config_loader import load_yaml, project_root


def load_jsonl(path: Path) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    root = project_root()

    train_path = root / cfg["train_file"]
    val_path = root / cfg["validation_file"]
    output_dir = root / cfg["output_dir"]

    train_ds = load_jsonl(train_path)
    eval_ds = load_jsonl(val_path) if val_path.exists() else None

    base_id = cfg["base_model_id"]
    torch_dtype = torch.bfloat16 if cfg.get("bf16") else torch.float16
    quant_config = None
    if cfg.get("load_in_4bit") and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model_kwargs: dict = {"trust_remote_code": True}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(base_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.get("load_in_4bit") and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    def to_text(batch):
        texts = []
        for messages in batch["messages"]:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    train_ds = train_ds.map(to_text, batched=True)
    if eval_ds is not None:
        eval_ds = eval_ds.map(to_text, batched=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        bf16=bool(cfg.get("bf16")) and torch.cuda.is_available(),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing")),
        report_to="none",
        max_seq_length=int(cfg["max_seq_length"]),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Saved adapter to", output_dir)


if __name__ == "__main__":
    main()
