from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm.backend_base import GenerationResult, LLMBackend
from utils.config_loader import load_yaml, project_root
from utils.device import clear_cuda_cache

logger = logging.getLogger(__name__)


def _read_system_prompt() -> str:
    p = Path(__file__).resolve().parent / "prompts" / "system_finance.txt"
    return p.read_text(encoding="utf-8").strip()


class HFLocalBackend(LLMBackend):
    """Primary backend: local Transformers model + optional PEFT LoRA adapter."""

    name = "hf_local"

    def __init__(
        self,
        config_path: str = "configs/models.yaml",
        *,
        use_primary: bool = True,
        low_vram: bool = False,
    ) -> None:
        self._cfg_path = config_path
        self._use_primary = use_primary
        self._low_vram = low_vram
        self._model = None
        self._tokenizer = None
        self._loaded_key: str | None = None

    def _section(self) -> dict[str, Any]:
        cfg = load_yaml(self._cfg_path)
        if self._use_primary:
            return cfg.get("primary", {})
        return cfg.get("comparison", {})

    def _effective_limits(self) -> tuple[int, int]:
        cfg = load_yaml(self._cfg_path)
        primary = cfg.get("primary", {})
        low = cfg.get("low_vram", {})
        if self._low_vram:
            return int(low.get("max_new_tokens", 256)), int(
                low.get("max_input_tokens", 1024)
            )
        return int(primary.get("max_new_tokens", 512)), int(
            primary.get("max_input_tokens", 2048)
        )

    def load(self) -> None:
        cfg = load_yaml(self._cfg_path)
        if self._use_primary:
            sec = cfg["primary"]
            base_id = sec["base_model_id"]
            adapter = sec.get("adapter_path")
            load_4 = sec.get("load_in_4bit", True)
            load_8 = sec.get("load_in_8bit", False)
        else:
            sec = cfg["comparison"]
            base_id = sec["alt_base_model_id"]
            adapter = sec.get("alt_adapter_path")
            load_4 = sec.get("load_in_4bit", True)
            load_8 = False

        key = f"{base_id}|{adapter}|{load_4}"
        if self._model is not None and self._loaded_key == key:
            return

        self.unload()

        dtype_str = cfg.get("primary", {}).get("torch_dtype", "bfloat16")
        torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            if dtype_str == "bfloat16":
                torch_dtype = torch.float16

        quant_config = None
        use_cuda = torch.cuda.is_available()
        if use_cuda and load_4:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif use_cuda and load_8:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
            kwargs["device_map"] = "auto" if use_cuda else None
        else:
            kwargs["torch_dtype"] = torch_dtype
            if use_cuda:
                kwargs["device_map"] = "auto"
            elif torch.backends.mps.is_available():
                kwargs["device_map"] = "mps"
            else:
                kwargs["device_map"] = "cpu"

        model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if adapter:
            adapter_path = Path(adapter)
            if not adapter_path.is_absolute():
                adapter_path = project_root() / adapter_path
            if adapter_path.exists():
                model = PeftModel.from_pretrained(model, str(adapter_path))
                logger.info("Loaded LoRA adapter from %s", adapter_path)
            else:
                logger.warning("Adapter path not found: %s", adapter_path)

        self._model = model
        self._tokenizer = tokenizer
        self._loaded_key = key
        self._torch_dtype = torch_dtype

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded_key = None
        clear_cuda_cache()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.2,
        max_input_tokens: int | None = None,
        truncation_side: str | None = None,
    ) -> GenerationResult:
        self.load()
        assert self._tokenizer is not None and self._model is not None
        default_max, base_max_input = self._effective_limits()
        max_input = int(max_input_tokens) if max_input_tokens is not None else base_max_input
        if max_new_tokens is None:
            max_new_tokens = default_max

        system = _read_system_prompt()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            input_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            input_text = f"System:\n{system}\n\nUser:\n{prompt}\n\nAssistant:\n"

        _tok_kw: dict[str, Any] = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": max_input,
        }
        if truncation_side is not None:
            _tok_kw["truncation_side"] = truncation_side
        inputs = self._tokenizer(input_text, **_tok_kw)
        dev = next(self._model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(gen, skip_special_tokens=True)
        return GenerationResult(text=text, raw={"backend": self.name})
