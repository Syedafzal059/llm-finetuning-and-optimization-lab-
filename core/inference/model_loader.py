"""Load base and adapter-augmented causal language models."""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(requested: str) -> str:
    """Use the requested device when available; fall back to CPU for local dev."""
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def load_model(model_name: str, device: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a base causal LM and move it to the requested device."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    return model, tokenizer


def load_model_with_lora(
    base_model_name: str,
    adapter_path: str,
    device: str,
) -> tuple[PeftModel, AutoTokenizer]:
    """Load a base causal LM and attach LoRA/QLoRA adapters from a PEFT checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer
