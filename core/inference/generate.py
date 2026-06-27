"""Text generation helpers for local inference."""

from __future__ import annotations

from typing import Any

import torch


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    config: dict[str, Any],
) -> str:
    """Generate text from a prompt using sampling settings from config."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inference_cfg = config.get("inference", {})
    model_cfg = config.get("model", {})

    with torch.no_grad():
        max_new_tokens = model_cfg.get("max_new_tokens", 100)
        saved_max_length = None
        if hasattr(model, "generation_config") and model.generation_config is not None:
            saved_max_length = model.generation_config.max_length
            model.generation_config.max_length = None
        try:
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=inference_cfg.get("temperature", 0.7),
                top_p=inference_cfg.get("top_p", 0.9),
                do_sample=True,
            )
        finally:
            if saved_max_length is not None and model.generation_config is not None:
                model.generation_config.max_length = saved_max_length
    return tokenizer.decode(output[0], skip_special_tokens=True)
