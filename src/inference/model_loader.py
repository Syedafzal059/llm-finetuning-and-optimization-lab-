import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)

    return model, tokenizer


def load_model_with_lora(base_model_name: str, adapter_path: str, device: str):
    """Load a base causal LM and attach LoRA adapters from a local PEFT checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer



