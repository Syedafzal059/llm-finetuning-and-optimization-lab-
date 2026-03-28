from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)

    return model, tokenizer



