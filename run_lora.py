"""Run inference with LoRA adapters (train first: python src/training/train_lora.py)."""

from pathlib import Path

from src.inference.generate import generate_text
from src.inference.model_loader import load_model_with_lora
from src.utils.config_loader import load_config

_PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    config = load_config(str(_PROJECT_ROOT / "configs.py" / "base.yaml"))
    adapter_dir = _PROJECT_ROOT / "model" / "lora"
    if not (adapter_dir / "adapter_config.json").is_file():
        raise SystemExit(
            f"No LoRA adapters found at {adapter_dir}. "
            "Run: python src/training/train_lora.py"
        )

    model, tokenizer = load_model_with_lora(
        config["model"]["name"],
        str(adapter_dir),
        config["model"]["device"],
    )

    prompt = "Explain what is LoRA in simple terms"
    output = generate_text(model, tokenizer, prompt, config)

    print("\n=== LoRA OUTPUT ===\n")
    print(output)


if __name__ == "__main__":
    main()
