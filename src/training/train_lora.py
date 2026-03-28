import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from src.inference.model_loader import load_model
from src.training.dataset_loader import load_dataset, tokenize_dataset
from src.utils.config_loader import load_config


def main():
    config = load_config(str(_PROJECT_ROOT / "configs.py" / "base.yaml"))

    model, tokenizer = load_model(
        config["model"]["name"],
        config["model"]["device"],
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data = load_dataset(str(_PROJECT_ROOT / "data" / "raw" / "sample_dataset.json"))
    tokenized_data = tokenize_dataset(data, tokenizer)

    lora_output_dir = _PROJECT_ROOT / "model" / "lora"
    training_args = TrainingArguments(
        output_dir=str(lora_output_dir),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=10,
        fp16=(config["model"]["device"] == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    trainer.train()
    model.save_pretrained(str(lora_output_dir))


if __name__ == "__main__":
    main()
