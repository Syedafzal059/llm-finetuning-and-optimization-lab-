import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from transformers import Trainer, TrainingArguments

from src.inference.model_loader import load_model
from src.training.dataset_loader import load_dataset, tokenize_dataset
from src.utils.config_loader import load_config


def main():
    config = load_config(str(_PROJECT_ROOT / "configs.py" / "base.yaml"))
    model, tokenizer = load_model(
        config["model"]["name"],
        config["model"]["device"]

    )

    data = load_dataset(str(_PROJECT_ROOT / "data" / "raw" / "sample_dataset.json"))
    tokenized_data = tokenize_dataset(data, tokenizer)

    Training_args = TrainingArguments(
        output_dir="./model/sft",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=10,
        fp16=(config["model"]["device"])=="cuda"
    )

    trainer = Trainer(
        model=model,
        args=Training_args,
        train_dataset=tokenized_data,
    )
    trainer.train()


if __name__ == "__main__":
    main()