"""Supervised fine-tuning (full weights) training entrypoint."""

from __future__ import annotations

from transformers import Trainer, TrainingArguments

from core.inference.model_loader import load_model
from core.training.dataset_loader import prepare_datasets
from core.utils.config_loader import get_model_output_dir, load_project_config, load_prompt_template
from core.utils.logger import setup_logger


def train_sft(project_name: str) -> None:
    """Run full supervised fine-tuning for a project."""
    config = load_project_config(project_name)
    logger = setup_logger(project_name, "training")
    logger.info("Starting SFT training for project=%s", project_name)

    format_prompt = load_prompt_template(project_name)
    device = config.get("model", {}).get("device", "cpu")
    training_cfg = config["training"]

    model, tokenizer = load_model(config["base_model"], device)
    train_dataset, val_dataset = prepare_datasets(config, format_prompt, tokenizer)

    output_dir = get_model_output_dir(config, "sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_cfg["batch_size"],
        num_train_epochs=training_cfg["epochs"],
        learning_rate=training_cfg["learning_rate"],
        logging_steps=training_cfg.get("logging_steps", 1),
        save_steps=training_cfg.get("save_steps", 10),
        fp16=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    logger.info("SFT training complete. Checkpoints saved to %s", output_dir)


def main(project_name: str | None = None) -> None:
    """CLI wrapper for SFT training."""
    if project_name is None:
        raise ValueError("--project is required for SFT training.")
    train_sft(project_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SFT training for a project.")
    parser.add_argument("--project", required=True, help="Project folder name under projects/")
    args = parser.parse_args()
    main(args.project)
