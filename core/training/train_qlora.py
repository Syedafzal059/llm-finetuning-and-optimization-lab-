"""QLoRA (4-bit base + LoRA) training entrypoint."""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from core.training.dataset_loader import prepare_datasets
from core.training.train_lora import _build_lora_config
from core.utils.config_loader import get_model_output_dir, load_project_config, load_prompt_template
from core.utils.logger import setup_logger


def _build_bnb_config(config: dict) -> BitsAndBytesConfig:
    """Build a bitsandbytes quantization config from project settings."""
    qlora_cfg = config["training"]["qlora"]
    return BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.get("bits", 4) == 4,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=qlora_cfg.get("double_quant", True),
        bnb_4bit_quant_type=qlora_cfg.get("quant_type", "nf4"),
    )


def train_qlora(project_name: str) -> None:
    """Run QLoRA fine-tuning for a project."""
    config = load_project_config(project_name)
    logger = setup_logger(project_name, "training")
    logger.info("Starting QLoRA training for project=%s", project_name)

    format_prompt = load_prompt_template(project_name)
    training_cfg = config["training"]

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=_build_bnb_config(config),
        device_map="auto",
    )

    model = get_peft_model(model, _build_lora_config(config))
    model.print_trainable_parameters()

    train_dataset, val_dataset = prepare_datasets(config, format_prompt, tokenizer)

    output_dir = get_model_output_dir(config, "qlora")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_cfg["batch_size"],
        num_train_epochs=training_cfg["epochs"],
        learning_rate=training_cfg["learning_rate"],
        logging_steps=training_cfg.get("logging_steps", 1),
        save_steps=training_cfg.get("save_steps", 10),
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    logger.info("QLoRA training complete. Adapters saved to %s", output_dir)


def main(project_name: str | None = None) -> None:
    """CLI wrapper for QLoRA training."""
    if project_name is None:
        raise ValueError("--project is required for QLoRA training.")
    train_qlora(project_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run QLoRA training for a project.")
    parser.add_argument("--project", required=True, help="Project folder name under projects/")
    args = parser.parse_args()
    main(args.project)
