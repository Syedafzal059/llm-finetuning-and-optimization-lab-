import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.training.dataset_loader import load_dataset, tokenize_dataset
from src.utils.config_loader import load_config


def main():
    config = load_config(str(_PROJECT_ROOT / "configs.py" / "base.yaml"))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
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

    qlora_output_dir = _PROJECT_ROOT / "model" / "qlora"
    training_args = TrainingArguments(
        output_dir=str(qlora_output_dir),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=1,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )
    trainer.train()
    model.save_pretrained(str(qlora_output_dir))


if __name__ == "__main__":
    main()
