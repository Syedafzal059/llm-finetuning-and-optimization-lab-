"""Generic JSON dataset loader with project-specific prompt templates."""

from __future__ import annotations

import json
import random
from typing import Any, Callable

from datasets import Dataset

from core.utils.config_loader import resolve_path


def load_json_dataset(path: str) -> list[dict[str, Any]]:
    """Load a JSON array of training samples from disk."""
    dataset_path = resolve_path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Dataset at {dataset_path} must be a JSON array.")
    return data


def split_dataset(
    data: list[dict[str, Any]],
    val_split: float,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Split data into train and optional validation sets."""
    if val_split <= 0 or len(data) < 2:
        return data, None

    shuffled = data.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    train_count = max(1, int(len(shuffled) * (1 - val_split)))
    train_data = shuffled[:train_count]
    val_data = shuffled[train_count:]
    if not val_data:
        return train_data, None
    return train_data, val_data


def format_samples(
    data: list[dict[str, Any]],
    format_prompt: Callable[[dict[str, Any]], str],
) -> list[str]:
    """Apply a project's format_prompt function to every sample."""
    return [format_prompt(sample) for sample in data]


def tokenize_dataset(
    texts: list[str],
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    """Tokenize formatted text samples for causal language modeling."""
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(example: dict[str, str]) -> dict[str, list[int]]:
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(tokenize)


def prepare_datasets(
    config: dict[str, Any],
    format_prompt: Callable[[dict[str, Any]], str],
    tokenizer: Any,
) -> tuple[Dataset, Dataset | None]:
    """Load, split, format, and tokenize a project's training data."""
    data_cfg = config["data"]
    training_cfg = config["training"]

    raw_data = load_json_dataset(data_cfg["train_path"])
    train_raw, val_raw = split_dataset(raw_data, data_cfg.get("val_split", 0.0))
    max_length = training_cfg.get("max_length", 256)

    train_texts = format_samples(train_raw, format_prompt)
    train_dataset = tokenize_dataset(train_texts, tokenizer, max_length)

    val_dataset = None
    if val_raw:
        val_texts = format_samples(val_raw, format_prompt)
        val_dataset = tokenize_dataset(val_texts, tokenizer, max_length)

    return train_dataset, val_dataset
