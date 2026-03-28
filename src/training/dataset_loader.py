import json

from datasets import Dataset


def load_dataset(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def format_example(example):
    return f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""


def tokenize_dataset(data, tokenizer):
    texts = [format_example(x) for x in data]

    dataset = Dataset.from_dict({"text": texts})

    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    dataset = dataset.map(tokenize)
    return dataset
