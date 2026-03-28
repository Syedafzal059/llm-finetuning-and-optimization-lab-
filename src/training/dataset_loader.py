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

    #creates HuggingFace dataset(Trainer expects structured dataset)
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,# Cut long text and prevent memory overflow
            padding="max_length",# makes eceh text of same size important batching
            max_length=256
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    dataset = dataset.map(tokenize) #applies tokenization to all rows
    return dataset
