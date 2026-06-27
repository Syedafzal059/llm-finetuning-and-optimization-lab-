"""Prompt formatting for clinical note summarization."""


def format_prompt(sample: dict) -> str:
    """Format a clinical note sample into an instruction-tuning prompt."""
    return f"""### Instruction:
Summarize the following clinical note into a structured SOAP format.

### Input:
{sample['clinical_note']}

### Response:
{sample['summary']}"""


def format_inference_prompt(clinical_note: str) -> str:
    """Format prompt for inference only (no expected output)."""
    return f"""### Instruction:
Summarize the following clinical note into a structured SOAP format.

### Input:
{clinical_note}

### Response:
"""
