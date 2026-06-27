"""Prompt formatting for patient support conversations."""


def format_prompt(sample: dict) -> str:
    """
    Format input for patient support.

    Expected sample keys:
    - patient_question: str
      (patient's question about health, medication, or procedure)
    - support_response: str
      (clear, empathetic, accurate answer)
    """
    return f"""### Instruction:
You are a helpful patient support assistant
for a healthcare provider. Answer the
patient's question clearly and empathetically.
Provide accurate medical information.
Always recommend consulting their doctor
for personal medical decisions.
Never diagnose or prescribe.

### Patient Question:
{sample['patient_question']}

### Support Response:
{sample['support_response']}"""


def format_inference_prompt(patient_question: str) -> str:
    """
    Format prompt for inference only.
    Used at serving time.
    """
    return f"""### Instruction:
You are a helpful patient support assistant
for a healthcare provider. Answer the
patient's question clearly and empathetically.
Provide accurate medical information.
Always recommend consulting their doctor
for personal medical decisions.
Never diagnose or prescribe.

### Patient Question:
{patient_question}

### Support Response:
"""
