"""Prompt formatting for ICD-10 medical coding."""


def format_prompt(sample: dict) -> str:
    """
    Format input for ICD-10 medical coding.

    Expected sample keys:
    - clinical_description: str
      (doctor's description of condition)
    - icd_codes: str
      (correct ICD-10 codes with descriptions)
    """
    return f"""### Instruction:
You are a medical coding specialist.
Analyze the clinical description and provide
the correct ICD-10 diagnosis codes with
full descriptions. List primary diagnosis
first, followed by secondary codes.

### Clinical Description:
{sample['clinical_description']}

### ICD-10 Codes:
{sample['icd_codes']}"""


def format_inference_prompt(clinical_description: str) -> str:
    """
    Format prompt for inference only (no expected output).
    Used at serving time.
    """
    return f"""### Instruction:
You are a medical coding specialist.
Analyze the clinical description and provide
the correct ICD-10 diagnosis codes with
full descriptions. List primary diagnosis
first, followed by secondary codes.

### Clinical Description:
{clinical_description}

### ICD-10 Codes:
"""
