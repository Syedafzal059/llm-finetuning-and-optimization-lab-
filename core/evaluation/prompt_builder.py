"""Build inference prompts and extract reference outputs from dataset samples."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

from core.utils.config_loader import PROJECTS_DIR

REFERENCE_KEYS = ("summary", "icd_codes", "support_response", "output", "response")
INPUT_KEYS = (
    "clinical_note",
    "clinical_description",
    "patient_question",
    "input",
    "question",
)


def _load_prompt_module(project_name: str) -> Any:
    """Import a project's prompt_template module."""
    template_path = PROJECTS_DIR / project_name / "prompt_template.py"
    if not template_path.is_file():
        raise FileNotFoundError(
            f"Prompt template not found: {template_path}",
        )

    module_name = f"eval_{project_name}_prompt_template"
    spec = importlib.util.spec_from_file_location(module_name, template_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load prompt template from {template_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_reference_output(sample: dict[str, Any]) -> str:
    """Return the expected model output field from a dataset sample."""
    for key in REFERENCE_KEYS:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(
        f"Sample has no known reference field. Expected one of: {REFERENCE_KEYS}",
    )


def get_input_text(sample: dict[str, Any]) -> str:
    """Return the human-readable input text for report examples."""
    for key in INPUT_KEYS:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(sample)


def build_inference_prompt(
    project_name: str,
    sample: dict[str, Any],
    format_prompt: Callable[[dict[str, Any]], str] | None = None,
) -> str:
    """
    Build a prompt for inference (no expected output appended).

    Uses format_inference_prompt from the project when available;
    otherwise strips the reference suffix from format_prompt output.
    """
    module = _load_prompt_module(project_name)
    inference_fn = getattr(module, "format_inference_prompt", None)

    if callable(inference_fn):
        input_key = next(
            (key for key in INPUT_KEYS if key in sample),
            None,
        )
        if input_key is not None:
            return inference_fn(sample[input_key])

    if format_prompt is None:
        format_prompt = getattr(module, "format_prompt", None)
    if not callable(format_prompt):
        raise AttributeError(
            f"projects/{project_name}/prompt_template.py must define format_prompt.",
        )

    full_prompt = format_prompt(sample)
    reference = get_reference_output(sample)
    if full_prompt.endswith(reference):
        return full_prompt[: -len(reference)]
    return full_prompt
