"""Load and merge base + project configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = PROJECT_ROOT / "base_config.yaml"
PROJECTS_DIR = PROJECT_ROOT / "projects"


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (PROJECT_ROOT / resolved).resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base; override values win."""
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a single YAML config file."""
    return _load_yaml(resolve_path(path))


def load_project_config(project_name: str) -> dict[str, Any]:
    """Load base_config.yaml, merge projects/{name}/config.yaml, and attach metadata."""
    if not (PROJECTS_DIR / project_name).is_dir():
        raise FileNotFoundError(
            f"Project '{project_name}' not found at {PROJECTS_DIR / project_name}"
        )

    base = _load_yaml(BASE_CONFIG_PATH)
    project_path = PROJECTS_DIR / project_name / "config.yaml"
    project = _load_yaml(project_path)
    merged = _deep_merge(base, project)

    merged["project_name"] = project_name
    merged["_project_dir"] = str(PROJECTS_DIR / project_name)
    return merged


def get_project_dir(config: dict[str, Any]) -> Path:
    """Return the absolute path to the project directory."""
    return PROJECT_ROOT / "projects" / config["project_name"]


def get_model_output_dir(config: dict[str, Any], mode: str) -> Path:
    """Return the checkpoint directory for a training mode (sft/lora/qlora)."""
    return get_project_dir(config) / "model" / mode


def get_logs_dir(config: dict[str, Any]) -> Path:
    """Return the logs directory for a project."""
    return get_project_dir(config) / "logs"


def load_prompt_template(project_name: str) -> Callable[[dict[str, Any]], str]:
    """Dynamically import format_prompt from a project's prompt_template.py."""
    template_path = PROJECTS_DIR / project_name / "prompt_template.py"
    if not template_path.is_file():
        raise FileNotFoundError(
            f"Prompt template not found: {template_path}"
        )

    module_name = f"project_{project_name}_prompt_template"
    spec = importlib.util.spec_from_file_location(module_name, template_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load prompt template from {template_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    format_prompt = getattr(module, "format_prompt", None)
    if not callable(format_prompt):
        raise AttributeError(
            f"{template_path} must define a format_prompt(sample: dict) -> str function."
        )
    return format_prompt


def _normalize_rate_limit(value: int | str) -> int:
    """Convert rate_limit config to an integer requests-per-minute value."""
    if isinstance(value, int):
        return value
    raw = str(value).strip()
    if "/" in raw:
        raw = raw.split("/", 1)[0].strip()
    return int(raw)


def build_serving_config(project_config: dict[str, Any]) -> dict[str, Any]:
    """Build the serving-layer config dict from a merged project config."""
    project_name = project_config["project_name"]
    training_mode = project_config.get("training", {}).get("mode", "qlora")
    adapter_rel = project_config.get(
        "adapter_path",
        f"projects/{project_name}/model/{training_mode}",
    )
    serving = project_config.get("serving", {})
    timeout = serving.get("timeout", serving.get("timeout_seconds", 60))

    return {
        "model": {
            "model_name": project_config["base_model"],
            "adapter_path": adapter_rel,
            "max_model_len": serving.get("max_model_len", 4096),
            "gpu_memory_utilization": serving.get("gpu_memory_utilization", 0.85),
            "max_lora_rank": serving.get("max_lora_rank", 64),
            "quantization_enabled": serving.get("quantization_enabled", True),
            "quantization": serving.get("quantization", "bitsandbytes"),
            "dtype": serving.get("dtype", "auto"),
            "trust_remote_code": serving.get("trust_remote_code", False),
        },
        "validation": project_config.get("validation", {"max_prompt_tokens": 2048}),
        "serving": {
            "host": serving.get("host", "0.0.0.0"),
            "port": serving.get("port", 8000),
            "timeout": timeout,
            "timeout_seconds": timeout,
            "rate_limit": _normalize_rate_limit(serving.get("rate_limit", 10)),
            "monthly_limit": serving.get("monthly_limit", 10000),
            "token_limit": serving.get("token_limit", 1_000_000),
            "max_tokens": serving.get("max_tokens", 512),
            "temperature": serving.get("temperature", 0.7),
            "top_p": serving.get("top_p", 0.9),
        },
        "alerts": project_config.get("alerts", {}),
        "redis": project_config.get("redis", {}),
        "auth": project_config.get("auth", {}),
        "queue": project_config.get(
            "queue",
            {
                "max_queue_depth": 50,
                "job_ttl_seconds": 600,
                "result_ttl_seconds": 600,
                "worker_threads": 1,
            },
        ),
        "sanitization": project_config.get(
            "sanitization",
            {
                "max_chars": 10_000,
                "max_tokens": project_config.get("validation", {}).get(
                    "max_prompt_tokens", 2048
                ),
                "redact_pii": True,
                "log_pii_audit": True,
                "injection_block_score": 0.6,
                "jailbreak_always_block": True,
                "max_injection_attempts": 3,
                "auto_revoke_key": True,
                "custom_blocked_phrases": [],
            },
        ),
        "project_name": project_name,
    }
