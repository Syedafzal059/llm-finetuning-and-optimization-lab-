"""Universal entry point for training, inference, and serving."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.inference.generate import generate_text
from core.inference.model_loader import load_model, load_model_with_lora, resolve_device
from core.training.train_lora import train_lora
from core.training.train_qlora import train_qlora
from core.training.train_sft import train_sft
from core.utils.config_loader import get_model_output_dir, load_project_config
from core.utils.logger import setup_logger


def _resolve_adapter_dir(config: dict) -> Path | None:
    """Return the adapter directory when LoRA/QLoRA weights exist."""
    mode = config.get("training", {}).get("mode", "qlora")
    if mode not in ("lora", "qlora"):
        return None

    adapter_dir = get_model_output_dir(config, mode)
    if (adapter_dir / "adapter_config.json").is_file():
        return adapter_dir
    return None


def run_inference(project_name: str, prompt: str | None) -> None:
    """Run local inference for a project using trained adapters when available."""
    config = load_project_config(project_name)
    logger = setup_logger(project_name, "training")
    requested_device = config.get("model", {}).get("device", "cpu")
    device = resolve_device(requested_device)
    if device != requested_device:
        logger.warning(
            "CUDA requested but unavailable; falling back to CPU inference.",
        )

    if prompt is None:
        prompt = (
            "### Instruction:\n"
            "Summarize the following clinical note into a structured SOAP format.\n\n"
            "### Input:\n"
            "Patient presented with chest pain and shortness of breath.\n\n"
            "### Response:\n"
        )

    adapter_dir = _resolve_adapter_dir(config)
    if adapter_dir is not None:
        logger.info("Loading base model with adapters from %s", adapter_dir)
        model, tokenizer = load_model_with_lora(
            config["base_model"],
            str(adapter_dir),
            device,
        )
    else:
        logger.info("No adapters found; loading base model only.")
        model, tokenizer = load_model(config["base_model"], device)

    output = generate_text(model, tokenizer, prompt, config)
    print("\n=== OUTPUT ===\n")
    print(output)


def run_serve(project_name: str, port: int | None = None) -> None:
    """Start the FastAPI + vLLM serving layer for a project."""
    import uvicorn

    from core.serving.auth import load_project_env
    from core.serving.api import create_app

    load_project_env(project_name)
    config = load_project_config(project_name)
    serving_cfg = config.get("serving", {})
    host = serving_cfg.get("host", "0.0.0.0")
    resolved_port = port if port is not None else serving_cfg.get("port", 8000)

    app = create_app(project_name)
    uvicorn.run(app, host=host, port=resolved_port)


def main() -> None:
    """Parse CLI arguments and dispatch to the requested mode."""
    parser = argparse.ArgumentParser(
        description="Universal entry point for LLM fine-tuning lab.",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Project folder name under projects/ (e.g. clinical-notes)",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["sft", "lora", "qlora", "inference", "serve", "eval"],
        help="Operation mode: training, evaluation, local inference, or API server",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text for inference mode",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port override for serve mode (default: from project config)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Adapter subdirectory for eval mode (e.g. v2, lora, qlora)",
    )
    args = parser.parse_args()

    if args.mode == "sft":
        train_sft(args.project)
    elif args.mode == "lora":
        train_lora(args.project)
    elif args.mode == "qlora":
        train_qlora(args.project)
    elif args.mode == "inference":
        run_inference(args.project, args.prompt)
    elif args.mode == "serve":
        run_serve(args.project, port=args.port)
    elif args.mode == "eval":
        from core.evaluation.run_eval import run_evaluation

        run_evaluation(args.project, adapter_override=args.adapter)


if __name__ == "__main__":
    main()
