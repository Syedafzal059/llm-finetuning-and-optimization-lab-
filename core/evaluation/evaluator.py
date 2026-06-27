"""Compare base and fine-tuned models on a held-out test set."""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import torch

from core.evaluation.data_split import split_test_set
from core.evaluation.metrics import (
    average_metric,
    check_metric_dependencies,
    compute_bleu,
    compute_exact_match_accuracy,
    compute_perplexity,
    compute_rouge,
    pct_improvement,
    pct_reduction,
)
from core.evaluation.prompt_builder import (
    build_inference_prompt,
    get_input_text,
    get_reference_output,
)
from core.evaluation.types import EvalReport, SampleComparison
from core.inference.generate import generate_text
from core.inference.model_loader import load_model, load_model_with_lora, resolve_device
from core.training.dataset_loader import load_json_dataset
from core.utils.config_loader import (
    get_model_output_dir,
    get_project_dir,
    load_prompt_template,
    resolve_path,
)

DEFAULT_TEST_SPLIT = 0.2
MEDICAL_CODING_PROJECT = "medical-coding"


def resolve_adapter_path(
    config: dict[str, Any],
    adapter_override: str | None = None,
) -> Path | None:
    """
    Resolve the adapter checkpoint directory.

    ``adapter_override`` may be a training mode (``lora``, ``qlora``) or a
    custom subdirectory name (``v2`` → ``projects/X/model/v2/``).
    """
    project_dir = get_project_dir(config)

    if adapter_override:
        candidate = project_dir / "model" / adapter_override
        if (candidate / "adapter_config.json").is_file():
            return candidate
        mode_candidate = get_model_output_dir(config, adapter_override)
        if (mode_candidate / "adapter_config.json").is_file():
            return mode_candidate
        return None

    mode = config.get("training", {}).get("mode", "qlora")
    if mode not in ("lora", "qlora"):
        return None

    adapter_dir = get_model_output_dir(config, mode)
    if (adapter_dir / "adapter_config.json").is_file():
        return adapter_dir

    configured = config.get("adapter_path")
    if configured:
        configured_path = resolve_path(configured)
        if (configured_path / "adapter_config.json").is_file():
            return configured_path

    return None


def _extract_generation(full_output: str, prompt: str) -> str:
    """Strip the prompt prefix from a decoded generation."""
    if full_output.startswith(prompt):
        return full_output[len(prompt) :].strip()

    for marker in ("### Response:", "### ICD-10 Codes:", "### Support Response:"):
        if marker in full_output:
            return full_output.split(marker, 1)[1].strip()

    return full_output.strip()


class ModelEvaluator:
    """Run base and fine-tuned models on the same test set and compute metrics."""

    def __init__(
        self,
        project_name: str,
        config: dict[str, Any],
        adapter_override: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.project_name = project_name
        self.config = config
        self.adapter_override = adapter_override
        self.log = logger or logging.getLogger(__name__)
        self.metric_warnings: list[str] = check_metric_dependencies()

        data_cfg = config.get("data", {})
        train_path = data_cfg.get(
            "train_path",
            f"projects/{project_name}/data/raw/sample_dataset.json",
        )
        fallback_path = get_project_dir(config) / "data" / "raw" / "sample.json"
        dataset_path = resolve_path(train_path)
        if not dataset_path.is_file() and fallback_path.is_file():
            dataset_path = fallback_path
            self.log.warning(
                "Dataset not found at %s; using %s",
                train_path,
                fallback_path,
            )

        raw_data = load_json_dataset(str(dataset_path))
        test_split = data_cfg.get("test_split", DEFAULT_TEST_SPLIT)
        self.test_samples = split_test_set(raw_data, test_split)
        self.format_prompt = load_prompt_template(project_name)

        requested_device = config.get("model", {}).get("device", "cpu")
        self.device = resolve_device(requested_device)
        if self.device != requested_device:
            self.log.warning(
                "CUDA requested but unavailable; using CPU for evaluation.",
            )

        self.base_model_name = config["base_model"]
        self.adapter_path = resolve_adapter_path(config, adapter_override)
        self.adapter_loaded = self.adapter_path is not None

        self.log.info(
            "Loaded %d test samples from %s (test_split=%.0f%%)",
            len(self.test_samples),
            dataset_path,
            test_split * 100,
        )

    def _load_models(
        self,
    ) -> tuple[torch.nn.Module, Any, torch.nn.Module | None, Any]:
        """Load base model and optionally the fine-tuned adapter model."""
        self.log.info("Loading base model: %s", self.base_model_name)
        base_model, base_tokenizer = load_model(self.base_model_name, self.device)
        base_model.eval()

        ft_model: torch.nn.Module | None = None
        ft_tokenizer = base_tokenizer

        if self.adapter_loaded and self.adapter_path is not None:
            self.log.info("Loading fine-tuned adapter from %s", self.adapter_path)
            try:
                ft_model, ft_tokenizer = load_model_with_lora(
                    self.base_model_name,
                    str(self.adapter_path),
                    self.device,
                )
            except Exception as exc:
                self.log.warning(
                    "Failed to load adapter at %s: %s",
                    self.adapter_path,
                    exc,
                )
                self.metric_warnings.append(
                    f"Adapter load failed: {exc}. Fine-tuned metrics unavailable.",
                )
                self.adapter_loaded = False
        else:
            self.log.warning(
                "No adapter found for project %s; fine-tuned metrics will mirror base.",
                self.project_name,
            )
            self.metric_warnings.append(
                "No adapter checkpoint found. Fine-tuned metrics mirror base model.",
            )

        return base_model, base_tokenizer, ft_model, ft_tokenizer

    def _generate_with_timing(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        prompt: str,
    ) -> tuple[str, float]:
        """Generate text and return the completion plus latency in milliseconds."""
        start = time.perf_counter()
        full_output = generate_text(model, tokenizer, prompt, self.config)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return _extract_generation(full_output, prompt), latency_ms

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        label: str,
    ) -> dict[str, Any]:
        """Run one model on all test samples and aggregate metrics."""
        rouge1_scores: list[float] = []
        rouge2_scores: list[float] = []
        rougeL_scores: list[float] = []
        bleu4_scores: list[float] = []
        perplexities: list[float] = []
        latencies: list[float] = []
        references: list[str] = []
        hypotheses: list[str] = []

        self.log.info("Running %s model on %d samples", label, len(self.test_samples))

        for index, sample in enumerate(self.test_samples, start=1):
            reference = get_reference_output(sample)
            prompt = build_inference_prompt(
                self.project_name,
                sample,
                self.format_prompt,
            )

            hypothesis, latency_ms = self._generate_with_timing(
                model,
                tokenizer,
                prompt,
            )

            rouge = compute_rouge(reference, hypothesis)
            bleu = compute_bleu(reference, hypothesis)
            ppl = compute_perplexity(
                model,
                tokenizer,
                reference,
                self.device,
                max_length=self.config.get("training", {}).get("max_length", 512),
            )

            rouge1_scores.append(rouge["rouge1"])
            rouge2_scores.append(rouge["rouge2"])
            rougeL_scores.append(rouge["rougeL"])
            bleu4_scores.append(bleu["bleu4"])
            perplexities.append(ppl)
            latencies.append(latency_ms)
            references.append(reference)
            hypotheses.append(hypothesis)

            if index % 5 == 0 or index == len(self.test_samples):
                self.log.info(
                    "%s progress: %d/%d samples",
                    label,
                    index,
                    len(self.test_samples),
                )

        exact_match: float | None = None
        if self.project_name == MEDICAL_CODING_PROJECT:
            exact_match = compute_exact_match_accuracy(references, hypotheses)

        return {
            "rouge1": average_metric(rouge1_scores),
            "rouge2": average_metric(rouge2_scores),
            "rougeL": average_metric(rougeL_scores),
            "bleu": average_metric(bleu4_scores),
            "perplexity": average_metric(perplexities),
            "latency_ms": average_metric(latencies),
            "exact_match": exact_match,
            "references": references,
            "hypotheses": hypotheses,
        }

    def _build_sample_comparisons(
        self,
        base_hypotheses: list[str],
        ft_hypotheses: list[str],
    ) -> list[SampleComparison]:
        """Collect up to five example outputs for the markdown report."""
        comparisons: list[SampleComparison] = []
        for sample, base_out, ft_out in zip(
            self.test_samples,
            base_hypotheses,
            ft_hypotheses,
            strict=True,
        ):
            comparisons.append(
                SampleComparison(
                    input_text=get_input_text(sample),
                    expected_output=get_reference_output(sample),
                    base_output=base_out,
                    fine_tuned_output=ft_out,
                ),
            )
        return comparisons[:5]

    def evaluate(self) -> EvalReport:
        """Run both models, compute metrics, save JSON report, and return results."""
        base_model, base_tokenizer, ft_model, ft_tokenizer = self._load_models()

        base_results = self._evaluate_model(base_model, base_tokenizer, "base")

        if ft_model is not None:
            ft_results = self._evaluate_model(ft_model, ft_tokenizer, "fine-tuned")
        else:
            ft_results = base_results.copy()
            ft_results["hypotheses"] = list(base_results["hypotheses"])

        comparisons = self._build_sample_comparisons(
            base_results["hypotheses"],
            ft_results["hypotheses"],
        )

        adapter_display = (
            str(self.adapter_path) if self.adapter_path else "none"
        )

        report = EvalReport(
            project_name=self.project_name,
            eval_date=date.today().isoformat(),
            num_test_samples=len(self.test_samples),
            base_model=self.base_model_name,
            adapter_path=adapter_display,
            base_rouge1=base_results["rouge1"],
            base_rouge2=base_results["rouge2"],
            base_rougeL=base_results["rougeL"],
            ft_rouge1=ft_results["rouge1"],
            ft_rouge2=ft_results["rouge2"],
            ft_rougeL=ft_results["rougeL"],
            base_bleu=base_results["bleu"],
            ft_bleu=ft_results["bleu"],
            base_perplexity=base_results["perplexity"],
            ft_perplexity=ft_results["perplexity"],
            base_latency_ms=base_results["latency_ms"],
            ft_latency_ms=ft_results["latency_ms"],
            rouge1_improvement=pct_improvement(
                base_results["rouge1"],
                ft_results["rouge1"],
            ),
            bleu_improvement=pct_improvement(
                base_results["bleu"],
                ft_results["bleu"],
            ),
            perplexity_reduction=pct_reduction(
                base_results["perplexity"],
                ft_results["perplexity"],
            ),
            base_exact_match=base_results["exact_match"],
            ft_exact_match=ft_results["exact_match"],
            adapter_loaded=self.adapter_loaded and ft_model is not None,
            metric_warnings=list(self.metric_warnings),
            sample_comparisons=comparisons,
        )

        return report
