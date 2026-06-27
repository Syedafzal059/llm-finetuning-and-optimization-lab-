"""Evaluation report dataclasses (no model dependencies)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SampleComparison:
    """One test sample with base and fine-tuned model outputs."""

    input_text: str
    expected_output: str
    base_output: str
    fine_tuned_output: str


@dataclass
class EvalReport:
    """Structured evaluation results for JSON export and markdown reports."""

    project_name: str
    eval_date: str
    num_test_samples: int
    base_model: str
    adapter_path: str

    base_rouge1: float
    base_rouge2: float
    base_rougeL: float
    ft_rouge1: float
    ft_rouge2: float
    ft_rougeL: float

    base_bleu: float
    ft_bleu: float

    base_perplexity: float
    ft_perplexity: float

    base_latency_ms: float
    ft_latency_ms: float

    rouge1_improvement: float
    bleu_improvement: float
    perplexity_reduction: float

    base_exact_match: float | None = None
    ft_exact_match: float | None = None
    adapter_loaded: bool = True
    metric_warnings: list[str] = field(default_factory=list)
    sample_comparisons: list[SampleComparison] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report, converting nested dataclasses to dicts."""
        payload = asdict(self)
        payload["sample_comparisons"] = [
            asdict(sample) for sample in self.sample_comparisons
        ]
        return payload
