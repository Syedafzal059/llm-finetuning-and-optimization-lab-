"""Generate human-readable markdown evaluation reports."""

from __future__ import annotations

from core.evaluation.types import EvalReport


class ReportGenerator:
    """Build markdown reports from EvalReport dataclass instances."""

    def generate(self, report: EvalReport) -> str:
        """Return a full markdown evaluation report."""
        sections = [
            self._header(report),
            self._summary_table(report),
        ]

        if report.base_exact_match is not None and report.ft_exact_match is not None:
            sections.append(self._exact_match_section(report))

        sections.append(self._example_outputs(report))
        sections.append(self._verdict(report))

        if report.metric_warnings:
            sections.append(self._warnings(report))

        return "\n\n".join(sections) + "\n"

    def compute_verdict_label(self, report: EvalReport) -> str:
        """Return a short verdict token for API responses."""
        return verdict_from_metrics(
            base_rouge1=report.base_rouge1,
            ft_rouge1=report.ft_rouge1,
            base_bleu=report.base_bleu,
            ft_bleu=report.ft_bleu,
            base_perplexity=report.base_perplexity,
            ft_perplexity=report.ft_perplexity,
            adapter_loaded=report.adapter_loaded,
        )

    @staticmethod
    def verdict_from_dict(report_data: dict) -> str:
        """Compute verdict label from a serialized eval report dict."""
        return verdict_from_metrics(
            base_rouge1=float(report_data.get("base_rouge1", 0.0)),
            ft_rouge1=float(report_data.get("ft_rouge1", 0.0)),
            base_bleu=float(report_data.get("base_bleu", 0.0)),
            ft_bleu=float(report_data.get("ft_bleu", 0.0)),
            base_perplexity=float(report_data.get("base_perplexity", 0.0)),
            ft_perplexity=float(report_data.get("ft_perplexity", 0.0)),
            adapter_loaded=bool(report_data.get("adapter_loaded", False)),
        )

    def _header(self, report: EvalReport) -> str:
        short_model = report.base_model.split("/")[-1]
        return (
            f"# Evaluation Report — {report.project_name}\n"
            f"Date: {report.eval_date}\n"
            f"Test samples: {report.num_test_samples}\n\n"
            f"## Summary\n"
            f"Fine-tuned model vs base {short_model}"
        )

    def _summary_table(self, report: EvalReport) -> str:
        rows = [
            self._metric_row(
                "ROUGE-1",
                report.base_rouge1,
                report.ft_rouge1,
                report.rouge1_improvement,
                higher_is_better=True,
            ),
            self._metric_row(
                "ROUGE-2",
                report.base_rouge2,
                report.ft_rouge2,
                pct_improvement(report.base_rouge2, report.ft_rouge2),
                higher_is_better=True,
            ),
            self._metric_row(
                "ROUGE-L",
                report.base_rougeL,
                report.ft_rougeL,
                pct_improvement(report.base_rougeL, report.ft_rougeL),
                higher_is_better=True,
            ),
            self._metric_row(
                "BLEU",
                report.base_bleu,
                report.ft_bleu,
                report.bleu_improvement,
                higher_is_better=True,
            ),
            self._metric_row(
                "Perplexity",
                report.base_perplexity,
                report.ft_perplexity,
                -report.perplexity_reduction,
                higher_is_better=False,
            ),
            self._metric_row(
                "Latency",
                report.base_latency_ms,
                report.ft_latency_ms,
                pct_improvement(report.base_latency_ms, report.ft_latency_ms)
                if report.ft_latency_ms < report.base_latency_ms
                else -pct_improvement(report.ft_latency_ms, report.base_latency_ms),
                higher_is_better=False,
                suffix="ms",
            ),
        ]

        header = (
            "| Metric | Base Model | Fine-tuned | Improvement |\n"
            "|--------|-----------|------------|-------------|"
        )
        return header + "\n" + "\n".join(rows)

    def _exact_match_section(self, report: EvalReport) -> str:
        base_pct = report.base_exact_match * 100 if report.base_exact_match else 0
        ft_pct = report.ft_exact_match * 100 if report.ft_exact_match else 0
        return (
            "## Exact Match (ICD-10 Primary Code)\n\n"
            f"- Base model: {base_pct:.0f}% exact match\n"
            f"- Fine-tuned: {ft_pct:.0f}% exact match"
        )

    def _example_outputs(self, report: EvalReport) -> str:
        if not report.sample_comparisons:
            return "## Example Outputs\n\nNo sample outputs available."

        blocks: list[str] = ["## Example Outputs"]
        for index, sample in enumerate(report.sample_comparisons, start=1):
            blocks.append(
                f"### Sample {index}\n"
                f"**Input:**\n{sample.input_text}\n\n"
                f"**Expected Output:**\n{sample.expected_output}\n\n"
                f"**Base Model Output:**\n{sample.base_output}\n\n"
                f"**Fine-tuned Output:**\n{sample.fine_tuned_output}\n\n"
                "---"
            )
        return "\n\n".join(blocks)

    def _verdict(self, report: EvalReport) -> str:
        label = self.compute_verdict_label(report)
        messages = {
            "production_ready": (
                "Fine-tuned model shows significant improvement across all metrics.\n"
                "Suitable for production deployment."
            ),
            "promising": (
                "Fine-tuned model shows meaningful improvement on key metrics.\n"
                "Consider additional training or validation before production."
            ),
            "needs_improvement": (
                "Fine-tuned model did not clearly outperform the base model.\n"
                "Review training data, hyperparameters, or adapter configuration."
            ),
            "adapter_missing": (
                "No fine-tuned adapter was found. Only base model metrics are reliable.\n"
                "Train an adapter and re-run evaluation."
            ),
        }
        return f"## Verdict\n{messages.get(label, messages['needs_improvement'])}"

    def _warnings(self, report: EvalReport) -> str:
        lines = "\n".join(f"- {warning}" for warning in report.metric_warnings)
        return f"## Warnings\n{lines}"

    def _metric_row(
        self,
        name: str,
        base: float,
        fine_tuned: float,
        improvement_pct: float,
        *,
        higher_is_better: bool,
        suffix: str = "",
    ) -> str:
        base_display = self._format_value(base, suffix)
        ft_display = self._format_value(fine_tuned, suffix)

        if higher_is_better:
            sign = "+" if improvement_pct >= 0 else ""
            improvement_display = f"{sign}{improvement_pct:.0f}%"
        else:
            reduction = abs(improvement_pct)
            sign = "-" if reduction > 0 else "+"
            improvement_display = f"{sign}{reduction:.0f}%"

        return f"| {name} | {base_display} | {ft_display} | {improvement_display} |"

    @staticmethod
    def _format_value(value: float, suffix: str) -> str:
        if suffix == "ms":
            return f"{value:.0f}{suffix}"
        if value >= 10:
            return f"{value:.1f}{suffix}"
        return f"{value:.2f}{suffix}"


def pct_improvement(base: float, improved: float) -> float:
    """Return percentage improvement from base to improved."""
    if base == 0:
        return 100.0 if improved > 0 else 0.0
    return ((improved - base) / base) * 100.0


def verdict_from_metrics(
    *,
    base_rouge1: float,
    ft_rouge1: float,
    base_bleu: float,
    ft_bleu: float,
    base_perplexity: float,
    ft_perplexity: float,
    adapter_loaded: bool,
) -> str:
    """Return verdict label from raw metric values."""
    if not adapter_loaded:
        return "adapter_missing"

    improved_rouge = ft_rouge1 > base_rouge1 * 1.2
    reduced_ppl = ft_perplexity < base_perplexity * 0.9

    if improved_rouge and reduced_ppl:
        return "production_ready"
    if improved_rouge or ft_bleu > base_bleu * 1.2:
        return "promising"
    return "needs_improvement"
