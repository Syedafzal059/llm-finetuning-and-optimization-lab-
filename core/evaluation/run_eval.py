"""Entry point for post-training model evaluation."""

from __future__ import annotations

import json
import logging

from core.evaluation.evaluator import ModelEvaluator
from core.evaluation.types import EvalReport
from core.evaluation.report_generator import ReportGenerator
from core.utils.config_loader import get_logs_dir, load_project_config
from core.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def print_summary(report: EvalReport) -> None:
    """Print a concise evaluation summary to the console."""
    print("\n" + "=" * 50)
    print(f"EVALUATION COMPLETE: {report.project_name}")
    print("=" * 50)
    print(
        f"ROUGE-1:    {report.base_rouge1:.3f} → {report.ft_rouge1:.3f} "
        f"(+{report.rouge1_improvement:.0f}%)",
    )
    print(
        f"BLEU:       {report.base_bleu:.3f} → {report.ft_bleu:.3f} "
        f"(+{report.bleu_improvement:.0f}%)",
    )
    print(
        f"Perplexity: {report.base_perplexity:.1f} → {report.ft_perplexity:.1f} "
        f"(-{report.perplexity_reduction:.0f}%)",
    )
    if report.base_exact_match is not None and report.ft_exact_match is not None:
        print(
            f"Exact Match: {report.base_exact_match * 100:.0f}% → "
            f"{report.ft_exact_match * 100:.0f}%",
        )
    if not report.adapter_loaded:
        print("NOTE: No adapter loaded — fine-tuned metrics mirror base model.")
        print("      Train first: python run.py --project "
              f"{report.project_name} --mode qlora")
    if report.metric_warnings:
        print("\nWarnings:")
        for warning in report.metric_warnings:
            print(f"  - {warning}")
    print("=" * 50)


def run_evaluation(
    project_name: str,
    adapter_override: str | None = None,
) -> EvalReport:
    """
    Run full evaluation for a project and write JSON + markdown reports.

    Args:
        project_name: Project folder under ``projects/``.
        adapter_override: Optional adapter subdirectory (e.g. ``v2``) to evaluate.
    """
    setup_logger(project_name, "eval")
    logger.info("Starting evaluation: %s", project_name)

    config = load_project_config(project_name)
    eval_logger = logging.getLogger(f"{project_name}.eval")
    evaluator = ModelEvaluator(
        project_name,
        config,
        adapter_override,
        logger=eval_logger,
    )
    report = evaluator.evaluate()

    generator = ReportGenerator()
    md_report = generator.generate(report)

    report_dir = get_logs_dir(config)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "eval_report.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)

    md_path = report_dir / "eval_report.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(md_report)

    print_summary(report)
    logger.info("Report saved to %s", report_dir)
    return report
