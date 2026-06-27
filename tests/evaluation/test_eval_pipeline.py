"""Tests for evaluation metrics and report generation."""

from __future__ import annotations

import pytest
from core.evaluation.data_split import split_test_set
from core.evaluation.metrics import (
    compute_exact_match_accuracy,
    extract_primary_icd_code,
    pct_improvement,
    pct_reduction,
)
from core.evaluation.prompt_builder import build_inference_prompt, get_reference_output
from core.evaluation.report_generator import ReportGenerator
from core.evaluation.types import EvalReport, SampleComparison


def test_split_test_set_uses_last_twenty_percent() -> None:
    data = [{"id": index} for index in range(10)]
    test_set = split_test_set(data, test_split=0.2, seed=42)
    assert len(test_set) == 2


def test_extract_primary_icd_code() -> None:
    text = "Primary: I21.9 — Acute myocardial infarction\nSecondary: R07.9"
    assert extract_primary_icd_code(text) == "I21.9"


def test_exact_match_accuracy() -> None:
    references = [
        "Primary: I21.9 — AMI",
        "Primary: J18.9 — Pneumonia",
    ]
    hypotheses = [
        "Primary: I21.9 — AMI",
        "Primary: J18.9 — Pneumonia",
    ]
    assert compute_exact_match_accuracy(references, hypotheses) == 1.0


def test_pct_improvement_and_reduction() -> None:
    assert pct_improvement(0.2, 0.6) == pytest.approx(200.0)
    assert pct_reduction(10.0, 2.0) == pytest.approx(80.0)


def test_build_inference_prompt_clinical_notes() -> None:
    sample = {
        "clinical_note": "Patient with chest pain.",
        "summary": "S: Chest pain.",
    }
    prompt = build_inference_prompt("clinical-notes", sample)
    assert "Patient with chest pain." in prompt
    assert "S: Chest pain." not in prompt
    assert prompt.endswith("### Response:\n")


def test_report_generator_includes_summary_table() -> None:
    report = EvalReport(
        project_name="clinical-notes",
        eval_date="2026-06-27",
        num_test_samples=2,
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        adapter_path="projects/clinical-notes/model/qlora",
        base_rouge1=0.18,
        base_rouge2=0.09,
        base_rougeL=0.15,
        ft_rouge1=0.67,
        ft_rouge2=0.54,
        ft_rougeL=0.61,
        base_bleu=0.07,
        ft_bleu=0.48,
        base_perplexity=45.2,
        ft_perplexity=8.3,
        base_latency_ms=1240.0,
        ft_latency_ms=980.0,
        rouge1_improvement=272.0,
        bleu_improvement=586.0,
        perplexity_reduction=82.0,
        adapter_loaded=True,
        sample_comparisons=[
            SampleComparison(
                input_text="Chest pain note",
                expected_output="S: Chest pain",
                base_output="Some chest issues",
                fine_tuned_output="S: Acute chest pain",
            ),
        ],
    )
    markdown = ReportGenerator().generate(report)
    assert "# Evaluation Report — clinical-notes" in markdown
    assert "| ROUGE-1 |" in markdown
    assert "## Example Outputs" in markdown
    assert ReportGenerator().compute_verdict_label(report) == "production_ready"


def test_get_reference_output_medical_coding() -> None:
    sample = {
        "clinical_description": "Chest pain",
        "icd_codes": "Primary: I21.9",
    }
    assert get_reference_output(sample) == "Primary: I21.9"
