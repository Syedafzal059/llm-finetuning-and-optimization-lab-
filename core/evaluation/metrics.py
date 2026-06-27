"""Metric computation helpers with graceful fallbacks when libraries are missing."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import torch

logger = logging.getLogger(__name__)

PRIMARY_ICD_PATTERN = re.compile(
    r"Primary:\s*([A-Z]\d{2}(?:\.\d+)?)",
    re.IGNORECASE,
)

_rouge_scorer: Any | None = None
_rouge_available: bool | None = None
_nltk_bleu_available: bool | None = None
_SmoothingFunction: Any | None = None


def _get_rouge_scorer() -> Any | None:
    """Lazy-load rouge_score; return None when the package is unavailable."""
    global _rouge_scorer, _rouge_available
    if _rouge_available is False:
        return None
    if _rouge_scorer is not None:
        return _rouge_scorer
    try:
        from rouge_score import rouge_scorer

        _rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )
        _rouge_available = True
        return _rouge_scorer
    except ImportError:
        logger.warning("rouge_score not installed; ROUGE metrics will be skipped.")
        _rouge_available = False
        return None


def _ensure_nltk_bleu() -> bool:
    """Lazy-load NLTK BLEU helpers (whitespace tokenization; no punkt data needed)."""
    global _nltk_bleu_available, _SmoothingFunction
    if _nltk_bleu_available is False:
        return False
    if _nltk_bleu_available is True:
        return True
    try:
        from nltk.translate.bleu_score import SmoothingFunction

        _SmoothingFunction = SmoothingFunction
        _nltk_bleu_available = True
        return True
    except ImportError:
        logger.warning("nltk not installed; BLEU metrics will be skipped.")
        _nltk_bleu_available = False
        return False
    except Exception as exc:
        logger.warning("NLTK BLEU unavailable (%s); BLEU metrics will be skipped.", exc)
        _nltk_bleu_available = False
        return False


def check_metric_dependencies() -> list[str]:
    """Return warnings for any missing optional metric libraries."""
    warnings: list[str] = []
    if _get_rouge_scorer() is None:
        warnings.append(
            "rouge-score not installed — run: pip install rouge-score",
        )
    if not _ensure_nltk_bleu():
        warnings.append(
            "nltk not installed — run: pip install nltk",
        )
    return warnings


def compute_rouge(reference: str, hypothesis: str) -> dict[str, float]:
    """Return ROUGE-1, ROUGE-2, and ROUGE-L F-measures for one sample pair."""
    scorer = _get_rouge_scorer()
    if scorer is None:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_bleu(reference: str, hypothesis: str) -> dict[str, float]:
    """Return BLEU-1, BLEU-2, and BLEU-4 scores for one sample pair."""
    if not _ensure_nltk_bleu():
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}

    from nltk.translate.bleu_score import sentence_bleu

    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}

    smoothing = _SmoothingFunction().method1
    weights = {
        "bleu1": (1.0, 0.0, 0.0, 0.0),
        "bleu2": (0.5, 0.5, 0.0, 0.0),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
    }
    return {
        name: float(
            sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=weight,
                smoothing_function=smoothing,
            )
        )
        for name, weight in weights.items()
    }


def extract_primary_icd_code(text: str) -> str | None:
    """Extract the primary ICD-10 code from a medical-coding model output."""
    match = PRIMARY_ICD_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1).upper()


def compute_exact_match_accuracy(
    references: list[str],
    hypotheses: list[str],
) -> float:
    """Return fraction of samples where the primary ICD-10 code matches exactly."""
    if not references:
        return 0.0

    matches = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        expected = extract_primary_icd_code(reference)
        predicted = extract_primary_icd_code(hypothesis)
        if expected is not None and predicted is not None and expected == predicted:
            matches += 1
    return matches / len(references)


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    device: str,
    max_length: int = 512,
) -> float:
    """Compute perplexity (exp of cross-entropy loss) for a text span."""
    if not text.strip():
        return float("inf")

    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("inf")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    if loss is None or math.isnan(loss.item()):
        return float("inf")
    return float(math.exp(min(loss.item(), 20.0)))


def average_metric(values: list[float]) -> float:
    """Average metric values, ignoring NaN and infinity."""
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 0.0
    return sum(finite) / len(finite)


def pct_improvement(base: float, improved: float) -> float:
    """Return percentage improvement from base to improved (higher is better)."""
    if base == 0:
        return 100.0 if improved > 0 else 0.0
    return ((improved - base) / base) * 100.0


def pct_reduction(base: float, reduced: float) -> float:
    """Return percentage reduction from base to reduced (lower is better)."""
    if base == 0:
        return 0.0
    return ((base - reduced) / base) * 100.0
