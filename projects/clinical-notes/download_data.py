"""Download medical flashcard data and save as clinical-notes training JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

from core.utils.config_loader import load_project_config, resolve_path

DATASET_ID = "medalpaca/medical_meadow_medical_flashcards"
DEFAULT_LIMIT = 500
MIN_NOTE_LEN = 50
MIN_SUMMARY_LEN = 20


def convert_flashcards(
    examples: list[dict[str, Any]],
    min_note_len: int = MIN_NOTE_LEN,
    min_summary_len: int = MIN_SUMMARY_LEN,
) -> list[dict[str, str]]:
    """Map Hugging Face rows to {clinical_note, summary} training records."""
    converted: list[dict[str, str]] = []
    for ex in examples:
        note = ex.get("input", "").strip()
        summary = ex.get("output", "").strip()
        if len(note) >= min_note_len and len(summary) >= min_summary_len:
            converted.append({"clinical_note": note, "summary": summary})
    return converted


def download_medical_data(
    limit: int = DEFAULT_LIMIT,
    output_path: Path | None = None,
    min_note_len: int = MIN_NOTE_LEN,
    min_summary_len: int = MIN_SUMMARY_LEN,
) -> Path:
    """Download, convert, and save medical training data."""
    print("Downloading real medical data...")

    dataset = load_dataset(DATASET_ID, split="train")
    print(f"Downloaded {len(dataset)} examples")

    converted = convert_flashcards(
        dataset,
        min_note_len=min_note_len,
        min_summary_len=min_summary_len,
    )
    if limit > 0:
        converted = converted[:limit]

    print(f"Using {len(converted)} examples")

    if output_path is None:
        raise ValueError("output_path is required")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, indent=2)

    print(f"Saved to: {output_path}")
    return output_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download medical flashcard data for clinical-notes training.",
    )
    parser.add_argument(
        "--project",
        default="clinical-notes",
        help="Project folder under projects/ (default: clinical-notes)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max examples to keep after filtering (0 = all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data.train_path from project config)",
    )
    parser.add_argument(
        "--min-note-len",
        type=int,
        default=MIN_NOTE_LEN,
        help="Minimum clinical_note character length",
    )
    parser.add_argument(
        "--min-summary-len",
        type=int,
        default=MIN_SUMMARY_LEN,
        help="Minimum summary character length",
    )
    args = parser.parse_args()

    if args.output:
        output_path = resolve_path(args.output)
    else:
        config = load_project_config(args.project)
        output_path = resolve_path(config["data"]["train_path"])

    download_medical_data(
        limit=args.limit,
        output_path=output_path,
        min_note_len=args.min_note_len,
        min_summary_len=args.min_summary_len,
    )


if __name__ == "__main__":
    main()
