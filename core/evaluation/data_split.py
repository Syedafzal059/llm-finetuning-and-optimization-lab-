"""Held-out test set splitting for evaluation."""

from __future__ import annotations

import random
from typing import Any


def split_test_set(
    data: list[dict[str, Any]],
    test_split: float,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return the last ``test_split`` fraction of shuffled data as the test set."""
    if test_split <= 0 or len(data) < 2:
        return data

    shuffled = data.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    test_count = max(1, int(len(shuffled) * test_split))
    return shuffled[-test_count:]
