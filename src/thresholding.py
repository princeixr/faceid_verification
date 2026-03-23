"""
Milestone 2: Thresholding logic (scores -> decisions) and threshold selection helpers.
"""

from __future__ import annotations

from typing import Any, Iterable

from src.config import Config


def apply_threshold(scores: Any, threshold: float, *, higher_means_same: bool) -> Any:
    """Convert continuous scores into binary decisions."""
    raise NotImplementedError("TODO: implement apply_threshold()")


def generate_threshold_grid(config: Config) -> list[float]:
    """Create the list of thresholds to test in sweep mode."""
    raise NotImplementedError("TODO: implement generate_threshold_grid()")


def select_threshold(sweep_table: Any, rule: str) -> float:
    """Choose one operating threshold using a fixed rule (e.g., max balanced accuracy)."""
    raise NotImplementedError("TODO: implement select_threshold()")


def is_higher_score_same_person(config: Config) -> bool:
    """Return score direction convention."""
    raise NotImplementedError("TODO: implement is_higher_score_same_person()")

