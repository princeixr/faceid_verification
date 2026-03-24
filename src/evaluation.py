"""
Milestone 2: Evaluation orchestration logic (scores + thresholding + metrics).

This module should contain reusable helpers that `scripts/run_eval.py` calls.
"""

from __future__ import annotations

from typing import Any

from src.config import Config


def compute_similarity_scores(pairs_df: Any, config: Config) -> Any:
    """
    Compute a score for each pair using the existing embedding + similarity pipeline.

    Note: In your repo, milestone-1 scoring exists in `src/similarity_score.py` and
    `scripts/similarity_lfw.py`. We'll fold that into this function later.
    """
    raise NotImplementedError("TODO: implement compute_similarity_scores()")


def load_or_compute_scores(pairs_df: Any, config: Config) -> Any:
    """Load saved scores if present; otherwise compute them."""
    raise NotImplementedError("TODO: implement load_or_compute_scores()")


def evaluate_at_threshold(pairs_df: Any, *, threshold: float, higher_means_same: bool) -> dict[str, Any]:
    """
    Apply threshold, compute predictions, metrics, confusion counts.

    Expected return shape (used by run_eval.py):
    {
      "metrics": {...},
      "confusion": {"tp":..., "fp":..., "tn":..., "fn":...},
      "predictions": <table with y_true, score, y_pred, threshold, ids/paths>
    }
    """
    raise NotImplementedError("TODO: implement evaluate_at_threshold()")


def run_threshold_sweep(pairs_df: Any, thresholds: list[float], *, higher_means_same: bool) -> Any:
    """Loop over thresholds and record metrics for each threshold."""
    raise NotImplementedError("TODO: implement run_threshold_sweep()")


def attach_predictions(pairs_df: Any, y_pred: Any, *, threshold: float) -> Any:
    """Attach prediction columns to the table and record the threshold used."""
    raise NotImplementedError("TODO: implement attach_predictions()")

