"""
Milestone 2: Evaluation orchestration logic (scores + thresholding + metrics).

This module should contain reusable helpers that `scripts/run_eval.py` calls.
"""

from __future__ import annotations

from typing import Any

from src.config import Config
from src.similarity_score import get_cosine_similarity_batch, euclidean_distance_batch, build_embedding_batches
# from src.io_utils import write_experiment_pairs_with_scores


def compute_similarity_scores(pair_df: Any, config: Config, similarity_type: str) -> Any:
    """
    Compute similarity scores for a given pair of images.
    """
    #create embedding batches 
    pair_left, pair_right, pair_label = build_embedding_batches(pair_df, config)
    #get cosine similarity using config epsilon
    if similarity_type == "cosine":
        scores = get_cosine_similarity_batch(pair_left, pair_right, config)
    elif similarity_type == "euclidean":
        scores = euclidean_distance_batch(pair_left, pair_right)
    else:
        raise ValueError(f"Invalid similarity type: {similarity_type}")

    # Attach score to each dict in the list (works for List[Dict])
    for i, row in enumerate(pair_df):
        row["similarity_score"] = float(scores[i])

    return pair_df

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

