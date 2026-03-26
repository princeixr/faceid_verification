"""
Milestone 2: Evaluation orchestration logic (scores + thresholding + metrics).

This module should contain reusable helpers that `scripts/run_eval.py` calls.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.config import Config
from src.similarity_score import get_cosine_similarity_batch, euclidean_distance_batch, build_embedding_batches
from src.metrics import summarize_metrics
from src.thresholding import apply_threshold
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
    df = pairs_df.copy() if isinstance(pairs_df, pd.DataFrame) else pd.DataFrame(pairs_df)
    if "similarity_score" not in df.columns:
        raise ValueError("pairs_df must contain similarity_score column")
    if "label" not in df.columns:
        raise ValueError("pairs_df must contain label column")

    y_true = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int).to_numpy()
    y_pred = apply_threshold(df["similarity_score"].to_numpy(), threshold, higher_means_same=higher_means_same)
    summary = summarize_metrics(y_true, y_pred)

    predictions = attach_predictions(df, y_pred, threshold=threshold)
    return {
        "metrics": {
            k: v for k, v in summary.items() if k != "confusion_matrix"
        },
        "confusion": summary["confusion_matrix"],
        "predictions": predictions,
    }


def run_threshold_sweep(pairs_df: Any, thresholds: list[float], *, higher_means_same: bool) -> Any:
    """Loop over thresholds and record metrics for each threshold."""
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        evaluated = evaluate_at_threshold(
            pairs_df,
            threshold=float(threshold),
            higher_means_same=higher_means_same,
        )
        row = {
            "threshold": float(threshold),
            **evaluated["metrics"],
            "confusion_matrix": evaluated["confusion"],
        }
        results.append(row)
    return pd.DataFrame(results)


def attach_predictions(pairs_df: Any, y_pred: Any, *, threshold: float) -> Any:
    """Attach prediction columns to the table and record the threshold used."""
    df = pairs_df.copy() if isinstance(pairs_df, pd.DataFrame) else pd.DataFrame(pairs_df)
    pred_arr = np.asarray(y_pred).astype(int)
    if len(df) != len(pred_arr):
        raise ValueError(f"Prediction length mismatch: rows={len(df)}, y_pred={len(pred_arr)}")

    df["y_pred"] = pred_arr
    df["threshold"] = float(threshold)
    return df

