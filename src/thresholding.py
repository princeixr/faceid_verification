"""
Milestone 2: Thresholding logic (scores -> decisions) and threshold selection helpers.
"""

from __future__ import annotations

from typing import Any, Iterable

from src.config import Config
import pandas as pd
import numpy as np
from src.metrics import summarize_metrics

def compute_metrics(pair_similarity_scores: pd.DataFrame) -> dict:
    y_pred = pair_similarity_scores["predicted_label"].astype(int)
    y_true = pair_similarity_scores["label"].astype(int)
    accuracy_report = summarize_metrics(y_true, y_pred)
    return accuracy_report

def get_best_threshold(
    pair_similarity_scores: pd.DataFrame,
    config: Config,
    *,
    rule: str = "max_accuracy",
) -> tuple[float, dict, list[dict]]:
    ## get threshold range from config
    ## divide that range with 0.1 increment 
    ## for each threshold range 
    ## apply the threshold to the similarity scores and create a predicted_label column
    ## calculate the metrics for each threshold 
    ## return the threshold with the best metrics
    threshold_cfg = config._config.get("similarity_threshold", {})
    threshold_range = threshold_cfg.get("range", [0.3, 0.95])
    threshold_increment = float(threshold_cfg.get("increment", 0.05))
    thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_increment)
    threshold_metrics = []
    best_threshold = None
    best_accuracy = float("-inf")
    best_metrics = {}
    metric_key_map = {
        "max_accuracy": "accuracy",
        "max_balanced_accuracy": "balanced_accuracy",
        "max_f1": "f1_score",
    }
    metric_key = metric_key_map.get(rule, "accuracy")

    for threshold in thresholds:
        pair_similarity_scores["predicted_label"] = (pair_similarity_scores["similarity_score"] > threshold).astype(int)
        metrics = compute_metrics(pair_similarity_scores)
        metrics["threshold"] = threshold
        current_value = float(metrics.get(metric_key, float("-inf")))
        if current_value > best_accuracy:
            best_accuracy = current_value
            best_threshold = float(threshold)
            best_metrics = metrics
        threshold_metrics.append(metrics)
    if best_threshold is None:
        raise ValueError("No valid threshold found during sweep")
    return float(best_threshold), best_metrics, threshold_metrics

def get_default_threshold(config: Config) -> float:
    threshold_cfg = config._config.get("similarity_threshold", {})
    return float(threshold_cfg.get("default", 0.7))

def evaluate_at_threshold(pair_similarity_scores: pd.DataFrame, threshold: float, config: Config) -> dict:
    pair_similarity_scores["predicted_label"] = (pair_similarity_scores["similarity_score"] > threshold).astype(int)
    metrics = compute_metrics(pair_similarity_scores)
    return metrics

def apply_threshold(scores: Any, threshold: float, *, higher_means_same: bool) -> Any:
    """Convert continuous scores into binary decisions."""
    arr = np.asarray(scores, dtype=float)
    if higher_means_same:
        return (arr >= threshold).astype(int)
    return (arr <= threshold).astype(int)


def generate_threshold_grid(config: Config) -> list[float]:
    """Create the list of thresholds to test in sweep mode."""
    threshold_cfg = config._config.get("similarity_threshold", {})
    threshold_range = threshold_cfg.get("range", [0.3, 0.95])
    threshold_increment = float(threshold_cfg.get("increment", 0.05))
    start = float(threshold_range[0])
    end = float(threshold_range[1])
    if threshold_increment <= 0:
        raise ValueError("similarity_threshold.increment must be > 0")
    if end <= start:
        raise ValueError("similarity_threshold.range must have end > start")
    return np.arange(start, end + (threshold_increment / 2.0), threshold_increment).tolist()


def select_threshold(sweep_table: Any, rule: str) -> float:
    """Choose one operating threshold using a fixed rule (e.g., max balanced accuracy)."""
    rows = sweep_table.to_dict(orient="records") if isinstance(sweep_table, pd.DataFrame) else list(sweep_table)
    if not rows:
        raise ValueError("sweep_table is empty")

    metric_map = {
        "max_accuracy": "accuracy",
        "max_balanced_accuracy": "balanced_accuracy",
        "max_f1": "f1_score",
    }
    metric_key = metric_map.get(rule, "accuracy")
    best_row = max(rows, key=lambda r: float(r.get(metric_key, float("-inf"))))
    if "threshold" not in best_row:
        raise ValueError("Each sweep row must include threshold")
    return float(best_row["threshold"])


def is_higher_score_same_person(config: Config) -> bool:
    """Return score direction convention."""
    similarity_cfg = config._config.get("similarity", {})
    return str(similarity_cfg.get("direction", "higher")).lower() != "lower"

