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

def get_best_threshold(pair_similarity_scores: pd.DataFrame, config: Config) -> float:
    ## get threshold range from config
    ## divide that range with 0.1 increment 
    ## for each threshold range 
    ## apply the threshold to the similarity scores and create a predicted_label column
    ## calculate the metrics for each threshold 
    ## return the threshold with the best metrics
    threshold_range = config.similarity_threshold.range
    threshold_increment = config.similarity_threshold.increment
    thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_increment)
    threshold_metrics = []
    best_threshold = None
    best_accuracy = 0
    best_metrics = {}
    for threshold in thresholds:
        pair_similarity_scores["predicted_label"] = (pair_similarity_scores["similarity_score"] > threshold).astype(int)
        metrics = compute_metrics(pair_similarity_scores)
        metrics["threshold"] = threshold
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_threshold = threshold
            best_metrics = metrics
        threshold_metrics.append(metrics)
    return best_threshold, best_metrics, threshold_metrics

def get_default_threshold(config: Config) -> float:
    return config.similarity_threshold.default

def evaluate_at_threshold(pair_similarity_scores: pd.DataFrame, threshold: float, config: Config) -> dict:
    pair_similarity_scores["predicted_label"] = (pair_similarity_scores["similarity_score"] > threshold).astype(int)
    metrics = compute_metrics(pair_similarity_scores)
    return metrics

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

