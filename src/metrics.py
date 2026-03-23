"""
Milestone 2: Metric formulas.

Keep these pure and testable on toy inputs.
"""

from __future__ import annotations

from typing import Any


def compute_confusion_counts(y_true: Any, y_pred: Any) -> dict[str, int]:
    """Return TP, FP, TN, FN counts."""
    raise NotImplementedError("TODO: implement compute_confusion_counts()")


def compute_accuracy(y_true: Any, y_pred: Any) -> float:
    """Compute standard accuracy."""
    raise NotImplementedError("TODO: implement compute_accuracy()")


def compute_precision_recall_f1(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute precision, recall, and F1."""
    raise NotImplementedError("TODO: implement compute_precision_recall_f1()")


def compute_balanced_accuracy(y_true: Any, y_pred: Any) -> float:
    """Compute balanced accuracy."""
    raise NotImplementedError("TODO: implement compute_balanced_accuracy()")


def summarize_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Build one dictionary of all metrics from predictions."""
    raise NotImplementedError("TODO: implement summarize_metrics()")

