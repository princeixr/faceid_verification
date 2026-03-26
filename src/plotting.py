"""
Milestone 2: Plotting helpers for evaluation artifacts (ROC-style curve, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_roc_style_curve(sweep_table: Any, path: Path) -> None:
    """Create ROC-style plot from sweep results."""
    df = sweep_table.copy() if isinstance(sweep_table, pd.DataFrame) else pd.DataFrame(sweep_table)
    if "confusion_matrix" not in df.columns:
        raise ValueError("sweep_table must include confusion_matrix column")

    fprs: list[float] = []
    tprs: list[float] = []
    for cm in df["confusion_matrix"].tolist():
        tp = float(cm.get("TP", 0))
        fp = float(cm.get("FP", 0))
        tn = float(cm.get("TN", 0))
        fn = float(cm.get("FN", 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fprs, tprs, "o-", color="steelblue")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-Style Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_vs_metric(sweep_table: Any, metric: str, path: Path) -> None:
    """Plot threshold vs chosen metric (balanced accuracy, F1, etc.)."""
    df = sweep_table.copy() if isinstance(sweep_table, pd.DataFrame) else pd.DataFrame(sweep_table)
    if "threshold" not in df.columns:
        raise ValueError("sweep_table must include threshold column")
    if metric not in df.columns:
        raise ValueError(f"metric '{metric}' not found in sweep_table")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["threshold"], df[metric], "o-", color="darkorange")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(metric)
    ax.set_title(f"Threshold vs {metric}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

