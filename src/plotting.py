"""
Milestone 2: Plotting helpers for evaluation artifacts (ROC-style curve, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_roc_style_curve(sweep_table: Any, path: Path) -> None:
    """Create ROC-style plot from sweep results."""
    raise NotImplementedError("TODO: implement plot_roc_style_curve()")


def plot_threshold_vs_metric(sweep_table: Any, metric: str, path: Path) -> None:
    """Plot threshold vs chosen metric (balanced accuracy, F1, etc.)."""
    raise NotImplementedError("TODO: implement plot_threshold_vs_metric()")

