"""
Milestone 2: Analysis & plotting — generates all evaluation artifacts for a run.

Plots saved under <run_dir>/plots/:
  1. roc_curve.png          – FPR vs TPR across thresholds (from sweep)
  2. threshold_vs_metrics.png – threshold on x-axis, metrics on y-axis
  3. confusion_matrix_<split>.png – heatmap per split at the selected threshold
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── helpers ────────────────────────────────────────────────────────────────

def _rates_from_cm(cm: dict) -> tuple[float, float]:
    """Return (FPR, TPR) from a confusion-matrix dict with keys TP/FP/TN/FN."""
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fpr, tpr


# ── main entry point ──────────────────────────────────────────────────────

def analyze_metrics(
    *,
    threshold_metrics: list[dict],
    selected_threshold: float,
    selection_metrics: dict,
    sweep_split_label: str,
    train_metrics: dict,
    test_metrics: dict,
    val_metrics: dict,
    run_dir: Path,
) -> None:
    """Generate and save all evaluation plots into <run_dir>/plots/."""

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. ROC-style curve (from threshold sweep) ─────────────────────
    fprs, tprs, thresholds = [], [], []
    for m in threshold_metrics:
        fpr, tpr = _rates_from_cm(m["confusion_matrix"])
        fprs.append(fpr)
        tprs.append(tpr)
        thresholds.append(m["threshold"])

    # Best-threshold point
    best_fpr, best_tpr = _rates_from_cm(selection_metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fprs, tprs, "o-", color="steelblue", label="Threshold sweep")
    ax.plot(best_fpr, best_tpr, "r*", markersize=16,
            label=f"Selected θ = {selected_threshold:.3f}")
    # annotate each point with its threshold value
    for fpr_i, tpr_i, t_i in zip(fprs, tprs, thresholds):
        ax.annotate(f"{t_i:.2f}", (fpr_i, tpr_i), textcoords="offset points",
                    xytext=(6, -8), fontsize=8, color="grey")
    ax.plot([0, 1], [0, 1], "--", color="lightgrey", label="Random")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC-Style Curve (Threshold Sweep on {sweep_split_label.title()})")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {plots_dir / 'roc_curve.png'}")

    # ── 2. Threshold vs metrics ───────────────────────────────────────
    metric_keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in metric_keys:
        vals = [m[key] for m in threshold_metrics]
        ax.plot(thresholds, vals, "o-", label=key)
    ax.axvline(selected_threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Selected θ = {selected_threshold:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Metrics vs Threshold ({sweep_split_label.title()} Sweep)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "threshold_vs_metrics.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {plots_dir / 'threshold_vs_metrics.png'}")

    # ── 3. Confusion matrices (one per split) ─────────────────────────
    splits = {"train": train_metrics, "test": test_metrics, "val": val_metrics}
    for split_name, metrics in splits.items():
        cm = metrics["confusion_matrix"]
        matrix = np.array([[cm["TP"], cm["FN"]],
                           [cm["FP"], cm["TN"]]])
        labels = np.array([["TP", "FN"],
                           ["FP", "TN"]])

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, cmap="Blues", aspect="equal")
        # annotate cells
        for i in range(2):
            for j in range(2):
                text_color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, f"{labels[i, j]}\n{matrix[i, j]}",
                        ha="center", va="center", fontsize=13,
                        fontweight="bold", color=text_color)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted 1", "Predicted 0"])
        ax.set_yticklabels(["Actual 1", "Actual 0"])
        acc = metrics.get("accuracy", 0)
        ax.set_title(f"Confusion Matrix — {split_name}  (θ={selected_threshold:.3f}, acc={acc:.3f})")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"confusion_matrix_{split_name}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {plots_dir / f'confusion_matrix_{split_name}.png'}")

    print(f"\n  All plots saved to {plots_dir}\n")
