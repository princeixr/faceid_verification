"""
Milestone 2: Error analysis runner.

Creates at least two concrete error slices from saved predictions:
  - False positives (label=0, predicted_label=1)
  - False negatives (label=1, predicted_label=0)

Outputs are stored under:
  <run_dir>/error_analysis/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run error slice analysis from saved predictions.")
    p.add_argument("--run-dir", required=True, help="Path to outputs/runs/<run_id> directory.")
    p.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Which prediction file to analyze.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many examples to save per error slice.",
    )
    return p.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _build_slice_summary(name: str, df: pd.DataFrame, total: int) -> dict[str, Any]:
    if df.empty:
        return {
            "slice_name": name,
            "count": 0,
            "rate": 0.0,
            "mean_similarity_score": None,
        }

    score_mean = _safe_float(df["similarity_score"].astype(float).mean())
    return {
        "slice_name": name,
        "count": int(len(df)),
        "rate": float(len(df) / total) if total > 0 else 0.0,
        "mean_similarity_score": score_mean,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    split = args.split
    top_k = max(1, int(args.top_k))

    pred_path = run_dir / f"{split}_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    df = pd.read_csv(pred_path)
    required_cols = {"label", "predicted_label", "left_path", "right_path"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {pred_path.name}: {missing}")

    # Ensure numeric labels for robust comparisons.
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce")
    valid_df = df.dropna(subset=["label", "predicted_label"]).copy()
    valid_df["label"] = valid_df["label"].astype(int)
    valid_df["predicted_label"] = valid_df["predicted_label"].astype(int)

    fp_df = valid_df[(valid_df["label"] == 0) & (valid_df["predicted_label"] == 1)].copy()
    fn_df = valid_df[(valid_df["label"] == 1) & (valid_df["predicted_label"] == 0)].copy()

    sort_by = "similarity_score" if "similarity_score" in valid_df.columns else None
    if sort_by:
        fp_examples = fp_df.sort_values(sort_by, ascending=False).head(top_k)
        fn_examples = fn_df.sort_values(sort_by, ascending=True).head(top_k)
    else:
        fp_examples = fp_df.head(top_k)
        fn_examples = fn_df.head(top_k)

    out_dir = run_dir / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_path = out_dir / f"{split}_false_positives.csv"
    fn_path = out_dir / f"{split}_false_negatives.csv"
    fp_examples.to_csv(fp_path, index=False)
    fn_examples.to_csv(fn_path, index=False)

    summary = {
        "run_dir": str(run_dir),
        "split": split,
        "total_rows": int(len(valid_df)),
        "slices": [
            _build_slice_summary("false_positives", fp_df, len(valid_df)),
            _build_slice_summary("false_negatives", fn_df, len(valid_df)),
        ],
        "hypotheses": {
            "false_positives": "Different identities with visually similar pose/lighting may be scoring too high.",
            "false_negatives": "Same-identity pairs with blur, occlusion, or large pose gaps may be scoring too low.",
        },
        "artifacts": {
            "false_positives_csv": str(fp_path),
            "false_negatives_csv": str(fn_path),
        },
    }

    summary_path = out_dir / f"{split}_error_slices_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Error analysis complete")
    print(f"  split: {split}")
    print(f"  summary: {summary_path}")
    print(f"  false positives: {fp_path}")
    print(f"  false negatives: {fn_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
