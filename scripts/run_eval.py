"""
Milestone 2: Tracked evaluation runner (sweep + locked threshold + test).

This script is intentionally thin. It orchestrates the evaluation by calling
reusable functions from `src/` so the logic is testable.

Blueprint reference: `milestone2_detailed_blueprint.pdf` (extracted to
`outputs/blueprint_m2_extracted.txt`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Reads --config and optional --note.

    Blueprint requires:
    - `--config`: path to YAML/JSON config
    - `--note`: optional short note recorded in run_info + run_summary
    """
    p = argparse.ArgumentParser(description="Run tracked evaluation for face verification.")
    p.add_argument(
        "--config",
        required=True,
        help="Path to an evaluation config (YAML or JSON).",
    )
    p.add_argument(
        "--note",
        default=None,
        help="Optional short note to attach to this run (stored in run artifacts).",
    )
    return p.parse_args(argv)


def print_run_summary(
    *,
    run_id: str,
    run_dir: Path,
    mode: str,
    split: str,
    selected_threshold: Optional[float],
    metrics: dict[str, Any],
) -> None:
    """
    Prints the final run id, threshold, metrics, and output path.
    Keeps terminal output readable and debuggable.
    """
    # Placeholder formatting; we will refine once metrics schema is finalized.
    print("\n=== RUN COMPLETE ===")
    print(f"run_id: {run_id}")
    print(f"mode: {mode}")
    print(f"split: {split}")
    print(f"run_dir: {run_dir}")
    if selected_threshold is not None:
        print(f"selected_threshold: {selected_threshold}")
    print("metrics:")
    for k in sorted(metrics.keys()):
        print(f"  - {k}: {metrics[k]}")
    print("====================\n")


def main(argv: Optional[list[str]] = None) -> int:
    """
    Pseudocode (from blueprint):
    1) Parse args, load config
    2) Create run metadata (run_id, timestamp, commit hash), create run dir
    3) Snapshot config + write run_info.json
    4) Load pairs, validate, filter split
    5) Load or compute scores, validate alignment
    6) If sweep: sweep thresholds, save sweep CSV, select threshold, save threshold JSON, plot ROC
    7) Evaluate at threshold (selected or fixed), save metrics + confusion + predictions
    8) Append one row to outputs/run_summary.csv
    9) Print summary
    """
    args = parse_args(argv)
    config_path = Path(args.config).expanduser()
    note = args.note

    # Load config (existing loader in milestone-1 codebase).
    config = Config.from_file(config_path)

    # --- The rest of this function intentionally delegates to src/ helpers. ---
    # We are only wiring the call graph here; each helper will be implemented separately.

    from src.io_utils import copy_config_snapshot, load_pairs_csv, save_json
    from src.validation import (
        check_split_integrity,
        validate_image_paths,
        validate_pairs_df,
        validate_scores_length,
        validate_threshold_config,
    )
    from src.thresholding import (
        generate_threshold_grid,
        is_higher_score_same_person,
        select_threshold,
    )
    from src.evaluation import (
        evaluate_at_threshold,
        load_or_compute_scores,
        run_threshold_sweep,
    )
    from src.plotting import plot_roc_style_curve
    from src.tracking import (
        append_run_summary,
        create_run_dir,
        get_git_commit_hash,
        get_timestamp,
        make_run_id,
        save_run_info,
    )

    # 1) Run identity/provenance
    run_id = make_run_id()
    timestamp = get_timestamp()
    commit_hash = get_git_commit_hash()

    # 2) Create run directory (under outputs/runs/ by default; controlled by config)
    run_dir = create_run_dir(config, run_id)

    # 3) Snapshot config + run metadata
    copy_config_snapshot(config, run_dir / "config_used.yaml")
    run_info = {
        "run_id": run_id,
        "timestamp": timestamp,
        "commit_hash": commit_hash,
        "config_path": str(config_path),
        "note": note,
    }
    save_run_info(run_dir / "run_info.json", run_info)

    # 4) Load pairs + validate
    pairs_df = load_pairs_csv(config)
    validate_pairs_df(pairs_df)
    validate_image_paths(pairs_df, config)
    check_split_integrity(pairs_df)

    # 5) Threshold config validation (mode, sweep range, fixed value, etc.)
    threshold_cfg = getattr(config, "threshold", None)
    validate_threshold_config(threshold_cfg)

    # 6) Choose split + scoring
    eval_split = getattr(getattr(config, "eval", None), "split", None)
    if eval_split is None:
        raise ValueError("Missing config.eval.split (expected 'train'|'val'|'test').")

    eval_df = filter_to_split(pairs_df, eval_split)
    scores = load_or_compute_scores(eval_df, config)
    validate_scores_length(eval_df, scores)

    higher_means_same = is_higher_score_same_person(config)

    # 7) Run mode
    mode = getattr(getattr(config, "eval", None), "mode", None)
    if mode is None:
        raise ValueError("Missing config.eval.mode (expected 'sweep' or 'fixed').")

    selected_threshold: Optional[float] = None
    sweep_df = None

    if mode == "sweep":
        thresholds = generate_threshold_grid(config)
        sweep_df = run_threshold_sweep(eval_df, thresholds, higher_means_same=higher_means_same)
        # Persist sweep artifacts
        save_sweep_table(sweep_df, run_dir / "threshold_sweep.csv")

        rule = getattr(threshold_cfg, "selection_rule", "max_balanced_accuracy")
        selected_threshold = select_threshold(sweep_df, rule)
        save_json({"selected_threshold": selected_threshold, "selection_rule": rule}, run_dir / "selected_threshold.json")
        plot_roc_style_curve(sweep_df, run_dir / "roc_curve.png")

    elif mode == "fixed":
        selected_threshold = getattr(threshold_cfg, "fixed_value", None)
        if selected_threshold is None:
            raise ValueError("Fixed mode requires config.threshold.fixed_value.")
    else:
        raise ValueError(f"Unknown eval mode: {mode!r} (expected 'sweep' or 'fixed').")

    # 8) Evaluate at the chosen threshold
    eval_out = evaluate_at_threshold(eval_df, threshold=selected_threshold, higher_means_same=higher_means_same)
    metrics = eval_out["metrics"]
    confusion = eval_out["confusion"]
    predictions_df = eval_out["predictions"]

    save_json(metrics, run_dir / "metrics.json")
    save_json(confusion, run_dir / "confusion_matrix.json")
    save_predictions(predictions_df, run_dir / "predictions.csv")

    # 9) Append row to master summary
    append_run_summary(config, {
        "run_id": run_id,
        "timestamp": timestamp,
        "commit_hash": commit_hash,
        "config_path": str(config_path),
        "mode": mode,
        "split": eval_split,
        "selected_threshold": selected_threshold,
        "note": note,
        # metrics will be flattened/selected inside tracking helper later
        "metrics": metrics,
    })

    # 10) Print summary
    print_run_summary(
        run_id=run_id,
        run_dir=run_dir,
        mode=mode,
        split=eval_split,
        selected_threshold=selected_threshold,
        metrics=metrics,
    )
    return 0


# --- Helpers that live in THIS file (we'll implement each separately) ---

def filter_to_split(pairs_df: Any, split: str) -> Any:
    """Return only rows belonging to `split`."""
    raise NotImplementedError("TODO: implement filter_to_split()")


def save_sweep_table(sweep_df: Any, out_path: Path) -> None:
    """Persist threshold sweep results to CSV."""
    raise NotImplementedError("TODO: implement save_sweep_table()")


def save_predictions(predictions_df: Any, out_path: Path) -> None:
    """Persist predictions (labels, scores, y_pred, threshold, ids/paths) to CSV."""
    raise NotImplementedError("TODO: implement save_predictions()")


if __name__ == "__main__":
    raise SystemExit(main())

