"""
Milestone 2: Tracked evaluation runner (sweep + locked threshold + test).

Pipeline flow:
  prepare_pairs.py (M1) -> saves pair artifact
  -> run_eval.py loads pair artifact
  -> validates input
  -> computes or loads scores
  -> runs threshold sweep or fixed-threshold evaluation
  -> saves metrics / confusion matrix / plot / predictions
  -> appends a run row to run_summary.csv
  -> (optional) run_error_analysis.py loads saved predictions

This script is intentionally thin. It orchestrates the evaluation by calling
reusable functions from `src/` so the logic is testable.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Optional
from similarity_lfw import get_pair_detail
import pandas as pd

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config

# ---------------------------------------------------------------------------
# Imports: tracking & provenance
# ---------------------------------------------------------------------------
from src.tracking import (
    make_run_id,
    get_timestamp,
    get_git_commit_hash,
    create_run_dir,
    save_run_info,
    append_run_summary,
)

# ---------------------------------------------------------------------------
# Imports: I/O utilities
# ---------------------------------------------------------------------------
from src.io_utils import (
    save_csv,
    save_json,
    copy_config_snapshot,
)

# ---------------------------------------------------------------------------
# Imports: validation
# ---------------------------------------------------------------------------
from src.validation import (
    validate_pairs_df,
    validate_image_paths,
    validate_threshold_config,
    validate_scores_length,
    check_split_integrity,
)

# ---------------------------------------------------------------------------
# Imports: scoring & evaluation
# ---------------------------------------------------------------------------
from src.evaluation import compute_similarity_scores

# ---------------------------------------------------------------------------
# Imports: thresholding
# ---------------------------------------------------------------------------
from src.thresholding import (
    get_best_threshold,
    evaluate_at_threshold,
)

# ---------------------------------------------------------------------------
# Imports: plotting
# ---------------------------------------------------------------------------
from src.analysis import (
    analyze_metrics
)
from src.inference import get_selected_threshold_artifact_path


LOGGER = logging.getLogger("run_eval")


def configure_logging() -> None:
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """CLI for tracked evaluation runs."""
    p = argparse.ArgumentParser(description="Run tracked evaluation for face verification.")
    p.add_argument("--config", required=True, help="Path to eval config (YAML/JSON).")
    p.add_argument("--note", default=None, help="Short note stored with run artifacts.")
    p.add_argument(
        "--mode",
        choices=["sweep", "fixed"],
        default="sweep",
        help="Thresholding mode. 'sweep' selects threshold from validation. 'fixed' uses a fixed threshold.",
    )
    p.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Threshold used when --mode fixed (defaults to config similarity_threshold.default).",
    )
    p.add_argument(
        "--selection-rule",
        choices=["max_accuracy", "max_balanced_accuracy", "max_f1"],
        default="max_accuracy",
        help="Selection rule used in sweep mode.",
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
    """Print final run id, threshold, metrics, and output path."""
    print("\n=== RUN COMPLETE ===")
    print(f"  run_id:    {run_id}")
    print(f"  mode:      {mode}")
    print(f"  split:     {split}")
    print(f"  run_dir:   {run_dir}")
    if selected_threshold is not None:
        print(f"  threshold: {selected_threshold}")
    print("  metrics:")
    for k in sorted(metrics.keys()):
        print(f"    {k}: {metrics[k]}")
    print("====================\n")


def build_pair_version_tag(out_root_abs: Path, pairs_dir: Path, pair_policy_file: str) -> str:
    """Build a compact pair-policy version tag for run provenance."""
    policy_path = out_root_abs / pairs_dir / pair_policy_file
    if not policy_path.exists():
        return "unknown"

    try:
        digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()[:12]
    except OSError:
        return "unknown"
    return f"{policy_path.name}:{digest}"


def persist_selected_threshold(
    config: Config,
    *,
    selected_threshold: float,
    selection_rule: str,
    selection_split: str,
    run_id: str,
    run_dir: Path,
) -> Path:
    artifact_path = get_selected_threshold_artifact_path(config)
    payload = {
        "threshold": float(selected_threshold),
        "selection_rule": selection_rule,
        "selection_split": selection_split,
        "run_id": run_id,
        "source_run_dir": str(run_dir),
        "source_run_info": str(run_dir / "run_info.json"),
    }
    save_json(payload, artifact_path)
    return artifact_path


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[list[str]] = None) -> int:
    """
    Pipeline stages (in order):
      1. Parse args & load config
      2. Create run metadata (run_id, timestamp, git hash)
      3. Create run directory & snapshot config + run_info.json
      4. Load pair artifact & validate inputs
      5. Load or compute similarity scores & validate alignment
      6. Threshold sweep  -OR-  fixed threshold
      7. Evaluate at selected threshold -> metrics, confusion, predictions
      8. Persist all artifacts to run directory
      9. Append one row to master run_summary.csv
     10. Print summary to terminal
    """

    # ------------------------------------------------------------------
    # Stage 1: Parse args & load config
    # ------------------------------------------------------------------
    configure_logging()
    overall_start = perf_counter()
    args = parse_args(argv)
    config_path = Path(args.config).expanduser()
    note = args.note
    mode = args.mode
    selection_rule = args.selection_rule
    fixed_threshold_arg = args.fixed_threshold

    LOGGER.info("Stage 1/10: loading config from %s", config_path)
    config = Config.from_file(config_path)

    LOGGER.info("Stage 2/10: creating tracked run metadata")
    run_id = make_run_id()
    timestamp = get_timestamp()
    commit_hash = get_git_commit_hash()
    run_dir = create_run_dir(config, run_id)       # ✅

    copy_config_snapshot(config, run_dir / "config_used.yaml")
    LOGGER.info("Run id: %s", run_id)
    LOGGER.info("Run directory: %s", run_dir)

    run_info = {
        "run_id": run_id,
        "timestamp": timestamp,
        "commit_hash": commit_hash,
        "config_path": str(config_path),
        "mode": mode,
        "selection_rule": selection_rule,
        "fixed_threshold_arg": fixed_threshold_arg,
        "note": note,
    }
    save_run_info(run_dir / "run_info.json", run_info) #save the run infor into run_info.json

    LOGGER.info("Stage 3/10: resolving pair artifact paths")
    # Use config paths from dict-style access for static-checker friendliness.
    paths_cfg = config._config.get("paths", {})
    files_cfg = config._config.get("files", {})
    project_root = Path(paths_cfg.get("project_root", Path.cwd()))
    out_root = Path(paths_cfg.get("out_root", "outputs"))
    out_root_abs = out_root if out_root.is_absolute() else (project_root / out_root)
    pairs_dir = Path(paths_cfg.get("pairs_dir", "pairs"))
    pair_policy_file = files_cfg.get("pair_policy_json", "pair_policy.json")
    pair_version = build_pair_version_tag(out_root_abs, pairs_dir, pair_policy_file)

    train_pair_path = out_root_abs / pairs_dir / files_cfg.get("train_pairs_csv", "train_pairs.csv")
    test_pair_path = out_root_abs / pairs_dir / files_cfg.get("test_pairs_csv", "test_pairs.csv")
    val_pair_path = out_root_abs / pairs_dir / files_cfg.get("val_pairs_csv", "val_pairs.csv")

    run_info["pair_version"] = pair_version
    save_run_info(run_dir / "run_info.json", run_info)

    LOGGER.info("Stage 4/10: loading train/val/test pair artifacts")
    #load pair records
    train_pair_df = get_pair_detail(train_pair_path)
    test_pair_df = get_pair_detail(test_pair_path)
    val_pair_df = get_pair_detail(val_pair_path)
    LOGGER.info(
        "Loaded pairs: train=%d, val=%d, test=%d",
        len(train_pair_df),
        len(val_pair_df),
        len(test_pair_df),
    )

    LOGGER.info("Stage 5/10: validating pair artifacts and config")
    # Validate the Pairs with the schema and missing-value checks
    train_pairs_pd = pd.DataFrame(train_pair_df)
    test_pairs_pd = pd.DataFrame(test_pair_df)
    val_pairs_pd = pd.DataFrame(val_pair_df)

    validate_pairs_df(train_pairs_pd)
    validate_pairs_df(test_pairs_pd)
    validate_pairs_df(val_pairs_pd)
    validate_image_paths(train_pairs_pd, config)
    validate_image_paths(test_pairs_pd, config)
    validate_image_paths(val_pairs_pd, config)
    validate_threshold_config(config.similarity_threshold)
    check_split_integrity(pd.concat([train_pairs_pd, test_pairs_pd, val_pairs_pd], ignore_index=True))

    LOGGER.info("Stage 6/10: computing similarity scores for each split")

    train_similarity_scores = pd.DataFrame(
        compute_similarity_scores(train_pair_df, config, "cosine", logger=LOGGER, progress_label="train")
    )
    test_similarity_scores = pd.DataFrame(
        compute_similarity_scores(test_pair_df, config, "cosine", logger=LOGGER, progress_label="test")
    )
    val_similarity_scores = pd.DataFrame(
        compute_similarity_scores(val_pair_df, config, "cosine", logger=LOGGER, progress_label="val")
    )

    validate_scores_length(train_pair_df, train_similarity_scores["similarity_score"])
    validate_scores_length(test_pair_df, test_similarity_scores["similarity_score"])
    validate_scores_length(val_pair_df, val_similarity_scores["similarity_score"])
    LOGGER.info("Similarity-score validation complete for all splits")

    LOGGER.info("Stage 7/10: selecting operating threshold using mode=%s", mode)
    if mode == "sweep":
        selected_threshold, val_selection_metrics, threshold_metrics = get_best_threshold(
            val_similarity_scores,
            config,
            rule=selection_rule,
        )
        LOGGER.info(
            "Selected threshold %.6f on validation using rule=%s",
            float(selected_threshold),
            selection_rule,
        )
    else:
        if fixed_threshold_arg is not None:
            selected_threshold = float(fixed_threshold_arg)
        else:
            selected_threshold = float(config._config.get("similarity_threshold", {}).get("default", 0.7))
        val_selection_metrics = evaluate_at_threshold(val_similarity_scores, selected_threshold, config)
        val_selection_metrics["threshold"] = selected_threshold
        threshold_metrics = [val_selection_metrics]
        LOGGER.info("Using fixed threshold %.6f", float(selected_threshold))

    run_info["selected_threshold"] = float(selected_threshold)
    run_info["selection_split"] = "val"
    run_info["selection_rule"] = selection_rule
    save_run_info(run_dir / "run_info.json", run_info)
    selected_threshold_artifact = persist_selected_threshold(
        config,
        selected_threshold=float(selected_threshold),
        selection_rule=selection_rule,
        selection_split="val",
        run_id=run_id,
        run_dir=run_dir,
    )
    run_info["selected_threshold_artifact"] = str(selected_threshold_artifact)
    save_run_info(run_dir / "run_info.json", run_info)
    LOGGER.info("Persisted selected threshold to %s", selected_threshold_artifact)
    
    print_run_summary(
        run_id=run_id,
        run_dir=run_dir,
        mode="threshold_selection",
        split="val",
        selected_threshold=selected_threshold,
        metrics=val_selection_metrics,
    )

    LOGGER.info("Stage 8/10: evaluating train/val/test at selected threshold")
    train_metrics = evaluate_at_threshold(train_similarity_scores, selected_threshold, config)
    test_metrics = evaluate_at_threshold(test_similarity_scores, selected_threshold, config)
    val_metrics = evaluate_at_threshold(val_similarity_scores, selected_threshold, config)

    print_run_summary(
        run_id=run_id,
        run_dir=run_dir,
        mode="evaluate",
        split="train",
        selected_threshold=selected_threshold,
        metrics=train_metrics,
    )

    print_run_summary(
        run_id=run_id,
        run_dir=run_dir,
        mode="evaluate",
        split="test",
        selected_threshold=selected_threshold,
        metrics=test_metrics,
    )
    print_run_summary(
        run_id=run_id,
        run_dir=run_dir,
        mode="evaluate",
        split="val",
        selected_threshold=selected_threshold,
        metrics=val_metrics,
    )

    LOGGER.info("Stage 9/10: saving metrics, predictions, and plots")
    save_json(threshold_metrics, run_dir / "threshold_metrics.json")
    save_json(train_metrics, run_dir / "train_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(val_metrics, run_dir / "val_metrics.json")
    save_csv(pd.DataFrame(threshold_metrics), run_dir / "threshold_metrics.csv")
    save_csv(test_similarity_scores, run_dir / "test_predictions.csv")
    save_csv(val_similarity_scores, run_dir / "val_predictions.csv")

    # Stage 7: Generate & save all plots (ROC, threshold-vs-metrics, confusion matrices)
    analyze_metrics(
        threshold_metrics=threshold_metrics,
        selected_threshold=selected_threshold,
        selection_metrics=val_selection_metrics,
        sweep_split_label="validation",
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        run_dir=run_dir,
    )

    LOGGER.info("Stage 10/10: appending run summary")
    append_run_summary(
        config,
        {
            "run_id": run_id,
            "timestamp": timestamp,
            "commit_hash": commit_hash,
            "config_path": str(config_path),
            "pair_version": pair_version,
            "mode": mode,
            "selection_rule": selection_rule,
            "fixed_threshold_arg": fixed_threshold_arg if fixed_threshold_arg is not None else "",
            "threshold": selected_threshold,
            "train_accuracy": train_metrics.get("accuracy"),
            "test_accuracy": test_metrics.get("accuracy"),
            "val_accuracy": val_metrics.get("accuracy"),
            "run_dir": str(run_dir),
            "note": note or "",
        },
    )
    LOGGER.info("Evaluation run complete in %.2fs", perf_counter() - overall_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
