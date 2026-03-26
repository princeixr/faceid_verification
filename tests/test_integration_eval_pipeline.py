from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import Config
from src.io_utils import save_csv, save_json
from src.thresholding import evaluate_at_threshold, get_best_threshold
from src.tracking import append_run_summary, create_run_dir, make_run_id, save_run_info
from src.validation import (
    check_split_integrity,
    validate_image_paths,
    validate_pairs_df,
    validate_scores_length,
    validate_threshold_config,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def test_mini_eval_pipeline_creates_tracking_artifacts(tmp_path: Path) -> None:
    """Small integration test for the tracked evaluation flow."""
    config = Config.from_dict(
        {
            "paths": {
                "project_root": str(tmp_path),
                "out_root": "outputs",
                "pairs_dir": "pairs",
            },
            "similarity_threshold": {
                "range": [0.1, 0.95],
                "increment": 0.1,
                "default": 0.5,
            },
        }
    )

    # Create tiny image files referenced by pair rows.
    rel_paths = [
        Path("data/lfw/images/A/000001.jpg"),
        Path("data/lfw/images/A/000002.jpg"),
        Path("data/lfw/images/B/000001.jpg"),
        Path("data/lfw/images/B/000002.jpg"),
    ]
    for rel in rel_paths:
        _touch(tmp_path / rel)

    pairs_df = pd.DataFrame(
        [
            {
                "left_path": str(rel_paths[0]),
                "right_path": str(rel_paths[1]),
                "label": 1,
                "split": "train",
            },
            {
                "left_path": str(rel_paths[2]),
                "right_path": str(rel_paths[3]),
                "label": 0,
                "split": "train",
            },
            {
                "left_path": str(rel_paths[0]),
                "right_path": str(rel_paths[2]),
                "label": 0,
                "split": "val",
            },
            {
                "left_path": str(rel_paths[1]),
                "right_path": str(rel_paths[1]),
                "label": 1,
                "split": "test",
            },
        ]
    )

    validate_pairs_df(pairs_df)
    validate_image_paths(pairs_df, config)
    validate_threshold_config(config.similarity_threshold)
    check_split_integrity(pairs_df)

    scored_df = pairs_df.copy()
    scored_df["similarity_score"] = [0.92, 0.15, 0.25, 0.88]
    validate_scores_length(pairs_df, scored_df["similarity_score"])

    selected_threshold, train_best_metrics, threshold_metrics = get_best_threshold(scored_df, config)
    test_metrics = evaluate_at_threshold(scored_df.copy(), selected_threshold, config)

    run_id = make_run_id()
    run_dir = create_run_dir(config, run_id)

    save_run_info(
        run_dir / "run_info.json",
        {
            "run_id": run_id,
            "mode": "sweep",
            "selection_rule": "max_accuracy",
        },
    )
    save_json(threshold_metrics, run_dir / "threshold_metrics.json")
    save_csv(pd.DataFrame(threshold_metrics), run_dir / "threshold_metrics.csv")

    append_run_summary(
        config,
        {
            "run_id": run_id,
            "mode": "sweep",
            "selection_rule": "max_accuracy",
            "threshold": selected_threshold,
            "train_accuracy": train_best_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "run_dir": str(run_dir),
            "note": "integration-test",
        },
    )

    summary_path = tmp_path / "outputs" / "run_summary.csv"
    assert run_dir.exists()
    assert (run_dir / "run_info.json").exists()
    assert (run_dir / "threshold_metrics.json").exists()
    assert (run_dir / "threshold_metrics.csv").exists()
    assert summary_path.exists()

    lines = summary_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert run_id in lines[1]

    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    assert info["run_id"] == run_id
