from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import Config
from src.validation import (
    check_split_integrity,
    validate_image_paths,
    validate_pairs_df,
    validate_scores_length,
    validate_threshold_config,
)


def _config_with_root(root: Path) -> Config:
    return Config.from_dict(
        {
            "paths": {"project_root": str(root), "out_root": "outputs"},
            "similarity_threshold": {"range": [0.3, 0.9], "increment": 0.1, "default": 0.7},
        }
    )


def _valid_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"left_path": "data/a.jpg", "right_path": "data/b.jpg", "label": 1, "split": "train"},
            {"left_path": "data/c.jpg", "right_path": "data/d.jpg", "label": 0, "split": "val"},
        ]
    )


def test_validate_pairs_df_accepts_valid_rows() -> None:
    validate_pairs_df(_valid_pairs())


def test_validate_pairs_df_rejects_bad_label() -> None:
    df = _valid_pairs().copy()
    df.loc[0, "label"] = 2
    with pytest.raises(ValueError):
        validate_pairs_df(df)


def test_validate_image_paths_passes_when_files_exist(tmp_path: Path) -> None:
    for rel in ["data/a.jpg", "data/b.jpg", "data/c.jpg", "data/d.jpg"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    cfg = _config_with_root(tmp_path)
    validate_image_paths(_valid_pairs(), cfg)


def test_validate_image_paths_fails_when_missing(tmp_path: Path) -> None:
    cfg = _config_with_root(tmp_path)
    with pytest.raises(FileNotFoundError):
        validate_image_paths(_valid_pairs(), cfg)


def test_validate_threshold_config_rejects_bad_range() -> None:
    bad_cfg = Config.from_dict({"similarity_threshold": {"range": [0.9, 0.3], "increment": 0.1, "default": 0.7}})
    with pytest.raises(ValueError):
        validate_threshold_config(bad_cfg.similarity_threshold)


def test_validate_scores_length_rejects_mismatch() -> None:
    with pytest.raises(ValueError):
        validate_scores_length([1, 2, 3], [0.1, 0.2])


def test_check_split_integrity_detects_leakage() -> None:
    df = pd.DataFrame(
        [
            {"left_path": "data/a.jpg", "right_path": "data/b.jpg", "label": 1, "split": "train"},
            {"left_path": "data/a.jpg", "right_path": "data/b.jpg", "label": 1, "split": "test"},
        ]
    )
    with pytest.raises(ValueError):
        check_split_integrity(df)
