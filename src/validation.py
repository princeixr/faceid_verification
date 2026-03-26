"""
Milestone 2: Validation helpers.

Blueprint goal: fail early when inputs or configs are malformed.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

from src.config import Config
import pandas as pd

def validate_pairs_df(pairs_df: Any) -> None:
    """Validate pair schema, missing values, label format, and split values."""
    # raise NotImplementedError("TODO: implement validate_pairs_df()")
    if not isinstance(pairs_df, pd.DataFrame):
        raise ValueError("pairs_df must be a pandas DataFrame")
    if not all(col in pairs_df.columns for col in ["left_path", "right_path", "label", "split"]):
        raise ValueError("pairs_df must contain the columns: left_path, right_path, label, split")
    labels = pd.to_numeric(pairs_df["label"], errors="coerce")
    if labels.isna().any() or not bool(labels.isin([0, 1]).all()):
        raise ValueError("label must be 0 or 1")
    if not all(pairs_df["split"].isin(["train", "val", "test"])):
        raise ValueError("split must be train, val, or test")



def validate_image_paths(pairs_df: Any, config: Config) -> None:
    """Confirm referenced image paths exist (or are otherwise resolvable)."""
    if not isinstance(pairs_df, pd.DataFrame):
        raise ValueError("pairs_df must be a pandas DataFrame")

    left_col = "left_path" if "left_path" in pairs_df.columns else "img1_path"
    right_col = "right_path" if "right_path" in pairs_df.columns else "img2_path"
    if left_col not in pairs_df.columns or right_col not in pairs_df.columns:
        raise ValueError("pairs_df must contain left/right path columns")

    paths_cfg = config._config.get("paths", {})
    project_root = Path(paths_cfg.get("project_root", Path.cwd()))

    missing: list[str] = []
    for rel_path in pairs_df[left_col].astype(str).tolist() + pairs_df[right_col].astype(str).tolist():
        p = Path(rel_path)
        full_path = p if p.is_absolute() else (project_root / p)
        if not full_path.exists():
            missing.append(str(full_path))
            if len(missing) >= 10:
                break

    if missing:
        sample = "\n  - ".join(missing)
        raise FileNotFoundError(f"Some image files are missing (showing up to 10):\n  - {sample}")



def validate_threshold_config(threshold_cfg: Any) -> None:
    """Validate threshold mode, range, selection rule, and fixed value format."""
    required = ["range", "increment", "default"]
    missing = [k for k in required if not hasattr(threshold_cfg, k)]
    if missing:
        raise ValueError(f"Missing threshold config keys: {missing}")

    threshold_range = getattr(threshold_cfg, "range")
    if not isinstance(threshold_range, (list, tuple)) or len(threshold_range) != 2:
        raise ValueError("similarity_threshold.range must be [start, end]")

    start = float(threshold_range[0])
    end = float(threshold_range[1])
    if end <= start:
        raise ValueError("similarity_threshold.range must have end > start")

    increment = float(getattr(threshold_cfg, "increment"))
    if increment <= 0:
        raise ValueError("similarity_threshold.increment must be > 0")

    default = float(getattr(threshold_cfg, "default"))
    if not (start <= default <= end):
        raise ValueError("similarity_threshold.default must lie within range")


def validate_scores_length(pairs_df: Any, scores: Any) -> None:
    """Ensure number of scores equals number of pairs."""
    if len(pairs_df) != len(scores):
        raise ValueError(f"scores length mismatch: pairs={len(pairs_df)}, scores={len(scores)}")


def check_split_integrity(pairs_df: Any) -> None:
    """Check for leakage or invalid split assignments."""
    if not isinstance(pairs_df, pd.DataFrame):
        raise ValueError("pairs_df must be a pandas DataFrame")
    if "split" not in pairs_df.columns:
        raise ValueError("pairs_df must contain split column")

    allowed = {"train", "val", "test"}
    observed = set(pairs_df["split"].astype(str).unique().tolist())
    if not observed.issubset(allowed):
        raise ValueError(f"Unexpected split values: {sorted(observed - allowed)}")

    left_col = "left_path" if "left_path" in pairs_df.columns else "img1_path"
    right_col = "right_path" if "right_path" in pairs_df.columns else "img2_path"
    if left_col in pairs_df.columns and right_col in pairs_df.columns:
        pair_keys = pairs_df[left_col].astype(str) + "||" + pairs_df[right_col].astype(str)
        pair_split_df = pd.DataFrame({"k": pair_keys, "s": pairs_df["split"].astype(str)}).drop_duplicates()
        split_count = pair_split_df.groupby("k")["s"].nunique()
        leakage_count = int((split_count > 1).sum())
        if leakage_count > 0:
            raise ValueError(f"Found {leakage_count} pair(s) assigned to multiple splits")

