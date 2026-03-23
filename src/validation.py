"""
Milestone 2: Validation helpers.

Blueprint goal: fail early when inputs or configs are malformed.
"""

from __future__ import annotations

from typing import Any

from src.config import Config


def validate_pairs_df(pairs_df: Any) -> None:
    """Validate pair schema, missing values, label format, and split values."""
    raise NotImplementedError("TODO: implement validate_pairs_df()")


def validate_image_paths(pairs_df: Any, config: Config) -> None:
    """Confirm referenced image paths exist (or are otherwise resolvable)."""
    raise NotImplementedError("TODO: implement validate_image_paths()")


def validate_threshold_config(threshold_cfg: Any) -> None:
    """Validate threshold mode, range, selection rule, and fixed value format."""
    raise NotImplementedError("TODO: implement validate_threshold_config()")


def validate_scores_length(pairs_df: Any, scores: Any) -> None:
    """Ensure number of scores equals number of pairs."""
    raise NotImplementedError("TODO: implement validate_scores_length()")


def check_split_integrity(pairs_df: Any) -> None:
    """Check for leakage or invalid split assignments."""
    raise NotImplementedError("TODO: implement check_split_integrity()")

