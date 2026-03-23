"""
Milestone 2: I/O utilities.

Keep file reading/writing helpers here so they are reusable across scripts and easy to test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import Config


def load_pairs_csv(config: Config) -> Any:
    """
    Load the saved pair artifact into an in-memory table.

    Expected columns (blueprint): pair_id, img1_path, img2_path, label, split
    Your milestone-1 pairs currently use: left_path, right_path, label, split
    """
    raise NotImplementedError("TODO: implement load_pairs_csv()")


def save_csv(table: Any, path: Path) -> None:
    """Write an in-memory table to CSV."""
    raise NotImplementedError("TODO: implement save_csv()")


def save_json(data: Any, path: Path) -> None:
    """Write JSON-serializable data to disk."""
    raise NotImplementedError("TODO: implement save_json()")


def copy_config_snapshot(config: Config, path: Path) -> None:
    """Save the exact config used in a run folder."""
    raise NotImplementedError("TODO: implement copy_config_snapshot()")

