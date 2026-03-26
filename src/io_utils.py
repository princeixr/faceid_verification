"""
Milestone 2: I/O utilities.

Keep file reading/writing helpers here so they are reusable across scripts and easy to test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
from src.config import Config
import json

def load_pairs_csv(config: Config) -> Any:
    base_path = config.paths.out_root / config.paths.pairs_dir

    train_pairs_df = pd.read_csv(base_path / config.files.train_pairs_csv)
    val_pairs_df = pd.read_csv(base_path / config.files.val_pairs_csv)
    test_pairs_df = pd.read_csv(base_path / config.files.test_pairs_csv)
    
    return train_pairs_df, val_pairs_df, test_pairs_df


def save_csv(table: Any, path: Path) -> None:
    """Write an in-memory table to CSV."""
    raise NotImplementedError("TODO: implement save_csv()")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def copy_config_snapshot(config: Config, path: Path) -> None:
    """Save the exact config used in a run folder."""
    raise NotImplementedError("TODO: implement copy_config_snapshot()")

