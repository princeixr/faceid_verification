"""
Milestone 2: I/O utilities.

Keep file reading/writing helpers here so they are reusable across scripts and easy to test.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import pandas as pd
from src.config import Config
import json

def load_pairs_csv(config: Config) -> Any:
    """
    Load the saved pair artifact into an in-memory table.

    Expected columns (blueprint): pair_id, img1_path, img2_path, label, split
    Your milestone-1 pairs currently use: left_path, right_path, label, split
    """
    paths_cfg = config._config.get("paths", {})
    files_cfg = config._config.get("files", {})

    project_root = Path(paths_cfg.get("project_root", Path.cwd()))
    out_root = Path(paths_cfg.get("out_root", "outputs"))
    out_root_abs = out_root if out_root.is_absolute() else (project_root / out_root)

    pairs_dir = out_root_abs / Path(paths_cfg.get("pairs_dir", "pairs"))

    # Prefer a single configured evaluation artifact when available.
    eval_cfg = getattr(config, "eval", None)
    explicit_pair_path = getattr(eval_cfg, "pairs_csv_path", None) if eval_cfg is not None else None
    explicit_pair_name = getattr(eval_cfg, "pairs_csv", None) if eval_cfg is not None else None

    if explicit_pair_path:
        candidate_paths = [Path(explicit_pair_path)]
    elif explicit_pair_name:
        candidate_paths = [pairs_dir / explicit_pair_name]
    else:
        candidate_paths = [
            pairs_dir / files_cfg.get("train_pairs_csv", "train_pairs.csv"),
            pairs_dir / files_cfg.get("val_pairs_csv", "val_pairs.csv"),
            pairs_dir / files_cfg.get("test_pairs_csv", "test_pairs.csv"),
        ]

    rows: list[dict[str, Any]] = []
    running_pair_id = 0

    for csv_path in candidate_paths:
        if not csv_path.exists():
            # Only skip split files in the default multi-file mode.
            if explicit_pair_path or explicit_pair_name:
                raise FileNotFoundError(f"Pairs CSV not found: {csv_path}")
            continue

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue

                normalized = dict(row)

                # Keep milestone-1 names while exposing the milestone-2 blueprint aliases.
                if "img1_path" not in normalized and "left_path" in normalized:
                    normalized["img1_path"] = normalized["left_path"]
                if "img2_path" not in normalized and "right_path" in normalized:
                    normalized["img2_path"] = normalized["right_path"]

                if "pair_id" not in normalized or normalized["pair_id"] in (None, ""):
                    normalized["pair_id"] = running_pair_id

                label_value = normalized.get("label")
                if isinstance(label_value, str) and label_value.strip() in {"0", "1"}:
                    normalized["label"] = int(label_value)

                rows.append(normalized)
                running_pair_id += 1

    if not rows:
        looked_in = ", ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(f"No pair rows found. Looked in: {looked_in}")

    return rows


def save_csv(table: Any, path: Path) -> None:
    """Write an in-memory table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle pandas-style tables when available without importing pandas directly.
    to_csv = getattr(table, "to_csv", None)
    if callable(to_csv):
        to_csv(path, index=False)
        return

    if isinstance(table, list):
        if not table:
            path.write_text("", encoding="utf-8")
            return
        if not isinstance(table[0], dict):
            raise TypeError("save_csv expects a list of dict rows when not using a DataFrame-like object.")

        fieldnames = list(table[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table)
        return

    raise TypeError("Unsupported table type for save_csv(). Expected DataFrame-like or list[dict].")


def save_json(data: Any, path: Path) -> None:
    """Write JSON-serializable data to disk."""
    def _json_default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        # NumPy scalars expose item(); this keeps the helper lightweight.
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def copy_config_snapshot(config: Config, path: Path) -> None:
    """Save the exact config used in a run folder."""
    def _to_builtin(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _to_builtin(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return [_to_builtin(v) for v in value]
        if isinstance(value, list):
            return [_to_builtin(v) for v in value]
        return value

    snapshot = _to_builtin(config._config)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml

            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(snapshot, f, sort_keys=False)
            return
        except Exception:
            # Fall back to JSON if YAML serialization is unavailable.
            pass

    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=False)
        f.write("\n")

