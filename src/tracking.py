"""
Milestone 2: Run tracking and provenance helpers.

Goal: Every `run_eval.py` invocation becomes a tracked run with a run folder and a
row appended to a master `run_summary.csv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import Config
from datetime import datetime, timezone
import uuid
import json
import csv

RUN_SUMMARY_FIELDS = [
    "run_id",
    "timestamp",
    "commit_hash",
    "config_path",
    "pair_version",
    "mode",
    "selection_rule",
    "fixed_threshold_arg",
    "threshold",
    "train_accuracy",
    "test_accuracy",
    "val_accuracy",
    "run_dir",
    "note",
]

OLD_RUN_SUMMARY_FIELDS = [
    "run_id",
    "timestamp",
    "commit_hash",
    "config_path",
    "threshold",
    "train_accuracy",
    "test_accuracy",
    "val_accuracy",
    "run_dir",
    "note",
]

def make_run_id() -> str:
    """Create a unique run identifier."""

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    ## Example: run_20260323T142530Z_a1b2c3d4
    return f"run_{ts}_{suffix}"


def get_timestamp() -> str:
    """Return run time string."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def get_git_commit_hash() -> str | None:
    """Read current git commit hash if available."""
    import shutil
    import subprocess
    from pathlib import Path

    # If git isn't installed, we can't retrieve a commit hash.
    if shutil.which("git") is None:
        return None

    # Heuristic: project root is the parent of `src/`.
    project_root = Path(__file__).resolve().parents[1]
    if not (project_root / ".git").exists():
        return None

    try:
        # Use -C to avoid relying on the current working directory.
        r = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    if r.returncode != 0:
        return None

    sha = (r.stdout or "").strip()
    return sha or None


def create_run_dir(config: Config, run_id: str) -> Path:
    """Create a folder for this run under outputs/runs/ (or configured root)."""
    # In this repo, `config.paths.out_root` is typically "outputs" (relative).
    # Resolve relative out_root against project_root so the script can be run
    # from any working directory.
    out_root: Path = config.paths.out_root
    project_root: Path = config.paths.project_root
    out_root_abs = out_root if out_root.is_absolute() else (project_root / out_root)

    run_dir = out_root_abs / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_info(path: Path, metadata: dict[str, Any]) -> None:
    """Write run_info.json with config name, note, split, pair version, commit hash, etc."""

    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _to_csv_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return ""
    return value


def _normalize_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {field: "" for field in RUN_SUMMARY_FIELDS}
    for field in RUN_SUMMARY_FIELDS:
        if field in row:
            normalized[field] = _to_csv_value(row[field])
    return normalized


def _parse_existing_row(values: list[str]) -> dict[str, Any] | None:
    if len(values) >= 12:
        # Legacy mixed rows written with dynamic field order:
        # run_id,timestamp,commit_hash,config_path,mode,selection_rule,threshold,
        # train_accuracy,test_accuracy,val_accuracy,run_dir,note
        return _normalize_summary_row(
            {
                "run_id": values[0],
                "timestamp": values[1],
                "commit_hash": values[2],
                "config_path": values[3],
                "mode": values[4],
                "selection_rule": values[5],
                "threshold": values[6],
                "train_accuracy": values[7],
                "test_accuracy": values[8],
                "val_accuracy": values[9],
                "run_dir": values[10],
                "note": values[11],
            }
        )

    if len(values) >= 10:
        parsed = dict(zip(OLD_RUN_SUMMARY_FIELDS, values[:10]))
        return _normalize_summary_row(parsed)

    return None


def _load_existing_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    if (not summary_path.exists()) or summary_path.stat().st_size == 0:
        return []

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    data_rows = rows[1:] if rows else []
    normalized_rows: list[dict[str, Any]] = []
    for values in data_rows:
        parsed = _parse_existing_row(values)
        if parsed is not None:
            normalized_rows.append(parsed)
    return normalized_rows

def append_run_summary(config: Config, row: dict[str, Any]) -> None:
    """Append one compact row to the master tracking file (e.g., outputs/run_summary.csv)."""
    paths_cfg = config._config.get("paths", {})
    project_root = Path(paths_cfg.get("project_root", Path.cwd()))
    out_root = Path(paths_cfg.get("out_root", "outputs"))
    out_root_abs = out_root if out_root.is_absolute() else (project_root / out_root)

    summary_path = out_root_abs / "run_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = _load_existing_summary_rows(summary_path)
    existing_rows.append(_normalize_summary_row(row))

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)

