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

def make_run_id() -> str:
    """Create a unique run identifier."""
    # Format goals:
    # - Unique across runs (even if launched within the same second)
    # - Filesystem-safe (no spaces/colons)
    # - Short enough to read in logs and filenames
    #
    # Example: run_20260323T142530Z_a1b2c3d4
    

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"run_{ts}_{suffix}"


def get_timestamp() -> str:
    """Return run time string."""
    # raise NotImplementedError("TODO: implement get_timestamp()")
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
    raise NotImplementedError("TODO: implement save_run_info()")


def append_run_summary(config: Config, row: dict[str, Any]) -> None:
    """Append one compact row to the master tracking file (e.g., outputs/run_summary.csv)."""
    raise NotImplementedError("TODO: implement append_run_summary()")

