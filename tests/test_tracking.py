from __future__ import annotations

import csv
import json
from pathlib import Path

from src.config import Config
from src.tracking import append_run_summary, create_run_dir, make_run_id, save_run_info


def _cfg(tmp_path: Path) -> Config:
    return Config.from_dict({"paths": {"project_root": str(tmp_path), "out_root": "outputs"}})


def test_create_run_dir_and_save_run_info(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_id = make_run_id()
    run_dir = create_run_dir(cfg, run_id)

    assert run_dir.exists()
    info_path = run_dir / "run_info.json"
    save_run_info(info_path, {"run_id": run_id, "note": "test"})
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id


def test_append_run_summary_writes_header_and_row(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    row = {
        "run_id": "run_x",
        "mode": "sweep",
        "threshold": 0.7,
        "note": "unit",
    }
    append_run_summary(cfg, row)

    summary_path = tmp_path / "outputs" / "run_summary.csv"
    assert summary_path.exists()

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_x"
    assert rows[0]["mode"] == "sweep"
