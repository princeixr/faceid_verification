from __future__ import annotations

import csv
import json
from pathlib import Path

from src.config import Config
from src.tracking import RUN_SUMMARY_FIELDS, append_run_summary, create_run_dir, make_run_id, save_run_info


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
        assert reader.fieldnames == RUN_SUMMARY_FIELDS
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_x"
    assert rows[0]["mode"] == "sweep"


def test_append_run_summary_migrates_legacy_row_layout(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    summary_path = tmp_path / "outputs" / "run_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Old 10-column layout written before mode/selection_rule fields existed.
    summary_path.write_text(
        "run_id,timestamp,commit_hash,config_path,threshold,train_accuracy,test_accuracy,val_accuracy,run_dir,note\n"
        "legacy_run,20260326T000000Z,abc123,configs/default.yaml,0.7,0.5,0.5,0.5,outputs/runs/legacy,old\n",
        encoding="utf-8",
    )

    append_run_summary(
        cfg,
        {
            "run_id": "run_new",
            "mode": "sweep",
            "selection_rule": "max_balanced_accuracy",
            "threshold": 0.75,
            "note": "new",
        },
    )

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == RUN_SUMMARY_FIELDS
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["run_id"] == "legacy_run"
    assert rows[0]["threshold"] == "0.7"
    assert rows[1]["run_id"] == "run_new"
    assert rows[1]["mode"] == "sweep"
