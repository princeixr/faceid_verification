from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = PROJECT_ROOT / "scripts" / "infer_pair.py"


def _write_rgb(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def _make_temp_config(tmp_path: Path) -> Path:
    config_dict = {
        "paths": {
            "project_root": str(tmp_path),
            "out_root": "outputs",
        },
        "image": {
            "size": [64, 64],
        },
        "embedding": {
            "backend": "deterministic_baseline",
            "dimension": 100,
            "normalization_value": 255.0,
            "preprocess_size": [64, 64],
            "spatial_size": [32, 32],
            "grid_size": [4, 4],
            "frequency_block": [8, 8],
        },
        "similarity": {
            "epsilon": 1e-12,
        },
        "similarity_threshold": {
            "default": 0.5,
        },
        "confidence": {
            "method": "logistic_margin",
            "sharpness": 10.0,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    return config_path


def test_cli_requires_right_path_with_left_path() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--left-path",
            "data/lfw/images/A/000001.jpg",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    assert proc.returncode != 0
    assert "--right-path is required when --left-path is provided." in proc.stderr


def test_cli_single_pair_smoke_json_output(tmp_path: Path) -> None:
    config_path = _make_temp_config(tmp_path)
    left_rel = Path("data/lfw/images/A/000001.jpg")
    right_rel = Path("data/lfw/images/B/000001.jpg")

    _write_rgb(tmp_path / left_rel, np.full((80, 80, 3), 30, dtype=np.uint8))
    _write_rgb(tmp_path / right_rel, np.full((80, 80, 3), 210, dtype=np.uint8))

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--config",
            str(config_path),
            "--left-path",
            str(left_rel),
            "--right-path",
            str(right_rel),
            "--output-format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    payload = json.loads(proc.stdout)
    assert payload["left_path"] == str(left_rel)
    assert payload["right_path"] == str(right_rel)
    assert isinstance(payload["similarity_score"], float)
    assert payload["decision"] in (0, 1)
    assert 0.0 <= float(payload["confidence"]) <= 1.0


def test_cli_text_output_includes_stage_latency_breakdown(tmp_path: Path) -> None:
    config_path = _make_temp_config(tmp_path)
    left_rel = Path("data/lfw/images/A/000001.jpg")
    right_rel = Path("data/lfw/images/B/000001.jpg")

    _write_rgb(tmp_path / left_rel, np.full((80, 80, 3), 60, dtype=np.uint8))
    _write_rgb(tmp_path / right_rel, np.full((80, 80, 3), 180, dtype=np.uint8))

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--config",
            str(config_path),
            "--left-path",
            str(left_rel),
            "--right-path",
            str(right_rel),
            "--output-format",
            "text",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "stage_latency_ms.preprocessing=" in proc.stdout
    assert "stage_latency_ms.embedding_generation=" in proc.stdout
    assert "stage_latency_ms.similarity_scoring=" in proc.stdout


def test_cli_can_write_comparison_plot_and_json(tmp_path: Path) -> None:
    config_path = _make_temp_config(tmp_path)
    left_rel = Path("data/lfw/images/A/000001.jpg")
    right_rel = Path("data/lfw/images/B/000001.jpg")
    output_json = tmp_path / "outputs" / "pair_result.json"
    output_plot = tmp_path / "outputs" / "pair_result.png"

    _write_rgb(tmp_path / left_rel, np.full((80, 80, 3), 90, dtype=np.uint8))
    _write_rgb(tmp_path / right_rel, np.full((80, 80, 3), 170, dtype=np.uint8))

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--config",
            str(config_path),
            "--left-path",
            str(left_rel),
            "--right-path",
            str(right_rel),
            "--output-format",
            "json",
            "--output-json",
            str(output_json),
            "--output-plot",
            str(output_plot),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert output_json.exists()
    assert output_plot.exists()
    assert output_plot.stat().st_size > 0


def test_cli_always_writes_default_single_pair_artifacts(tmp_path: Path) -> None:
    config_path = _make_temp_config(tmp_path)
    left_rel = Path("data/lfw/images/A/000001.jpg")
    right_rel = Path("data/lfw/images/B/000001.jpg")

    _write_rgb(tmp_path / left_rel, np.full((80, 80, 3), 100, dtype=np.uint8))
    _write_rgb(tmp_path / right_rel, np.full((80, 80, 3), 110, dtype=np.uint8))

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--config",
            str(config_path),
            "--left-path",
            str(left_rel),
            "--right-path",
            str(right_rel),
            "--output-format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    inference_root = tmp_path / "outputs" / "inference"
    run_dirs = sorted(p for p in inference_root.iterdir() if p.is_dir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "results.json").exists()
    assert (run_dir / "run_info.json").exists()
    assert (run_dir / "pairs" / "pair_000001.json").exists()
    assert (run_dir / "plots" / "pair_000001.png").exists()


def test_cli_batch_respects_max_pairs_and_writes_batch_artifacts(tmp_path: Path) -> None:
    config_path = _make_temp_config(tmp_path)
    left_a = Path("data/lfw/images/A/000001.jpg")
    right_a = Path("data/lfw/images/B/000001.jpg")
    left_b = Path("data/lfw/images/A/000002.jpg")
    right_b = Path("data/lfw/images/B/000002.jpg")
    left_c = Path("data/lfw/images/A/000003.jpg")
    right_c = Path("data/lfw/images/B/000003.jpg")
    _write_rgb(tmp_path / left_a, np.full((80, 80, 3), 50, dtype=np.uint8))
    _write_rgb(tmp_path / right_a, np.full((80, 80, 3), 150, dtype=np.uint8))
    _write_rgb(tmp_path / left_b, np.full((80, 80, 3), 60, dtype=np.uint8))
    _write_rgb(tmp_path / right_b, np.full((80, 80, 3), 160, dtype=np.uint8))
    _write_rgb(tmp_path / left_c, np.full((80, 80, 3), 70, dtype=np.uint8))
    _write_rgb(tmp_path / right_c, np.full((80, 80, 3), 170, dtype=np.uint8))

    pairs_csv = tmp_path / "pairs.csv"
    pairs_csv.write_text(
        "\n".join(
            [
                "pair_id,left_path,right_path",
                f"pair_a,{left_a},{right_a}",
                f"pair_b,{left_b},{right_b}",
                f"pair_c,{left_c},{right_c}",
            ]
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--config",
            str(config_path),
            "--pairs-csv",
            str(pairs_csv),
            "--max-pairs",
            "2",
            "--output-format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload["results"]) == 2
    assert payload["results"][0]["pair_id"] == "pair_a"
    assert payload["results"][1]["pair_id"] == "pair_b"

    inference_root = tmp_path / "outputs" / "inference"
    run_dirs = sorted(p for p in inference_root.iterdir() if p.is_dir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "results.json").exists()
    assert (run_dir / "pairs" / "pair_a.json").exists()
    assert (run_dir / "pairs" / "pair_b.json").exists()
    assert not (run_dir / "pairs" / "pair_c.json").exists()
    assert (run_dir / "plots" / "pair_a.png").exists()
    assert (run_dir / "plots" / "pair_b.png").exists()
