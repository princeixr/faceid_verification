from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.config import Config
from src.inference import infer_pair


def _make_config(tmp_path: Path) -> Config:
    return Config.from_dict(
        {
            "paths": {
                "project_root": str(tmp_path),
                "out_root": "outputs",
            },
            "image": {
                "size": [64, 64],
            },
            "embedding": {
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
        }
    )


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def test_infer_pair_returns_pair_level_fields(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    left_rel = Path("data/lfw/images/A/000001.jpg")
    right_rel = Path("data/lfw/images/B/000001.jpg")
    _write_image(tmp_path / left_rel, np.full((80, 80, 3), 20, dtype=np.uint8))
    _write_image(tmp_path / right_rel, np.full((80, 80, 3), 220, dtype=np.uint8))

    result = infer_pair(str(left_rel), str(right_rel), config, threshold=0.5)

    assert result["left_path"] == str(left_rel)
    assert result["right_path"] == str(right_rel)
    assert isinstance(result["similarity_score"], float)
    assert isinstance(result["decision"], int)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["latency_ms"] >= 0.0