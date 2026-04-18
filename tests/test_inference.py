from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.config import Config
from src.inference import _derive_confidence, compute_similarity_score, infer_pair


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
    assert isinstance(result["stage_latency_ms"], dict)
    expected_stages = {
        "preprocessing",
        "embedding_generation",
        "similarity_scoring",
        "threshold_decision",
        "confidence_computation",
        "total",
    }
    assert expected_stages.issubset(set(result["stage_latency_ms"].keys()))
    for key in expected_stages:
        assert float(result["stage_latency_ms"][key]) >= 0.0


def test_compute_similarity_score_on_toy_embeddings(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    same_left = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    same_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    diff_right = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    same_score = compute_similarity_score(same_left, same_right, config)
    diff_score = compute_similarity_score(same_left, diff_right, config)

    assert abs(same_score - 1.0) <= 1e-9
    assert abs(diff_score) <= 1e-12


def test_derive_confidence_respects_margin_direction() -> None:
    threshold = 0.5

    above_threshold = _derive_confidence(0.8, threshold, higher_means_same=True, sharpness=10.0)
    at_threshold = _derive_confidence(0.5, threshold, higher_means_same=True, sharpness=10.0)
    below_threshold = _derive_confidence(0.2, threshold, higher_means_same=True, sharpness=10.0)

    assert above_threshold > at_threshold > below_threshold
    assert abs(at_threshold - 0.5) < 1e-12

    # For lower-means-same mode, smaller score should mean higher confidence.
    lower_mode_high = _derive_confidence(0.2, threshold, higher_means_same=False, sharpness=10.0)
    lower_mode_low = _derive_confidence(0.8, threshold, higher_means_same=False, sharpness=10.0)
    assert lower_mode_high > lower_mode_low