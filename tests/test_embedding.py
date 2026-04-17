from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.config import Config
from src.embedding import build_face_embedding, build_face_embedding_batches


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
        }
    )


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def test_build_face_embedding_is_deterministic_and_dimensional(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    rel_path = Path("data/lfw/images/A/000001.jpg")
    image_path = tmp_path / rel_path
    pixels = np.zeros((80, 80, 3), dtype=np.uint8)
    pixels[..., 0] = 32
    pixels[..., 1] = 64
    pixels[..., 2] = 128
    _write_image(image_path, pixels)

    embedding_a = build_face_embedding(str(rel_path), config)
    embedding_b = build_face_embedding(str(rel_path), config)

    assert embedding_a.shape == (100,)
    assert np.allclose(embedding_a, embedding_b)


def test_build_face_embedding_batches_shapes(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    rel_left = Path("data/lfw/images/A/000001.jpg")
    rel_right = Path("data/lfw/images/B/000001.jpg")
    _write_image(tmp_path / rel_left, np.full((80, 80, 3), 40, dtype=np.uint8))
    _write_image(tmp_path / rel_right, np.full((80, 80, 3), 200, dtype=np.uint8))

    pairs = [
        {"left_path": str(rel_left), "right_path": str(rel_right), "label": 0},
        {"left_path": str(rel_right), "right_path": str(rel_left), "label": 1},
    ]

    left_batch, right_batch, labels = build_face_embedding_batches(pairs, config)

    assert left_batch.shape == (2, 100)
    assert right_batch.shape == (2, 100)
    assert labels.tolist() == [0, 1]