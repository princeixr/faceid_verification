"""Deterministic face-image embedding helpers.

The project currently uses a local, reproducible embedding stage that is
explicit in the codebase and can be swapped later for a heavier pretrained
face model. The implementation below keeps the preprocessing pipeline clear,
stable, and testable without introducing a hidden dependency on model
downloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.config import Config

_RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def _embedding_section(config: Config) -> Dict[str, Any]:
    return dict(config._config.get("embedding", {}))


def _resolve_image_path(image_path: str, config: Config) -> Path:
    project_root = config.paths.project_root
    return project_root / image_path


def _as_tuple(value: Any, *, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return fallback
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(value, Sequence) and len(value) == 2:
        return int(value[0]), int(value[1])
    return fallback


def load_face_image(image_path: str, config: Config, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load a face image as a normalized RGB float array."""
    embedding_cfg = _embedding_section(config)
    if size is None:
        size = _as_tuple(embedding_cfg.get("preprocess_size"), fallback=tuple(config.image.size))

    path = _resolve_image_path(image_path, config)
    image = Image.open(path).convert("RGB").resize(size, _RESAMPLE_BILINEAR)
    image_array = np.asarray(image, dtype=np.float32)
    normalization_value = float(embedding_cfg.get("normalization_value", config.embedding.normalization_value))
    return image_array / normalization_value


def preprocess_face_image(image_path: str, config: Config) -> np.ndarray:
    """Apply the deterministic preprocessing stage used by the embedding path."""
    return load_face_image(image_path, config)


def _fit_dimension(vector: np.ndarray, dimension: int) -> np.ndarray:
    if dimension <= 0:
        raise ValueError("Embedding dimension must be positive.")
    if vector.size == dimension:
        return vector.astype(np.float32, copy=False)
    if vector.size > dimension:
        indices = np.linspace(0, vector.size - 1, num=dimension, dtype=np.int64)
        return vector[indices].astype(np.float32, copy=False)
    padded = np.zeros(dimension, dtype=np.float32)
    padded[: vector.size] = vector.astype(np.float32, copy=False)
    return padded


def _block_statistics(gray_image: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
    rows, cols = grid_size
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_size must contain positive integers.")

    height, width = gray_image.shape
    row_edges = np.linspace(0, height, num=rows + 1, dtype=np.int64)
    col_edges = np.linspace(0, width, num=cols + 1, dtype=np.int64)

    means = []
    stds = []
    for row_index in range(rows):
        row_start, row_end = row_edges[row_index], row_edges[row_index + 1]
        for col_index in range(cols):
            col_start, col_end = col_edges[col_index], col_edges[col_index + 1]
            block = gray_image[row_start:row_end, col_start:col_end]
            means.append(float(block.mean()))
            stds.append(float(block.std()))

    return np.asarray(means + stds, dtype=np.float32)


def _frequency_features(gray_image: np.ndarray, block_size: Tuple[int, int]) -> np.ndarray:
    freq_rows, freq_cols = block_size
    if freq_rows <= 0 or freq_cols <= 0:
        raise ValueError("frequency_block must contain positive integers.")

    spectrum = np.fft.rfft2(gray_image)
    low_freq = spectrum[:freq_rows, :freq_cols]
    features = np.log1p(np.abs(low_freq)).astype(np.float32).reshape(-1)
    return features


def build_face_embedding_from_preprocessed(
    preprocessed_image: np.ndarray,
    config: Config,
    dimension: Optional[int] = None,
) -> np.ndarray:
    """Create a deterministic embedding from a preprocessed RGB image array."""
    embedding_cfg = _embedding_section(config)
    if dimension is None:
        dimension = int(embedding_cfg.get("dimension", config.embedding.dimension))

    spatial_size = _as_tuple(embedding_cfg.get("spatial_size"), fallback=(32, 32))
    grid_size = _as_tuple(embedding_cfg.get("grid_size"), fallback=(4, 4))
    frequency_block = _as_tuple(embedding_cfg.get("frequency_block"), fallback=(8, 8))

    gray = np.dot(preprocessed_image, np.asarray([0.299, 0.587, 0.114], dtype=np.float32))
    gray_image = Image.fromarray(np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    gray_small = np.asarray(gray_image.resize(spatial_size, _RESAMPLE_BILINEAR), dtype=np.float32) / 255.0

    frequency_features = _frequency_features(gray_small, frequency_block)
    block_features = _block_statistics(gray_small, grid_size)
    global_features = np.asarray(
        [
            float(gray_small.mean()),
            float(gray_small.std()),
            float(gray_small.min()),
            float(gray_small.max()),
        ],
        dtype=np.float32,
    )

    raw_embedding = np.concatenate([frequency_features, block_features, global_features]).astype(np.float32)
    return _fit_dimension(raw_embedding, int(dimension))


def build_face_embedding(image_path: str, config: Config, dimension: Optional[int] = None) -> np.ndarray:
    """Create a deterministic embedding vector for a single face image.

    The embedding is composed of low-frequency image-spectrum features plus
    coarse block statistics. This keeps the representation compact and
    reproducible while making the embedding stage explicit in the codebase.
    """

    embedding_cfg = _embedding_section(config)
    if dimension is None:
        dimension = int(embedding_cfg.get("dimension", config.embedding.dimension))

    preprocess_size = _as_tuple(embedding_cfg.get("preprocess_size"), fallback=(64, 64))
    image = load_face_image(image_path, config, size=preprocess_size)
    return build_face_embedding_from_preprocessed(image, config, dimension=dimension)


def build_face_embedding_batches(pairs: Iterable[Dict[str, Any]], config: Config, dimension: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build embedding batches from a list of pair records."""
    if dimension is None:
        dimension = int(_embedding_section(config).get("dimension", config.embedding.dimension))

    left_vecs = []
    right_vecs = []
    labels = []
    for pair in pairs:
        left_vecs.append(build_face_embedding(pair["left_path"], config, dimension=dimension))
        right_vecs.append(build_face_embedding(pair["right_path"], config, dimension=dimension))
        labels.append(int(pair["label"]))

    left_batch = np.stack(left_vecs, axis=0)
    right_batch = np.stack(right_vecs, axis=0)
    label_batch = np.asarray(labels, dtype=np.int64)
    return left_batch, right_batch, label_batch