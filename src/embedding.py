"""Face-image embedding helpers.

The main Milestone 3 path uses a documented face-embedding model:
`InceptionResnetV1` from `facenet-pytorch` with pretrained weights.

The previous deterministic handcrafted embedding remains available as a
lightweight fallback backend for tests and controlled offline scenarios.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.config import Config

_RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
LOGGER = logging.getLogger("embedding")


def _embedding_section(config: Config) -> Dict[str, Any]:
    return dict(config._config.get("embedding", {}))


def _embedding_backend(config: Config) -> str:
    return str(_embedding_section(config).get("backend", "deterministic_baseline")).strip().lower()


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
    LOGGER.debug("Loading face image from %s with resize=%s", path, size)
    image = Image.open(path).convert("RGB").resize(size, _RESAMPLE_BILINEAR)
    image_array = np.asarray(image, dtype=np.float32)
    normalization_value = float(embedding_cfg.get("normalization_value", config.embedding.normalization_value))
    LOGGER.debug(
        "Loaded face image from %s; array_shape=%s normalization_value=%.3f",
        path,
        image_array.shape,
        normalization_value,
    )
    return image_array / normalization_value


def preprocess_face_image(image_path: str, config: Config) -> np.ndarray:
    """Apply the deterministic preprocessing stage used by the embedding path."""
    LOGGER.debug("Preprocessing image %s", image_path)
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


def _build_deterministic_embedding_from_preprocessed(
    preprocessed_image: np.ndarray,
    config: Config,
    dimension: int,
) -> np.ndarray:
    LOGGER.debug(
        "Embedding backend deterministic_baseline: computing handcrafted features for input shape=%s target_dim=%d",
        preprocessed_image.shape,
        dimension,
    )
    started_at = perf_counter()
    embedding_cfg = _embedding_section(config)
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
    output = _fit_dimension(raw_embedding, int(dimension))
    LOGGER.debug(
        "Embedding backend deterministic_baseline: finished in %.2fs output_dim=%d",
        perf_counter() - started_at,
        output.shape[0],
    )
    return output


def _get_inceptionresnet_model(config: Config) -> Any:
    embedding_cfg = _embedding_section(config)
    pretrained_name = str(embedding_cfg.get("pretrained", "vggface2"))
    requested_device = str(embedding_cfg.get("device", "cpu")).strip().lower()
    cache_dir = embedding_cfg.get("model_cache_dir")

    try:
        import torch
        from facenet_pytorch import InceptionResnetV1
    except ImportError as exc:
        raise ImportError(
            "InceptionResnetV1 embedding backend requires `torch` and `facenet-pytorch`."
        ) from exc

    if cache_dir:
        torch.hub.set_dir(str(Path(cache_dir).expanduser()))

    if requested_device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = requested_device

    cache_key = (pretrained_name, device_name)
    cached_model = _MODEL_CACHE.get(cache_key)
    if cached_model is not None:
        LOGGER.debug(
            "Embedding backend inceptionresnetv1: reusing cached model pretrained=%s device=%s",
            pretrained_name,
            device_name,
        )
        return cached_model

    LOGGER.info(
        "Embedding backend inceptionresnetv1: loading model pretrained=%s device=%s",
        pretrained_name,
        device_name,
    )
    started_at = perf_counter()
    model = InceptionResnetV1(pretrained=pretrained_name).eval().to(device_name)
    _MODEL_CACHE[cache_key] = model
    LOGGER.info(
        "Embedding backend inceptionresnetv1: model ready in %.2fs",
        perf_counter() - started_at,
    )
    return model


def _build_inceptionresnet_embedding_from_preprocessed(
    preprocessed_image: np.ndarray,
    config: Config,
    dimension: int,
) -> np.ndarray:
    import torch

    LOGGER.debug(
        "Embedding backend inceptionresnetv1: preparing tensor for input shape=%s target_dim=%d",
        preprocessed_image.shape,
        dimension,
    )
    started_at = perf_counter()
    model = _get_inceptionresnet_model(config)
    input_tensor = torch.from_numpy(preprocessed_image.transpose(2, 0, 1)).float().unsqueeze(0)
    # Match the standard FaceNet preprocessing convention.
    input_tensor = (input_tensor - 0.5) / 0.5
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    LOGGER.debug(
        "Embedding backend inceptionresnetv1: running forward pass on device=%s",
        device,
    )

    with torch.inference_mode():
        embedding = model(input_tensor).detach().cpu().numpy().reshape(-1)

    output = _fit_dimension(embedding.astype(np.float32, copy=False), int(dimension))
    LOGGER.debug(
        "Embedding backend inceptionresnetv1: forward pass complete in %.2fs output_dim=%d",
        perf_counter() - started_at,
        output.shape[0],
    )
    return output


def build_face_embedding_from_preprocessed(
    preprocessed_image: np.ndarray,
    config: Config,
    dimension: Optional[int] = None,
) -> np.ndarray:
    """Create an embedding from a preprocessed RGB image array."""
    embedding_cfg = _embedding_section(config)
    if dimension is None:
        dimension = int(embedding_cfg.get("dimension", config.embedding.dimension))

    backend = _embedding_backend(config)
    LOGGER.debug(
        "Building embedding from preprocessed image using backend=%s target_dim=%d",
        backend,
        int(dimension),
    )
    if backend == "inceptionresnetv1":
        return _build_inceptionresnet_embedding_from_preprocessed(preprocessed_image, config, int(dimension))
    if backend == "deterministic_baseline":
        return _build_deterministic_embedding_from_preprocessed(preprocessed_image, config, int(dimension))
    raise ValueError(f"Unsupported embedding backend: {backend}")


def build_face_embedding(image_path: str, config: Config, dimension: Optional[int] = None) -> np.ndarray:
    """Create an embedding vector for a single face image."""
    embedding_cfg = _embedding_section(config)
    if dimension is None:
        dimension = int(embedding_cfg.get("dimension", config.embedding.dimension))

    backend = _embedding_backend(config)
    started_at = perf_counter()
    LOGGER.debug(
        "Embedding request started for image=%s backend=%s target_dim=%d",
        image_path,
        backend,
        int(dimension),
    )
    preprocess_size = _as_tuple(embedding_cfg.get("preprocess_size"), fallback=(64, 64))
    image = load_face_image(image_path, config, size=preprocess_size)
    output = build_face_embedding_from_preprocessed(image, config, dimension=dimension)
    LOGGER.debug(
        "Embedding request complete for image=%s in %.2fs output_dim=%d",
        image_path,
        perf_counter() - started_at,
        output.shape[0],
    )
    return output


def build_face_embedding_batches(
    pairs: Iterable[Dict[str, Any]],
    config: Config,
    dimension: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build embedding batches from a list of pair records."""
    pairs_list = list(pairs)
    if dimension is None:
        dimension = int(_embedding_section(config).get("dimension", config.embedding.dimension))

    backend = _embedding_backend(config)
    total_pairs = len(pairs_list)
    LOGGER.info(
        "Batch embedding build started: pairs=%d backend=%s target_dim=%d",
        total_pairs,
        backend,
        int(dimension),
    )
    started_at = perf_counter()
    left_vecs = []
    right_vecs = []
    labels = []
    progress = tqdm(
        pairs_list,
        desc=f"Embedding pairs ({backend})",
        unit="pair",
        leave=False,
        disable=(total_pairs == 0),
    )
    for pair in progress:
        left_vecs.append(build_face_embedding(pair["left_path"], config, dimension=dimension))
        right_vecs.append(build_face_embedding(pair["right_path"], config, dimension=dimension))
        labels.append(int(pair["label"]))

    left_batch = np.stack(left_vecs, axis=0)
    right_batch = np.stack(right_vecs, axis=0)
    label_batch = np.asarray(labels, dtype=np.int64)
    LOGGER.info(
        "Batch embedding build complete in %.2fs left_shape=%s right_shape=%s",
        perf_counter() - started_at,
        left_batch.shape,
        right_batch.shape,
    )
    return left_batch, right_batch, label_batch
