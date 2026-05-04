"""Pair-level face verification inference helpers.

This module exposes the per-pair runtime path that Milestone 3 needs:
load two images, build embeddings, compute similarity, apply a threshold,
derive a confidence score, and report latency for the full call.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.config import Config
from src.embedding import build_face_embedding_from_preprocessed, preprocess_face_image


def _cosine_similarity(left_embedding: np.ndarray, right_embedding: np.ndarray, eps: float) -> float:
    dot_product = float(np.dot(left_embedding, right_embedding))
    left_norm = float(np.linalg.norm(left_embedding))
    right_norm = float(np.linalg.norm(right_embedding))
    return dot_product / ((left_norm * right_norm) + eps)


def get_selected_threshold_artifact_path(config: Config) -> Path:
    paths_cfg = config._config.get("paths", {})
    project_root = Path(paths_cfg.get("project_root", Path.cwd()))
    out_root = Path(paths_cfg.get("out_root", "outputs"))
    out_root_abs = out_root if out_root.is_absolute() else (project_root / out_root)
    return out_root_abs / "inference" / "selected_threshold.json"


def load_persisted_inference_threshold(config: Config) -> Optional[float]:
    artifact_path = get_selected_threshold_artifact_path(config)
    if not artifact_path.exists():
        return None

    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    threshold = payload.get("threshold")
    if threshold is None:
        return None
    return float(threshold)


def resolve_inference_threshold(config: Config, explicit_threshold: Optional[float] = None) -> float:
    if explicit_threshold is not None:
        return float(explicit_threshold)

    persisted_threshold = load_persisted_inference_threshold(config)
    if persisted_threshold is not None:
        return float(persisted_threshold)

    return float(config._config.get("similarity_threshold", {}).get("default", 0.7))


def preprocess_pair_inputs(left_path: str, right_path: str, config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess the left/right images for inference."""
    return preprocess_face_image(left_path, config), preprocess_face_image(right_path, config)


def generate_pair_embeddings(
    left_preprocessed: np.ndarray,
    right_preprocessed: np.ndarray,
    config: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate embeddings from preprocessed image arrays."""
    embedding_cfg = config._config.get("embedding", {})
    dimension = int(embedding_cfg.get("dimension", 100))
    left_embedding = build_face_embedding_from_preprocessed(left_preprocessed, config, dimension=dimension)
    right_embedding = build_face_embedding_from_preprocessed(right_preprocessed, config, dimension=dimension)
    return left_embedding, right_embedding


def compute_similarity_score(left_embedding: np.ndarray, right_embedding: np.ndarray, config: Config) -> float:
    """Compute the pair similarity score from embeddings."""
    similarity_cfg = config._config.get("similarity", {})
    eps = float(similarity_cfg.get("epsilon", 1e-12))
    return _cosine_similarity(left_embedding, right_embedding, eps=eps)


def make_threshold_decision(score: float, threshold: float, *, higher_means_same: bool) -> int:
    """Apply thresholding to get the binary same/different decision."""
    return int(score >= threshold) if higher_means_same else int(score <= threshold)


def _derive_confidence(score: float, threshold: float, *, higher_means_same: bool, sharpness: float = 10.0) -> float:
    """Map score margin to a bounded confidence value.

    The confidence is a logistic transform of the signed margin between the
    score and the operating threshold. This is a simple, documented rule that
    can be replaced later by a calibrated model if the project adds one.
    """

    margin = score - threshold if higher_means_same else threshold - score
    confidence = 1.0 / (1.0 + np.exp(-sharpness * margin))
    return float(confidence)


def infer_pair(
    left_path: str,
    right_path: str,
    config: Config,
    *,
    threshold: Optional[float] = None,
    higher_means_same: bool = True,
) -> Dict[str, Any]:
    """Run the end-to-end pair-level inference path for two face images."""

    threshold = resolve_inference_threshold(config, explicit_threshold=threshold)
    confidence_cfg = config._config.get("confidence", {})
    confidence_method = str(confidence_cfg.get("method", "logistic_margin"))
    confidence_sharpness = float(confidence_cfg.get("sharpness", 10.0))

    total_start = perf_counter()

    preprocess_start = perf_counter()
    left_preprocessed, right_preprocessed = preprocess_pair_inputs(left_path, right_path, config)
    preprocessing_latency_ms = (perf_counter() - preprocess_start) * 1000.0

    embedding_start = perf_counter()
    left_embedding, right_embedding = generate_pair_embeddings(left_preprocessed, right_preprocessed, config)
    embedding_latency_ms = (perf_counter() - embedding_start) * 1000.0

    scoring_start = perf_counter()
    similarity_score = compute_similarity_score(left_embedding, right_embedding, config)
    scoring_latency_ms = (perf_counter() - scoring_start) * 1000.0

    threshold_start = perf_counter()
    decision = make_threshold_decision(similarity_score, float(threshold), higher_means_same=higher_means_same)
    threshold_latency_ms = (perf_counter() - threshold_start) * 1000.0

    confidence_start = perf_counter()
    confidence = _derive_confidence(
        similarity_score,
        float(threshold),
        higher_means_same=higher_means_same,
        sharpness=confidence_sharpness,
    )
    confidence_latency_ms = (perf_counter() - confidence_start) * 1000.0

    latency_ms = (perf_counter() - total_start) * 1000.0
    stage_latency_ms = {
        "preprocessing": float(preprocessing_latency_ms),
        "embedding_generation": float(embedding_latency_ms),
        "similarity_scoring": float(scoring_latency_ms),
        "threshold_decision": float(threshold_latency_ms),
        "confidence_computation": float(confidence_latency_ms),
        "total": float(latency_ms),
    }

    return {
        "left_path": left_path,
        "right_path": right_path,
        "similarity_score": float(similarity_score),
        "threshold": float(threshold),
        "decision": decision,
        "confidence": confidence,
        "confidence_method": confidence_method,
        "confidence_sharpness": float(confidence_sharpness),
        "latency_ms": float(latency_ms),
        "stage_latency_ms": stage_latency_ms,
    }
