"""Pair-level face verification inference helpers.

This module exposes the per-pair runtime path that Milestone 3 needs:
load two images, build embeddings, compute similarity, apply a threshold,
derive a confidence score, and report latency for the full call.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np

from src.config import Config
from src.embedding import build_face_embedding


def _cosine_similarity(left_embedding: np.ndarray, right_embedding: np.ndarray, eps: float) -> float:
    dot_product = float(np.dot(left_embedding, right_embedding))
    left_norm = float(np.linalg.norm(left_embedding))
    right_norm = float(np.linalg.norm(right_embedding))
    return dot_product / ((left_norm * right_norm) + eps)


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

    embedding_cfg = config._config.get("embedding", {})
    similarity_cfg = config._config.get("similarity", {})
    if threshold is None:
        threshold = float(config._config.get("similarity_threshold", {}).get("default", 0.7))

    eps = float(similarity_cfg.get("epsilon", 1e-12))
    start = perf_counter()
    left_embedding = build_face_embedding(left_path, config, dimension=int(embedding_cfg.get("dimension", config.embedding.dimension)))
    right_embedding = build_face_embedding(right_path, config, dimension=int(embedding_cfg.get("dimension", config.embedding.dimension)))
    similarity_score = _cosine_similarity(left_embedding, right_embedding, eps=eps)
    decision = int(similarity_score >= threshold) if higher_means_same else int(similarity_score <= threshold)
    confidence = _derive_confidence(similarity_score, float(threshold), higher_means_same=higher_means_same)
    latency_ms = (perf_counter() - start) * 1000.0

    return {
        "left_path": left_path,
        "right_path": right_path,
        "similarity_score": float(similarity_score),
        "threshold": float(threshold),
        "decision": decision,
        "confidence": confidence,
        "latency_ms": float(latency_ms),
    }