import math
from typing import Dict, List, Tuple, Optional

import numpy as np

from src.config import Config
from src.embedding import build_face_embedding, build_face_embedding_batches, load_face_image

def get_image_array(
    image_path: str, 
    config: Config,
    size: Optional[Tuple[int, int]] = None, 
    dtype: np.dtype = np.uint8
    ) -> np.ndarray:
    """Backward-compatible wrapper around the explicit embedding preprocessing stage."""
    image = load_face_image(image_path, config, size=size)
    return (image * config.embedding.normalization_value).astype(dtype)

def get_image_embedding(
    image_path: str, 
    config: Config,
    normalize: Optional[bool] = None,
    D: Optional[int] = None
    ) -> np.ndarray:
    """Backward-compatible wrapper for the explicit face embedding stage."""
    if D is None:
        D = config.embedding.dimension
    return build_face_embedding(image_path, config, dimension=D)

def get_cosine_similarity_batch(
    a: np.ndarray, 
    b: np.ndarray, 
    config: Config,
    eps: Optional[float] = None
) -> np.ndarray:
    """
    a, b: (N, D)
    returns: (N,) cosine similarity per row
    """
    if eps is None:
        eps = config.similarity.epsilon
    
    # dot per row
    dot_product = np.sum(a*b, axis = 1) #(N,)

    #norm per row
    na = np.linalg.norm(a, axis=1) 
    nb = np.linalg.norm(b, axis=1)

    return dot_product / (na*nb + eps)


def euclidean_distance_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a, b: (N, D)
    returns: (N,) L2 distance per row
    """
    return np.linalg.norm(a - b, axis=1)


def get_cosine_similarity_loop(
    a_list: List[List[float]], 
    b_list: List[List[float]], 
    config: Config,
    eps: Optional[float] = None
) -> List[float]:
    """
    Naive Python implementation of cosine similarity using for-loops.
    """
    if eps is None:
        eps = config.similarity.epsilon
    
    scores = []
    # Loop over every pair (N)
    for i in range(len(a_list)):
        dot_product = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0
        
        # Loop over every feature in the vector (D)
        for j in range(len(a_list[i])):
            dot_product += a_list[i][j] * b_list[i][j]
            norm_a_sq += a_list[i][j] ** 2
            norm_b_sq += b_list[i][j] ** 2
            
        norm_a = math.sqrt(norm_a_sq)
        norm_b = math.sqrt(norm_b_sq)
        
        score = dot_product / ((norm_a * norm_b) + eps)
        scores.append(score)
        
    return scores

def euclidean_distance_loop(a_list: List[List[float]], b_list: List[List[float]]) -> List[float]:
    """
    Naive Python implementation of Euclidean distance using for-loops.
    """
    scores = []
    for i in range(len(a_list)):
        dist_sq = 0.0
        for j in range(len(a_list[i])):
            diff = a_list[i][j] - b_list[i][j]
            dist_sq += diff ** 2
            
        scores.append(math.sqrt(dist_sq))
        
    return scores


def build_embedding_batches(
    pairs: List[Dict], 
    config: Config,
    D: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build embedding batches from pairs."""
    return build_face_embedding_batches(pairs, config, dimension=D)