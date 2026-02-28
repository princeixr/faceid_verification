import numpy as np 
from pathlib import Path 
from typing import Dict, List, Tuple, Optional
import math
import csv 
import json 
import sys
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent

def get_image_array(
    image_path: str, 
    size=(250,250), 
    dtype: np.dtype = np.uint8
    ) -> np.ndarray:
    
    path = BASE_DIR / image_path
    img = Image.open(path).convert("RGB").resize(size) #making sure the image stays 250*250*3
    img_array = np.asarray(img, dtype=dtype)

    return img_array 

def get_image_embedding(
    image_path: str, 
    normalize: bool=True,
    D: Optional[int]=None
    ) -> np.ndarray:
    """
    Convert the image array (H, W, 3) to a 1D vector
    Later on we will build the embedding logic
    """
    img_array = get_image_array(image_path)
    x = img_array.astype(np.float32)
    #normalizing the values to [0, 1]
    if normalize:
        x /= 255.0

    flat = x.reshape(-1)

    if D is None or D == flat.shape[0]:
        return flat 
    if D <= 0:
        raise ValueError("D must be a positive integer.")
    
    # simple deterministic sampling
    idx = np.linspace(0, flat.shape[0] - 1, num=D, dtype=np.int64)
    return flat[idx]

def get_cosine_similarity_batch(a: np.ndarray, b:np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    a, b: (N, D)
    returns: (N,) cosine similarity per row
    """
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


def get_cosine_similarity_loop(a_list: List[List[float]], b_list: List[List[float]], eps: float = 1e-12) -> List[float]:
    """
    Naive Python implementation of cosine similarity using for-loops.
    """
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