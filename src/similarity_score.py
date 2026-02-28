import numpy as np 
from pathlib import Path 
from typing import Dict, List, Tuple, Optional
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