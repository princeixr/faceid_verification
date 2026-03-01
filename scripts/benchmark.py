import time
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.config import Config
from src.similarity_score import (
    get_cosine_similarity_batch,
    get_cosine_similarity_loop,
    euclidean_distance_batch,
    euclidean_distance_loop
)

def main():
    # Load config
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config = Config.from_file(config_path)
    
    # 1. Create dummy data: 10,000 pairs of vectors, using config embedding dimension
    N = 10000
    D = config.embedding.dimension
    print(f"Benchmarking with N={N} pairs and D={D} dimensions...\n")
    
    # NumPy arrays for the vectorized functions
    a_np = np.random.rand(N, D).astype(np.float32)
    b_np = np.random.rand(N, D).astype(np.float32)
    
    # Pure Python lists for the naive loop functions
    a_list = a_np.tolist()
    b_list = b_np.tolist()

    TOLERANCE = 1e-6
    # --- COSINE SIMILARITY BENCHMARK ---
    print("--- Cosine Similarity ---")
    
    start_time = time.perf_counter()
    loop_cos = get_cosine_similarity_loop(a_list, b_list, config)
    loop_time = time.perf_counter() - start_time
    print(f"Python For-Loop: {loop_time:.4f} seconds")
    
    start_time = time.perf_counter()
    vec_cos = get_cosine_similarity_batch(a_np, b_np, config)
    vectorized_time = time.perf_counter() - start_time
    print(f"NumPy Vectorized: {vectorized_time:.4f} seconds")
    
    print(f"Speedup: NumPy is {loop_time / vectorized_time:.1f}x faster!\n")
    
    # CORRECTNESS CHECK
    cos_diff = np.max(np.abs(np.array(loop_cos) - vec_cos))
    print(f"Max absolute difference between loop and vectorized cosine similarity: {cos_diff:.3e}\n")
    if cos_diff < TOLERANCE:
        print("Correctness check passed for cosine similarity!\n")
    else:
        print("Warning: Cosine similarity results differ beyond tolerance!")

    # --- EUCLIDEAN DISTANCE BENCHMARK ---
    print("--- Euclidean Distance ---")
    
    start_time = time.perf_counter()
    loop_euclid = euclidean_distance_loop(a_list, b_list)
    loop_time = time.perf_counter() - start_time
    print(f"Python For-Loop: {loop_time:.4f} seconds")
    
    start_time = time.perf_counter()
    vec_euclid = euclidean_distance_batch(a_np, b_np)
    vectorized_time = time.perf_counter() - start_time
    print(f"NumPy Vectorized: {vectorized_time:.4f} seconds")
    
    print(f"Speedup: NumPy is {loop_time / vectorized_time:.1f}x faster!\n")
    
    # CORRECTNESS CHECK
    euclid_diff = np.max(np.abs(np.array(loop_euclid) - vec_euclid))
    print(f"Max absolute difference between loop and vectorized Euclidean distance: {euclid_diff:.3e}\n")
    if euclid_diff < TOLERANCE:
        print("Correctness check passed for Euclidean distance!")
    else:
        print("Warning: Euclidean distance results differ beyond tolerance!")    
    
if __name__ == "__main__":
    main()