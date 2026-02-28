import numpy as np 
from pathlib import Path 
import csv 
import json 
import sys
from typing import Dict, List, Tuple, Optional
from src.similarity_score import get_image_embedding, get_cosine_similarity_batch, euclidean_distance_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def get_pair_detail(csv_path: Path) -> List[Dict]:
    """Load pair csv from relative path"""
    record = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record.append(row)
    return record

def build_embedding_batches(pairs: List[Dict], D: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_vecs, right_vecs, labels = [], [], []
    for p in pairs:
        left_vecs.append(get_image_embedding(p["left_path"], D=D))
        right_vecs.append(get_image_embedding(p["right_path"], D=D))
        labels.append(int(p["label"]))

    L = np.stack(left_vecs, axis=0)    # (N, D)
    R = np.stack(right_vecs, axis=0)   # (N, D)
    y = np.asarray(labels)             # (N,)

    return L, R, y

def write_pairs_with_scores(
    pairs: List[Dict],
    cos_scores: np.ndarray,
    out_csv: Path,
    *,
    l2_scores: Optional[np.ndarray] = None,
) -> None:
    """
    Writes a new CSV with all original columns + cos_sim (+ optional l2_dist).
    Keeps row order identical to the input `pairs`.
    """
    if len(pairs) != cos_scores.shape[0]:
        raise ValueError(f"Row mismatch: pairs={len(pairs)} vs cos_scores={cos_scores.shape[0]}")
    if l2_scores is not None and l2_scores.shape[0] != len(pairs):
        raise ValueError(f"Row mismatch: pairs={len(pairs)} vs l2_scores={l2_scores.shape[0]}")

    # Determine output columns (preserve original order)
    base_fields = list(pairs[0].keys()) if pairs else []
    extra_fields = ["cos_sim"] + (["l2_dist"] if l2_scores is not None else [])
    fieldnames = base_fields + [f for f in extra_fields if f not in base_fields]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(pairs):
            out_row = dict(row)
            out_row["cos_sim"] = float(cos_scores[i])
            if l2_scores is not None:
                out_row["l2_dist"] = float(l2_scores[i])
            writer.writerow(out_row)

def main():
    #Load train, test and val samples' path and label
    out_root = PROJECT_ROOT / "outputs"
    train_pair_path = out_root / "pairs" / "train_pairs.csv"
    test_pair_path = out_root / "pairs" / "test_pairs.csv"
    val_pair_path = out_root / "pairs" / "val_pairs.csv"

    #load pair records
    train_pair = get_pair_detail(train_pair_path)
    test_pair = get_pair_detail(test_pair_path)
    val_pair = get_pair_detail(val_pair_path)

    D = 100 #embedding the image into 100 dimention
    #create embedding batches 
    train_left, train_right, train_label = build_embedding_batches(train_pair, D=D)
    test_left, test_right, test_label = build_embedding_batches(test_pair, D=D)
    val_left, val_right, val_label = build_embedding_batches(val_pair, D=D)

    #get cosine similarity
    train_score = get_cosine_similarity_batch(train_left, train_right)
    test_score = get_cosine_similarity_batch(test_left, test_right)
    val_score = get_cosine_similarity_batch(val_left, val_right)

    #get euclidean distance
    train_l2 = euclidean_distance_batch(train_left, train_right)
    test_l2 = euclidean_distance_batch(test_left, test_right)
    val_l2 = euclidean_distance_batch(val_left, val_right)

    scored_dir = out_root / "similarity_score"
    write_pairs_with_scores(train_pair, train_score, scored_dir / "train_pairs_scored.csv", l2_scores=train_l2)
    write_pairs_with_scores(test_pair, test_score, scored_dir / "test_pairs_scored.csv", l2_scores=test_l2)
    write_pairs_with_scores(val_pair, val_score, scored_dir / "val_pairs_scored.csv", l2_scores=val_l2)


if __name__ == "__main__":
    main()