#!/usr/bin/env python3
"""
Deterministic LFW ingestion via TensorFlow Datasets (TFDS).

What this script does:
- Downloads/loads the TFDS "lfw" dataset (split='train') deterministically.
- Builds a manifest with: counts, seed, split policy, TFDS version/source, and per-sample split assignment.
- Saves:
  - outputs/manifests/lfw_manifest.json
  - outputs/manifests/lfw_samples.csv

Notes:
- TFDS caches datasets under ~/tensorflow_datasets by default (or --data_dir if provided).
- TFDS "lfw" provides a single split named "train"; we create our own split assignment deterministically.

Usage:
  python scripts/ingest_lfw.py --out_dir outputs --seed 1337 --split_policy identity --train 0.8 --val 0.1 --test 0.1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import tensorflow_datasets as tfds
except Exception as e:
    raise RuntimeError(
        "tensorflow_datasets is required. Install it with: pip install tensorflow-datasets"
    ) from e


@dataclass
class Manifest:
    dataset: str
    dataset_version: str
    tfds_data_dir: str | None
    created_utc: str
    seed: int
    split_policy: str
    split_ratios: Dict[str, float]
    counts: Dict[str, int]
    identity_counts: Dict[str, int]
    files: Dict[str, str]


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    s = train + val + test
    if s <= 0:
        raise ValueError("Split ratios must sum to a positive number.")
    return train / s, val / s, test / s


def _assign_identities_identity_disjoint(
    identities: List[str],
    seed: int,
    train_r: float,
    val_r: float,
    test_r: float,
) -> Dict[str, str]:
    """
    Deterministically assign each identity to exactly one split.
    We sort identities first (stable), then shuffle with RNG(seed).
    """
    identities_sorted = sorted(identities)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(identities_sorted))

    n = len(identities_sorted)
    n_train = int(round(train_r * n))
    n_val = int(round(val_r * n))
    # ensure all assigned, avoid rounding drift
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    split_map: Dict[str, str] = {}
    idxs = [identities_sorted[i] for i in perm.tolist()]
    for ident in idxs[:n_train]:
        split_map[ident] = "train"
    for ident in idxs[n_train : n_train + n_val]:
        split_map[ident] = "val"
    for ident in idxs[n_train + n_val :]:
        split_map[ident] = "test"

    assert len(split_map) == n
    assert list(split_map.values()).count("train") == n_train
    assert list(split_map.values()).count("val") == n_val
    assert list(split_map.values()).count("test") == n_test
    return split_map


def ingest_lfw(
    out_dir: Path,
    seed: int,
    split_policy: str,
    train: float,
    val: float,
    test: float,
    data_dir: str | None,
) -> Manifest:
    out_dir = out_dir.resolve()
    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_r, val_r, test_r = _normalize_ratios(train, val, test)

    # Deterministic load: don't shuffle TFDS files.
    # LFW in TFDS is a single split named "train".
    ds, ds_info = tfds.load(
        "lfw",
        split="train",
        with_info=True,
        shuffle_files=False,
        data_dir=data_dir,
        as_supervised=False,
    )

    # Convert to numpy deterministically by iterating in order.
    # We do NOT write images to disk; we only create a manifest of sample IDs and labels.
    # sample_id is stable for a given TFDS version + split ordering.
    records: List[Tuple[int, str]] = []
    for i, ex in enumerate(tfds.as_numpy(ds)):
        # label is a scalar tf.string -> bytes in numpy
        label_bytes = ex["label"]
        person = label_bytes.decode("utf-8") if isinstance(label_bytes, (bytes, bytearray)) else str(label_bytes)
        records.append((i, person))

    total_images = len(records)
    identities = sorted({p for _, p in records})
    total_identities = len(identities)

    if split_policy.lower() != "identity":
        raise ValueError("For Milestone 1, this script supports split_policy='identity' only.")

    identity_to_split = _assign_identities_identity_disjoint(
        identities=identities,
        seed=seed,
        train_r=train_r,
        val_r=val_r,
        test_r=test_r,
    )

    # Per-sample split
    samples_path = manifest_dir / "lfw_samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "person", "split", "person_hash"])
        for sample_id, person in records:
            split = identity_to_split[person]
            w.writerow([sample_id, person, split, _stable_hash(person)])

    # Counts
    split_counts = {"train": 0, "val": 0, "test": 0}
    for _, person in records:
        split_counts[identity_to_split[person]] += 1

    # Identity counts by split
    ident_counts = {"train": 0, "val": 0, "test": 0}
    for ident, sp in identity_to_split.items():
        ident_counts[sp] += 1

    manifest = Manifest(
        dataset="tfds:lfw",
        dataset_version=str(getattr(ds_info, "version", "unknown")),
        tfds_data_dir=data_dir,
        created_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        split_policy="identity-disjoint",
        split_ratios={"train": float(train_r), "val": float(val_r), "test": float(test_r)},
        counts={
            "images_total": total_images,
            "identities_total": total_identities,
            "images_train": split_counts["train"],
            "images_val": split_counts["val"],
            "images_test": split_counts["test"],
        },
        identity_counts={
            "identities_train": ident_counts["train"],
            "identities_val": ident_counts["val"],
            "identities_test": ident_counts["test"],
        },
        files={
            "samples_csv": str(samples_path.relative_to(out_dir).as_posix()),
        },
    )

    manifest_path = manifest_dir / "lfw_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest) | {"files": manifest.files | {"manifest_json": str(manifest_path.relative_to(out_dir).as_posix())}},
                  f, indent=2, sort_keys=True)

    # Update file pointers after writing
    manifest.files["manifest_json"] = str(manifest_path.relative_to(out_dir).as_posix())
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic LFW ingestion (TFDS) + manifest export.")
    p.add_argument("--out_dir", type=Path, default=Path("outputs"), help="Output directory (default: outputs/).")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for deterministic splitting.")
    p.add_argument(
        "--split_policy",
        type=str,
        default="identity",
        choices=["identity"],
        help="Split policy. 'identity' = identity-disjoint split by person.",
    )
    p.add_argument("--train", type=float, default=0.8, help="Train ratio (renormalized with val/test).")
    p.add_argument("--val", type=float, default=0.1, help="Val ratio (renormalized with train/test).")
    p.add_argument("--test", type=float, default=0.1, help="Test ratio (renormalized with train/val).")
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional TFDS data_dir (where TFDS stores downloads/extracts).",
    )
    args = p.parse_args()

    manifest = ingest_lfw(
        out_dir=args.out_dir,
        seed=args.seed,
        split_policy=args.split_policy,
        train=args.train,
        val=args.val,
        test=args.test,
        data_dir=args.data_dir,
    )

    print("✅ LFW ingestion complete")
    print(f"Output dir: {args.out_dir.resolve()}")
    print(f"Manifest:   {args.out_dir / manifest.files['manifest_json']}")
    print(f"Samples:    {args.out_dir / manifest.files['samples_csv']}")
    print("Counts:", manifest.counts)
    print("Identity counts:", manifest.identity_counts)
    print("Split policy:", manifest.split_policy)
    print("Seed:", manifest.seed)


if __name__ == "__main__":
    main()