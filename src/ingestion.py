from __future__ import annotations

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image  # pip install pillow
from collections import Counter
from pathlib import Path
import json
from typing import Dict, List
from collections import defaultdict
import csv


def write_samples_csv(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "person", "rel_path", "split"])
        for r in records:
            w.writerow([r["sample_id"], r["person"], r["rel_path"], r["split"]])


def sort_records_deterministically(records: list[dict]) -> list[dict]:
    def key_fn(r: dict):
        fname = Path(r["rel_path"]).name  # just the filename
        return (r["person"], fname)

    return sorted(records, key=key_fn)


def build_manifest(
    records: list[dict],
    seed: int,
    split_policy: str,
    data_source: dict,
) -> dict:

    people = [r["person"] for r in records]
    total_images = len(records)
    total_identities = len(set(people))

    # If split exists, compute per-split counts, otherwise just totals.
    has_split = all(("split" in r) for r in records)
    if has_split:
        split_counts = Counter(r["split"] for r in records)
        # identities per split
        ids_by_split = {}
        for r in records:
            ids_by_split.setdefault(r["split"], set()).add(r["person"])
        identity_counts = {k: len(v) for k, v in ids_by_split.items()}
    else:
        split_counts = {}
        identity_counts = {}

    manifest = {
        "seed": seed,
        "split_policy": split_policy,
        "data_source": data_source,  
        "counts": {
            "images_total": total_images,
            "identities_total": total_identities,
            "images_by_split": dict(split_counts),
            "identities_by_split": dict(identity_counts),
        },
    }
    return manifest


def write_manifest(manifest: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def make_identity_split_map(
    identities: List[str],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Dict[str, str]:

    # normalize ratios (in case they don't sum to 1.0 exactly)
    s = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = train_ratio / s, val_ratio / s, test_ratio / s

    ids_sorted = sorted(identities)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids_sorted))
    ids_shuffled = [ids_sorted[i] for i in perm]

    n = len(ids_shuffled)
    n_train = int(np.floor(train_ratio * n))
    n_val = int(np.floor(val_ratio * n))
    # remainder goes to test
    n_test = n - n_train - n_val

    split_map: Dict[str, str] = {}
    for ident in ids_shuffled[:n_train]:
        split_map[ident] = "train"
    for ident in ids_shuffled[n_train:n_train + n_val]:
        split_map[ident] = "val"
    for ident in ids_shuffled[n_train + n_val:]:
        split_map[ident] = "test"

    # basic sanity checks
    assert len(split_map) == n
    assert list(split_map.values()).count("train") == n_train
    assert list(split_map.values()).count("val") == n_val
    assert list(split_map.values()).count("test") == n_test

    return split_map

def assign_splits_to_records(records: List[Dict], identity_split_map: Dict[str, str]) -> List[Dict]:
    """
    Adds record["split"] based on record["person"] using the identity_split_map.
    """
    out = []
    for r in records:
        person = r["person"]
        split = identity_split_map[person]
        rr = dict(r)
        rr["split"] = split
        out.append(rr)
    return out

def compute_split_counts(records: List[Dict]) -> Dict:
    """
    Returns counts of images and identities per split.
    """
    images_by_split = defaultdict(int)
    identities_by_split = defaultdict(set)

    for r in records:
        sp = r["split"]
        images_by_split[sp] += 1
        identities_by_split[sp].add(r["person"])

    return {
        "images_by_split": {k: int(v) for k, v in images_by_split.items()},
        "identities_by_split": {k: len(v) for k, v in identities_by_split.items()},
        "images_total": len(records),
        "identities_total": len({r["person"] for r in records}),
    }


def download_and_save_lfw_images(data_root: Path, overwrite: bool = False) -> list[dict]:

    skipped = 0
    written = 0
    images_dir = data_root / "lfw" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ds = tfds.load("lfw", split="train", shuffle_files=False)  # determinism knob

    records: list[dict] = []

    for sample_id, ex in enumerate(tfds.as_numpy(ds)):
        person = ex["label"].decode("utf-8")          # bytes -> str
        img = ex["image"]                             # numpy array uint8 (250,250,3)

        # Person directory
        person_dir = images_dir / person
        person_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic filename: sample_id-based (stable)
        filename = f"{sample_id:06d}.jpg"
        out_path = person_dir / filename

        if out_path.exists() and not overwrite:
            skipped += 1
        else:
            Image.fromarray(img).save(out_path, format="JPEG", quality=95) # Write image (idempotent: overwrites consistently)
            written += 1

        # Store a RELATIVE path so it works on another machine
        # Resolve to absolute path first, then compute relative to project root
        out_path_abs = out_path.resolve()
        project_root = Path.cwd().resolve()
        rel_path = out_path_abs.relative_to(project_root).as_posix()  
        records.append(
            {"sample_id": sample_id, "person": person, "rel_path": rel_path}
        )

    print(f"✅ Image written: {written}")
    print(f"❌ Image skipped: {skipped}")
    print(f"🔍 Total images processed: {written + skipped}")

    # Return only the records list; counts are logged above
    return records