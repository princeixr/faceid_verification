import numpy as np
import tensorflow_datasets as tfds
from PIL import Image  # pip install pillow
from collections import Counter
from pathlib import Path
import json
from typing import Dict, List
from collections import defaultdict
import csv
from src.ingestion import download_and_save_lfw_images, sort_records_deterministically, make_identity_split_map, assign_splits_to_records, compute_split_counts, build_manifest, write_manifest, write_samples_csv

def main():
    # configure
    seed = 1337
    split_policy = "identity_disjoint"
    split_ratios = {"train": 0.7, "val": 0.1, "test": 0.2}
    data_root = Path("data")
    out_root = Path("outputs")

    records = download_and_save_lfw_images(data_root, overwrite=False) # download + write images + produce base records

    records = sort_records_deterministically(records) #deterministic ordering

    identities = sorted({r["person"] for r in records}) # extract identities

    # build identity_split map deterministically (i.e. same split for same identity)
    split_map = make_identity_split_map(
        identities=identities,
        seed=seed,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )

    # assign split to each record
    records = assign_splits_to_records(records, split_map)

    samples_csv_path = out_root / "manifests" / "lfw_samples.csv"
    
    counts = compute_split_counts(records)
    data_source = {"name": "tfds:lfw"} 

    # build manifest file (including counts and files)
    manifest = build_manifest(records, seed=seed, split_policy=f"{split_policy}_{split_ratios['train']}_{split_ratios['val']}_{split_ratios['test']}", data_source=data_source)
    
    # add counts and files to manifest
    manifest["counts"] = counts
    manifest["files"] = {"samples_csv": samples_csv_path.as_posix()}
    
    # write manifest file (including counts and files)
    write_manifest(manifest, out_root / "manifests" / "lfw_manifest.json")

    # write samples csv file
    write_samples_csv(records, samples_csv_path)
    
    # print results
    print("✅ Ingestion complete")
    print(f"Samples CSV: {samples_csv_path}")
    print(f"Manifest:    {out_root / 'manifests' / 'lfw_manifest.json'}")

if __name__ == "__main__":
    main()