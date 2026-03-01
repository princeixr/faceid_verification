import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.ingestion import (
    download_and_save_lfw_images, 
    sort_records_deterministically, 
    make_identity_split_map, 
    assign_splits_to_records, 
    compute_split_counts, 
    build_manifest, 
    write_manifest, 
    write_samples_csv
)

def main():
    # Load config
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config = Config.from_file(config_path)
    
    # Use config paths
    data_root = config.paths.data_root
    out_root = config.paths.out_root

    records = download_and_save_lfw_images(data_root, config) # download + write images + produce base records

    records = sort_records_deterministically(records) #deterministic ordering

    identities = sorted({r["person"] for r in records}) # extract identities

    # build identity_split map deterministically (i.e. same split for same identity)
    split_map = make_identity_split_map(identities=identities, config=config)

    # assign split to each record
    records = assign_splits_to_records(records, split_map)

    samples_csv_path = out_root / config.paths.manifests_dir / config.files.samples_csv
    
    counts = compute_split_counts(records)
    data_source = {"name": config.data_source.name} 

    # build manifest file (including counts and files)
    split_policy = f"{config.split.policy}_{config.split.train_ratio}_{config.split.val_ratio}_{config.split.test_ratio}"
    manifest = build_manifest(
        records, 
        seed=config.random.seed, 
        split_policy=split_policy, 
        data_source=data_source
    )
    
    # add counts and files to manifest
    manifest["counts"] = counts
    manifest["files"] = {"samples_csv": samples_csv_path.as_posix()}
    
    # write manifest file (including counts and files)
    manifest_path = out_root / config.paths.manifests_dir / config.files.manifest_json
    write_manifest(manifest, manifest_path)

    # write samples csv file
    write_samples_csv(records, samples_csv_path)
    
    # print results
    print("✅ Ingestion complete")
    print(f"Samples CSV: {samples_csv_path}")
    print(f"Manifest:    {manifest_path}")

if __name__ == "__main__":
    main()