import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.pairing import load_samples_csv, generate_pairs, write_pairs_csv

def main():
    # Load config
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config = Config.from_file(config_path)
    
    # Use config paths
    out_root = config.paths.out_root
    
    # Load samples
    samples_csv = out_root / config.paths.manifests_dir / config.files.samples_csv
    
    records = load_samples_csv(samples_csv)
    
    # Generate pairs
    train_pairs, val_pairs, test_pairs = generate_pairs(records, config)
    
    # Write pairs using config file names
    pairs_dir = out_root / config.paths.pairs_dir
    write_pairs_csv(train_pairs, pairs_dir / config.files.train_pairs_csv)
    write_pairs_csv(val_pairs, pairs_dir / config.files.val_pairs_csv)
    write_pairs_csv(test_pairs, pairs_dir / config.files.test_pairs_csv)
    
    # Save pair generation policy using config values
    pair_policy = {
        "seed": config.random.seed,
        "num_positive_pairs": config.pairs.num_positive_pairs,
        "num_negative_pairs": config.pairs.num_negative_pairs,
        "pair_policy": config.pairs.policy,
        "description": config.pairs.description
    }
    policy_path = out_root / config.paths.pairs_dir / config.files.pair_policy_json
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(pair_policy, indent=2), encoding="utf-8")
    
    print("Success: Pair generation complete")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print(f"Test pairs:  {len(test_pairs)}")
    print(f"Pair policy saved to: {policy_path}")


if __name__ == "__main__":
    main()