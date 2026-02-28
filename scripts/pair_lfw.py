import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pairing import load_samples_csv, generate_pairs, write_pairs_csv

def main():
    seed = 1337
    num_positive_pairs = 3000
    num_negative_pairs = 3000
    out_root = PROJECT_ROOT / "outputs"
    
    # Load samples
    samples_csv = out_root / "manifests" / "lfw_samples.csv"
    
    records = load_samples_csv(samples_csv)
    
    # Generate pairs
    train_pairs, val_pairs, test_pairs = generate_pairs(
        records, 
        seed=seed,
        num_positive_pairs=num_positive_pairs,
        num_negative_pairs=num_negative_pairs
    )
    
    # Write pairs
    pairs_dir = out_root / "pairs"
    write_pairs_csv(train_pairs, pairs_dir / "train_pairs.csv")
    write_pairs_csv(val_pairs, pairs_dir / "val_pairs.csv")
    write_pairs_csv(test_pairs, pairs_dir / "test_pairs.csv")
    
    # Save pair generation policy
    pair_policy = {
        "seed": seed,
        "num_positive_pairs": num_positive_pairs,
        "num_negative_pairs": num_negative_pairs,
        "pair_policy": "identity_disjoint_verification",
        "description": "Positive pairs: same identity within split. Negative pairs: different identities within split."
    }
    policy_path = out_root / "pairs" / "pair_policy.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(pair_policy, indent=2), encoding="utf-8")
    
    print("✅ Pair generation complete")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print(f"Test pairs:  {len(test_pairs)}")
    print(f"Pair policy saved to: {policy_path}")


if __name__ == "__main__":
    main()