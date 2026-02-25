import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# filepath: src/pair_lfw.py


def load_samples_csv(csv_path: Path) -> List[Dict]:
    """Load samples from CSV file."""
    records = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def load_manifest(manifest_path: Path) -> Dict:
    """Load manifest JSON file."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def group_samples_by_person(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group samples by person identity."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["person"]].append(record)
    return grouped


def generate_pairs(
    records: List[Dict],
    seed: int,
    num_positive_pairs: int = 300,
    num_negative_pairs: int = 300,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate deterministic positive and negative pairs split by train/val/test.
    
    Returns:
        (train_pairs, val_pairs, test_pairs)
    """
    rng = np.random.default_rng(seed)
    
    # Group by person
    person_samples = group_samples_by_person(records)
    
    # Separate by split
    split_samples = defaultdict(lambda: defaultdict(list))
    for record in records:
        split = record["split"]
        person = record["person"]
        split_samples[split][person].append(record)
    
    all_pairs = {"train": [], "val": [], "test": []}
    splits_list = ["train", "val", "test"]
    
    for split_idx, split in enumerate(splits_list):
        people = list(split_samples[split].keys())
        people_sorted = sorted(people)
        
        # Positive pairs (same identity)
        positive_pairs = []
        for person in people_sorted:
            samples = split_samples[split][person]
            if len(samples) >= 2:
                # Generate all possible pairs for this person in this split
                indices = list(range(len(samples)))
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        positive_pairs.append({
                            "left_path": samples[i]["rel_path"],
                            "right_path": samples[j]["rel_path"],
                            "label": 1,
                            "split": split,
                        })

        available_positive_pairs = len(positive_pairs)
        if available_positive_pairs < num_positive_pairs:
            print(
                f"Split '{split}': requested {num_positive_pairs} positive pairs, generated {available_positive_pairs} available pairs."
            )
        
        # Shuffle positive pairs deterministically (use split index, not hash)
        rng_pos = np.random.default_rng(seed + split_idx)
        pos_indices = rng_pos.permutation(len(positive_pairs))
        positive_pairs = [positive_pairs[i] for i in pos_indices[:num_positive_pairs]]
        
        # Negative pairs (different identity)
        negative_pairs_list = []
        negative_pairs_seen = set()
        people_list = people_sorted.copy()

        if len(people_list) < 2:
            print(
                f"Split '{split}' has fewer than 2 identities; skipping negative pair generation."
            )
        else:
            # Use separate RNG for negative pairs to avoid dependency on positive pair generation
            rng_neg = np.random.default_rng(seed + 1000 + split_idx)

            attempts = 0
            max_attempts = num_negative_pairs * 10  # Prevent infinite loops
            while len(negative_pairs_seen) < num_negative_pairs and attempts < max_attempts:
                person1, person2 = rng_neg.choice(people_list, size=2, replace=False)
                # Get a random index from 0 to the length of the list
                idx1 = rng_neg.integers(0, len(split_samples[split][person1]))
                idx2 = rng_neg.integers(0, len(split_samples[split][person2]))

                # Use the index to grab the dictionary
                sample1 = split_samples[split][person1][idx1]
                sample2 = split_samples[split][person2][idx2]

                # Create a hashable tuple to detect duplicates
                pair_tuple = (sample1["rel_path"], sample2["rel_path"])
                if pair_tuple not in negative_pairs_seen:
                    negative_pairs_seen.add(pair_tuple)
                    negative_pairs_list.append({
                        "left_path": sample1["rel_path"],
                        "right_path": sample2["rel_path"],
                        "label": 0,
                        "split": split,
                    })
                attempts += 1

            if len(negative_pairs_list) < num_negative_pairs:
                print(
                    f"Split '{split}': requested {num_negative_pairs} negative pairs, generated {len(negative_pairs_list)} unique pairs."
                )
        
        combined_pairs = positive_pairs + negative_pairs_list
        
        rng_final = np.random.default_rng(seed + 2000 + split_idx)
        rng_final.shuffle(combined_pairs)
        all_pairs[split] = combined_pairs
    
    return all_pairs["train"], all_pairs["val"], all_pairs["test"]


def write_pairs_csv(pairs: List[Dict], out_path: Path) -> None:
    """Write pairs to CSV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["left_path", "right_path", "label", "split"])
        writer.writeheader()
        writer.writerows(pairs)


def main():
    seed = 1337
    num_positive_pairs = 3000
    num_negative_pairs = 3000
    data_root = Path("data")
    out_root = Path("outputs")
    
    # Load manifest and samples
    manifest_path = out_root / "manifests" / "lfw_manifest.json"
    samples_csv = out_root / "manifests" / "lfw_samples.csv"
    
    manifest = load_manifest(manifest_path)
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