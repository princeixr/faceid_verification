## **FaceID_Verification System**

This project implements an end-to-end face verification system inspired by FaceID-style authentication workflows. Given two facial images, the system determines whether they belong to the same individual and outputs:

* A similarity score between the two face embeddings
* A calibrated confidence score for the match decision
* A binary verification result (same identity / different identity)
* Basic latency measurements for performance analysis

The focus of this project is systems-oriented engineering rather than model accuracy alone. Emphasis is placed on:

* Reproducible experimentation and evaluation
* Performance-aware implementation and latency tracking
* Proper score calibration and threshold selection
* Clean, modular architecture
* Production-minded packaging and deployment readiness

The goal is to design and evaluate a reliable, well-engineered authentication service that reflects real-world identity verification constraints and best practices.

## Project Overview

Milestone 1 builds the foundational pipeline for face verification: data ingestion from TensorFlow Datasets, deterministic pair generation for train/val/test splits, similarity scoring using cosine similarity and Euclidean distance, and performance benchmarking. The pipeline emphasizes reproducibility through fixed random seeds, deterministic file ordering, and identity-disjoint splits. This ensures consistent results across runs and enables reliable evaluation of similarity metrics.

## Repo Layout

* **`src/`**: Core modules (`config.py`, `ingestion.py`, `pairing.py`, `similarity_score.py`)
* **`scripts/`**: Executable scripts (`ingest_lfw.py`, `pair_lfw.py`, `similarity_lfw.py`, `benchmark.py`)
* **`configs/`**: Configuration files (`default.yaml` with all pipeline parameters)
* **`outputs/`**: Generated outputs (manifests, pairs, similarity scores) - **ignored by git**
* **`data/`**: Downloaded LFW dataset images - **ignored by git**

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/princeixr/FaceID_Verification.git
cd facial_recognition
```

### 2. Environment Setup

```bash
# Create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Pipeline

Execute the three milestone commands in order:

```bash
# 1. Ingest LFW dataset and create train/val/test splits
python3 scripts/ingest_lfw.py

# 2. Generate positive and negative pairs for each split
python3 scripts/pair_lfw.py

# 3. Compute similarity scores (cosine similarity and Euclidean distance)
python3 scripts/similarity_lfw.py

# 4. Benchmark similarity computation performance
python3 scripts/benchmark.py
```

All scripts automatically load configuration from `configs/default.yaml`.

## Outputs

After running the pipeline, the following files are generated in `outputs/`:

**Manifests** (`outputs/manifests/`):

* `lfw_samples.csv` - Sample records with person, path, and split assignments
* `lfw_manifest.json` - Dataset metadata including counts and split information

**Pairs** (`outputs/pairs/`):

* `train_pairs.csv` - Training pairs (left_path, right_path, label, split)
* `val_pairs.csv` - Validation pairs
* `test_pairs.csv` - Test pairs
* `pair_policy.json` - Pair generation policy and parameters

**Similarity Scores** (`outputs/similarity_score/`):

* `train_pairs_scored.csv` - Training pairs with cosine similarity and L2 distance
* `val_pairs_scored.csv` - Validation pairs with scores
* `test_pairs_scored.csv` - Test pairs with scores

## Determinism Notes

The pipeline is fully deterministic with **random seed 1337**. Determinism is ensured by:

* Fixed random seed (1337) for all random operations
* Deterministic file ordering (no shuffling in TensorFlow Datasets)
* Identity-disjoint splits (same identity always in same split)
* Deterministic pair generation using seeded RNG streams with fixed offsets
* Deterministic image filename generation based on sample_id

Running the pipeline multiple times with the same configuration produces identical outputs.

## Features

* Similarity scoring
* Calibration
* Latency Tracking
* Reproducibility focus

## Architecture Overview

## Evaluation Section

## Performance

## Project Structure

## Limitations

## Future Work

## License
