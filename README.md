## **FaceID_Verification System**

This project implements an end-to-end face verification system inspired by FaceID-style authentication workflows. Given two facial images, the system determines whether they belong to the same individual and outputs:

* A similarity score between the two face embeddings
* A decision-confidence score derived from the score margin around the operating threshold
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

Milestone 2 adds tracked evaluation and data-centric iteration:

* Baseline system: `configs/default.yaml` with threshold selected by validation sweep using `max_balanced_accuracy`.
* Improved system: `configs/milestone2_identity_cap.yaml` enabling identity participation cap (`max_pairs_per_identity=120`).
* Fair comparison policy: same split roles and same threshold-selection rule for both runs.

For a Milestone 2 reproduction and results summary, see:

* `reports/Milestone2_Report.md`

Milestone 3 adds deployable pair-level inference, embedding-based scoring, persisted threshold usage for inference, automatic inference artifact generation under `outputs/inference/`, Docker packaging, and local load testing.

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
cd FaceID_Verification
```

### 2. Environment Setup

```bash
# Create virtual environment (optional but recommended)
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Pipeline

Execute the three milestone commands in order:

```bash
# 1. Ingest LFW dataset and create train/val/test splits
python scripts/ingest_lfw.py

# 2. Generate positive and negative pairs for each split
python scripts/pair_lfw.py

# 3. Compute similarity scores (cosine similarity and Euclidean distance)
python scripts/similarity_lfw.py

# 4. Benchmark similarity computation performance
python scripts/benchmark.py
```

All scripts automatically load configuration from `configs/default.yaml`.

### 4. Milestone 2 Evaluation (Tracked Runs)

Use the evaluation runner for threshold selection and final reporting:

Split roles used by `run_eval.py`:

* Threshold selection is done on the validation split in sweep mode.
* Final reporting is read from the held-out test split at the selected threshold.
* Score direction is `higher-is-more-same` for cosine similarity.

```bash
# Sweep mode (default rule: max_accuracy)
python scripts/run_eval.py --config configs/default.yaml --mode sweep --note "baseline-sweep"

# Sweep mode with a different selection rule
python scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_balanced_accuracy --note "sweep-balanced"

# Fixed mode (lock threshold for reporting)
python scripts/run_eval.py --config configs/default.yaml --mode fixed --fixed-threshold 0.7 --note "final-fixed"
```

Threshold rule behavior:

* You can change the rule any run using `--selection-rule`.
* Supported rules: `max_accuracy`, `max_balanced_accuracy`, `max_f1`.
* If not provided, the default is `max_accuracy`.

This means threshold policy is not hardcoded. You can explicitly choose the rule per run for fair, reproducible comparisons.

### 4.2 Milestone 3 Inference

The Milestone 3 inference path uses the selected threshold from evaluation when available. The persisted threshold artifact is stored at:

* `outputs/inference/selected_threshold.json`

This file is generated by `scripts/run_eval.py` after threshold sweep completes on the validation split. In sweep mode, `run_eval.py`:

* computes similarity scores for the configured embedding system
* evaluates a grid of candidate thresholds on `val`
* selects the best threshold using the requested rule such as `max_f1`
* writes the selected threshold to `outputs/inference/selected_threshold.json`

The artifact records both the threshold value and its provenance. A typical file looks like:

```json
{
  "run_id": "run_20260419T004952Z_a7f2c22b",
  "selection_rule": "max_f1",
  "selection_split": "val",
  "source_run_dir": "outputs/runs/run_20260419T004952Z_a7f2c22b",
  "source_run_info": "outputs/runs/run_20260419T004952Z_a7f2c22b/run_info.json",
  "threshold": 0.4
}
```

Field meaning:

* `threshold` - the operating threshold used by inference when no explicit `--threshold` override is passed
* `selection_rule` - the metric rule used to choose the threshold, such as `max_f1`
* `selection_split` - the split used for threshold selection; this should be `val`, not `test`
* `run_id` - the tracked evaluation run that produced the threshold
* `source_run_dir` and `source_run_info` - pointers back to the tracked evaluation artifacts for reproducibility

Inference uses threshold precedence in this order:

* explicit `--threshold` argument
* persisted `outputs/inference/selected_threshold.json`
* fallback config default from `configs/default.yaml`

You can run single-pair inference like this:

```bash
python scripts/infer_pair.py \
  --config configs/default.yaml \
  --left-path data/lfw/images/Barbara_Walters/004492.jpg \
  --right-path data/lfw/images/Barbara_Walters/007353.jpg \
  --output-format json
```

You can run batch inference from a pair CSV like this:

```bash
python scripts/infer_pair.py \
  --config configs/default.yaml \
  --pairs-csv outputs/pairs/test_pairs.csv \
  --output-format json
```

If you want to inspect only the first `N` rows from the batch CSV, use `--max-pairs`:

```bash
python scripts/infer_pair.py \
  --config configs/default.yaml \
  --pairs-csv outputs/pairs/test_pairs.csv \
  --max-pairs 25 \
  --output-format json
```

Every inference invocation now stores artifacts automatically under `outputs/inference/`:

* `infer_single_<timestamp>/results.json` or `infer_batch_<timestamp>/results.json`
* `run_info.json`
* `pairs/<pair_id>.json`
* `plots/<pair_id>.png`

You can also request explicit output copies in addition to the default artifact folder:

```bash
python scripts/infer_pair.py \
  --config configs/default.yaml \
  --left-path data/lfw/images/Barbara_Walters/004492.jpg \
  --right-path data/lfw/images/Barbara_Walters/007353.jpg \
  --output-format json \
  --output-json outputs/cli_test_infer_pair.json \
  --output-plot outputs/cli_test_infer_pair.png
```

### 4.3 Confidence Interpretation

The stored `confidence` value is confidence in the final binary decision, not an independent class probability.

The intended interpretation is:

* if the score is above the threshold, the decision is `1` and confidence should increase as the score moves farther above the threshold
* if the score is below the threshold, the decision is `0` and confidence should increase as the score moves farther below the threshold
* confidence is therefore tied to the distance between the score and the operating threshold in the direction of the chosen decision

Operationally:

* scores near the threshold correspond to lower decision confidence
* scores far from the threshold correspond to higher decision confidence
* this value should be read as decision confidence from threshold margin, not as a calibrated posterior probability of identity match

The inference output includes:

* `similarity_score`
* `threshold`
* `decision`
* `confidence`
* `latency_ms`
* `stage_latency_ms`

### 4.4 Milestone 2 Quick Path

If you want only the Milestone 2 workflow from a clean clone, run these commands in order:

```bash
python scripts/ingest_lfw.py
python scripts/pair_lfw.py --config configs/default.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_balanced_accuracy --note "baseline-default"
python scripts/run_error_analysis.py --run-dir outputs/runs/<baseline_run_id> --split test --top-k 20

python scripts/pair_lfw.py --config configs/milestone2_identity_cap.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/milestone2_identity_cap.yaml --mode sweep --selection-rule max_balanced_accuracy --note "data-centric-improved-identity-cap"
python scripts/run_error_analysis.py --run-dir outputs/runs/<improved_run_id> --split test --top-k 20
```

For additional context and exact artifact locations, see `reports/Milestone2_Report.md`.

### 5. Tests

Run unit and integration tests with:

```bash
python -m pytest -q tests
```

Run only the Milestone 2 miniature integration pipeline test:

```bash
python -m pytest -q tests/test_integration_eval_pipeline.py
```

### 6. Error Slice Analysis (Milestone 2)

After running evaluation, generate at least two required error slices (false positives and false negatives):

```bash
python scripts/run_error_analysis.py --run-dir outputs/runs/<run_id> --split test --top-k 20
```

Artifacts are written under:

* `outputs/runs/<run_id>/error_analysis/test_error_slices_summary.json`
* `outputs/runs/<run_id>/error_analysis/test_false_positives.csv`
* `outputs/runs/<run_id>/error_analysis/test_false_negatives.csv`

### 7. Data-Centric Improvement Workflow

This repo includes one data-centric option to reduce identity overrepresentation during pair generation:

* `pairs.identity_cap_enabled`
* `pairs.max_pairs_per_identity`

Use the dedicated config for the improved variant:

```bash
python scripts/pair_lfw.py --config configs/milestone2_identity_cap.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/milestone2_identity_cap.yaml --mode sweep --selection-rule max_balanced_accuracy --note "data-centric-improved-identity-cap"
```

After running baseline and improved variants, compare artifacts in:

* `outputs/comparisons/baseline_vs_identity_cap.json`
* `outputs/comparisons/baseline_vs_identity_cap.csv`

### 8. Report and Submission-Visible Artifacts

Primary report:

* `reports/Milestone2_Report.md`

Submission-visible evidence copied under `reports/evidence/`:

* Figures:
	* `reports/evidence/figures/baseline_roc_curve.png`
	* `reports/evidence/figures/baseline_confusion_matrix_test.png`
	* `reports/evidence/figures/improved_roc_curve.png`
	* `reports/evidence/figures/improved_confusion_matrix_test.png`
* Comparison summaries:
	* `reports/evidence/comparisons/baseline_vs_identity_cap.json`
	* `reports/evidence/comparisons/baseline_vs_identity_cap.csv`

### 9. Reproduce Selected Threshold and Main Reported Result

Threshold policy used for reported results:

* Selection split: validation
* Rule: `max_balanced_accuracy`
* Score direction: `higher-is-more-same` (cosine similarity)
* Selected threshold in the current reported runs: `0.70` (baseline and improved)

Main reported comparison from current tracked runs:

* Baseline test accuracy: `0.5640`
* Improved test accuracy: `0.5483`
* Baseline test FPR/FNR: `0.2518 / 0.1842`
* Improved test FPR/FNR: `0.2518 / 0.1998`

These values are traceable in `outputs/run_summary.csv` and summarized in `reports/Milestone2_Report.md`.

### 10. Clean-Clone Reproducibility Check (Before Tagging)

From a fresh clone, run this exact sequence:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/ingest_lfw.py
python scripts/pair_lfw.py --config configs/default.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_balanced_accuracy --note "baseline-default"

python scripts/pair_lfw.py --config configs/milestone2_identity_cap.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/milestone2_identity_cap.yaml --mode sweep --selection-rule max_balanced_accuracy --note "data-centric-improved-identity-cap"

python -m pytest -q tests
```

Confirm after running:

* `outputs/run_summary.csv` contains both new run rows.
* Each run folder under `outputs/runs/<run_id>/` has `run_info.json`, `threshold_metrics.csv`, `test_metrics.json`, and `plots/confusion_matrix_test.png`.
* The selected threshold in both runs is `0.70` for the currently reported result path.

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
