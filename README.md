# FaceID_Verification System

End-to-end face verification pipeline. The repo covers deterministic data ingestion, pair generation, tracked evaluation, an embedding-based Milestone 3 inference path built around `InceptionResnetV1`, Docker packaging, and a local load test.

<<<<<<< HEAD
* A similarity score between the two face embeddings
* A decision-confidence score derived from the score margin around the operating threshold
* A binary verification result (same identity / different identity)
* Basic latency measurements for performance analysis
=======
## What This Repo Produces
>>>>>>> 3de426606baa35318d9b77153e3411b1e63271fa

The system outputs:

* similarity score
* binary decision
* calibrated confidence
* latency for each inference
* tracked run artifacts for evaluation and reproducibility

## Getting Started

<<<<<<< HEAD
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
=======
Clone the repo and enter the project directory:
>>>>>>> 3de426606baa35318d9b77153e3411b1e63271fa

```bash
git clone "https://github.com/princeixr/FaceID_Verification"
cd FaceID_Verification
```

Create the environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to Run

If you want the full project flow, use this order:

1. Start with Milestone 1 to build the baseline data and evaluation pipeline.
2. Run Milestone 2 to compare the baseline with the improved pair-generation setup.
3. Run Milestone 3 to use the explicit embedding inference path, CLI, Docker, and load test.
4. Inspect the outputs under `outputs/` for run metadata, thresholds, and summaries.

## Milestones

Milestone 1 established the baseline pipeline: dataset ingestion, pair generation, similarity scoring, tracked evaluation, and the first saved run artifacts.

Milestone 2 extended that baseline with the identity-cap pair-generation variant, threshold selection, and error analysis for comparison.

Milestone 3 turned that pipeline into a deployable inference system: explicit face embeddings with `InceptionResnetV1`, separate inference stages, CLI entrypoint, confidence reporting, Docker packaging, and load testing.

## Milestone 1 Pipeline

<<<<<<< HEAD
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
=======
This is the base workflow that everything else builds on:
>>>>>>> 3de426606baa35318d9b77153e3411b1e63271fa

```bash
python scripts/ingest_lfw.py
python scripts/pair_lfw.py --config configs/default.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_balanced_accuracy --note "baseline-default"
python scripts/run_error_analysis.py --run-dir outputs/runs/<run_id> --split test --top-k 20
```

The command sequence above reproduces the tracked baseline pipeline and writes the core run artifacts under `outputs/runs/`.

## Milestone 2 Evaluation

Milestone 2 keeps the Milestone 1 pipeline and adds the improved pair-generation variant for comparison. Start from the Milestone 1 pipeline above, then run the identity-cap pair-generation variant below.

Baseline vs data-centric improvement:

* Baseline uses `configs/default.yaml` for pair generation and evaluation.
* Data-centric improvement uses `configs/milestone2_identity_cap.yaml` to cap identity contribution in pair sampling and reduce dominance by heavily represented identities.
* Both runs use the same threshold sweep rule (`max_balanced_accuracy`) so comparisons stay fair and reproducible.

Improved pair-generation variant:

```bash
python scripts/pair_lfw.py --config configs/milestone2_identity_cap.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/milestone2_identity_cap.yaml --mode sweep --selection-rule max_balanced_accuracy --note "data-centric-improved-identity-cap"
python scripts/run_error_analysis.py --run-dir outputs/runs/<run_id> --split test --top-k 20
```

Milestone 2 reproducible command block (environment, pair generation, evaluation, run logging, tests):

```bash
python scripts/pair_lfw.py --config configs/default.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_balanced_accuracy --note "baseline-default"

python scripts/pair_lfw.py --config configs/milestone2_identity_cap.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/milestone2_identity_cap.yaml --mode sweep --selection-rule max_balanced_accuracy --note "data-centric-improved-identity-cap"

python scripts/run_error_analysis.py --run-dir outputs/runs/<run_id> --split test --top-k 20
pytest tests/test_metrics.py tests/test_thresholding.py tests/test_tracking.py tests/test_validation.py tests/test_integration_eval_pipeline.py
```

Milestone 2 report and artifacts:

* Report: `reports/Milestone2_Report.md`
* Comparison outputs: `outputs/comparisons/baseline_vs_identity_cap.csv` and `outputs/comparisons/baseline_vs_identity_cap.json`
* Pair manifests: `outputs/manifests/lfw_manifest.json` and `outputs/manifests/lfw_samples.csv`
* Pair files: `outputs/pairs/train_pairs.csv`, `outputs/pairs/val_pairs.csv`, and `outputs/pairs/test_pairs.csv`
* Scored pairs: `outputs/similarity_score/train_pairs_scored.csv`, `outputs/similarity_score/val_pairs_scored.csv`, and `outputs/similarity_score/test_pairs_scored.csv`
* Run tracking: `outputs/run_summary.csv` and `outputs/runs/<run_id>/run_info.json`

Notes for reproducing selected threshold and main Milestone 2 result:

* Keep `--mode sweep --selection-rule max_balanced_accuracy` unchanged for both baseline and improved runs.
* Read each run's selected threshold and split metadata from `outputs/runs/<run_id>/run_info.json`.
* Use `outputs/comparisons/baseline_vs_identity_cap.csv` as the primary baseline-vs-improvement summary table referenced by the report.

## Milestone 3 Inference

Milestone 3 is the runnable deployment-oriented path in this repo. If you are treating the current embedding implementation as the active embedding stage, the practical flow is:

1. create the environment
2. ingest the dataset
3. generate deterministic train/val/test pairs
4. run tracked evaluation and threshold sweep
5. persist the selected threshold for inference
6. run pair-level inference
7. run the local load test
8. verify the Docker path

### End-to-End Commands

If you want to regenerate the Milestone 3 outputs from scratch, start from the project root and run the commands below in order.

Optional cleanup of generated artifacts:

```bash
rm -rf outputs data/lfw
```

Create the environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ingest the LFW dataset into the repo-local data directory:

```bash
python3 scripts/ingest_lfw.py
```

Generate deterministic verification pairs for train, validation, and test:

```bash
python3 scripts/pair_lfw.py --config configs/default.yaml
```

Run tracked evaluation, sweep thresholds on validation, and persist the selected threshold for inference. If you want F1-based threshold selection for the current `InceptionResnetV1` embedding system, use:

```bash
python3 scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_f1 --note "milestone3-embedding-threshold"
```

If you prefer the Milestone 2 rule instead, use:

```bash
python3 scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_balanced_accuracy --note "milestone3-embedding-threshold"
```

Inspect the persisted selected-threshold artifact that inference will use by default:

```bash
cat outputs/inference/selected_threshold.json
```

### CLI

Run single-pair inference locally. If `outputs/inference/selected_threshold.json` exists, the CLI will use that threshold automatically unless `--threshold` is passed explicitly:

```bash
python3 scripts/infer_pair.py --config configs/default.yaml --left-path data/lfw/images/Barbara_Walters/004492.jpg --right-path data/lfw/images/Barbara_Walters/007353.jpg --output-format json
```

Run batch inference over a CSV of pairs:

```bash
python3 scripts/infer_pair.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --output-format json
```

If you want to evaluate only the first `N` rows from the CSV, use `--max-pairs`:

```bash
python3 scripts/infer_pair.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --max-pairs 25 --output-format json
```

Every inference invocation now writes artifacts automatically under `outputs/inference/`. Single-pair runs create a folder like `outputs/inference/infer_single_<timestamp>/`, and batch runs create `outputs/inference/infer_batch_<timestamp>/`.

Each inference artifact folder contains:

* `results.json` - full result payload for the run
* `run_info.json` - run metadata, including pair count and threshold override if used
* `pairs/<pair_id>.json` - one JSON result per processed pair
* `plots/<pair_id>.png` - one side-by-side comparison plot per processed pair

Optional explicit output paths still work when you want an extra copy in a specific location:

```bash
python3 scripts/infer_pair.py --config configs/default.yaml --left-path data/lfw/images/Barbara_Walters/004492.jpg --right-path data/lfw/images/Barbara_Walters/007353.jpg --output-format json --output-json outputs/cli_test_infer_pair.json --output-plot outputs/cli_test_infer_pair.png
```

The CLI prints:

* pair identifier and input paths
* similarity score
* threshold
* binary decision
* calibrated confidence
* total latency
* per-stage latency breakdown for preprocessing, embedding, scoring, thresholding, and confidence

### Confidence

Confidence uses a deterministic logistic-margin rule. The stored confidence is confidence in the current binary decision, not an independent class probability. It measures how far the similarity score is from the operating threshold in the direction of the selected decision.

* `margin = similarity_score - threshold` when higher scores mean same identity
* `margin = threshold - similarity_score` when lower scores mean same identity
* `confidence = 1 / (1 + exp(-sharpness * margin))`
* default `sharpness = 10.0`

Range and interpretation:

* range: `(0, 1)`
* around `0.5`: the score is near the decision boundary, so the system has weak support for the current decision
* near `1.0`: the score is farther past the threshold in the decision direction, so the system has stronger support for the current decision
* this value should be read as threshold-margin decision confidence, not as a calibrated posterior probability of identity match

### Threshold

The current Milestone 3 operating threshold is selected on validation with the same sweep discipline used in Milestone 2.

* selected threshold: `0.30`
* selection rule: `max_balanced_accuracy`
* selection split: `val`
* persisted for inference in: `outputs/inference/selected_threshold.json`
* fallback config default in: `configs/default.yaml`
* recorded from the tracked validation sweep in: `outputs/runs/run_20260417T160805Z_6b612ae4/run_info.json`

### Docker

Build the container image for the CLI:

```bash
docker build -t faceid-verification:m3 .
```

Smoke-test the container entrypoint:

```bash
docker run --rm faceid-verification:m3 --help
```

Run single-pair inference inside Docker. Because the image does not bundle your local `data/` and `outputs/` directories, mount the repo into `/app`:

```bash
docker run --rm -v ${PWD}:/app -w /app faceid-verification:m3 --config configs/default.yaml --left-path data/lfw/images/Barbara_Walters/004492.jpg --right-path data/lfw/images/Barbara_Walters/007353.jpg --output-format json
```

Run batch CSV inference inside Docker:

```bash
docker run --rm -v ${PWD}:/app -w /app faceid-verification:m3 --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --output-format json
```

The image excludes `data/` and `outputs/` through `.dockerignore`, so mount the working directory when you run inference in the container.

### Embedding Model

The main Milestone 3 embedding backend is `InceptionResnetV1` from `facenet-pytorch` with pretrained `vggface2` weights. The repo keeps the older handcrafted embedding backend only as a lightweight fallback for tests; the default config uses the pretrained face model.

On the local machine, the first model-backed inference run may download the pretrained weights if they are not already cached. The Docker image prefetches those weights during build so the container path is more reproducible.

### Load Test

Run the local concurrency/load test after threshold selection and CLI verification:

```bash
python3 scripts/load_test.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --workers 2 --repeat 1 --output-json outputs/load_test_summary.json
```

The load-test summary includes:

* total requests processed
* successful requests
* failed requests
* total wall-clock time
* throughput in requests/sec
* latency distribution, including p95
* per-request records with latency or error text

### Tests

Run the Milestone 3-relevant tests from the repo root:

```bash
python3 -m pytest tests/test_embedding.py tests/test_inference.py tests/test_infer_pair_cli.py tests/test_thresholding.py tests/test_metrics.py tests/test_tracking.py tests/test_validation.py tests/test_integration_eval_pipeline.py
```

### Milestone 3 Artifacts

* Inference run artifacts: `outputs/inference/infer_single_<timestamp>/` or `outputs/inference/infer_batch_<timestamp>/`
* Optional explicit CLI JSON output: `outputs/cli_test_infer_pair.json`
* Load-test summary: `outputs/load_test_summary.json`
* Persisted inference threshold: `outputs/inference/selected_threshold.json`
* Selected-threshold metadata: `outputs/runs/run_20260417T160805Z_6b612ae4/run_info.json`
* Threshold sweep metrics: `outputs/runs/<run_id>/threshold_metrics.csv`

## Reproducibility Checklist

Use this command sequence from a clean workspace to reproduce the current Milestone 3 flow from scratch:

```bash
git clone "https://github.com/princeixr/FaceID_Verification"
cd FaceID_Verification

rm -rf outputs data/lfw

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 scripts/ingest_lfw.py
python3 scripts/pair_lfw.py --config configs/default.yaml
python3 scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_f1 --note "milestone3-embedding-threshold"

cat outputs/inference/selected_threshold.json

python3 scripts/infer_pair.py --config configs/default.yaml --left-path data/lfw/images/Barbara_Walters/004492.jpg --right-path data/lfw/images/Barbara_Walters/007353.jpg --output-format json
python3 scripts/infer_pair.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --max-pairs 25 --output-format json
python3 scripts/load_test.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --workers 2 --repeat 1 --output-json outputs/load_test_summary.json
python3 -m pytest tests/test_embedding.py tests/test_inference.py tests/test_infer_pair_cli.py tests/test_thresholding.py tests/test_metrics.py tests/test_tracking.py tests/test_validation.py tests/test_integration_eval_pipeline.py

docker build -t faceid-verification:m3 .
docker run --rm faceid-verification:m3 --help
docker run --rm -v "${PWD}:/app" -w /app faceid-verification:m3 --config configs/default.yaml --left-path data/lfw/images/Barbara_Walters/004492.jpg --right-path data/lfw/images/Barbara_Walters/007353.jpg --output-format json
```

Artifacts:

* `outputs/inference/infer_single_<timestamp>/results.json` - full single-pair inference result
* `outputs/inference/infer_batch_<timestamp>/results.json` - full batch inference result
* `outputs/inference/infer_batch_<timestamp>/pairs/<pair_id>.json` - per-pair batch results
* `outputs/inference/infer_batch_<timestamp>/plots/<pair_id>.png` - per-pair comparison plots
* `outputs/load_test_summary.json` - load-test summary and latency distribution
* `outputs/inference/selected_threshold.json` - persisted threshold used by inference
* `outputs/run_summary.csv` - tracked evaluation history
* `outputs/runs/<run_id>/run_info.json` - run metadata, including selected threshold
* `outputs/runs/<run_id>/threshold_metrics.csv` - validation sweep summary

## Repo Layout

* `src/` - core logic for config, embedding, inference, evaluation, thresholding, and tracking
* `scripts/` - runnable entrypoints for ingestion, pair generation, evaluation, CLI inference, and load test
* `configs/` - YAML configuration files
* `outputs/` - generated artifacts
* `data/` - downloaded LFW images
* `Dockerfile` - container entrypoint for the CLI

## Notes

* The default embedding stage now uses pretrained `InceptionResnetV1` face embeddings; the deterministic handcrafted backend remains available for tests and fallback use.
* The Milestone 3 inference path is split into preprocessing, embedding generation, similarity scoring, threshold decision, confidence computation, and latency measurement.
* Milestone 2 and Milestone 3 artifacts are preserved so the repo still supports tracked evaluation and comparison.
