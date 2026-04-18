# FaceID_Verification System

End-to-end face verification pipeline. The repo covers deterministic data ingestion, pair generation, tracked evaluation, an embedding-based Milestone 3 inference path, Docker packaging, and a local load test.

## What This Repo Produces

The system outputs:

* similarity score
* binary decision
* calibrated confidence
* latency for each inference
* tracked run artifacts for evaluation and reproducibility

## Getting Started

Clone the repo and enter the project directory:

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

Milestone 3 turned that pipeline into a deployable inference system: explicit embedding, separate inference stages, CLI entrypoint, confidence reporting, Docker packaging, and load testing.

## Milestone 1 Pipeline

This is the base workflow that everything else builds on:

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

### CLI

Single pair:

```bash
python scripts/infer_pair.py --config configs/default.yaml --left-path data/lfw/images/Barbara_Walters/004492.jpg --right-path data/lfw/images/Barbara_Walters/007353.jpg --output-format json --output-json outputs/cli_test_infer_pair.json
```

Batch CSV:

```bash
python scripts/infer_pair.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --output-format json
```

The CLI prints:

* pair identifier and input paths
* similarity score
* threshold
* binary decision
* calibrated confidence
* latency and per-stage latency breakdown

### Confidence

Confidence uses a deterministic logistic-margin rule:

* `margin = similarity_score - threshold` when higher scores mean same identity
* `margin = threshold - similarity_score` when lower scores mean same identity
* `confidence = 1 / (1 + exp(-sharpness * margin))`
* default `sharpness = 10.0`

Range and interpretation:

* range: `(0, 1)`
* around `0.5`: near the decision boundary
* near `1.0`: stronger support for the predicted class

### Threshold

The current Milestone 3 operating threshold is selected on validation with the same sweep discipline used in Milestone 2.

* selected threshold: `0.30`
* selection rule: `max_balanced_accuracy`
* selection split: `val`
* recorded in: `outputs/runs/run_20260417T160805Z_6b612ae4/run_info.json`

### Docker

Build:

```bash
docker build -t faceid-verification:m3 .
```

Smoke test:

```bash
docker run --rm faceid-verification:m3 --help
```

Single pair inside container:

```bash
docker run --rm -v ${PWD}:/app -w /app faceid-verification:m3 --config configs/default.yaml --left-path data/lfw/images/Barbara_Walters/004492.jpg --right-path data/lfw/images/Barbara_Walters/007353.jpg --output-format json
```

Batch CSV inside container:

```bash
docker run --rm -v ${PWD}:/app -w /app faceid-verification:m3 --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --output-format json
```

The image excludes `data/` and `outputs/` through `.dockerignore`, so mount the working directory when you run inference in the container.

### Load Test

```bash
python scripts/load_test.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --workers 4 --repeat 2 --output-json outputs/load_test_summary.json
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

```bash
pytest tests/test_embedding.py tests/test_inference.py tests/test_infer_pair_cli.py tests/test_thresholding.py tests/test_metrics.py tests/test_tracking.py tests/test_validation.py tests/test_integration_eval_pipeline.py
```

### Milestone 3 Artifacts

* CLI sample output: `outputs/cli_test_infer_pair.json`
* Load-test summary: `outputs/load_test_summary.json`
* Selected-threshold metadata: `outputs/runs/run_20260417T160805Z_6b612ae4/run_info.json`
* Threshold sweep metrics: `outputs/runs/<run_id>/threshold_metrics.csv`

## Reproducibility Checklist

Use this command sequence from a clean workspace to reproduce the full Milestone 1 to Milestone 3 flow:

```bash
git clone "https://github.com/princeixr/FaceID_Verification"
cd FaceID_Verification

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/ingest_lfw.py
python scripts/pair_lfw.py --config configs/default.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_balanced_accuracy --note "baseline-default"
python scripts/run_error_analysis.py --run-dir outputs/runs/<run_id> --split test --top-k 20

python scripts/pair_lfw.py --config configs/milestone2_identity_cap.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/milestone2_identity_cap.yaml --mode sweep --selection-rule max_balanced_accuracy --note "data-centric-improved-identity-cap"
python scripts/run_error_analysis.py --run-dir outputs/runs/<run_id> --split test --top-k 20

docker build -t faceid-verification:m3 .
docker run --rm faceid-verification:m3 --help

python scripts/infer_pair.py --config configs/default.yaml --left-path data/lfw/images/Barbara_Walters/004492.jpg --right-path data/lfw/images/Barbara_Walters/007353.jpg --output-format json --output-json outputs/cli_test_infer_pair.json
python scripts/load_test.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --workers 2 --repeat 1 --output-json outputs/load_test_summary.json
```

Artifacts:

* `outputs/cli_test_infer_pair.json` - sample CLI result
* `outputs/load_test_summary.json` - load-test summary and latency distribution
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

* The embedding stage is deterministic and local for now, so the pipeline stays reproducible and easy to package.
* The Milestone 3 inference path is split into preprocessing, embedding generation, similarity scoring, threshold decision, confidence computation, and latency measurement.
* Milestone 2 and Milestone 3 artifacts are preserved so the repo still supports tracked evaluation and comparison.
