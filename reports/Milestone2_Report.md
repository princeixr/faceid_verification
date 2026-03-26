# MSML 605 Milestone 2 Report

## 1. Scope

This report is organized to mirror the milestone document requirements and to make grading straightforward.

## 2. Reproduction Commands (Copy-Paste)

### 2.1 Environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2.2 Baseline Pipeline

```bash
python scripts/ingest_lfw.py
python scripts/pair_lfw.py --config configs/default.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_balanced_accuracy --note "baseline-default"
python scripts/run_error_analysis.py --run-dir outputs/runs/run_20260326T203818Z_c9daa6e2 --split test --top-k 20
```

### 2.3 Data-Centric Improved Pipeline

```bash
python scripts/pair_lfw.py --config configs/milestone2_identity_cap.yaml
python scripts/similarity_lfw.py
python scripts/run_eval.py --config configs/milestone2_identity_cap.yaml --mode sweep --selection-rule max_balanced_accuracy --note "data-centric-improved-identity-cap"
python scripts/run_error_analysis.py --run-dir outputs/runs/run_20260326T212540Z_86ba5024 --split test --top-k 20
```

### 2.4 Tests

```bash
python -m pytest -q tests
python -m pytest -q tests/test_integration_eval_pipeline.py
```

Unit test coverage includes:

- metrics (tests/test_metrics.py)
- thresholding logic (tests/test_thresholding.py)
- validation checks (tests/test_validation.py)
- run tracking helpers (tests/test_tracking.py)

Integration test:

- tests/test_integration_eval_pipeline.py

## 3. What Was Implemented

### 3.1 Baseline and Threshold Selection

The baseline run used the default configuration and selected threshold via sweep using the balanced-accuracy rule.

- baseline run id: run_20260326T203818Z_c9daa6e2
- baseline config: configs/default.yaml
- threshold rule used: max_balanced_accuracy

Threshold selection is configurable per run through:

- scripts/run_eval.py (`--selection-rule`)
- src/thresholding.py (max_accuracy, max_balanced_accuracy, max_f1)

### 3.2 Tracked Runs and Provenance

Each evaluation run writes tracked artifacts including:

- outputs/run_summary.csv
- outputs/runs/<run_id>/run_info.json
- outputs/runs/<run_id>/config_used.yaml
- outputs/runs/<run_id>/threshold_metrics.json
- outputs/runs/<run_id>/threshold_metrics.csv
- outputs/runs/<run_id>/train_metrics.json
- outputs/runs/<run_id>/test_metrics.json
- outputs/runs/<run_id>/val_metrics.json
- outputs/runs/<run_id>/plots/*

### 3.3 Evaluation Modes

The evaluation runner supports both:

- threshold sweep mode (`--mode sweep`)
- fixed threshold mode (`--mode fixed --fixed-threshold <value>`)

### 3.4 Validation Checks

The pipeline validates:

- pair schema and required columns
- binary labels and split values
- referenced image paths
- threshold config range/increment/default
- score count vs pair count
- split integrity leakage

Implementation is in src/validation.py and is called from scripts/run_eval.py.

### 3.5 Tests

Tests included in this milestone:

- unit tests: tests/test_metrics.py
- integration test: tests/test_integration_eval_pipeline.py

### 3.6 Data-Centric Improvement

The implemented data-centric change is identity participation capping during pair generation.

- config switches:
  - pairs.identity_cap_enabled
  - pairs.max_pairs_per_identity
- improved config used: configs/milestone2_identity_cap.yaml

Related implementation and artifacts:

- src/pairing.py
- scripts/pair_lfw.py
- outputs/pairs/pair_policy.json

### 3.7 Error Slice Analysis

Error analysis generates two required slices:

- false positives
- false negatives

Artifacts are produced per run under:

- outputs/runs/<run_id>/error_analysis/test_error_slices_summary.json
- outputs/runs/<run_id>/error_analysis/test_false_positives.csv
- outputs/runs/<run_id>/error_analysis/test_false_negatives.csv

Runs analyzed in this report:

- outputs/runs/run_20260326T203818Z_c9daa6e2/error_analysis/test_error_slices_summary.json
- outputs/runs/run_20260326T212540Z_86ba5024/error_analysis/test_error_slices_summary.json

### 3.8 Baseline vs Improved Comparison Artifact

Comparison artifacts:

- outputs/comparisons/baseline_vs_identity_cap.json
- outputs/comparisons/baseline_vs_identity_cap.csv

## 4. Quantitative Summary (Baseline -> Improved)

- Selected threshold: 0.70 -> 0.75
- Test accuracy: 0.5642 -> 0.5427 (delta -0.0215)
- Val accuracy: 0.5753 -> 0.5783 (delta +0.0029)
- False positives rate (test): 0.2518 -> 0.1530 (delta -0.0988)
- False negatives rate (test): 0.1840 -> 0.3043 (delta +0.1203)

Interpretation:

- The improved policy is more conservative.
- It reduces false accepts significantly, but increases false rejects.
- This is a valid operating-point tradeoff and is reported transparently.

## 5. Required Narrative Elements

### 5.1 Threshold-Selection Rule (Concise)

- Rule used: maximize balanced accuracy on sweep results.

### 5.2 Before-vs-After Data-Centric Summary

- Before: no identity cap in pair participation.
- After: enabled identity cap (max_pairs_per_identity=120).

### 5.3 Two Error Slices + Hypotheses

- False positives hypothesis:
  - different identities with similar pose/lighting are scored too high.
- False negatives hypothesis:
  - same-identity pairs with blur/occlusion/pose gap are scored too low.

### 5.4 Iteration Lesson

- Data balancing changes operating behavior, not just scalar accuracy.
- The preferred variant depends on whether the use case prioritizes false-accept control or false-reject control.

## 6. Remaining Submission Item

- Milestone tag still pending:
  - create and push v0.2 on the final reproducible commit.
