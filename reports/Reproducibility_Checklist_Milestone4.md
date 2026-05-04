# Milestone 4 Reproducibility Checklist

Use this checklist from a clean clone to reproduce the final system path, core metrics, profiling summary, Docker CLI.

## 1. Clone And Install

```bash
git clone "https://github.com/princeixr/FaceID_Verification"
cd FaceID_Verification

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Recreate Data And Pairs

```bash
python3 scripts/ingest_lfw.py
python3 scripts/pair_lfw.py --config configs/default.yaml
```

## 3. Reproduce Threshold And Metrics

```bash
python3 scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_f1 --note "milestone4-final"
cat outputs/inference/selected_threshold.json
```

Expected final operating threshold: `0.40`, selected on validation with `max_f1`.

Inspect generated metrics:

```bash
cat outputs/runs/<run_id>/val_metrics.json
cat outputs/runs/<run_id>/test_metrics.json
```

The final release report references source run `run_20260419T004952Z_a7f2c22b`, where the held-out test metrics at threshold `0.40` were accuracy `0.9770`, balanced accuracy `0.9770`, precision `0.9773`, recall `0.9767`, and F1 `0.9770`.

## 4. Run Local CLI Inference

Single pair:

```bash
python3 scripts/infer_pair.py --config configs/default.yaml --left-path data/lfw/images/Barbara_Walters/004492.jpg --right-path data/lfw/images/Barbara_Walters/007353.jpg --output-format json
```

Batch sample:

```bash
python3 scripts/infer_pair.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --max-pairs 25 --output-format json
```

Inference artifacts are written under `outputs/inference/infer_single_<timestamp>/` and `outputs/inference/infer_batch_<timestamp>/`.

## 5. Reproduce CPU Profiling

```bash
python3 scripts/profile_inference.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --repeats 3 --warmup 1 --batch-sizes 1,2,4 --output-dir reports/evidence/profiling
```

Profiling artifacts:

* `reports/evidence/profiling/profile_summary.json`
* `reports/evidence/profiling/single_pair_stage_records.csv`
* `reports/evidence/profiling/batch_size_sensitivity.csv`

## 6. Run Tests

```bash
python3 -m pytest tests/test_embedding.py tests/test_inference.py tests/test_infer_pair_cli.py tests/test_thresholding.py tests/test_metrics.py tests/test_tracking.py tests/test_validation.py tests/test_integration_eval_pipeline.py
```

## 7. Build And Run Docker CLI

```bash
docker build -t faceid-verification:v1.0-final .
docker run --rm faceid-verification:v1.0-final --help
docker run --rm -v "${PWD}:/app" -w /app faceid-verification:v1.0-final --config configs/default.yaml --left-path data/lfw/images/Barbara_Walters/004492.jpg --right-path data/lfw/images/Barbara_Walters/007353.jpg --output-format json
```

The Docker image does not include `data/` or `outputs/`; mount the repo as shown above.

## 8. Verify Final Artifact Locations

* README: `README.md`
* System Card: `reports/System_Card_Milestone4.md`
* Profiling report: `reports/Profiling_Report_Milestone4.md`
* Reproducibility checklist: `reports/Reproducibility_Checklist_Milestone4.md`
* Final config: `configs/default.yaml`
* CLI entrypoint: `scripts/infer_pair.py`
* Profiling entrypoint: `scripts/profile_inference.py`
