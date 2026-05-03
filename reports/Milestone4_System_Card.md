# MSML 605 Milestone 4 System Card

## 1. System Overview

This system is a face verification pipeline that takes two face images, produces embeddings, computes a similarity score, applies an operating threshold, and returns a binary same/different decision with confidence and latency information.

The current project implementation includes deterministic data ingestion and pair generation, tracked evaluation, embedding-based inference, Docker packaging, and local load testing. Milestone 4 does not change the core verifier; it documents the final system, its intended use, its limits, and its runtime behavior.

## 2. Intended Use

This system is intended for controlled face verification experiments and classroom demonstration of a reproducible ML pipeline. It supports comparing two aligned face images and reporting whether they appear to belong to the same identity under the selected threshold.

Appropriate uses:

* reproducible evaluation on the prepared LFW-based benchmark pairs
* offline pairwise verification experiments
* controlled local inference demonstrations
* profiling and benchmarking of the deployed verifier

## 3. Out of Scope

This system is not intended for:

* identity search across a large gallery
* access control or authentication in high-security settings
* surveillance or watchlist deployment
* automated decisions with legal, employment, or safety consequences
* demographic attribute inference

The system should also not be presented as a general-purpose face recognition product. It is a pairwise verifier for a specific benchmark and a specific release configuration.

## 4. System Inputs and Outputs

Inputs:

* two face image paths, or a CSV of left/right image pairs
* a configuration file that defines preprocessing, embedding, threshold, and output settings

Outputs:

* similarity score
* binary decision
* confidence derived from distance to the threshold
* end-to-end latency
* stage-level latency breakdown for inference

The verifier uses the persisted threshold artifact when present, and otherwise falls back to the configured default threshold.

## 5. Model and Data

The main embedding backend is `InceptionResnetV1` from `facenet-pytorch` with pretrained `vggface2` weights. The pipeline is built around the LFW dataset, with deterministic ingestion and pair generation.

The project also keeps a deterministic handcrafted embedding path available for tests and fallback use. That path exists to support reproducibility and automated checks, but the deployed Milestone 3/4 verifier path is the embedding-based one.

## 6. Operating Threshold and Decision Rule

The current project treats higher similarity scores as more likely same-identity matches.

Threshold precedence during inference is:

1. explicit command-line override
2. persisted selected-threshold artifact
3. configuration default

The same threshold governs both the binary decision and the reported decision confidence. Confidence is a margin-based score, not a calibrated probability.

## 7. Failure Modes and Limitations

Observed and expected failure modes include:

* false positives on different-identity pairs with similar pose, lighting, or appearance
* false negatives on same-identity pairs with blur, occlusion, pose variation, or low image quality
* sensitivity to preprocessing quality and input image path validity
* runtime variation caused by model warm-up, CPU load, and batch size

The system also inherits dataset limitations from LFW-style benchmark evaluation. Performance measured on that benchmark should not be interpreted as guaranteed performance in unconstrained real-world settings.

## 8. Fairness and Responsible Use

This system card does not claim demographic fairness guarantees. The project is evaluated on LFW-derived benchmark pairs, which is not enough to support strong claims about population-wide fairness, robustness across demographic groups, or performance across capture conditions.

Responsible-use notes:

* do not use the verifier as the sole basis for a consequential decision
* do not overstate confidence as a calibrated probability of identity match
* do not imply generalization beyond the evaluated benchmark without new evidence
* document any deployment setting where identity verification may create harm from false accepts or false rejects

If the final submission includes a fairness-risk discussion, it should be framed as a limitation analysis rather than as a claim of parity or bias elimination.

## 9. Hardware-Aware Profiling Status

Need to add profiling results...

## 10. Reproducibility Checklist

Use the following commands from a clean clone to recreate the current release artifacts:

```bash
git clone "https://github.com/princeixr/FaceID_Verification"
cd FaceID_Verification

python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

python scripts/ingest_lfw.py
python scripts/pair_lfw.py --config configs/default.yaml
python scripts/run_eval.py --config configs/default.yaml --mode sweep --selection-rule max_f1 --note "milestone4-system-card"
python scripts/infer_pair.py --config configs/default.yaml --left-path data/lfw/images/Barbara_Walters/004492.jpg --right-path data/lfw/images/Barbara_Walters/007353.jpg --output-format json
python scripts/load_test.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --workers 2 --repeat 1 --output-json outputs/load_test_summary.json
python -m pytest tests/test_embedding.py tests/test_inference.py tests/test_infer_pair_cli.py tests/test_thresholding.py tests/test_metrics.py tests/test_tracking.py tests/test_validation.py tests/test_integration_eval_pipeline.py
```

Key artifacts referenced by this system card:

* `outputs/inference/selected_threshold.json`
* `outputs/load_test_summary.json`
* `outputs/run_summary.csv`
* `outputs/runs/<run_id>/run_info.json`
* `outputs/runs/<run_id>/threshold_metrics.csv`
