# Milestone 4 Profiling Report

This report profiles the final pair-level inference path used by `scripts/infer_pair.py`. The profiling command runs `src.inference.infer_pair`, so the measured stages match the CLI output.

## Methodology

Command used:

```bash
.venv/bin/python scripts/profile_inference.py --config configs/default.yaml --pairs-csv outputs/pairs/test_pairs.csv --repeats 3 --warmup 1 --batch-sizes 1,2,4 --output-dir reports/evidence/profiling
```

Timing uses `time.perf_counter()`. One warm-up call runs before timed measurements so the model and filesystem cache are initialized. The per-stage profile repeats one pair three times. Batch-size sensitivity processes the first `N` test pairs sequentially for `N` in `[1, 2, 4]`.

## Measurement Environment

| Item | Value |
|---|---|
| OS/platform | macOS-15.2-arm64-arm-64bit |
| Machine | arm64 |
| Processor | arm |
| Python | 3.11.0 |
| Torch | 2.11.0 |
| CUDA available | false |
| Embedding backend | `inceptionresnetv1` |
| Pretrained weights | `vggface2` |
| Device | CPU |
| Embedding dimension | 512 |
| Preprocess size | 160 x 160 |

## CPU Baseline: Per-Stage Latency

Single-pair stage latency summary:

| Stage | Mean ms | Median ms | p95 ms |
|---|---:|---:|---:|
| Preprocessing | 1.055 | 1.033 | 1.120 |
| Embedding generation | 43.204 | 44.307 | 44.617 |
| Similarity scoring | 0.023 | 0.022 | 0.025 |
| Threshold decision | 0.001 | 0.001 | 0.001 |
| Confidence computation | 0.004 | 0.004 | 0.004 |
| Combined scoring | 0.027 | 0.027 | 0.029 |
| End-to-end total | 44.287 | 45.457 | 45.656 |

The CPU baseline is dominated by embedding generation. Preprocessing is approximately 1 ms, while cosine scoring plus threshold and confidence is well below 0.1 ms.

## Batch-Size Sensitivity

Here "batch size" means the number of pairs processed sequentially in one timed loop, using the same single-pair inference function as the CLI.

| Batch size | Elapsed ms | Mean pair latency ms | Throughput pairs/sec |
|---:|---:|---:|---:|
| 1 | 42.287 | 42.136 | 23.648 |
| 2 | 87.485 | 43.577 | 22.861 |
| 4 | 182.465 | 45.448 | 21.922 |

Throughput remains roughly flat because the current CLI path runs pairs one at a time. Larger sequential batches add work nearly linearly and do not provide true vectorized model batching. A production service could improve throughput by batching tensor inference inside the embedding model, but that is outside the final project scope.

## Artifacts

Profiling outputs:

* `reports/evidence/profiling/profile_summary.json`
* `reports/evidence/profiling/single_pair_stage_records.csv`
* `reports/evidence/profiling/batch_size_sensitivity.csv`

Optional notebook:

* `reports/Profile_Inference_Milestone4.ipynb`

No GPU comparison is reported for the final release because the measured environment did not expose CUDA.
