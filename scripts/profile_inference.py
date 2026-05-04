"""Profile the final pair-level inference path for Milestone 4.

The script times the same `src.inference.infer_pair` function used by the CLI
and writes lightweight CPU profiling summaries for documentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.inference import infer_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile pair-level face verification inference.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"), help="Path to config file.")
    parser.add_argument("--pairs-csv", default=str(PROJECT_ROOT / "outputs" / "pairs" / "test_pairs.csv"), help="CSV with left_path,right_path columns.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repeats per single-pair profile.")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up calls before timing.")
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="Comma-separated pair counts for batch sensitivity.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "evidence" / "profiling"), help="Directory for profiling summaries.")
    return parser.parse_args()


def read_pairs(csv_path: Path, needed: int) -> list[tuple[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"left_path", "right_path"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
        pairs = [(row["left_path"], row["right_path"]) for row in reader]
    if len(pairs) < needed:
        raise ValueError(f"Need at least {needed} pairs in {csv_path}, found {len(pairs)}.")
    return pairs[:needed]


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": mean(values),
        "median_ms": median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": float(min(values)) if values else 0.0,
        "max_ms": float(max(values)) if values else 0.0,
    }


def environment_summary(config: Config) -> dict[str, Any]:
    torch_summary: dict[str, Any]
    try:
        import torch

        torch_summary = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:  # pragma: no cover
        torch_summary = {"error": str(exc)}

    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "embedding": dict(config._config.get("embedding", {})),
        "torch": torch_summary,
    }


def profile_single_pair(
    config: Config,
    pair: tuple[str, str],
    *,
    repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage_names = [
        "preprocessing",
        "embedding_generation",
        "similarity_scoring",
        "threshold_decision",
        "confidence_computation",
        "total",
    ]
    stage_values: dict[str, list[float]] = {name: [] for name in stage_names}
    records: list[dict[str, Any]] = []

    for index in range(1, repeats + 1):
        result = infer_pair(pair[0], pair[1], config)
        stage_latency = result["stage_latency_ms"]
        record = {"repeat": index, **{name: float(stage_latency[name]) for name in stage_names}}
        record["combined_scoring"] = (
            record["similarity_scoring"] + record["threshold_decision"] + record["confidence_computation"]
        )
        records.append(record)
        for name in stage_names:
            stage_values[name].append(record[name])

    stage_values["combined_scoring"] = [row["combined_scoring"] for row in records]
    summary = {name: summarize_values(values) for name, values in stage_values.items()}
    return summary, records


def profile_batch_sizes(
    config: Config,
    pairs: list[tuple[str, str]],
    *,
    batch_sizes: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        batch_pairs = pairs[:batch_size]
        start = perf_counter()
        results = [infer_pair(left, right, config) for left, right in batch_pairs]
        elapsed_ms = (perf_counter() - start) * 1000.0
        per_pair_total = [float(result["stage_latency_ms"]["total"]) for result in results]
        rows.append(
            {
                "batch_size": batch_size,
                "elapsed_ms": float(elapsed_ms),
                "mean_pair_latency_ms": mean(per_pair_total),
                "throughput_pairs_per_second": float(batch_size / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")

    batch_sizes = [int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()]
    if not batch_sizes or min(batch_sizes) <= 0:
        raise ValueError("--batch-sizes must contain positive integers.")

    config = Config.from_file(Path(args.config).expanduser())
    pairs = read_pairs(Path(args.pairs_csv).expanduser(), max(max(batch_sizes), 1))
    first_pair = pairs[0]

    for _ in range(args.warmup):
        infer_pair(first_pair[0], first_pair[1], config)

    stage_summary, single_pair_records = profile_single_pair(config, first_pair, repeats=args.repeats)
    batch_rows = profile_batch_sizes(config, pairs, batch_sizes=batch_sizes)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "methodology": {
            "timer": "time.perf_counter",
            "warmup_calls": int(args.warmup),
            "single_pair_repeats": int(args.repeats),
            "batch_sizes": batch_sizes,
            "pair_source": str(Path(args.pairs_csv).expanduser()),
        },
        "environment": environment_summary(config),
        "single_pair_stage_latency_ms": stage_summary,
        "batch_size_sensitivity": batch_rows,
    }

    (output_dir / "profile_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output_dir / "single_pair_stage_records.csv", single_pair_records)
    write_csv(output_dir / "batch_size_sensitivity.csv", batch_rows)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
