from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.inference import infer_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local concurrent load test for pair inference.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"), help="Path to config YAML/JSON file.")
    parser.add_argument("--pairs-csv", required=True, help="CSV with left_path,right_path columns to invoke repeatedly.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads to use.")
    parser.add_argument("--repeat", type=int, default=1, help="How many times to repeat the input pair list.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold override.")
    parser.add_argument("--output-json", default=None, help="Optional path to write summary JSON.")
    return parser.parse_args()


def _read_pairs(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {"left_path", "right_path"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing)}")
    return rows


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def main() -> int:
    args = parse_args()
    config = Config.from_file(Path(args.config).expanduser())
    pair_rows = _read_pairs(Path(args.pairs_csv).expanduser())
    tasks = pair_rows * max(1, int(args.repeat))

    if not tasks:
        raise ValueError("No pairs found for load testing.")

    start = time.perf_counter()
    latencies_ms: list[float] = []
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        for row in tasks:
            futures.append(
                executor.submit(
                    infer_pair,
                    row["left_path"],
                    row["right_path"],
                    config,
                    threshold=args.threshold,
                )
            )

        for future in as_completed(futures):
            result = future.result()
            latencies_ms.append(float(result["latency_ms"]))

    total_elapsed_s = time.perf_counter() - start
    total_requests = len(tasks)
    throughput_rps = total_requests / total_elapsed_s if total_elapsed_s > 0 else 0.0

    summary: dict[str, Any] = {
        "total_requests": total_requests,
        "workers": int(args.workers),
        "repeat": int(args.repeat),
        "elapsed_seconds": float(total_elapsed_s),
        "throughput_rps": float(throughput_rps),
        "latency_ms": {
            "mean": float(statistics.fmean(latencies_ms)) if latencies_ms else 0.0,
            "median": float(statistics.median(latencies_ms)) if latencies_ms else 0.0,
            "p95": _percentile(latencies_ms, 0.95),
            "min": float(min(latencies_ms)) if latencies_ms else 0.0,
            "max": float(max(latencies_ms)) if latencies_ms else 0.0,
        },
    }

    summary_json = json.dumps(summary, indent=2)
    print(summary_json)

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary_json, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())