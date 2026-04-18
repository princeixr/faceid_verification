"""Run one face pair through the Milestone 3 inference path.

This script is the CLI entrypoint for single-pair or CSV-batch verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.inference import infer_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pair-level face verification inference.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"), help="Path to config YAML/JSON file.")
    pair_source = parser.add_mutually_exclusive_group(required=True)
    pair_source.add_argument("--pairs-csv", default=None, help="CSV with left_path,right_path and optional pair_id columns.")
    pair_source.add_argument("--left-path", default=None, help="Relative path to the left face image.")
    parser.add_argument("--right-path", default=None, help="Relative path to the right face image (required with --left-path).")
    parser.add_argument("--pair-id", default=None, help="Optional identifier for single-pair mode.")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold; defaults to config threshold.")
    score_group = parser.add_mutually_exclusive_group()
    score_group.add_argument(
        "--higher-means-same",
        dest="higher_means_same",
        action="store_true",
        help="Interpret larger scores as more likely to be the same identity.",
    )
    score_group.add_argument(
        "--lower-means-same",
        dest="higher_means_same",
        action="store_false",
        help="Interpret smaller scores as more likely to be the same identity.",
    )
    parser.set_defaults(higher_means_same=True)
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="How to print inference output to stdout.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write the result JSON.")
    args = parser.parse_args()

    if args.left_path is not None and not args.right_path:
        parser.error("--right-path is required when --left-path is provided.")
    if args.right_path is not None and not args.left_path:
        parser.error("--left-path is required when --right-path is provided.")
    return args


def _read_pairs_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"left_path", "right_path"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing)}")

        rows = [dict(row) for row in reader]
    return rows


def _build_pairs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.pairs_csv is not None:
        return _read_pairs_csv(Path(args.pairs_csv).expanduser())

    return [
        {
            "pair_id": args.pair_id or "pair_000001",
            "left_path": str(args.left_path),
            "right_path": str(args.right_path),
        }
    ]


def _format_result_text(index: int, result: dict[str, Any], pair_id: str) -> str:
    lines = [
        f"pair_index={index}",
        f"pair_id={pair_id}",
        f"left_path={result['left_path']}",
        f"right_path={result['right_path']}",
        f"similarity_score={float(result['similarity_score']):.6f}",
        f"threshold={float(result['threshold']):.6f}",
        f"decision={int(result['decision'])}",
        f"confidence={float(result['confidence']):.6f}",
        f"latency_ms={float(result['latency_ms']):.3f}",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config = Config.from_file(Path(args.config).expanduser())
    pairs = _build_pairs(args)

    results: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        result = infer_pair(
            pair["left_path"],
            pair["right_path"],
            config,
            threshold=args.threshold,
            higher_means_same=args.higher_means_same,
        )
        pair_id = str(pair.get("pair_id") or f"pair_{index:06d}")
        result["pair_id"] = pair_id
        results.append(result)

    if args.output_format == "json":
        stdout_payload = results[0] if len(results) == 1 else {"results": results}
        print(json.dumps(stdout_payload, indent=2))
    else:
        rendered = [
            _format_result_text(i, result, str(result["pair_id"]))
            for i, result in enumerate(results, start=1)
        ]
        print("\n\n".join(rendered))

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_payload = results[0] if len(results) == 1 else {"results": results}
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())