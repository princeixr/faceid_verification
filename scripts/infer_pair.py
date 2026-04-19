"""Run one face pair through the Milestone 3 inference path.

This script is the CLI entrypoint for single-pair or CSV-batch verification.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

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
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Optional path for a comparison plot. For batch mode, provide a directory.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on how many pairs to process from --pairs-csv.",
    )
    args = parser.parse_args()

    if args.left_path is not None and not args.right_path:
        parser.error("--right-path is required when --left-path is provided.")
    if args.right_path is not None and not args.left_path:
        parser.error("--left-path is required when --right-path is provided.")
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be > 0 when provided.")
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
        rows = _read_pairs_csv(Path(args.pairs_csv).expanduser())
        if args.max_pairs is not None:
            return rows[: int(args.max_pairs)]
        return rows

    return [
        {
            "pair_id": args.pair_id or "pair_000001",
            "left_path": str(args.left_path),
            "right_path": str(args.right_path),
        }
    ]


def _format_result_text(index: int, result: dict[str, Any], pair_id: str) -> str:
    stage_latency_ms = result.get("stage_latency_ms", {})
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
    for stage_name in (
        "preprocessing",
        "embedding_generation",
        "similarity_scoring",
        "threshold_decision",
        "confidence_computation",
    ):
        if stage_name in stage_latency_ms:
            lines.append(f"stage_latency_ms.{stage_name}={float(stage_latency_ms[stage_name]):.3f}")
    return "\n".join(lines)


def _resolve_plot_output_path(base_path: Path, *, pair_id: str, batch_mode: bool) -> Path:
    if not batch_mode:
        return base_path

    if base_path.suffix:
        stem = base_path.stem
        suffix = base_path.suffix
        return base_path.with_name(f"{stem}_{pair_id}{suffix}")
    return base_path / f"{pair_id}.png"


def _resolve_inference_root(config: Config) -> Path:
    out_root = config.paths.out_root
    project_root = config.paths.project_root
    out_root_abs = out_root if out_root.is_absolute() else (project_root / out_root)
    return out_root_abs / "inference"


def _make_inference_run_dir(config: Config, *, batch_mode: bool) -> Path:
    root = _resolve_inference_root(config)
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_type = "batch" if batch_mode else "single"
    run_dir = root / f"infer_{run_type}_{ts}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"infer_{run_type}_{ts}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_comparison_plot(config: Config, result: dict[str, Any], output_path: Path) -> None:
    project_root = config.paths.project_root
    left_path = project_root / result["left_path"]
    right_path = project_root / result["right_path"]
    left_image = Image.open(left_path).convert("RGB")
    right_image = Image.open(right_path).convert("RGB")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(left_image)
    axes[0].set_title("Left Image")
    axes[0].axis("off")

    axes[1].imshow(right_image)
    axes[1].set_title("Right Image")
    axes[1].axis("off")

    fig.suptitle(
        "\n".join(
            [
                f"pair_id={result['pair_id']}",
                f"score={float(result['similarity_score']):.6f}  threshold={float(result['threshold']):.6f}",
                f"decision={int(result['decision'])}  confidence={float(result['confidence']):.6f}  latency_ms={float(result['latency_ms']):.3f}",
            ]
        ),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_default_inference_artifacts(
    config: Config,
    results: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> Path:
    batch_mode = len(results) > 1
    run_dir = _make_inference_run_dir(config, batch_mode=batch_mode)
    plots_dir = run_dir / "plots"
    pairs_dir = run_dir / "pairs"
    plots_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    run_payload = results[0] if not batch_mode else {"results": results}
    (run_dir / "results.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    metadata = {
        "mode": "batch" if batch_mode else "single",
        "pair_count": len(results),
        "config_path": str(Path(args.config).expanduser()),
        "threshold_override": args.threshold,
        "higher_means_same": bool(args.higher_means_same),
        "max_pairs": args.max_pairs,
        "results_json": str(run_dir / "results.json"),
    }
    (run_dir / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for result in results:
        pair_id = str(result["pair_id"])
        pair_json_path = pairs_dir / f"{pair_id}.json"
        pair_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _save_comparison_plot(config, result, plots_dir / f"{pair_id}.png")

    return run_dir


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

    _save_default_inference_artifacts(config, results, args=args)

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

    if args.output_plot is not None:
        plot_base_path = Path(args.output_plot).expanduser()
        batch_mode = len(results) > 1
        for result in results:
            pair_id = str(result["pair_id"])
            plot_path = _resolve_plot_output_path(plot_base_path, pair_id=pair_id, batch_mode=batch_mode)
            _save_comparison_plot(config, result, plot_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
