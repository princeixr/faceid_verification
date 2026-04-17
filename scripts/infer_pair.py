from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.inference import infer_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pair-level face verification inference.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"), help="Path to config YAML/JSON file.")
    parser.add_argument("--left-path", required=True, help="Relative path to the left face image.")
    parser.add_argument("--right-path", required=True, help="Relative path to the right face image.")
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
    parser.add_argument("--output-json", default=None, help="Optional path to write the result JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config.from_file(Path(args.config).expanduser())
    result = infer_pair(
        args.left_path,
        args.right_path,
        config,
        threshold=args.threshold,
        higher_means_same=args.higher_means_same,
    )

    result_json = json.dumps(result, indent=2)
    print(result_json)

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())