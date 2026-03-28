from __future__ import annotations

import argparse
import json

from src.analysis.report import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a finished experiment run.")
    parser.add_argument("--run-dir", required=True, help="Raw run directory created by the runner.")
    parser.add_argument(
        "--processed-root",
        default="results/processed",
        help="Directory where processed analysis outputs should be written.",
    )
    parser.add_argument(
        "--digits",
        default="1,2,3,4,5,6,7,8,9",
        help="Comma-separated digit domain for the analysis.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    digits = [int(item.strip()) for item in args.digits.split(",") if item.strip()]
    result = run_analysis(run_dir=args.run_dir, processed_root=args.processed_root, digits=digits)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
