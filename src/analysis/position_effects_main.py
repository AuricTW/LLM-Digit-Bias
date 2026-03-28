from __future__ import annotations

import argparse
import json

from src.analysis.position_effects import write_position_effect_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize position effects for a finished experiment run.")
    parser.add_argument("--run-dir", required=True, help="Raw run directory created by the runner.")
    parser.add_argument(
        "--processed-root",
        default="results/processed",
        help="Directory where processed analysis outputs were written.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = write_position_effect_summary(run_dir=args.run_dir, processed_root=args.processed_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
