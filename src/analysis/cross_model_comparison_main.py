from __future__ import annotations

import argparse
import json

from src.analysis.cross_model_comparison import write_cross_model_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a pooled comparison table across completed ordering-study runs.")
    parser.add_argument(
        "--processed-root",
        default="results/processed",
        help="Directory containing processed experiment outputs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = write_cross_model_comparison(processed_root=args.processed_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
