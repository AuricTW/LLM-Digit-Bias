from __future__ import annotations

import argparse
import json

from src.analysis.run_registry import build_run_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a registry of completed experiment runs.")
    parser.add_argument("--raw-root", default="results/raw", help="Directory containing raw run folders.")
    parser.add_argument(
        "--processed-root",
        default="results/processed",
        help="Directory containing processed analysis outputs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = build_run_registry(raw_root=args.raw_root, processed_root=args.processed_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
