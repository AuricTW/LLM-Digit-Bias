from __future__ import annotations

import argparse
import json

from src.analysis.report_artifacts import build_report_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build report-ready comparison tables and narrative summaries.")
    parser.add_argument(
        "--processed-root",
        default="results/processed",
        help="Directory containing processed experiment outputs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = build_report_artifacts(processed_root=args.processed_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
