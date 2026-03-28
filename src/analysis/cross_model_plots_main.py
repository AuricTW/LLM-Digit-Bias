from __future__ import annotations

import argparse
import json

from src.analysis.cross_model_plots import write_cross_model_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pooled cross-model comparison figures.")
    parser.add_argument(
        "--details-csv",
        default="results/processed/cross_model_temperature_details.csv",
        help="Cross-model temperature detail CSV produced by cross_model_comparison_main.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/processed/cross_model_figures",
        help="Directory where figures should be written.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = write_cross_model_plots(details_csv=args.details_csv, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
