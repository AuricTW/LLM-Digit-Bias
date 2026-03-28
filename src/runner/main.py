from __future__ import annotations

import argparse
import json

from src.runner.config import load_experiment_config
from src.runner.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an LLM digit-bias experiment.")
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    parser.add_argument(
        "--raw-output-root",
        default="results/raw",
        help="Directory where raw experiment runs should be stored.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    result = run_experiment(config=config, config_path=args.config, raw_output_root=args.raw_output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
