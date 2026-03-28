from __future__ import annotations

import argparse
import json

from src.runner.protocol_compatibility import run_protocol_compatibility_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a protocol-compatibility study for wrapper-prone models.")
    parser.add_argument("--config", required=True, help="JSON config for the compatibility study.")
    parser.add_argument("--raw-root", default="results/raw", help="Directory for raw outputs.")
    parser.add_argument("--processed-root", default="results/processed", help="Directory for processed outputs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_protocol_compatibility_study(
        config_path=args.config,
        raw_output_root=args.raw_root,
        processed_root=args.processed_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
