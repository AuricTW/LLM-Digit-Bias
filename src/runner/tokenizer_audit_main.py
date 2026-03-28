from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.clients.factory import build_client
from src.io_utils import ensure_directory, write_csv, write_json
from src.runner.config import load_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tokenizer audit for the configured client.")
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    parser.add_argument(
        "--output-dir",
        default="results/tokenizer_audits",
        help="Directory where tokenizer audit artifacts should be written.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    client = build_client(config.client, seed=config.seed)
    try:
        audit = client.audit_tokenizer(config.digits)
    finally:
        client.close()

    if audit is None:
        raise SystemExit(f"Client provider {config.client['provider']!r} does not expose tokenizer audit output.")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_model_name = config.client["model_name"].replace("/", "__")
    output_dir = ensure_directory(Path(args.output_dir) / f"{config.client['provider']}__{safe_model_name}__{timestamp}")
    json_path = output_dir / "tokenizer_audit.json"
    csv_path = output_dir / "tokenizer_audit.csv"

    payload = {
        "config_path": args.config,
        "digits": config.digits,
        **audit,
    }
    write_json(json_path, payload)
    write_csv(csv_path, audit.get("surface_rows", []))

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "tokenizer_audit_json": str(json_path),
                "tokenizer_audit_csv": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
