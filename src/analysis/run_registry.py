from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.io_utils import ensure_directory, write_csv


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_run_registry(raw_root: str | Path = "results/raw", processed_root: str | Path = "results/processed") -> dict[str, Any]:
    raw_path = Path(raw_root)
    processed_path = Path(processed_root)
    output_dir = ensure_directory(processed_path)

    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(raw_path.glob("*/manifest.json")):
        run_dir = manifest_path.parent
        manifest = _load_json(manifest_path)
        processed_dir = processed_path / run_dir.name
        metrics_path = processed_dir / "condition_metrics.csv"

        metrics_frame = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        mean_invalid_rate = (
            float(metrics_frame["invalid_rate"].mean()) if not metrics_frame.empty else float("nan")
        )
        max_invalid_rate = (
            float(metrics_frame["invalid_rate"].max()) if not metrics_frame.empty else float("nan")
        )

        rows.append(
            {
                "run_id": manifest.get("run_id", run_dir.name),
                "experiment_name": manifest.get("experiment_name"),
                "created_at_utc": manifest.get("created_at_utc"),
                "client_provider": manifest.get("client_provider"),
                "model_name": manifest.get("model_name"),
                "condition_count": manifest.get("condition_count"),
                "repetitions_per_condition": manifest.get("repetitions_per_condition"),
                "logprob_enabled": manifest.get("logprob_enabled"),
                "tokenizer_audit_available": manifest.get("tokenizer_audit_available"),
                "mean_invalid_rate": mean_invalid_rate,
                "max_invalid_rate": max_invalid_rate,
                "run_dir": str(run_dir),
                "processed_dir": str(processed_dir) if processed_dir.exists() else "",
                "summary_markdown": str(processed_dir / "summary.md") if (processed_dir / "summary.md").exists() else "",
                "ordering_summary_markdown": (
                    str(processed_dir / "ordering_effect_summary.md")
                    if (processed_dir / "ordering_effect_summary.md").exists()
                    else ""
                ),
            }
        )

    csv_path = output_dir / "run_registry.csv"
    write_csv(csv_path, rows)

    lines = [
        "# Run Registry",
        "",
        "| run_id | model | provider | conditions | reps | mean invalid | max invalid | summary | ordering summary |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        summary_name = Path(row["summary_markdown"]).name if row["summary_markdown"] else ""
        ordering_name = Path(row["ordering_summary_markdown"]).name if row["ordering_summary_markdown"] else ""
        lines.append(
            "| {run_id} | {model_name} | {client_provider} | {condition_count} | {repetitions_per_condition} | {mean_invalid_rate:.4f} | {max_invalid_rate:.4f} | {summary_name} | {ordering_name} |".format(
                **row,
                summary_name=summary_name,
                ordering_name=ordering_name,
            )
        )

    markdown_path = output_dir / "run_registry.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "run_registry_csv": str(csv_path),
        "run_registry_markdown": str(markdown_path),
        "run_count": len(rows),
    }
