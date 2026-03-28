from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.io_utils import ensure_directory, write_csv


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _top_digit(rows: pd.DataFrame, value_column: str) -> tuple[str | None, float | None]:
    if rows.empty:
        return None, None
    filtered = rows.loc[pd.notna(rows[value_column])].copy()
    if filtered.empty:
        return None, None
    ordered = filtered.sort_values(by=[value_column, "digit"], ascending=[False, True])
    top = ordered.iloc[0]
    return str(top["digit"]), float(top[value_column])


def _support_string(rows: pd.DataFrame, value_column: str) -> str:
    supports = rows.loc[pd.notna(rows[value_column]) & (rows[value_column] > 0), ["digit", value_column]].sort_values(
        by=[value_column, "digit"],
        ascending=[False, True],
    )
    if supports.empty:
        return ""
    parts = []
    for _, row in supports.iterrows():
        parts.append(f"{int(row['digit'])}:{float(row[value_column]):.4f}")
    return ", ".join(parts)


def _format_optional_number(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def _format_optional_label_value(label: str | None, value: float | None, precision: int = 4) -> str:
    if label is None or value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{label}:{value:.{precision}f}"


def write_ordering_effect_summary(run_dir: str | Path, processed_root: str | Path = "results/processed") -> dict[str, Any]:
    run_path = Path(run_dir)
    processed_dir = ensure_directory(Path(processed_root) / run_path.name)

    records = _load_jsonl(run_path / "records.jsonl")
    raw_frame = pd.DataFrame(records)
    freq_frame = pd.read_csv(processed_dir / "digit_frequencies.csv")
    candidate_frame = pd.read_csv(processed_dir / "candidate_probabilities.csv")
    metric_frame = pd.read_csv(processed_dir / "condition_metrics.csv")

    order_map = (
        raw_frame.groupby("condition_id", sort=False)["digits_order"]
        .first()
        .apply(lambda value: ",".join(str(item) for item in value))
        .to_dict()
    )

    summary_rows: list[dict[str, Any]] = []
    for condition_id, metric_rows in metric_frame.groupby("condition_id", sort=True):
        freq_rows = freq_frame.loc[freq_frame["condition_id"] == condition_id].copy()
        cand_rows = candidate_frame.loc[candidate_frame["condition_id"] == condition_id].copy()

        top_observed_digit, top_observed_proportion = _top_digit(freq_rows, "observed_proportion")
        top_model_digit, top_model_prob = _top_digit(cand_rows, "mean_candidate_prob")
        top_policy_digit, top_policy_prob = _top_digit(cand_rows, "mean_policy_candidate_prob")
        metric_row = metric_rows.iloc[0]

        summary_rows.append(
            {
                "condition_id": condition_id,
                "ordering_label": metric_row["ordering_label"],
                "temperature": float(metric_row["temperature"]),
                "digits_order": order_map.get(condition_id, ""),
                "top_observed_digit": top_observed_digit,
                "top_observed_proportion": top_observed_proportion,
                "observed_support": _support_string(freq_rows, "observed_proportion"),
                "top_model_digit": top_model_digit,
                "top_model_prob": top_model_prob,
                "top_policy_digit": top_policy_digit,
                "top_policy_prob": top_policy_prob,
                "invalid_rate": float(metric_row["invalid_rate"]),
                "entropy_bits": float(metric_row["entropy_bits"]),
                "observed_vs_model_corr": float(metric_row["observed_vs_candidate_corr"]),
                "observed_vs_policy_corr": float(metric_row["observed_vs_policy_candidate_corr"]),
            }
        )

    csv_path = processed_dir / "ordering_effect_summary.csv"
    write_csv(csv_path, summary_rows)

    lines = [
        "# Ordering Effect Summary",
        "",
        "| condition_id | order | temp | top observed | observed support | top model | top policy | entropy | obs-vs-model | obs-vs-policy |",
        "| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {condition_id} | {digits_order} | {temperature:.1f} | {top_observed} | {observed_support} | {top_model} | {top_policy} | {entropy_bits} | {observed_vs_model_corr} | {observed_vs_policy_corr} |".format(
                condition_id=row["condition_id"],
                digits_order=row["digits_order"],
                temperature=row["temperature"],
                top_observed=_format_optional_label_value(
                    row["top_observed_digit"],
                    row["top_observed_proportion"],
                ),
                observed_support=row["observed_support"] or "n/a",
                top_model=_format_optional_label_value(row["top_model_digit"], row["top_model_prob"]),
                top_policy=_format_optional_label_value(row["top_policy_digit"], row["top_policy_prob"]),
                entropy_bits=_format_optional_number(row["entropy_bits"]),
                observed_vs_model_corr=_format_optional_number(row["observed_vs_model_corr"]),
                observed_vs_policy_corr=_format_optional_number(row["observed_vs_policy_corr"]),
            )
        )

    markdown_path = processed_dir / "ordering_effect_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "output_dir": str(processed_dir),
        "ordering_effect_summary_csv": str(csv_path),
        "ordering_effect_summary_markdown": str(markdown_path),
        "condition_count": len(summary_rows),
    }
