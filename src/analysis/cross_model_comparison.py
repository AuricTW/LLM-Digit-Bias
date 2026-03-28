from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.io_utils import ensure_directory, write_csv


def _family_label(model_name: str) -> str:
    lowered = model_name.lower()
    if "gemma" in lowered:
        return "Gemma"
    if "llama" in lowered:
        return "Llama"
    if "qwen3" in lowered:
        return "Qwen3"
    if "qwen2.5" in lowered:
        return "Qwen2.5"
    if "smollm" in lowered:
        return "SmolLM"
    return "Other"


def _scale_label(model_name: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)([BM])", model_name, flags=re.IGNORECASE)
    if not match:
        return "unknown"
    value = match.group(1)
    unit = match.group(2).upper()
    return f"{value}{unit}"


def _scale_sort_key(scale_label: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([BM])", scale_label, flags=re.IGNORECASE)
    if not match:
        return float("inf")
    value = float(match.group(1))
    unit = match.group(2).upper()
    return value * (1000.0 if unit == "B" else 1.0)


def _mechanism_label(values: list[str]) -> str:
    unique = {value for value in values if value}
    if not unique:
        return "unknown"
    if len(unique) == 1:
        return next(iter(unique))
    return "hybrid"


def _top_digit_pattern(ordering_frame: pd.DataFrame) -> str:
    ordered = ordering_frame.sort_values(by=["ordering_label"])
    parts = []
    for _, row in ordered.iterrows():
        parts.append(f"{row['ordering_label']}={row['top_observed_digit']}")
    return ", ".join(parts)


def _temp_slice(frame: pd.DataFrame, temperature: float) -> pd.DataFrame:
    return frame.loc[pd.to_numeric(frame["temperature"], errors="coerce") == temperature].copy()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _iter_run_dirs(processed_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for child in processed_root.iterdir():
        if not child.is_dir():
            continue
        if not (child / "position_effect_summary.csv").exists():
            continue
        if not (child / "ordering_effect_summary.csv").exists():
            continue
        if not (child / "condition_metrics.csv").exists():
            continue
        run_dirs.append(child)
    return sorted(run_dirs, key=lambda path: path.name)


def write_cross_model_comparison(processed_root: str | Path = "results/processed") -> dict[str, Any]:
    processed_dir = ensure_directory(Path(processed_root))
    run_dirs = _iter_run_dirs(processed_dir)

    if not run_dirs:
        output_path = processed_dir / "cross_model_comparison.csv"
        output_path.write_text("", encoding="utf-8")
        return {
            "processed_root": str(processed_dir),
            "run_count": 0,
            "cross_model_comparison_csv": str(output_path),
            "cross_model_comparison_markdown": str(processed_dir / "cross_model_comparison.md"),
        }

    run_rows: list[dict[str, Any]] = []
    temp_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        position_frame = _read_csv(run_dir / "position_effect_summary.csv")
        ordering_frame = _read_csv(run_dir / "ordering_effect_summary.csv")
        metric_frame = _read_csv(run_dir / "condition_metrics.csv")

        model_name = str(position_frame.iloc[0]["model_name"])
        family = _family_label(model_name)
        scale = _scale_label(model_name)

        temp_mechanisms: list[str] = []
        temp_summary: dict[float, dict[str, Any]] = {}
        for temperature in (0.0, 0.2):
            temp_position = _temp_slice(position_frame, temperature)
            temp_ordering = _temp_slice(ordering_frame, temperature)
            temp_metrics = _temp_slice(metric_frame, temperature)
            if temp_position.empty or temp_ordering.empty or temp_metrics.empty:
                continue

            position_row = temp_position.iloc[0]
            unique_top_digits = sorted(
                {
                    str(value)
                    for value in temp_ordering["top_observed_digit"].dropna().astype(str).tolist()
                },
                key=lambda value: (len(value), value),
            )
            unique_top_positions = sorted(
                {
                    str(value)
                    for value in temp_position["top_observed_position"].dropna().astype(str).tolist()
                }
            )
            invalid_mean = float(pd.to_numeric(temp_metrics["invalid_rate"], errors="coerce").mean())
            invalid_max = float(pd.to_numeric(temp_metrics["invalid_rate"], errors="coerce").max())

            summary = {
                "observed_focus_heuristic": str(position_row["observed_focus_heuristic"]),
                "policy_focus_heuristic": str(position_row["policy_focus_heuristic"]),
                "top_observed_digit": str(position_row["top_observed_digit"]),
                "top_observed_digit_prob": float(position_row["top_observed_digit_prob"]),
                "top_observed_position": str(position_row["top_observed_position"]),
                "top_observed_position_prob": float(position_row["top_observed_position_prob"]),
                "observed_position_js_to_uniform_bits": float(
                    position_row["observed_position_js_to_uniform_bits"]
                ),
                "observed_digit_js_to_uniform_bits": float(
                    position_row["observed_digit_js_to_uniform_bits"]
                ),
                "ordering_diversity_count": int(len(unique_top_digits)),
                "ordering_diversity_digits": ",".join(unique_top_digits),
                "ordering_top_digit_pattern": _top_digit_pattern(temp_ordering),
                "invalid_rate_mean": invalid_mean,
                "invalid_rate_max": invalid_max,
            }
            temp_summary[temperature] = summary
            temp_mechanisms.append(summary["observed_focus_heuristic"])
            temp_rows.append(
                {
                    "run_id": run_dir.name,
                    "model_name": model_name,
                    "family": family,
                    "scale": scale,
                    "temperature": temperature,
                    **summary,
                }
            )

        row: dict[str, Any] = {
            "run_id": run_dir.name,
            "model_name": model_name,
            "family": family,
            "scale": scale,
            "mechanism_overall": _mechanism_label(temp_mechanisms),
        }
        for temperature, prefix in ((0.0, "temp0"), (0.2, "temp02")):
            summary = temp_summary.get(temperature, {})
            row.update(
                {
                    f"{prefix}_mechanism": summary.get("observed_focus_heuristic"),
                    f"{prefix}_policy_mechanism": summary.get("policy_focus_heuristic"),
                    f"{prefix}_top_digit": summary.get("top_observed_digit"),
                    f"{prefix}_top_digit_prob": summary.get("top_observed_digit_prob"),
                    f"{prefix}_top_position": summary.get("top_observed_position"),
                    f"{prefix}_top_position_prob": summary.get("top_observed_position_prob"),
                    f"{prefix}_position_js": summary.get("observed_position_js_to_uniform_bits"),
                    f"{prefix}_digit_js": summary.get("observed_digit_js_to_uniform_bits"),
                    f"{prefix}_ordering_diversity_count": summary.get("ordering_diversity_count"),
                    f"{prefix}_ordering_diversity_digits": summary.get("ordering_diversity_digits"),
                    f"{prefix}_ordering_top_digit_pattern": summary.get("ordering_top_digit_pattern"),
                    f"{prefix}_invalid_rate_mean": summary.get("invalid_rate_mean"),
                    f"{prefix}_invalid_rate_max": summary.get("invalid_rate_max"),
                }
            )
        run_rows.append(row)

    run_rows.sort(
        key=lambda row: (
            row["family"],
            _scale_sort_key(str(row["scale"])),
            str(row["model_name"]),
        )
    )
    temp_rows.sort(
        key=lambda row: (
            row["family"],
            _scale_sort_key(str(row["scale"])),
            str(row["model_name"]),
            float(row["temperature"]),
        )
    )

    run_csv_path = processed_dir / "cross_model_comparison.csv"
    temp_csv_path = processed_dir / "cross_model_temperature_details.csv"
    write_csv(run_csv_path, run_rows)
    write_csv(temp_csv_path, temp_rows)

    lines = [
        "# Cross-Model Comparison",
        "",
        "Mechanism labels are based on the pooled position-analysis heuristic over the five fixed ordering conditions.",
        "",
        "| family | scale | model | overall | temp0 | temp0.2 | temp0 top digit | temp0 top pos | temp0 unique top digits | temp0.2 top digit | temp0.2 top pos | temp0.2 unique top digits |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: |",
    ]
    for row in run_rows:
        lines.append(
            "| {family} | {scale} | {model_name} | {mechanism_overall} | {temp0_mechanism} | {temp02_mechanism} | {temp0_top_digit}:{temp0_top_digit_prob:.4f} | {temp0_top_position}:{temp0_top_position_prob:.4f} | {temp0_ordering_diversity_count} | {temp02_top_digit}:{temp02_top_digit_prob:.4f} | {temp02_top_position}:{temp02_top_position_prob:.4f} | {temp02_ordering_diversity_count} |".format(
                **row
            )
        )
        lines.append(
            "| pattern | pattern | pattern |  | {temp0_ordering_top_digit_pattern} | {temp02_ordering_top_digit_pattern} |  |  |  |  |  |  |".format(
                **row
            )
        )

    markdown_path = processed_dir / "cross_model_comparison.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "processed_root": str(processed_dir),
        "run_count": len(run_rows),
        "cross_model_comparison_csv": str(run_csv_path),
        "cross_model_temperature_details_csv": str(temp_csv_path),
        "cross_model_comparison_markdown": str(markdown_path),
    }
