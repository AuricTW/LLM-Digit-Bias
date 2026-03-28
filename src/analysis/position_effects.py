from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.metrics import (
    entropy_bits,
    jensen_shannon_divergence_to_uniform,
    pearson_correlation,
)
from src.io_utils import ensure_directory, write_csv


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ordered_distribution(
    frame: pd.DataFrame,
    key_column: str,
    value_column: str,
    keys: list[int],
) -> dict[str, float]:
    if frame.empty:
        return {str(key): 0.0 for key in keys}
    keyed = frame.set_index(key_column)[value_column].to_dict()
    distribution: dict[str, float] = {}
    for key in keys:
        value = keyed.get(key, 0.0)
        if pd.isna(value):
            distribution[str(key)] = 0.0
        else:
            distribution[str(key)] = float(value)
    return distribution


def _top_key(distribution: dict[str, float]) -> tuple[str | None, float | None]:
    if not distribution:
        return None, None
    if not any(value > 0 for value in distribution.values()):
        return None, None
    ordered = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    return ordered[0][0], float(ordered[0][1])


def _support_string(distribution: dict[str, float]) -> str:
    supports = sorted(
        [(key, value) for key, value in distribution.items() if (not math.isnan(value)) and value > 0],
        key=lambda item: (-item[1], item[0]),
    )
    return ", ".join(f"{key}:{value:.4f}" for key, value in supports)


def _focus_label(position_js: float, digit_js: float, tolerance: float = 0.02) -> str:
    if math.isnan(position_js) or math.isnan(digit_js):
        return "n/a"
    delta = position_js - digit_js
    if delta > tolerance:
        return "position-leaning"
    if delta < -tolerance:
        return "digit-leaning"
    return "mixed"


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


def _position_rows(raw_frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_id, group in raw_frame.groupby("condition_id", sort=False):
        first = group.iloc[0]
        digits_order = [int(value) for value in first["digits_order"]]
        digits_order_text = ",".join(str(value) for value in digits_order)
        for position_index, digit in enumerate(digits_order, start=1):
            rows.append(
                {
                    "condition_id": condition_id,
                    "digit": digit,
                    "position_index": position_index,
                    "digits_order_text": digits_order_text,
                    "model_name": first["model_name"],
                    "prompt_id": first["prompt_id"],
                    "ordering_label": first["ordering_label"],
                    "ordering_mode": first["ordering_mode"],
                    "temperature": float(first["temperature"]),
                }
            )
    return rows


def write_position_effect_summary(
    run_dir: str | Path,
    processed_root: str | Path = "results/processed",
) -> dict[str, Any]:
    run_path = Path(run_dir)
    processed_dir = ensure_directory(Path(processed_root) / run_path.name)

    raw_frame = pd.DataFrame(_load_jsonl(run_path / "records.jsonl"))
    candidate_frame = pd.read_csv(processed_dir / "candidate_probabilities.csv")
    position_frame = pd.DataFrame(_position_rows(raw_frame))

    merged = candidate_frame.merge(
        position_frame,
        on=["condition_id", "digit"],
        how="left",
        suffixes=("", "_position"),
    )
    merged = merged[
        [
            "condition_id",
            "model_name",
            "prompt_id_position",
            "ordering_mode",
            "ordering_label_position",
            "temperature_position",
            "digit",
            "position_index",
            "digits_order_text",
            "observed_proportion",
            "mean_candidate_prob",
            "mean_policy_candidate_prob",
        ]
    ].rename(
        columns={
            "prompt_id_position": "prompt_id",
            "ordering_label_position": "ordering_label",
            "temperature_position": "temperature",
        }
    )

    condition_output_path = processed_dir / "condition_position_probabilities.csv"
    merged.to_csv(condition_output_path, index=False)

    group_keys = ["model_name", "prompt_id", "temperature"]
    position_summary_rows: list[dict[str, Any]] = []
    digit_summary_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    positions = list(range(1, 10))
    digits = list(range(1, 10))

    for (model_name, prompt_id, temperature), group in merged.groupby(group_keys, sort=True):
        n_conditions = int(group["condition_id"].nunique())
        has_model_probs = bool(group["mean_candidate_prob"].notna().any())
        has_policy_probs = bool(group["mean_policy_candidate_prob"].notna().any())

        position_view = (
            group.groupby("position_index", sort=True)
            .agg(
                mean_observed_proportion=("observed_proportion", "mean"),
                mean_candidate_prob=("mean_candidate_prob", "mean"),
                mean_policy_candidate_prob=("mean_policy_candidate_prob", "mean"),
            )
            .reset_index()
        )
        digit_view = (
            group.groupby("digit", sort=True)
            .agg(
                mean_observed_proportion=("observed_proportion", "mean"),
                mean_candidate_prob=("mean_candidate_prob", "mean"),
                mean_policy_candidate_prob=("mean_policy_candidate_prob", "mean"),
            )
            .reset_index()
        )

        for _, row in position_view.iterrows():
            occupants = (
                group.loc[group["position_index"] == row["position_index"], "digit"]
                .dropna()
                .astype(int)
                .sort_values()
                .unique()
            )
            position_summary_rows.append(
                {
                    "model_name": model_name,
                    "prompt_id": prompt_id,
                    "temperature": float(temperature),
                    "n_conditions": n_conditions,
                    "position_index": int(row["position_index"]),
                    "digits_seen": ",".join(str(value) for value in occupants),
                    "mean_observed_proportion": float(row["mean_observed_proportion"]),
                    "mean_candidate_prob": float(row["mean_candidate_prob"]),
                    "mean_policy_candidate_prob": float(row["mean_policy_candidate_prob"]),
                }
            )

        for _, row in digit_view.iterrows():
            seen_positions = (
                group.loc[group["digit"] == row["digit"], "position_index"]
                .dropna()
                .astype(int)
                .sort_values()
                .unique()
            )
            digit_summary_rows.append(
                {
                    "model_name": model_name,
                    "prompt_id": prompt_id,
                    "temperature": float(temperature),
                    "n_conditions": n_conditions,
                    "digit": int(row["digit"]),
                    "positions_seen": ",".join(str(value) for value in seen_positions),
                    "mean_observed_proportion": float(row["mean_observed_proportion"]),
                    "mean_candidate_prob": float(row["mean_candidate_prob"]),
                    "mean_policy_candidate_prob": float(row["mean_policy_candidate_prob"]),
                }
            )

        observed_position_dist = _ordered_distribution(
            position_view,
            key_column="position_index",
            value_column="mean_observed_proportion",
            keys=positions,
        )
        observed_digit_dist = _ordered_distribution(
            digit_view,
            key_column="digit",
            value_column="mean_observed_proportion",
            keys=digits,
        )
        model_position_dist = (
            _ordered_distribution(
                position_view,
                key_column="position_index",
                value_column="mean_candidate_prob",
                keys=positions,
            )
            if has_model_probs
            else {}
        )
        model_digit_dist = (
            _ordered_distribution(
                digit_view,
                key_column="digit",
                value_column="mean_candidate_prob",
                keys=digits,
            )
            if has_model_probs
            else {}
        )
        policy_position_dist = (
            _ordered_distribution(
                position_view,
                key_column="position_index",
                value_column="mean_policy_candidate_prob",
                keys=positions,
            )
            if has_policy_probs
            else {}
        )
        policy_digit_dist = (
            _ordered_distribution(
                digit_view,
                key_column="digit",
                value_column="mean_policy_candidate_prob",
                keys=digits,
            )
            if has_policy_probs
            else {}
        )

        top_observed_position, top_observed_position_prob = _top_key(observed_position_dist)
        top_observed_digit, top_observed_digit_prob = _top_key(observed_digit_dist)
        top_policy_position, top_policy_position_prob = _top_key(policy_position_dist)
        top_policy_digit, top_policy_digit_prob = _top_key(policy_digit_dist)

        observed_position_js = jensen_shannon_divergence_to_uniform(observed_position_dist)
        observed_digit_js = jensen_shannon_divergence_to_uniform(observed_digit_dist)
        policy_position_js = (
            jensen_shannon_divergence_to_uniform(policy_position_dist) if has_policy_probs else float("nan")
        )
        policy_digit_js = (
            jensen_shannon_divergence_to_uniform(policy_digit_dist) if has_policy_probs else float("nan")
        )

        summary_rows.append(
            {
                "model_name": model_name,
                "prompt_id": prompt_id,
                "temperature": float(temperature),
                "n_conditions": n_conditions,
                "top_observed_position": top_observed_position,
                "top_observed_position_prob": top_observed_position_prob,
                "top_observed_digit": top_observed_digit,
                "top_observed_digit_prob": top_observed_digit_prob,
                "top_policy_position": top_policy_position,
                "top_policy_position_prob": top_policy_position_prob,
                "top_policy_digit": top_policy_digit,
                "top_policy_digit_prob": top_policy_digit_prob,
                "observed_position_support": _support_string(observed_position_dist),
                "observed_digit_support": _support_string(observed_digit_dist),
                "policy_position_support": _support_string(policy_position_dist),
                "policy_digit_support": _support_string(policy_digit_dist),
                "observed_position_entropy_bits": entropy_bits(observed_position_dist),
                "observed_digit_entropy_bits": entropy_bits(observed_digit_dist),
                "policy_position_entropy_bits": entropy_bits(policy_position_dist),
                "policy_digit_entropy_bits": entropy_bits(policy_digit_dist),
                "observed_position_js_to_uniform_bits": observed_position_js,
                "observed_digit_js_to_uniform_bits": observed_digit_js,
                "policy_position_js_to_uniform_bits": policy_position_js,
                "policy_digit_js_to_uniform_bits": policy_digit_js,
                "observed_position_minus_digit_js_bits": observed_position_js - observed_digit_js,
                "policy_position_minus_digit_js_bits": (
                    policy_position_js - policy_digit_js if has_policy_probs else float("nan")
                ),
                "observed_position_vs_model_position_corr": pearson_correlation(
                    observed_position_dist, model_position_dist
                )
                if has_model_probs
                else float("nan"),
                "observed_position_vs_policy_position_corr": pearson_correlation(
                    observed_position_dist, policy_position_dist
                )
                if has_policy_probs
                else float("nan"),
                "observed_digit_vs_model_digit_corr": pearson_correlation(
                    observed_digit_dist, model_digit_dist
                )
                if has_model_probs
                else float("nan"),
                "observed_digit_vs_policy_digit_corr": pearson_correlation(
                    observed_digit_dist, policy_digit_dist
                )
                if has_policy_probs
                else float("nan"),
                "observed_focus_heuristic": _focus_label(observed_position_js, observed_digit_js),
                "policy_focus_heuristic": _focus_label(policy_position_js, policy_digit_js),
            }
        )

    position_output_path = processed_dir / "position_summary.csv"
    digit_output_path = processed_dir / "digit_identity_summary.csv"
    summary_output_path = processed_dir / "position_effect_summary.csv"
    write_csv(position_output_path, position_summary_rows)
    write_csv(digit_output_path, digit_summary_rows)
    write_csv(summary_output_path, summary_rows)

    lines = [
        "# Position Effect Summary",
        "",
        "Higher Jensen-Shannon divergence means the distribution is more concentrated away from uniform. If position JS exceeds digit JS, the model is more position-leaning under this heuristic.",
        "",
        "| model | prompt | temp | top obs pos | top obs digit | obs pos JS | obs digit JS | policy pos JS | policy digit JS | obs heuristic | policy heuristic |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model_name} | {prompt_id} | {temperature:.1f} | {top_observed_position} | {top_observed_digit} | {observed_position_js} | {observed_digit_js} | {policy_position_js} | {policy_digit_js} | {observed_focus_heuristic} | {policy_focus_heuristic} |".format(
                model_name=row["model_name"],
                prompt_id=row["prompt_id"],
                temperature=row["temperature"],
                top_observed_position=_format_optional_label_value(
                    row["top_observed_position"],
                    row["top_observed_position_prob"],
                ),
                top_observed_digit=_format_optional_label_value(
                    row["top_observed_digit"],
                    row["top_observed_digit_prob"],
                ),
                observed_position_js=_format_optional_number(row["observed_position_js_to_uniform_bits"]),
                observed_digit_js=_format_optional_number(row["observed_digit_js_to_uniform_bits"]),
                policy_position_js=_format_optional_number(row["policy_position_js_to_uniform_bits"]),
                policy_digit_js=_format_optional_number(row["policy_digit_js_to_uniform_bits"]),
                observed_focus_heuristic=row["observed_focus_heuristic"],
                policy_focus_heuristic=row["policy_focus_heuristic"],
            )
        )
        lines.append(
            "| support | support |  | {observed_position_support} | {observed_digit_support} |  |  | {policy_position_support} | {policy_digit_support} |  |  |".format(
                observed_position_support=row["observed_position_support"] or "n/a",
                observed_digit_support=row["observed_digit_support"] or "n/a",
                policy_position_support=row["policy_position_support"] or "n/a",
                policy_digit_support=row["policy_digit_support"] or "n/a",
            )
        )

    markdown_path = processed_dir / "position_effect_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "output_dir": str(processed_dir),
        "condition_position_probabilities_csv": str(condition_output_path),
        "position_summary_csv": str(position_output_path),
        "digit_identity_summary_csv": str(digit_output_path),
        "position_effect_summary_csv": str(summary_output_path),
        "position_effect_summary_markdown": str(markdown_path),
        "group_count": len(summary_rows),
    }
