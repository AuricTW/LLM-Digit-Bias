from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.metrics import (
    chi_square_against_uniform,
    entropy_bits,
    frequency_distribution,
    jensen_shannon_divergence_to_uniform,
    kl_divergence_to_uniform,
    pearson_correlation,
)
from src.io_utils import ensure_directory, write_csv, write_json
from src.visualization.plots import plot_digit_distribution


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean_dict(group: pd.DataFrame, column: str, keys: list[str]) -> dict[str, float]:
    totals = {key: 0.0 for key in keys}
    count = 0
    for item in group[column].tolist():
        if not isinstance(item, dict) or not item:
            continue
        count += 1
        for key in keys:
            totals[key] += float(item.get(key, 0.0))
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def write_markdown_summary(path: Path, metric_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Experiment Summary",
        "",
        "| condition_id | n_total | n_valid | invalid_rate | chi_square_pvalue | entropy_bits | model_mass | policy_mass | obs_vs_model_corr | obs_vs_policy_corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            "| {condition_id} | {n_total} | {n_valid} | {invalid_rate:.4f} | {chi_square_pvalue:.6f} | {entropy_bits:.4f} | {mean_candidate_mass_total:.6f} | {mean_policy_candidate_mass_total:.6f} | {observed_vs_candidate_corr:.4f} | {observed_vs_policy_candidate_corr:.4f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(run_dir: str | Path, processed_root: str | Path, digits: list[int]) -> dict[str, Any]:
    run_path = Path(run_dir)
    records = _load_jsonl(run_path / "records.jsonl")
    output_dir = ensure_directory(Path(processed_root) / run_path.name)
    figures_dir = ensure_directory(output_dir / "figures")

    frame = pd.DataFrame(records)
    digit_keys = [str(digit) for digit in digits]
    frequency_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []

    for condition_id, group in frame.groupby("condition_id", sort=True):
        valid_values = [
            int(value)
            for value in group.loc[group["is_valid"] == True, "parsed_digit"].tolist()  # noqa: E712
            if value == value
        ]
        counts, proportions = frequency_distribution(valid_values, digits=digits)
        chi2, chi2_pvalue = chi_square_against_uniform(counts)
        invalid_rate = float((group["is_valid"] == False).mean())  # noqa: E712
        entropy = entropy_bits(proportions)
        kl_div = kl_divergence_to_uniform(proportions)
        js_div = jensen_shannon_divergence_to_uniform(proportions)

        mean_digit_probs = _mean_dict(group, "digit_probs", digit_keys) if "digit_probs" in group else {}
        mean_digit_probs_raw = _mean_dict(group, "digit_probs_raw", digit_keys) if "digit_probs_raw" in group else {}
        mean_policy_digit_probs = (
            _mean_dict(group, "policy_digit_probs", digit_keys) if "policy_digit_probs" in group else {}
        )
        mean_policy_digit_probs_raw = (
            _mean_dict(group, "policy_digit_probs_raw", digit_keys)
            if "policy_digit_probs_raw" in group
            else {}
        )
        mean_candidate_mass_total = (
            float(pd.to_numeric(group["candidate_mass_total"], errors="coerce").mean())
            if "candidate_mass_total" in group
            else float("nan")
        )
        mean_policy_candidate_mass_total = (
            float(pd.to_numeric(group["policy_candidate_mass_total"], errors="coerce").mean())
            if "policy_candidate_mass_total" in group
            else float("nan")
        )
        observed_vs_candidate_corr = (
            pearson_correlation(proportions, mean_digit_probs) if mean_digit_probs else float("nan")
        )
        observed_vs_policy_candidate_corr = (
            pearson_correlation(proportions, mean_policy_digit_probs)
            if mean_policy_digit_probs
            else float("nan")
        )

        surface_accumulator: dict[tuple[str, str], dict[str, Any]] = {}
        surface_count = 0
        for items in group["logprob_raw_candidates"].tolist():
            if not isinstance(items, list):
                continue
            surface_count += 1
            for item in items:
                key = (str(item.get("digit")), str(item.get("surface_display", item.get("surface", ""))))
                row = surface_accumulator.setdefault(
                    key,
                    {
                        "digit": str(item.get("digit")),
                        "surface": str(item.get("surface_display", item.get("surface", ""))),
                        "prefix_type": item.get("prefix_type"),
                        "token_count": item.get("token_count"),
                        "is_single_token": item.get("is_single_token"),
                        "probability_sum": 0.0,
                        "normalized_probability_sum": 0.0,
                        "policy_probability_sum": 0.0,
                        "policy_normalized_probability_sum": 0.0,
                    },
                )
                row["probability_sum"] += float(item.get("probability", 0.0))
                row["normalized_probability_sum"] += float(item.get("normalized_probability") or 0.0)
                row["policy_probability_sum"] += float(item.get("policy_probability", 0.0))
                row["policy_normalized_probability_sum"] += float(
                    item.get("policy_normalized_probability") or 0.0
                )

        for digit in digits:
            digit_key = str(digit)
            frequency_rows.append(
                {
                    "condition_id": condition_id,
                    "prompt_id": group["prompt_id"].iloc[0],
                    "ordering_label": group["ordering_label"].iloc[0],
                    "temperature": group["temperature"].iloc[0],
                    "digit": digit,
                    "count": counts[digit_key],
                    "observed_proportion": proportions[digit_key],
                }
            )
            candidate_rows.append(
                {
                    "condition_id": condition_id,
                    "prompt_id": group["prompt_id"].iloc[0],
                    "ordering_label": group["ordering_label"].iloc[0],
                    "temperature": group["temperature"].iloc[0],
                    "digit": digit,
                    "observed_proportion": proportions[digit_key],
                    "mean_candidate_prob": mean_digit_probs.get(digit_key),
                    "mean_candidate_prob_raw": mean_digit_probs_raw.get(digit_key),
                    "mean_policy_candidate_prob": mean_policy_digit_probs.get(digit_key),
                    "mean_policy_candidate_prob_raw": mean_policy_digit_probs_raw.get(digit_key),
                    "difference_observed_minus_candidate": (
                        proportions[digit_key] - mean_digit_probs.get(digit_key, 0.0)
                        if mean_digit_probs
                        else None
                    ),
                    "difference_observed_minus_policy_candidate": (
                        proportions[digit_key] - mean_policy_digit_probs.get(digit_key, 0.0)
                        if mean_policy_digit_probs
                        else None
                    ),
                }
            )

        for row in surface_accumulator.values():
            surface_rows.append(
                {
                    "condition_id": condition_id,
                    "prompt_id": group["prompt_id"].iloc[0],
                    "ordering_label": group["ordering_label"].iloc[0],
                    "temperature": group["temperature"].iloc[0],
                    "digit": row["digit"],
                    "surface": row["surface"],
                    "prefix_type": row["prefix_type"],
                    "token_count": row["token_count"],
                    "is_single_token": row["is_single_token"],
                    "mean_surface_prob": (row["probability_sum"] / surface_count if surface_count else None),
                    "mean_surface_prob_normalized": (
                        row["normalized_probability_sum"] / surface_count if surface_count else None
                    ),
                    "mean_policy_surface_prob": (
                        row["policy_probability_sum"] / surface_count if surface_count else None
                    ),
                    "mean_policy_surface_prob_normalized": (
                        row["policy_normalized_probability_sum"] / surface_count if surface_count else None
                    ),
                }
            )

        metric_rows.append(
            {
                "condition_id": condition_id,
                "prompt_id": group["prompt_id"].iloc[0],
                "ordering_mode": group["ordering_mode"].iloc[0],
                "ordering_label": group["ordering_label"].iloc[0],
                "temperature": group["temperature"].iloc[0],
                "n_total": int(len(group)),
                "n_valid": int(sum(group["is_valid"] == True)),  # noqa: E712
                "n_invalid": int(sum(group["is_valid"] == False)),  # noqa: E712
                "invalid_rate": invalid_rate,
                "chi_square_stat": chi2,
                "chi_square_pvalue": chi2_pvalue,
                "entropy_bits": entropy,
                "kl_to_uniform_bits": kl_div,
                "js_to_uniform_bits": js_div,
                "mean_candidate_mass_total": mean_candidate_mass_total,
                "mean_policy_candidate_mass_total": mean_policy_candidate_mass_total,
                "observed_vs_candidate_corr": observed_vs_candidate_corr,
                "observed_vs_policy_candidate_corr": observed_vs_policy_candidate_corr,
            }
        )

        plot_digit_distribution(
            counts=counts,
            proportions=proportions,
            title=f"{condition_id} (valid n={len(valid_values)})",
            output_path=figures_dir / f"{condition_id}.png",
        )

    write_csv(output_dir / "digit_frequencies.csv", frequency_rows)
    write_csv(output_dir / "condition_metrics.csv", metric_rows)
    write_csv(output_dir / "candidate_probabilities.csv", candidate_rows)
    write_csv(output_dir / "surface_probabilities.csv", surface_rows)
    write_json(output_dir / "analysis_manifest.json", {"run_dir": str(run_path), "digits": digits})
    write_markdown_summary(output_dir / "summary.md", metric_rows)

    return {
        "output_dir": str(output_dir),
        "digit_frequencies_csv": str(output_dir / "digit_frequencies.csv"),
        "condition_metrics_csv": str(output_dir / "condition_metrics.csv"),
        "candidate_probabilities_csv": str(output_dir / "candidate_probabilities.csv"),
        "surface_probabilities_csv": str(output_dir / "surface_probabilities.csv"),
        "summary_markdown": str(output_dir / "summary.md"),
        "figure_count": len(metric_rows),
    }
