from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.io_utils import ensure_directory, write_csv


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _model_short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def _format_rate(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


def _format_pair(label: Any, value: Any) -> str:
    if pd.isna(label) or pd.isna(value):
        return "n/a"
    return f"{label}:{float(value):.3f}"


def _main_note(row: pd.Series) -> str:
    overall = str(row["mechanism_overall"])
    max_invalid = float(row.get("max_invalid_rate_overall", 0.0) or 0.0)
    if overall == "position-leaning":
        return "position concentration exceeds digit concentration"
    if overall == "mixed":
        return "digit and position concentration are comparable"
    if overall == "hybrid":
        return "mechanism changes with temperature"
    if max_invalid >= 0.1:
        return "digit-leaning, but strict-format compliance is fragile"
    return "digit identity dominates position"


def _appendix_note(case_id: str) -> str:
    if case_id == "qwen3_1p7b_protocol":
        return "semantically answers the task, but outputs think-wrapper text under the strict parser"
    if case_id == "codex_cli_gpt54":
        return "strong output-only non-uniformity, but no logprobs, tokenizer audit, or explicit decode controls"
    return ""


def build_report_artifacts(processed_root: str | Path = "results/processed") -> dict[str, Any]:
    processed_dir = ensure_directory(Path(processed_root))
    report_dir = ensure_directory(processed_dir / "report_ready")

    comparison_frame = _read_csv(processed_dir / "cross_model_comparison.csv")
    temp_frame = _read_csv(processed_dir / "cross_model_temperature_details.csv")
    protocol_frame = _read_csv(
        processed_dir
        / "phase6_qwen3_1p7b_protocol_compatibility__20260328T161603Z"
        / "protocol_compatibility_summary.csv"
    )
    codex_position_frame = _read_csv(
        processed_dir / "phase7_codex_cli_ordering_study__20260328T163919Z" / "position_effect_summary.csv"
    )
    codex_metrics_frame = _read_csv(
        processed_dir / "phase7_codex_cli_ordering_study__20260328T163919Z" / "condition_metrics.csv"
    )

    comparison_frame = comparison_frame.copy()
    comparison_frame["max_invalid_rate_overall"] = comparison_frame[
        ["temp0_invalid_rate_max", "temp02_invalid_rate_max"]
    ].max(axis=1)
    comparison_frame["mean_invalid_rate_overall"] = comparison_frame[
        ["temp0_invalid_rate_mean", "temp02_invalid_rate_mean"]
    ].mean(axis=1)

    main_rows: list[dict[str, Any]] = []
    for _, row in comparison_frame.iterrows():
        main_rows.append(
            {
                "family": row["family"],
                "scale": row["scale"],
                "model": row["model_name"],
                "model_short": _model_short_name(str(row["model_name"])),
                "mechanism_overall": row["mechanism_overall"],
                "mechanism_temp0": row["temp0_mechanism"],
                "mechanism_temp02": row["temp02_mechanism"],
                "top_digit_temp0": _format_pair(row["temp0_top_digit"], row["temp0_top_digit_prob"]),
                "top_digit_temp02": _format_pair(row["temp02_top_digit"], row["temp02_top_digit_prob"]),
                "top_position_temp0": _format_pair(row["temp0_top_position"], row["temp0_top_position_prob"]),
                "top_position_temp02": _format_pair(row["temp02_top_position"], row["temp02_top_position_prob"]),
                "ordering_diversity_temp0": int(row["temp0_ordering_diversity_count"]),
                "ordering_diversity_temp02": int(row["temp02_ordering_diversity_count"]),
                "max_invalid_rate": float(row["max_invalid_rate_overall"]),
                "interpretation_note": _main_note(row),
            }
        )

    main_csv_path = report_dir / "main_comparison_table.csv"
    write_csv(main_csv_path, main_rows)

    main_lines = [
        "# Main Comparison Table",
        "",
        "This table covers the cleanly comparable local ordering-study runs that support the strict parser and the digit-vs-position mechanism analysis.",
        "",
        "| family | scale | model | overall | temp0 | temp0.2 | top digit temp0 | top digit temp0.2 | max invalid | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in main_rows:
        main_lines.append(
            "| {family} | {scale} | {model_short} | {mechanism_overall} | {mechanism_temp0} | {mechanism_temp02} | {top_digit_temp0} | {top_digit_temp02} | {max_invalid_rate:.3f} | {interpretation_note} |".format(
                **row
            )
        )
    main_md_path = report_dir / "main_comparison_table.md"
    main_md_path.write_text("\n".join(main_lines) + "\n", encoding="utf-8")

    supplementary_rows: list[dict[str, Any]] = []
    for _, row in temp_frame.iterrows():
        supplementary_rows.append(
            {
                "family": row["family"],
                "scale": row["scale"],
                "model": row["model_name"],
                "model_short": _model_short_name(str(row["model_name"])),
                "temperature": float(row["temperature"]),
                "observed_focus_heuristic": row["observed_focus_heuristic"],
                "top_observed_digit": _format_pair(row["top_observed_digit"], row["top_observed_digit_prob"]),
                "top_observed_position": _format_pair(
                    row["top_observed_position"],
                    row["top_observed_position_prob"],
                ),
                "digit_js_bits": float(row["observed_digit_js_to_uniform_bits"]),
                "position_js_bits": float(row["observed_position_js_to_uniform_bits"]),
                "ordering_diversity_count": int(row["ordering_diversity_count"]),
                "ordering_diversity_digits": row["ordering_diversity_digits"],
                "ordering_pattern": row["ordering_top_digit_pattern"],
                "invalid_rate_mean": float(row["invalid_rate_mean"]),
                "invalid_rate_max": float(row["invalid_rate_max"]),
            }
        )

    supplementary_csv_path = report_dir / "supplementary_temperature_table.csv"
    write_csv(supplementary_csv_path, supplementary_rows)

    supp_lines = [
        "# Supplementary Temperature Table",
        "",
        "| model | temp | focus | top digit | top position | digit JS | position JS | diversity | invalid mean | invalid max |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in supplementary_rows:
        supp_lines.append(
            "| {model_short} | {temperature:.1f} | {observed_focus_heuristic} | {top_observed_digit} | {top_observed_position} | {digit_js_bits:.4f} | {position_js_bits:.4f} | {ordering_diversity_count} | {invalid_rate_mean:.3f} | {invalid_rate_max:.3f} |".format(
                **row
            )
        )
        supp_lines.append(
            "| pattern | pattern |  | {ordering_diversity_digits} | {ordering_pattern} |  |  |  |  |  |".format(
                **row
            )
        )
    supplementary_md_path = report_dir / "supplementary_temperature_table.md"
    supplementary_md_path.write_text("\n".join(supp_lines) + "\n", encoding="utf-8")

    protocol_best = protocol_frame.sort_values(
        by=["strict_valid_rate", "recovered_valid_rate", "max_output_tokens"],
        ascending=[False, False, False],
    ).iloc[0]
    protocol_recovered = protocol_frame.sort_values(
        by=["recovered_valid_rate", "max_output_tokens"],
        ascending=[False, False],
    ).iloc[0]
    codex_position = codex_position_frame.iloc[0]

    appendix_rows = [
        {
            "case_id": "qwen3_1p7b_protocol",
            "route": "transformers",
            "model": "Qwen/Qwen3-1.7B",
            "status": "protocol-mismatch",
            "included_in_main_table": "no",
            "strict_valid_rate_best": float(protocol_best["strict_valid_rate"]),
            "recovered_valid_rate_best": float(protocol_recovered["recovered_valid_rate"]),
            "support_or_pattern": str(protocol_recovered["recovered_digit_support"] or ""),
            "note": _appendix_note("qwen3_1p7b_protocol"),
        },
        {
            "case_id": "codex_cli_gpt54",
            "route": "codex_cli",
            "model": "gpt-5.4",
            "status": "auxiliary-output-only",
            "included_in_main_table": "no",
            "strict_valid_rate_best": 1.0 - float(codex_metrics_frame["invalid_rate"].max()),
            "recovered_valid_rate_best": float("nan"),
            "support_or_pattern": str(codex_position["observed_digit_support"]),
            "note": _appendix_note("codex_cli_gpt54"),
        },
    ]

    appendix_csv_path = report_dir / "appendix_cases_table.csv"
    write_csv(appendix_csv_path, appendix_rows)

    appendix_lines = [
        "# Appendix Cases Table",
        "",
        "These cases are informative, but they are not merged into the main local logprob-aware comparison table.",
        "",
        "| case | route | model | status | main table | strict valid | recovered valid | support | note |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in appendix_rows:
        appendix_lines.append(
            "| {case_id} | {route} | {model} | {status} | {included_in_main_table} | {strict_valid_rate_best} | {recovered_valid_rate_best} | {support_or_pattern} | {note} |".format(
                case_id=row["case_id"],
                route=row["route"],
                model=row["model"],
                status=row["status"],
                included_in_main_table=row["included_in_main_table"],
                strict_valid_rate_best=_format_rate(float(row["strict_valid_rate_best"])),
                recovered_valid_rate_best=_format_rate(float(row["recovered_valid_rate_best"])),
                support_or_pattern=row["support_or_pattern"] or "n/a",
                note=row["note"],
            )
        )
    appendix_md_path = report_dir / "appendix_cases_table.md"
    appendix_md_path.write_text("\n".join(appendix_lines) + "\n", encoding="utf-8")

    narrative_lines = [
        "# Report-Ready Results Skeleton",
        "",
        "## Methods Snapshot",
        "",
        "We evaluated a strict single-digit generation task in which each model was asked to choose one digit from 1 to 9 and output exactly one Arabic numeral. The main comparative analysis used one list-based prompt, five fixed digit orderings, and two temperatures (`0.0`, `0.2`). Runs were parsed with a strict validator: only a bare single digit in `1` to `9` counted as valid.",
        "",
        "For the local Hugging Face `transformers` backend, the main comparison additionally included tokenizer audits and audited-surface candidate probabilities, which supported digit-vs-position mechanism analysis. Mechanism labels were assigned from pooled ordering results using a Jensen-Shannon divergence heuristic: a model was labeled `digit-leaning` when digit concentration exceeded position concentration, `position-leaning` when the reverse held, `mixed` when they were close, and `hybrid` when the label changed across temperatures.",
        "",
        "## Main Comparative Result",
        "",
        "Across the cleanly comparable local ordering-study runs, non-uniformity was universal, but the mechanism of bias varied by model family rather than collapsing into a single universal pattern.",
        "",
        "- `Gemma 3 1B` was the clearest `position-leaning` model.",
        "- `Llama 3.2 1B`, `Llama 3.2 3B`, `Qwen2.5 0.5B`, `Qwen2.5 1.5B`, and `SmolLM2-135M` were `digit-leaning`.",
        "- `Qwen3 4B` was `mixed`.",
        "- `SmolLM2-1.7B` was `hybrid`, shifting from `mixed` at `temp=0.0` to `digit-leaning` at `temp=0.2`.",
        "",
        "This pattern supports a stronger conclusion than simple non-uniformity: the single-digit random-choice task reveals family-dependent bias mechanisms, with some models concentrating on specific digit identities, some concentrating on list positions, and some showing intermediate or temperature-sensitive behavior.",
        "",
        "## Format Robustness Result",
        "",
        "Strict protocol compliance was high for most modern 1B to 4B models in the main comparison. The main exception within the pooled local runs was `SmolLM2-135M`, which remained strongly digit-leaning but also showed substantial formatting fragility under some orderings, with maximum invalid-rate conditions above `0.7` at `temp=0.2`.",
        "",
        "## Appendix-Scope Cases",
        "",
        "`Qwen3-1.7B` was excluded from the main comparison because it systematically emitted think-wrapper text under the strict protocol. Under `/no_think` plus a longer output budget, the wrapper could be stripped and the recovered answer was consistently digit `5`, showing semantic task competence but transport-layer incompatibility with the main protocol.",
        "",
        "`gpt-5.4` via `codex CLI` was also strongly non-uniform in an auxiliary five-ordering output-only study. However, this route did not expose tokenizer audits, logprobs, or explicit decode controls, so it is better treated as supporting frontier evidence than as a directly pooled main-table comparison.",
        "",
        "## Suggested Reporting Structure",
        "",
        "Main text:",
        "- use `main_comparison_table.md` as the compact cross-family table",
        "- use `cross_model_js_temp_0p0.png` and `cross_model_js_temp_0p2.png` as visual support",
        "",
        "Supplement:",
        "- use `supplementary_temperature_table.md` for per-temperature details",
        "- use `appendix_cases_table.md` for protocol mismatch and auxiliary output-only cases",
        "",
        "Open next step:",
        "- add one OpenAI-compatible API run with top-logprob support so the frontier route can be compared at both the behavior layer and the probability layer",
    ]
    narrative_path = report_dir / "report_results_skeleton.md"
    narrative_path.write_text("\n".join(narrative_lines) + "\n", encoding="utf-8")

    return {
        "report_dir": str(report_dir),
        "main_comparison_table_csv": str(main_csv_path),
        "main_comparison_table_markdown": str(main_md_path),
        "supplementary_temperature_table_csv": str(supplementary_csv_path),
        "supplementary_temperature_table_markdown": str(supplementary_md_path),
        "appendix_cases_table_csv": str(appendix_csv_path),
        "appendix_cases_table_markdown": str(appendix_md_path),
        "report_results_skeleton_markdown": str(narrative_path),
    }
