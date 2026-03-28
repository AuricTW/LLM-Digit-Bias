from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.clients.factory import build_client
from src.io_utils import ensure_directory, write_csv, write_json, write_jsonl
from src.parsing.digit_parser import parse_single_digit
from src.types import GenerationRequest

EMPTY_THINK_WRAPPER_PATTERN = re.compile(
    r"^\s*<think>\s*</think>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strip_empty_think_wrapper(text: str) -> tuple[str, bool]:
    stripped, count = EMPTY_THINK_WRAPPER_PATTERN.subn("", text, count=1)
    return stripped, count > 0


def _prepare_run_dir(experiment_name: str, raw_root: str | Path = "results/raw") -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ensure_directory(Path(raw_root) / f"{experiment_name}__{timestamp}")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _top_outputs(records: list[dict[str, Any]], limit: int = 5) -> str:
    counter = Counter(record["normalized_text"] for record in records)
    parts = []
    for text, count in counter.most_common(limit):
        display = text.encode("unicode_escape").decode()
        parts.append(f"{display}:{count}")
    return ", ".join(parts)


def run_protocol_compatibility_study(
    config_path: str | Path,
    raw_output_root: str | Path = "results/raw",
    processed_root: str | Path = "results/processed",
) -> dict[str, Any]:
    config = _load_json(config_path)
    run_dir = _prepare_run_dir(config["experiment_name"], raw_root=raw_output_root)
    processed_dir = ensure_directory(Path(processed_root) / run_dir.name)

    client = build_client(config["client"], seed=int(config.get("seed", 0)))
    digits = [int(value) for value in config["digits"]]
    tokenizer_audit = client.audit_tokenizer(digits)
    if tokenizer_audit:
        write_json(run_dir / "tokenizer_audit.json", tokenizer_audit)
        write_csv(run_dir / "tokenizer_audit.csv", tokenizer_audit.get("surface_rows", []))

    records: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    trial_index = 0
    temperatures = [float(value) for value in config.get("temperatures", [0.0])]
    max_output_tokens_list = [int(value) for value in config["max_output_tokens_list"]]
    prompt_variants = config["prompt_variants"]
    repetitions = int(config["repetitions_per_condition"])

    try:
        total_trials = len(prompt_variants) * len(max_output_tokens_list) * len(temperatures) * repetitions
        for prompt_variant in prompt_variants:
            for max_output_tokens in max_output_tokens_list:
                for temperature in temperatures:
                    condition_id = (
                        f"{prompt_variant['id']}__max_{max_output_tokens}__temp_{temperature:.2f}"
                        .replace(".", "p")
                        .replace("-", "m")
                    )
                    condition_records: list[dict[str, Any]] = []
                    for repetition_index in range(repetitions):
                        trial_index += 1
                        started = time.perf_counter()
                        response = client.generate(
                            request=GenerationRequest(
                                prompt=prompt_variant["text"],
                                digits=digits,
                                temperature=temperature,
                                top_p=config.get("top_p"),
                                max_output_tokens=max_output_tokens,
                                logprobs_enabled=False,
                                top_logprobs=None,
                            )
                        )
                        latency_ms = round((time.perf_counter() - started) * 1000, 3)
                        strict_parse = parse_single_digit(response.text)
                        stripped_text, had_empty_wrapper = _strip_empty_think_wrapper(response.text)
                        recovered_parse = parse_single_digit(stripped_text)

                        record = {
                            "run_id": run_dir.name,
                            "trial_index": trial_index,
                            "total_trials": total_trials,
                            "timestamp_utc": datetime.now(UTC).isoformat(),
                            "condition_id": condition_id,
                            "prompt_variant_id": prompt_variant["id"],
                            "prompt_description": prompt_variant.get("description", ""),
                            "prompt_text": prompt_variant["text"],
                            "temperature": temperature,
                            "max_output_tokens": max_output_tokens,
                            "repetition_index": repetition_index,
                            "raw_text": response.text,
                            "normalized_text": strict_parse.normalized_text,
                            "strict_is_valid": strict_parse.is_valid,
                            "strict_invalid_reason": strict_parse.invalid_reason,
                            "strict_parsed_digit": strict_parse.parsed_digit,
                            "had_empty_think_wrapper": had_empty_wrapper,
                            "wrapper_stripped_text": stripped_text,
                            "wrapper_stripped_display": stripped_text.encode("unicode_escape").decode(),
                            "recovered_is_valid": recovered_parse.is_valid,
                            "recovered_invalid_reason": recovered_parse.invalid_reason,
                            "recovered_parsed_digit": recovered_parse.parsed_digit,
                            "finish_reason": response.finish_reason,
                            "latency_ms": latency_ms,
                            "provider": config["client"]["provider"],
                            "model_name": config["client"]["model_name"],
                        }
                        records.append(record)
                        condition_records.append(record)
                        if trial_index % 10 == 0 or trial_index == total_trials:
                            print(f"[protocol] completed {trial_index}/{total_trials} trials")

                    strict_valid_rate = sum(record["strict_is_valid"] for record in condition_records) / repetitions
                    wrapper_rate = sum(record["had_empty_think_wrapper"] for record in condition_records) / repetitions
                    recovered_valid_rate = (
                        sum(record["recovered_is_valid"] for record in condition_records) / repetitions
                    )
                    recovered_only_rate = (
                        sum(
                            (not record["strict_is_valid"]) and record["recovered_is_valid"]
                            for record in condition_records
                        )
                        / repetitions
                    )
                    recovered_digits = [
                        int(record["recovered_parsed_digit"])
                        for record in condition_records
                        if record["recovered_is_valid"] and record["recovered_parsed_digit"] is not None
                    ]
                    recovered_digit_support = ""
                    if recovered_digits:
                        counts = Counter(recovered_digits)
                        recovered_digit_support = ", ".join(
                            f"{digit}:{count / len(recovered_digits):.4f}"
                            for digit, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                        )

                    condition_rows.append(
                        {
                            "run_id": run_dir.name,
                            "condition_id": condition_id,
                            "prompt_variant_id": prompt_variant["id"],
                            "temperature": temperature,
                            "max_output_tokens": max_output_tokens,
                            "n_total": repetitions,
                            "strict_valid_rate": strict_valid_rate,
                            "empty_think_wrapper_rate": wrapper_rate,
                            "recovered_valid_rate": recovered_valid_rate,
                            "recovered_only_rate": recovered_only_rate,
                            "recovered_digit_support": recovered_digit_support,
                            "top_raw_outputs": _top_outputs(condition_records),
                            "mean_latency_ms": float(
                                pd.to_numeric(
                                    pd.Series([record["latency_ms"] for record in condition_records]),
                                    errors="coerce",
                                ).mean()
                            ),
                        }
                    )
    finally:
        client.close()

    write_json(run_dir / "manifest.json", config)
    write_jsonl(run_dir / "records.jsonl", records)
    write_csv(run_dir / "records.csv", records)
    write_csv(processed_dir / "protocol_compatibility_summary.csv", condition_rows)

    lines = [
        "# Protocol Compatibility Summary",
        "",
        "Strict validity uses the unchanged single-digit parser. Recovered validity strips one empty `<think></think>` wrapper before re-parsing.",
        "",
        "| condition_id | strict_valid_rate | empty_wrapper_rate | recovered_valid_rate | recovered_only_rate | recovered_digit_support | top_raw_outputs |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in condition_rows:
        lines.append(
            "| {condition_id} | {strict_valid_rate:.4f} | {empty_think_wrapper_rate:.4f} | {recovered_valid_rate:.4f} | {recovered_only_rate:.4f} | {recovered_digit_support} | {top_raw_outputs} |".format(
                **row
            )
        )
    markdown_path = processed_dir / "protocol_compatibility_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "processed_dir": str(processed_dir),
        "summary_csv": str(processed_dir / "protocol_compatibility_summary.csv"),
        "summary_markdown": str(markdown_path),
        "record_count": len(records),
        "condition_count": len(condition_rows),
    }
