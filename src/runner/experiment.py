from __future__ import annotations

import json
import random
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from src.analysis.report import run_analysis
from src.clients.factory import build_client
from src.io_utils import ensure_directory, write_csv, write_json, write_jsonl
from src.parsing.digit_parser import parse_single_digit
from src.prompts import is_order_sensitive, load_prompt_templates, render_prompt
from src.runner.config import ExperimentConfig
from src.types import ExperimentCondition, GenerationRequest, Ordering, RunArtifacts


def build_orderings(config: ExperimentConfig) -> list[Ordering]:
    rng = random.Random(config.seed)
    base_digits = list(config.digits)
    orderings: list[Ordering] = []
    for mode in config.ordering_modes:
        if mode == "ascending":
            orderings.append(Ordering(mode="ascending", label="ascending", digits=sorted(base_digits)))
        elif mode == "descending":
            orderings.append(
                Ordering(mode="descending", label="descending", digits=sorted(base_digits, reverse=True))
            )
        elif mode == "random":
            seen: set[tuple[int, ...]] = set()
            for index in range(config.random_orderings_per_prompt):
                while True:
                    digits = base_digits[:]
                    rng.shuffle(digits)
                    key = tuple(digits)
                    if key not in seen:
                        seen.add(key)
                        break
                orderings.append(
                    Ordering(
                        mode="random",
                        label=f"random_{index + 1:03d}",
                        digits=digits,
                        index=index + 1,
                        is_randomized=True,
                    )
                )
        else:
            raise ValueError(f"Unsupported ordering mode: {mode!r}")
    return orderings


def build_conditions(config: ExperimentConfig) -> list[ExperimentCondition]:
    templates = {item.id: item for item in load_prompt_templates(config.prompt_template_file)}
    orderings = build_orderings(config)
    invariant_ordering = Ordering(mode="invariant", label="invariant", digits=sorted(config.digits))
    conditions: list[ExperimentCondition] = []

    for prompt_id in config.prompt_ids:
        template = templates[prompt_id]
        sensitive = is_order_sensitive(template)
        prompt_orderings = orderings if sensitive else [invariant_ordering]
        for ordering in prompt_orderings:
            prompt_text = render_prompt(template, ordering)
            for temperature in config.temperatures:
                ordering_label = ordering.label if sensitive else "invariant"
                condition_id = (
                    f"{prompt_id}__{ordering_label}__temp_{temperature:.2f}"
                    .replace(".", "p")
                    .replace("-", "m")
                )
                conditions.append(
                    ExperimentCondition(
                        condition_id=condition_id,
                        prompt_id=prompt_id,
                        prompt_text=prompt_text,
                        prompt_description=template.description,
                        ordering_mode=ordering.mode if sensitive else "invariant",
                        ordering_label=ordering_label,
                        ordering_index=ordering.index if sensitive else 0,
                        order_sensitive=sensitive,
                        digits_order=ordering.digits[:],
                        temperature=temperature,
                        top_p=config.top_p,
                    )
                )
    return conditions


def prepare_run_artifacts(experiment_name: str, raw_root: str | Path) -> RunArtifacts:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{experiment_name}__{timestamp}"
    run_dir = ensure_directory(Path(raw_root) / run_id)
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        raw_jsonl_path=run_dir / "records.jsonl",
        raw_csv_path=run_dir / "records.csv",
        manifest_path=run_dir / "manifest.json",
        resolved_config_path=run_dir / "resolved_config.json",
        tokenizer_audit_json_path=run_dir / "tokenizer_audit.json",
        tokenizer_audit_csv_path=run_dir / "tokenizer_audit.csv",
    )


def collect_environment_metadata(client_provider: str) -> dict[str, Any]:
    package_names = ["numpy", "pandas", "matplotlib", "scipy"]
    provider_packages = {
        "openai_compatible": ["openai"],
        "transformers": ["transformers", "torch"],
        "vllm_local": ["vllm"],
    }
    package_names.extend(provider_packages.get(client_provider, []))

    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = None

    return {
        "python_version": sys.version,
        "package_versions": versions,
    }


def flatten_record_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    flat = record.copy()
    flat["digits_order"] = ",".join(str(item) for item in record["digits_order"])
    flat["digit_logprob_keys"] = ",".join(sorted(record.get("digit_probs", {}).keys()))
    flat["digit_prob_sum"] = sum(record.get("digit_probs", {}).values()) if record.get("digit_probs") else None
    flat["candidate_mass_total"] = record.get("candidate_mass_total")
    flat["policy_name"] = record.get("policy_name")
    flat["policy_candidate_mass_total"] = record.get("policy_candidate_mass_total")
    flat["digit_probs_json"] = json.dumps(record.get("digit_probs", {}), ensure_ascii=False, sort_keys=True)
    flat["digit_probs_raw_json"] = json.dumps(record.get("digit_probs_raw", {}), ensure_ascii=False, sort_keys=True)
    flat["policy_digit_probs_json"] = json.dumps(
        record.get("policy_digit_probs", {}),
        ensure_ascii=False,
        sort_keys=True,
    )
    flat["policy_digit_probs_raw_json"] = json.dumps(
        record.get("policy_digit_probs_raw", {}),
        ensure_ascii=False,
        sort_keys=True,
    )
    flat["surface_sequence_probs_json"] = json.dumps(
        record.get("surface_sequence_probs", {}),
        ensure_ascii=False,
        sort_keys=True,
    )
    flat["surface_sequence_probs_normalized_json"] = json.dumps(
        record.get("surface_sequence_probs_normalized", {}),
        ensure_ascii=False,
        sort_keys=True,
    )
    flat["policy_surface_sequence_probs_json"] = json.dumps(
        record.get("policy_surface_sequence_probs", {}),
        ensure_ascii=False,
        sort_keys=True,
    )
    flat["policy_surface_sequence_probs_normalized_json"] = json.dumps(
        record.get("policy_surface_sequence_probs_normalized", {}),
        ensure_ascii=False,
        sort_keys=True,
    )
    flat["logprob_notes"] = " | ".join(record.get("logprob_notes", []))
    flat.pop("digit_probs", None)
    flat.pop("digit_probs_raw", None)
    flat.pop("digit_logprobs", None)
    flat.pop("policy_digit_probs", None)
    flat.pop("policy_digit_probs_raw", None)
    flat.pop("policy_digit_logprobs", None)
    flat.pop("surface_sequence_probs", None)
    flat.pop("surface_sequence_probs_normalized", None)
    flat.pop("surface_sequence_logprobs", None)
    flat.pop("policy_surface_sequence_probs", None)
    flat.pop("policy_surface_sequence_probs_normalized", None)
    flat.pop("policy_surface_sequence_logprobs", None)
    flat.pop("logprob_raw_candidates", None)
    return flat


def run_experiment(
    config: ExperimentConfig,
    config_path: str | Path,
    raw_output_root: str | Path = "results/raw",
) -> dict[str, Any]:
    artifacts = prepare_run_artifacts(config.experiment_name, raw_root=raw_output_root)
    conditions = build_conditions(config)
    client = build_client(config.client, seed=config.seed)
    tokenizer_audit = client.audit_tokenizer(config.digits)
    if tokenizer_audit:
        write_json(artifacts.tokenizer_audit_json_path, tokenizer_audit)
        write_csv(artifacts.tokenizer_audit_csv_path, tokenizer_audit.get("surface_rows", []))

    metadata = {
        "run_id": artifacts.run_id,
        "experiment_name": config.experiment_name,
        "description": config.description,
        "config_path": str(config_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "condition_count": len(conditions),
        "repetitions_per_condition": config.repetitions_per_condition,
        "client_provider": config.client["provider"],
        "model_name": config.client["model_name"],
        "client_params": config.client.get("params", {}),
        "logprob_enabled": bool(config.logprob.get("enabled", False)),
        "tokenizer_audit_available": tokenizer_audit is not None,
        "environment": collect_environment_metadata(config.client["provider"]),
    }
    write_json(artifacts.resolved_config_path, config.to_dict())

    raw_records: list[dict[str, Any]] = []
    total_trials = len(conditions) * config.repetitions_per_condition
    trial_counter = 0

    try:
        for condition in conditions:
            for repetition_index in range(config.repetitions_per_condition):
                trial_counter += 1
                request = GenerationRequest(
                    prompt=condition.prompt_text,
                    digits=config.digits,
                    temperature=condition.temperature,
                    top_p=condition.top_p,
                    max_output_tokens=config.max_output_tokens,
                    logprobs_enabled=bool(config.logprob.get("enabled", False)),
                    top_logprobs=int(config.logprob.get("top_k", 20))
                    if config.logprob.get("enabled", False)
                    else None,
                )
                started = time.perf_counter()
                error_message = None
                response = None
                try:
                    response = client.generate(request)
                except Exception as exc:  # pragma: no cover
                    error_message = str(exc)
                    if not config.continue_on_error:
                        raise
                latency_ms = round((time.perf_counter() - started) * 1000, 3)

                text = response.text if response else ""
                parse_result = parse_single_digit(text)
                summary = response.logprob_summary if response else None

                raw_record = {
                    "run_id": artifacts.run_id,
                    "trial_index": trial_counter,
                    "total_trials": total_trials,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "condition_id": condition.condition_id,
                    "prompt_id": condition.prompt_id,
                    "prompt_description": condition.prompt_description,
                    "prompt_text": condition.prompt_text,
                    "ordering_mode": condition.ordering_mode,
                    "ordering_label": condition.ordering_label,
                    "ordering_index": condition.ordering_index,
                    "order_sensitive": condition.order_sensitive,
                    "digits_order": condition.digits_order,
                    "temperature": condition.temperature,
                    "top_p": condition.top_p,
                    "repetition_index": repetition_index,
                    "raw_text": text,
                    "normalized_text": parse_result.normalized_text,
                    "parsed_digit": parse_result.parsed_digit,
                    "is_valid": parse_result.is_valid,
                    "invalid_reason": parse_result.invalid_reason,
                    "finish_reason": response.finish_reason if response else None,
                    "latency_ms": latency_ms,
                    "provider": config.client["provider"],
                    "model_name": config.client["model_name"],
                    "error": error_message,
                    "logprob_status": summary.status if summary else "not_requested",
                    "logprob_visibility": summary.visibility if summary else "not_requested",
                    "candidate_mass_total": summary.candidate_mass_total if summary else None,
                    "surface_sequence_logprobs": summary.surface_sequence_logprobs if summary else {},
                    "surface_sequence_probs": summary.surface_sequence_probs if summary else {},
                    "surface_sequence_probs_normalized": (
                        summary.surface_sequence_probs_normalized if summary else {}
                    ),
                    "digit_probs_raw": summary.digit_probs_raw if summary else {},
                    "digit_probs": summary.digit_probs if summary else {},
                    "digit_logprobs": summary.digit_logprobs if summary else {},
                    "policy_name": summary.policy_name if summary else None,
                    "policy_candidate_mass_total": (
                        summary.policy_candidate_mass_total if summary else None
                    ),
                    "policy_surface_sequence_logprobs": (
                        summary.policy_surface_sequence_logprobs if summary else {}
                    ),
                    "policy_surface_sequence_probs": (
                        summary.policy_surface_sequence_probs if summary else {}
                    ),
                    "policy_surface_sequence_probs_normalized": (
                        summary.policy_surface_sequence_probs_normalized if summary else {}
                    ),
                    "policy_digit_probs_raw": summary.policy_digit_probs_raw if summary else {},
                    "policy_digit_probs": summary.policy_digit_probs if summary else {},
                    "policy_digit_logprobs": summary.policy_digit_logprobs if summary else {},
                    "logprob_raw_candidates": summary.raw_candidates if summary else [],
                    "logprob_notes": summary.notes if summary else [],
                }
                raw_records.append(raw_record)
                if trial_counter % 25 == 0 or trial_counter == total_trials:
                    print(f"[runner] completed {trial_counter}/{total_trials} trials")
    finally:
        client.close()

    write_jsonl(artifacts.raw_jsonl_path, raw_records)
    write_csv(artifacts.raw_csv_path, [flatten_record_for_csv(item) for item in raw_records])
    write_json(artifacts.manifest_path, metadata)

    result = {
        "artifacts": {
            "run_id": artifacts.run_id,
            "run_dir": str(artifacts.run_dir),
            "raw_jsonl_path": str(artifacts.raw_jsonl_path),
            "raw_csv_path": str(artifacts.raw_csv_path),
            "manifest_path": str(artifacts.manifest_path),
            "resolved_config_path": str(artifacts.resolved_config_path),
            "tokenizer_audit_json_path": (
                str(artifacts.tokenizer_audit_json_path) if tokenizer_audit is not None else None
            ),
            "tokenizer_audit_csv_path": (
                str(artifacts.tokenizer_audit_csv_path) if tokenizer_audit is not None else None
            ),
        },
        "record_count": len(raw_records),
        "condition_count": len(conditions),
    }

    if config.analysis.get("enabled", False):
        analysis_result = run_analysis(
            run_dir=artifacts.run_dir,
            processed_root=config.analysis.get("output_dir", "results/processed"),
            digits=config.digits,
        )
        result["analysis"] = analysis_result

    return result
