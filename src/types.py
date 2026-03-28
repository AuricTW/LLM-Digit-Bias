from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PromptTemplate:
    id: str
    text: str
    description: str = ""


@dataclass(slots=True)
class Ordering:
    mode: str
    label: str
    digits: list[int]
    index: int = 0
    is_randomized: bool = False


@dataclass(slots=True)
class ExperimentCondition:
    condition_id: str
    prompt_id: str
    prompt_text: str
    prompt_description: str
    ordering_mode: str
    ordering_label: str
    ordering_index: int
    order_sensitive: bool
    digits_order: list[int]
    temperature: float
    top_p: float | None


@dataclass(slots=True)
class GenerationRequest:
    prompt: str
    digits: list[int]
    temperature: float
    top_p: float | None
    max_output_tokens: int
    logprobs_enabled: bool = False
    top_logprobs: int | None = None


@dataclass(slots=True)
class LogprobSummary:
    status: str
    visibility: str
    candidate_mass_total: float | None = None
    surface_sequence_logprobs: dict[str, float | None] = field(default_factory=dict)
    surface_sequence_probs: dict[str, float] = field(default_factory=dict)
    surface_sequence_probs_normalized: dict[str, float] = field(default_factory=dict)
    digit_probs_raw: dict[str, float] = field(default_factory=dict)
    digit_logprobs: dict[str, float | None] = field(default_factory=dict)
    digit_probs: dict[str, float] = field(default_factory=dict)
    policy_name: str | None = None
    policy_candidate_mass_total: float | None = None
    policy_surface_sequence_logprobs: dict[str, float | None] = field(default_factory=dict)
    policy_surface_sequence_probs: dict[str, float] = field(default_factory=dict)
    policy_surface_sequence_probs_normalized: dict[str, float] = field(default_factory=dict)
    policy_digit_probs_raw: dict[str, float] = field(default_factory=dict)
    policy_digit_logprobs: dict[str, float | None] = field(default_factory=dict)
    policy_digit_probs: dict[str, float] = field(default_factory=dict)
    raw_candidates: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationResponse:
    text: str
    finish_reason: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)
    logprob_summary: LogprobSummary | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParseResult:
    raw_text: str
    normalized_text: str
    parsed_digit: int | None
    is_valid: bool
    invalid_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    raw_jsonl_path: Path
    raw_csv_path: Path
    manifest_path: Path
    resolved_config_path: Path
    tokenizer_audit_json_path: Path
    tokenizer_audit_csv_path: Path
