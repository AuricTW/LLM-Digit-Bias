from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    description: str
    seed: int
    repetitions_per_condition: int
    digits: list[int]
    prompt_template_file: str
    prompt_ids: list[str]
    ordering_modes: list[str]
    random_orderings_per_prompt: int
    temperatures: list[float]
    top_p: float | None
    max_output_tokens: int
    continue_on_error: bool
    analysis: dict[str, Any]
    client: dict[str, Any]
    logprob: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "seed": self.seed,
            "repetitions_per_condition": self.repetitions_per_condition,
            "digits": self.digits,
            "prompt_template_file": self.prompt_template_file,
            "prompt_ids": self.prompt_ids,
            "ordering_modes": self.ordering_modes,
            "random_orderings_per_prompt": self.random_orderings_per_prompt,
            "temperatures": self.temperatures,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "continue_on_error": self.continue_on_error,
            "analysis": self.analysis,
            "client": self.client,
            "logprob": self.logprob,
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        experiment_name=payload["experiment_name"],
        description=payload.get("description", ""),
        seed=int(payload.get("seed", 0)),
        repetitions_per_condition=int(payload["repetitions_per_condition"]),
        digits=[int(item) for item in payload.get("digits", list(range(1, 10)))],
        prompt_template_file=payload["prompt_template_file"],
        prompt_ids=list(payload["prompt_ids"]),
        ordering_modes=list(payload.get("ordering_modes", ["ascending"])),
        random_orderings_per_prompt=int(payload.get("random_orderings_per_prompt", 1)),
        temperatures=[float(item) for item in payload.get("temperatures", [0.0])],
        top_p=float(payload["top_p"]) if payload.get("top_p") is not None else None,
        max_output_tokens=int(payload.get("max_output_tokens", 4)),
        continue_on_error=bool(payload.get("continue_on_error", False)),
        analysis=dict(payload.get("analysis", {"enabled": False})),
        client=dict(payload["client"]),
        logprob=dict(payload.get("logprob", {"enabled": False})),
    )
