from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from src.types import LogprobSummary


def normalize_token_surface(token_text: str) -> str:
    return token_text.strip()


def summarize_openai_top_logprobs(
    top_logprobs: list[dict[str, Any]] | None,
    digits: list[int],
) -> LogprobSummary:
    if not top_logprobs:
        return LogprobSummary(
            status="unavailable",
            visibility="none",
            notes=["No top_logprobs were returned for the first generated token."],
        )

    digit_strings = {str(digit) for digit in digits}
    digit_logprob_candidates: dict[str, list[float]] = defaultdict(list)
    raw_candidates: list[dict[str, Any]] = []

    for item in top_logprobs:
        token_text = str(item.get("token", ""))
        logprob = float(item.get("logprob", float("-inf")))
        normalized = normalize_token_surface(token_text)
        raw_candidates.append(
            {
                "token": token_text,
                "normalized_token": normalized,
                "logprob": logprob,
                "bytes": item.get("bytes"),
            }
        )
        if normalized in digit_strings:
            digit_logprob_candidates[normalized].append(logprob)

    digit_probs: dict[str, float] = {}
    digit_logprobs: dict[str, float] = {}
    if digit_logprob_candidates:
        unnormalized_probs = {
            digit: sum(math.exp(logprob) for logprob in values)
            for digit, values in digit_logprob_candidates.items()
        }
        mass = sum(unnormalized_probs.values())
        if mass > 0:
            digit_probs = {digit: value / mass for digit, value in unnormalized_probs.items()}
            digit_logprobs = {digit: math.log(value / mass) for digit, value in unnormalized_probs.items()}

    return LogprobSummary(
        status="partial_top_k",
        visibility="top_k_only",
        digit_logprobs=digit_logprobs,
        digit_probs=digit_probs,
        raw_candidates=raw_candidates,
        notes=[
            "Distribution is aggregated from returned top_logprobs token surfaces only.",
            "Missing digits may be absent because the provider did not expose the full vocabulary.",
        ],
    )
