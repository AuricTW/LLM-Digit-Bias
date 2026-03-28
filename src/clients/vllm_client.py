from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from src.clients.base import BaseLLMClient
from src.types import GenerationRequest, GenerationResponse, LogprobSummary


class VLLMLocalClient(BaseLLMClient):
    def __init__(self, model_name: str, **engine_kwargs: Any) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("vllm package is not installed. Install it with `pip install .[vllm]`.") from exc

        self.provider_name = "vllm_local"
        self.model_name = model_name
        self._engine = LLM(model=model_name, **engine_kwargs)

    def _summarize_vllm_logprobs(self, first_step: dict[int, Any] | None, digits: list[int]) -> LogprobSummary:
        if not first_step:
            return LogprobSummary(
                status="unavailable",
                visibility="none",
                notes=["vLLM did not return first-step token logprobs for this request."],
            )

        digit_strings = {str(digit) for digit in digits}
        digit_logprob_candidates: dict[str, list[float]] = defaultdict(list)
        raw_candidates: list[dict[str, Any]] = []

        for token_id, item in first_step.items():
            decoded = getattr(item, "decoded_token", None) or ""
            normalized = decoded.strip()
            logprob = float(getattr(item, "logprob", float("-inf")))
            raw_candidates.append(
                {
                    "token_id": int(token_id),
                    "token": decoded,
                    "normalized_token": normalized,
                    "logprob": logprob,
                }
            )
            if normalized in digit_strings:
                digit_logprob_candidates[normalized].append(logprob)

        digit_probs: dict[str, float] = {}
        digit_logprobs: dict[str, float] = {}
        if digit_logprob_candidates:
            unnormalized = {
                digit: sum(math.exp(value) for value in values)
                for digit, values in digit_logprob_candidates.items()
            }
            mass = sum(unnormalized.values())
            if mass > 0:
                digit_probs = {digit: value / mass for digit, value in unnormalized.items()}
                digit_logprobs = {digit: math.log(value) for digit, value in digit_probs.items()}

        return LogprobSummary(
            status="partial_top_k",
            visibility="top_k_only",
            digit_logprobs=digit_logprobs,
            digit_probs=digit_probs,
            raw_candidates=raw_candidates,
            notes=[
                "Aggregated from the logprob candidates returned by vLLM for the first generated token.",
                "The vLLM docs note that token logprobs are not guaranteed to be perfectly stable across runs.",
            ],
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            from vllm import SamplingParams
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("vllm package is not installed. Install it with `pip install .[vllm]`.") from exc

        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p if request.top_p is not None else 1.0,
            max_tokens=request.max_output_tokens,
            logprobs=request.top_logprobs if request.logprobs_enabled else None,
            prompt_logprobs=None,
        )
        outputs = self._engine.generate([request.prompt], sampling_params=params)
        item = outputs[0]
        candidate = item.outputs[0]
        first_step = candidate.logprobs[0] if getattr(candidate, "logprobs", None) else None
        summary = self._summarize_vllm_logprobs(first_step=first_step, digits=request.digits)
        return GenerationResponse(
            text=candidate.text,
            finish_reason=getattr(candidate, "finish_reason", None),
            raw_payload={"request_id": getattr(item, "request_id", None)},
            logprob_summary=summary,
            provider_metadata={"provider": self.provider_name, "model_name": self.model_name},
        )
