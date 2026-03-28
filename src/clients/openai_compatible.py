from __future__ import annotations

import os
from typing import Any

from src.clients.base import BaseLLMClient
from src.parsing.logprobs import summarize_openai_top_logprobs
from src.types import GenerationRequest, GenerationResponse


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        token_limit_field: str = "max_tokens",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is not installed. Install it with `pip install .[openai]`."
            ) from exc

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env!r} is not set.")

        self.provider_name = "openai_compatible"
        self.model_name = model_name
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._token_limit_field = token_limit_field

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            self._token_limit_field: request.max_output_tokens,
        }
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.logprobs_enabled:
            kwargs["logprobs"] = True
            if request.top_logprobs is not None:
                kwargs["top_logprobs"] = request.top_logprobs

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = _safe_getattr(_safe_getattr(choice, "message"), "content", "") or ""
        logprob_content = _safe_getattr(_safe_getattr(choice, "logprobs"), "content", None)

        summary = None
        if logprob_content:
            first_position = logprob_content[0]
            top_logprobs = [
                {
                    "token": _safe_getattr(item, "token"),
                    "logprob": _safe_getattr(item, "logprob"),
                    "bytes": _safe_getattr(item, "bytes"),
                }
                for item in (_safe_getattr(first_position, "top_logprobs", []) or [])
            ]
            summary = summarize_openai_top_logprobs(top_logprobs=top_logprobs, digits=request.digits)

        return GenerationResponse(
            text=content,
            finish_reason=_safe_getattr(choice, "finish_reason"),
            raw_payload=response.model_dump(mode="json"),
            logprob_summary=summary,
            provider_metadata={"provider": self.provider_name, "model_name": self.model_name},
        )
