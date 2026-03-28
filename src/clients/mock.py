from __future__ import annotations

import random

from src.clients.base import BaseLLMClient
from src.types import GenerationRequest, GenerationResponse


class MockClient(BaseLLMClient):
    def __init__(self, model_name: str, seed: int, mode: str = "uniform") -> None:
        self.provider_name = "mock"
        self.model_name = model_name
        self._rng = random.Random(seed)
        self._mode = mode

    def _pick_digit(self, digits: list[int]) -> int:
        if self._mode == "biased_center":
            weights = [1.0 / (1 + abs(digit - 5)) for digit in digits]
            return self._rng.choices(digits, weights=weights, k=1)[0]
        return self._rng.choice(digits)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        digit = self._pick_digit(request.digits)
        return GenerationResponse(
            text=str(digit),
            finish_reason="stop",
            raw_payload={"mock_mode": self._mode},
            provider_metadata={"provider": self.provider_name, "model_name": self.model_name},
        )
