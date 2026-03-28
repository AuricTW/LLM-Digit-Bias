from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import GenerationRequest, GenerationResponse


class BaseLLMClient(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def audit_tokenizer(self, digits: list[int]) -> dict | None:
        return None
