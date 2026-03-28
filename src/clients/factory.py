from __future__ import annotations

from src.clients.base import BaseLLMClient
from src.clients.mock import MockClient
from src.clients.openai_compatible import OpenAICompatibleClient
from src.clients.transformers_client import TransformersClient
from src.clients.vllm_client import VLLMLocalClient


def build_client(client_config: dict, seed: int) -> BaseLLMClient:
    provider = client_config["provider"]
    model_name = client_config["model_name"]
    params = client_config.get("params", {})

    if provider == "mock":
        return MockClient(model_name=model_name, seed=seed, mode=params.get("mode", "uniform"))
    if provider == "openai_compatible":
        return OpenAICompatibleClient(
            model_name=model_name,
            base_url=params["base_url"],
            api_key_env=params.get("api_key_env", "OPENAI_API_KEY"),
            token_limit_field=params.get("token_limit_field", "max_tokens"),
        )
    if provider == "transformers":
        return TransformersClient(
            model_name=model_name,
            device=params.get("device", "cpu"),
            torch_dtype=params.get("torch_dtype", "auto"),
            load_in_4bit=bool(params.get("load_in_4bit", False)),
            load_in_8bit=bool(params.get("load_in_8bit", False)),
            bnb_4bit_compute_dtype=params.get("bnb_4bit_compute_dtype", "float16"),
            bnb_4bit_quant_type=params.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(params.get("bnb_4bit_use_double_quant", True)),
            device_map=params.get("device_map"),
            apply_chat_template=bool(params.get("apply_chat_template", False)),
            chat_role=params.get("chat_role", "user"),
            candidate_surface_prefixes=params.get("candidate_surface_prefixes"),
            neutralize_generation_defaults=bool(params.get("neutralize_generation_defaults", True)),
        )
    if provider == "vllm_local":
        return VLLMLocalClient(model_name=model_name, **params)
    raise ValueError(f"Unsupported client provider: {provider!r}")
