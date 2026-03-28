from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from src.clients.base import BaseLLMClient
from src.types import GenerationRequest, GenerationResponse, LogprobSummary


class TransformersClient(BaseLLMClient):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        device_map: str | None = None,
        apply_chat_template: bool = False,
        chat_role: str = "user",
        candidate_surface_prefixes: list[str] | None = None,
        neutralize_generation_defaults: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers/torch packages are not installed. Install them with `pip install .[transformers]`."
            ) from exc

        self.provider_name = "transformers"
        self.model_name = model_name
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = self._resolve_torch_dtype(torch_dtype=torch_dtype, torch_module=torch)
        model_kwargs: dict[str, Any] = {}
        quantization_mode = "none"
        quantization_config = None
        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit cannot both be enabled.")
        if load_in_4bit or load_in_8bit:
            quantization_mode = "4bit" if load_in_4bit else "8bit"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=self._resolve_torch_dtype(
                    torch_dtype=bnb_4bit_compute_dtype,
                    torch_module=torch,
                ),
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = device_map or ("auto" if device.startswith("cuda") else None)
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **{key: value for key, value in model_kwargs.items() if value is not None},
        )
        if quantization_mode == "none":
            self._model.to(device)
        self._model.eval()
        self._device = device
        self._input_device = device
        self._torch_dtype_name = str(dtype).replace("torch.", "") if dtype is not None else "auto"
        self._quantization_mode = quantization_mode
        self._bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self._bnb_4bit_quant_type = bnb_4bit_quant_type
        self._bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self._device_map = model_kwargs.get("device_map")
        self._apply_chat_template = apply_chat_template
        self._chat_role = chat_role
        self._candidate_surface_prefixes = candidate_surface_prefixes or ["", " ", "\n"]
        self._neutralize_generation_defaults = neutralize_generation_defaults
        self._summary_cache: dict[tuple[Any, ...], LogprobSummary] = {}

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: str, torch_module: Any) -> Any:
        if torch_dtype == "auto":
            return None
        if hasattr(torch_module, torch_dtype):
            return getattr(torch_module, torch_dtype)
        raise ValueError(f"Unsupported torch_dtype value: {torch_dtype!r}")

    def _encode_prompt(self, prompt: str) -> dict[str, Any]:
        if self._apply_chat_template:
            if not hasattr(self._tokenizer, "apply_chat_template"):
                raise RuntimeError(
                    f"Tokenizer for {self.model_name!r} does not expose apply_chat_template."
                )
            messages = [{"role": self._chat_role, "content": prompt}]
            batch = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            batch = self._tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self._input_device) for key, value in batch.items()}

    @lru_cache(maxsize=8)
    def _surface_catalog(self, digits_key: tuple[int, ...]) -> tuple[dict[str, Any], ...]:
        prefix_labels = {
            "": "bare",
            " ": "leading_space",
            "\n": "leading_newline",
        }
        rows: list[dict[str, Any]] = []
        for digit in digits_key:
            digit_key = str(digit)
            for prefix in self._candidate_surface_prefixes:
                surface = f"{prefix}{digit_key}"
                token_ids = self._tokenizer.encode(surface, add_special_tokens=False)
                rows.append(
                    {
                        "digit": digit_key,
                        "surface": surface,
                        "surface_display": surface.encode("unicode_escape").decode(),
                        "prefix_type": prefix_labels.get(prefix, "custom_prefix"),
                        "token_ids": [int(token_id) for token_id in token_ids],
                        "token_count": int(len(token_ids)),
                        "is_single_token": len(token_ids) == 1,
                    }
                )
        return tuple(rows)

    def audit_tokenizer(self, digits: list[int]) -> dict[str, Any]:
        surface_rows = [dict(item) for item in self._surface_catalog(tuple(digits))]
        return {
            "provider": self.provider_name,
            "model_name": self.model_name,
            "device": self._device,
            "torch_dtype": self._torch_dtype_name,
            "quantization_mode": self._quantization_mode,
            "device_map": self._device_map,
            "apply_chat_template": self._apply_chat_template,
            "chat_role": self._chat_role if self._apply_chat_template else None,
            "surface_rows": surface_rows,
        }

    def _policy_name(self, temperature: float, top_p: float | None) -> str:
        if temperature <= 0:
            return "greedy"
        if top_p is not None and 0.0 < top_p < 1.0:
            return "temperature_top_p_sampling"
        return "temperature_sampling"

    def _apply_top_p(self, logits: Any, top_p: float | None) -> Any:
        if top_p is None or top_p >= 1.0 or top_p <= 0.0:
            return logits

        torch = self._torch
        filtered_logits = logits.clone()
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
        sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs > top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False
        filtered_logits[sorted_indices[remove_mask]] = float("-inf")
        return filtered_logits

    def _build_policy_logprobs(
        self,
        logits: Any,
        temperature: float,
        top_p: float | None,
    ) -> Any:
        torch = self._torch
        if temperature <= 0:
            policy_log_probs = torch.full_like(logits, float("-inf"))
            greedy_index = int(torch.argmax(logits).item())
            policy_log_probs[greedy_index] = 0.0
            return policy_log_probs

        adjusted_logits = logits
        if temperature != 1.0:
            adjusted_logits = adjusted_logits / temperature
        adjusted_logits = self._apply_top_p(adjusted_logits, top_p)
        return torch.nn.functional.log_softmax(adjusted_logits, dim=-1)

    def _score_surface_sequences(
        self,
        encoded_prompt: dict[str, Any],
        surface_rows: list[dict[str, Any]],
        temperature: float,
        top_p: float | None,
    ) -> tuple[list[dict[str, Any]], str]:
        torch = self._torch
        prompt_ids = encoded_prompt["input_ids"][0]
        prompt_prefix = prompt_ids.tolist()
        next_token_cache: dict[tuple[int, ...], tuple[Any, Any]] = {}
        policy_name = self._policy_name(temperature=temperature, top_p=top_p)

        def next_token_logprobs(prefix_tokens: tuple[int, ...]) -> tuple[Any, Any]:
            cached = next_token_cache.get(prefix_tokens)
            if cached is not None:
                return cached

            sequence = prompt_prefix + list(prefix_tokens)
            input_ids = torch.tensor([sequence], dtype=torch.long, device=self._device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1]
            model_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            policy_log_probs = self._build_policy_logprobs(
                logits=logits,
                temperature=temperature,
                top_p=top_p,
            )
            next_token_cache[prefix_tokens] = (model_log_probs, policy_log_probs)
            return model_log_probs, policy_log_probs

        scored_rows: list[dict[str, Any]] = []
        for row in surface_rows:
            token_ids = row["token_ids"]
            sequence_logprob = 0.0
            policy_sequence_logprob = 0.0
            prefix_tokens: tuple[int, ...] = ()
            for token_offset, token_id in enumerate(token_ids):
                model_log_probs, policy_log_probs = next_token_logprobs(prefix_tokens)
                token_logprob = float(model_log_probs[token_id].item())
                policy_token_logprob = float(policy_log_probs[token_id].item())
                sequence_logprob += token_logprob
                if not math.isinf(policy_sequence_logprob):
                    if math.isinf(policy_token_logprob) and policy_token_logprob < 0:
                        policy_sequence_logprob = float("-inf")
                    else:
                        policy_sequence_logprob += policy_token_logprob
                prefix_tokens = prefix_tokens + (token_id,)
            scored_rows.append(
                {
                    **row,
                    "logprob": sequence_logprob,
                    "probability": math.exp(sequence_logprob),
                    "policy_logprob": (
                        None
                        if math.isinf(policy_sequence_logprob) and policy_sequence_logprob < 0
                        else policy_sequence_logprob
                    ),
                    "policy_probability": (
                        0.0
                        if math.isinf(policy_sequence_logprob) and policy_sequence_logprob < 0
                        else math.exp(policy_sequence_logprob)
                    ),
                }
            )
        return scored_rows, policy_name

    def _aggregate_probability_view(
        self,
        scored_rows: list[dict[str, Any]],
        digits: list[int],
        *,
        logprob_key: str,
        probability_key: str,
        normalized_probability_key: str,
    ) -> dict[str, Any]:
        candidate_mass_total = sum(float(row[probability_key]) for row in scored_rows)
        surface_sequence_logprobs = {
            row["surface_display"]: row[logprob_key]
            for row in scored_rows
        }
        surface_sequence_probs = {
            row["surface_display"]: float(row[probability_key])
            for row in scored_rows
        }

        surface_sequence_probs_normalized = {}
        if candidate_mass_total > 0:
            surface_sequence_probs_normalized = {
                row["surface_display"]: float(row[probability_key]) / candidate_mass_total
                for row in scored_rows
            }

        digit_probs_raw: dict[str, float] = {str(digit): 0.0 for digit in digits}
        for row in scored_rows:
            digit_probs_raw[row["digit"]] += float(row[probability_key])

        digit_total = sum(digit_probs_raw.values())
        digit_probs = {}
        digit_logprobs = {}
        if digit_total > 0:
            digit_probs = {
                digit: value / digit_total
                for digit, value in digit_probs_raw.items()
            }
            digit_logprobs = {
                digit: math.log(value)
                for digit, value in digit_probs.items()
                if value > 0
            }

        normalized_rows = []
        for row in scored_rows:
            normalized_rows.append(
                {
                    **row,
                    normalized_probability_key: (
                        float(row[probability_key]) / candidate_mass_total if candidate_mass_total > 0 else None
                    ),
                }
            )

        return {
            "candidate_mass_total": candidate_mass_total,
            "surface_sequence_logprobs": surface_sequence_logprobs,
            "surface_sequence_probs": surface_sequence_probs,
            "surface_sequence_probs_normalized": surface_sequence_probs_normalized,
            "digit_probs_raw": digit_probs_raw,
            "digit_probs": digit_probs,
            "digit_logprobs": digit_logprobs,
            "normalized_rows": normalized_rows,
        }

    def _summarize_candidate_surfaces(
        self,
        encoded_prompt: dict[str, Any],
        digits: list[int],
        temperature: float,
        top_p: float | None,
    ) -> LogprobSummary:
        surface_rows = [dict(item) for item in self._surface_catalog(tuple(digits))]
        scored_rows, policy_name = self._score_surface_sequences(
            encoded_prompt=encoded_prompt,
            surface_rows=surface_rows,
            temperature=temperature,
            top_p=top_p,
        )
        model_view = self._aggregate_probability_view(
            scored_rows=scored_rows,
            digits=digits,
            logprob_key="logprob",
            probability_key="probability",
            normalized_probability_key="normalized_probability",
        )
        policy_view = self._aggregate_probability_view(
            scored_rows=scored_rows,
            digits=digits,
            logprob_key="policy_logprob",
            probability_key="policy_probability",
            normalized_probability_key="policy_normalized_probability",
        )
        normalized_rows = []
        for model_row, policy_row in zip(
            model_view["normalized_rows"],
            policy_view["normalized_rows"],
            strict=True,
        ):
            normalized_rows.append(
                {
                    **model_row,
                    "policy_normalized_probability": policy_row.get("policy_normalized_probability"),
                }
            )

        default_note = (
            "This backend explicitly disables top-k warping and neutralizes hidden generation defaults so the recorded policy distribution matches the configured temperature/top_p settings."
            if self._neutralize_generation_defaults
            else "This backend preserves the model's native generation defaults, so policy-level probabilities may reflect provider-specific repetition or sampling settings beyond temperature/top_p."
        )

        return LogprobSummary(
            status="surface_sequence_exact",
            visibility="audited_surface_subset_exact",
            candidate_mass_total=model_view["candidate_mass_total"],
            surface_sequence_logprobs=model_view["surface_sequence_logprobs"],
            surface_sequence_probs=model_view["surface_sequence_probs"],
            surface_sequence_probs_normalized=model_view["surface_sequence_probs_normalized"],
            digit_probs_raw=model_view["digit_probs_raw"],
            digit_logprobs=model_view["digit_logprobs"],
            digit_probs=model_view["digit_probs"],
            policy_name=policy_name,
            policy_candidate_mass_total=policy_view["candidate_mass_total"],
            policy_surface_sequence_logprobs=policy_view["surface_sequence_logprobs"],
            policy_surface_sequence_probs=policy_view["surface_sequence_probs"],
            policy_surface_sequence_probs_normalized=policy_view["surface_sequence_probs_normalized"],
            policy_digit_probs_raw=policy_view["digit_probs_raw"],
            policy_digit_logprobs=policy_view["digit_logprobs"],
            policy_digit_probs=policy_view["digit_probs"],
            raw_candidates=normalized_rows,
            notes=[
                "Model-level surface probabilities are exact sequence probabilities for the audited surfaces under the active tokenizer before decode-time warping.",
                "Policy-level surface probabilities apply the active decode policy step by step: greedy for temperature=0, or temperature/top_p warping for sampled decoding.",
                default_note,
                "Aggregated digit probabilities sum audited surface forms and are normalized within the audited 1-9 candidate set.",
                "Probability mass assigned to outputs outside the audited candidate surfaces is excluded from candidate probabilities and tracked separately via candidate_mass_total and policy_candidate_mass_total.",
            ],
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        torch = self._torch
        encoded = self._encode_prompt(request.prompt)
        do_sample = request.temperature > 0
        generation_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": request.max_output_tokens,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self._neutralize_generation_defaults:
            generation_kwargs["repetition_penalty"] = 1.0
        if do_sample:
            generation_kwargs["temperature"] = request.temperature
            generation_kwargs["top_k"] = 0
            generation_kwargs["top_p"] = request.top_p if request.top_p is not None else 1.0

        with torch.no_grad():
            outputs = self._model.generate(**generation_kwargs)

        prompt_length = int(encoded["input_ids"].shape[1])
        generated_ids = outputs.sequences[0][prompt_length:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        summary = None
        if request.logprobs_enabled:
            summary_cache_key = (
                request.prompt,
                tuple(request.digits),
                float(request.temperature),
                request.top_p,
            )
            summary = self._summary_cache.get(summary_cache_key)
            if summary is None:
                summary = self._summarize_candidate_surfaces(
                    encoded_prompt=encoded,
                    digits=request.digits,
                    temperature=request.temperature,
                    top_p=request.top_p,
                )
                self._summary_cache[summary_cache_key] = summary

        return GenerationResponse(
            text=text,
            finish_reason="stop",
            raw_payload={"generated_token_ids": generated_ids.tolist()},
            logprob_summary=summary,
            provider_metadata={
                "provider": self.provider_name,
                "model_name": self.model_name,
                "device": self._device,
                "torch_dtype": self._torch_dtype_name,
                "quantization_mode": self._quantization_mode,
                "device_map": self._device_map,
                "bnb_4bit_compute_dtype": self._bnb_4bit_compute_dtype if self._quantization_mode == "4bit" else None,
                "bnb_4bit_quant_type": self._bnb_4bit_quant_type if self._quantization_mode == "4bit" else None,
                "bnb_4bit_use_double_quant": (
                    self._bnb_4bit_use_double_quant if self._quantization_mode == "4bit" else None
                ),
                "apply_chat_template": self._apply_chat_template,
                "neutralize_generation_defaults": self._neutralize_generation_defaults,
            },
        )
