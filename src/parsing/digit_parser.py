from __future__ import annotations

import re

from src.types import ParseResult

VALID_DIGIT_PATTERN = re.compile(r"^[1-9]$")


def parse_single_digit(raw_text: str) -> ParseResult:
    normalized = raw_text.strip()
    if VALID_DIGIT_PATTERN.fullmatch(normalized):
        return ParseResult(
            raw_text=raw_text,
            normalized_text=normalized,
            parsed_digit=int(normalized),
            is_valid=True,
            invalid_reason=None,
        )
    if normalized == "":
        reason = "empty_after_strip"
    elif len(normalized) != 1:
        reason = "not_single_character"
    elif not normalized.isdigit():
        reason = "not_digit"
    elif normalized == "0":
        reason = "out_of_range"
    else:
        reason = "invalid_digit_token"
    return ParseResult(
        raw_text=raw_text,
        normalized_text=normalized,
        parsed_digit=None,
        is_valid=False,
        invalid_reason=reason,
    )
