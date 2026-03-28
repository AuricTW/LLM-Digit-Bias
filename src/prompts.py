from __future__ import annotations

import json
from pathlib import Path

from src.types import Ordering, PromptTemplate

ORDER_PLACEHOLDERS = ("{numbers_csv}", "{numbers_spaced}")


def load_prompt_templates(path: str | Path) -> list[PromptTemplate]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [PromptTemplate(**item) for item in payload]


def is_order_sensitive(template: PromptTemplate) -> bool:
    return any(marker in template.text for marker in ORDER_PLACEHOLDERS)


def render_prompt(template: PromptTemplate, ordering: Ordering) -> str:
    digits = ordering.digits
    substitutions = {
        "numbers_csv": ",".join(str(digit) for digit in digits),
        "numbers_spaced": " ".join(str(digit) for digit in digits),
        "range_start": min(digits),
        "range_end": max(digits),
        "range_phrase": f"{min(digits)} 到 {max(digits)}",
    }
    return template.text.format(**substitutions)
