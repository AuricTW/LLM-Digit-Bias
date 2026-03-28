# Report-Ready Results Skeleton

## Methods Snapshot

We evaluated a strict single-digit generation task in which each model was asked to choose one digit from 1 to 9 and output exactly one Arabic numeral. The main comparative analysis used one list-based prompt, five fixed digit orderings, and two temperatures (`0.0`, `0.2`). Runs were parsed with a strict validator: only a bare single digit in `1` to `9` counted as valid.

For the local Hugging Face `transformers` backend, the main comparison additionally included tokenizer audits and audited-surface candidate probabilities, which supported digit-vs-position mechanism analysis. Mechanism labels were assigned from pooled ordering results using a Jensen-Shannon divergence heuristic: a model was labeled `digit-leaning` when digit concentration exceeded position concentration, `position-leaning` when the reverse held, `mixed` when they were close, and `hybrid` when the label changed across temperatures.

## Main Comparative Result

Across the cleanly comparable local ordering-study runs, non-uniformity was universal, but the mechanism of bias varied by model family rather than collapsing into a single universal pattern.

- `Gemma 3 1B` was the clearest `position-leaning` model.
- `Llama 3.2 1B`, `Llama 3.2 3B`, `Qwen2.5 0.5B`, `Qwen2.5 1.5B`, and `SmolLM2-135M` were `digit-leaning`.
- `Qwen3 4B` was `mixed`.
- `SmolLM2-1.7B` was `hybrid`, shifting from `mixed` at `temp=0.0` to `digit-leaning` at `temp=0.2`.

This pattern supports a stronger conclusion than simple non-uniformity: the single-digit random-choice task reveals family-dependent bias mechanisms, with some models concentrating on specific digit identities, some concentrating on list positions, and some showing intermediate or temperature-sensitive behavior.

## Format Robustness Result

Strict protocol compliance was high for most modern 1B to 4B models in the main comparison. The main exception within the pooled local runs was `SmolLM2-135M`, which remained strongly digit-leaning but also showed substantial formatting fragility under some orderings, with maximum invalid-rate conditions above `0.7` at `temp=0.2`.

## Appendix-Scope Cases

`Qwen3-1.7B` was excluded from the main comparison because it systematically emitted think-wrapper text under the strict protocol. Under `/no_think` plus a longer output budget, the wrapper could be stripped and the recovered answer was consistently digit `5`, showing semantic task competence but transport-layer incompatibility with the main protocol.

`gpt-5.4` via `codex CLI` was also strongly non-uniform in an auxiliary five-ordering output-only study. However, this route did not expose tokenizer audits, logprobs, or explicit decode controls, so it is better treated as supporting frontier evidence than as a directly pooled main-table comparison.

## Suggested Reporting Structure

Main text:
- use `main_comparison_table.md` as the compact cross-family table
- use `cross_model_js_temp_0p0.png` and `cross_model_js_temp_0p2.png` as visual support

Supplement:
- use `supplementary_temperature_table.md` for per-temperature details
- use `appendix_cases_table.md` for protocol mismatch and auxiliary output-only cases

Open next step:
- add one OpenAI-compatible API run with top-logprob support so the frontier route can be compared at both the behavior layer and the probability layer
