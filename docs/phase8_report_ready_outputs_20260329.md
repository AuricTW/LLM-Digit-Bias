# Phase 8 Report-Ready Outputs

## What was added

This round did not add a new model run. Instead, it converted the existing cross-model results into report-ready artifacts.

Generated outputs:

- `results/processed/report_ready/main_comparison_table.csv`
- `results/processed/report_ready/main_comparison_table.md`
- `results/processed/report_ready/supplementary_temperature_table.csv`
- `results/processed/report_ready/supplementary_temperature_table.md`
- `results/processed/report_ready/appendix_cases_table.csv`
- `results/processed/report_ready/appendix_cases_table.md`
- `results/processed/report_ready/report_results_skeleton.md`

Script entry point:

- `python -m src.analysis.report_artifacts_main`

## Main table scope

The main comparison table includes only the cleanly comparable local ordering-study runs:

- Gemma 3 1B
- Llama 3.2 1B
- Llama 3.2 3B
- Qwen2.5 0.5B
- Qwen2.5 1.5B
- Qwen3 4B
- SmolLM2-135M
- SmolLM2-1.7B

Excluded from the main table:

- `Qwen3-1.7B`, because it is a protocol-mismatch case under the strict parser
- `gpt-5.4` via `codex CLI`, because it is an auxiliary output-only route without logprobs or tokenizer audit

## Main table headline

The report-ready table makes the current synthesis easier to state cleanly:

- one model is clearly `position-leaning`: `Gemma 3 1B`
- five models are clearly `digit-leaning`: `Llama 3.2 1B`, `Llama 3.2 3B`, `Qwen2.5 0.5B`, `Qwen2.5 1.5B`, `SmolLM2-135M`
- one model is clearly `mixed`: `Qwen3 4B`
- one model is `hybrid`: `SmolLM2-1.7B`

## Recommended use in writing

Main text:

- use `main_comparison_table.md`
- cite the two pooled JS scatter plots in `results/processed/cross_model_figures/`

Supplement:

- use `supplementary_temperature_table.md` for per-temperature detail
- use `appendix_cases_table.md` for exclusion logic and auxiliary evidence

Draft prose:

- start from `report_results_skeleton.md`
