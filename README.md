# LLM Digit Bias

[繁體中文](./README.zh-TW.md)

This repository contains a public release of a research framework for studying bias in discrete choice tasks for large language models. The current project focuses on a simple but diagnostic setting: a model is asked to choose one digit from `1` to `9` and to return exactly one Arabic numeral.

The framework is designed to support both behavioral analysis and probability-layer analysis. In addition to repeated sampling and output-frequency statistics, the local `transformers` path supports tokenizer audits and audited-surface candidate probability analysis, making it possible to compare final outputs with model-side preferences.

## Scope

- repeated sampling across prompt templates, orderings, and temperatures
- strict parsing: only a bare single digit in `1` to `9` is counted as valid
- modular client layer for `transformers`, OpenAI-compatible APIs, and vLLM
- analysis scripts for frequency, invalid rate, chi-square, entropy, KL divergence, and Jensen-Shannon divergence
- ordering and position analyses for comparing digit identity effects against list-position effects
- report-ready cross-model summaries and tables

## Repository Layout

```text
src/                 core framework and analysis code
prompts/             prompt templates
configs/             representative experiment configurations
results/processed/   curated aggregate outputs included in this release
docs/                public notes and report skeletons
```

## Installation

Base install:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e .[openai]
python -m pip install -e .[transformers]
python -m pip install -e .[vllm]
python -m pip install accelerate bitsandbytes
```

## Reproducing Basic Runs

Framework smoke test:

```bash
python -m src.runner.main --config configs/quickstart_mock.json
```

Build the aggregated comparison outputs included in this release:

```bash
python -m src.analysis.cross_model_comparison_main
python -m src.analysis.cross_model_plots_main
python -m src.analysis.report_artifacts_main
```

## Released Results

This public release includes aggregated outputs under `results/processed/`, including:

- `cross_model_comparison.csv`
- `cross_model_comparison.md`
- `cross_model_temperature_details.csv`
- `cross_model_figures/`
- `report_ready/main_comparison_table.md`
- `report_ready/supplementary_temperature_table.md`
- `report_ready/appendix_cases_table.md`
- `report_ready/report_results_skeleton.md`

These files summarize the current cross-model findings without requiring the full raw run corpus.

## Release Policy

This repository includes the code, representative configs, and curated aggregate outputs needed to understand and reproduce the released analyses. It does not include the full set of raw experimental dumps or local intermediate artifacts.

Excluded from this public release:

- `results/raw/`
- raw tokenizer-audit dumps
- local temporary files
- build artifacts such as `*.egg-info`

The rationale for these boundaries is documented in [PUBLIC_EXPORT_NOTES.md](./PUBLIC_EXPORT_NOTES.md).

## Current Research Picture

The released analyses support three high-level conclusions.

- LLM outputs in this task are typically far from uniform.
- The bias is not exhausted by final outputs; it is also visible in model-side candidate probabilities when available.
- The mechanism of bias differs across model families: some models are more digit-identity-driven, some are more position-sensitive, and some exhibit mixed or temperature-dependent behavior.
