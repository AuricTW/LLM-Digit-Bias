# Public Export Notes

This folder was curated conservatively for public release.

## Included

- `src/`
- `prompts/`
- selected `configs/`
- aggregated processed outputs under `results/processed/`
- public summary notes under `docs/`
- `pyproject.toml`

## Excluded On Purpose

### Raw experimental dumps

Not included:

- `results/raw/`
- per-run tokenizer audit raw files
- large per-trial JSONL/CSV outputs

Reason:

- they contain much more operational detail than is needed for a public code release
- they are bulky
- some provider-specific routes also expose execution-mode details that are unnecessary in the public snapshot

### Local temp and generated artifacts

Not included:

- temp files such as `tmp_codex_last_message.txt`
- generated package metadata such as `*.egg-info`
- Python cache artifacts

Reason:

- they are not part of the research contribution
- they add noise and increase the chance of accidental over-sharing

### Conservative treatment of provider-specific material

This export keeps the main framework code, but it does not include raw Codex CLI run directories or other raw provider dumps. The public snapshot also omits the `codex_cli` backend implementation itself.

Reason:

- no secrets were found
- but a safety-first public snapshot should avoid publishing unnecessary execution metadata when the main research conclusions are already preserved in aggregated outputs

## Notes

- No hardcoded Hugging Face token, OpenAI API key, or private key was intentionally copied into this export.
- The aggregated results in `results/processed/` are sufficient to understand the current cross-model conclusions without exposing the full raw run corpus.
