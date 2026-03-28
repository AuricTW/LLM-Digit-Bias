# Appendix Cases Table

These cases are informative, but they are not merged into the main local logprob-aware comparison table.

| case | route | model | status | main table | strict valid | recovered valid | support | note |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| qwen3_1p7b_protocol | transformers | Qwen/Qwen3-1.7B | protocol-mismatch | no | 0.000 | 1.000 | 5:1.0000 | semantically answers the task, but outputs think-wrapper text under the strict parser |
| codex_cli_gpt54 | codex_cli | gpt-5.4 | auxiliary-output-only | no | 1.000 | n/a | 4:0.4000, 7:0.3850, 2:0.1350, 5:0.0700, 6:0.0100 | strong output-only non-uniformity, but no logprobs, tokenizer audit, or explicit decode controls |
