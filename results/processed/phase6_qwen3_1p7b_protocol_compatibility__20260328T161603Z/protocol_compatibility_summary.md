# Protocol Compatibility Summary

Strict validity uses the unchanged single-digit parser. Recovered validity strips one empty `<think></think>` wrapper before re-parsing.

| condition_id | strict_valid_rate | empty_wrapper_rate | recovered_valid_rate | recovered_only_rate | recovered_digit_support | top_raw_outputs |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| default_prompt__max_4__temp_0p00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | <think>\nOkay,:20 |
| default_prompt__max_20__temp_0p00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | <think>\nOkay, let's see. I need to pick exactly one digit from the list::20 |
| no_think_prefix__max_4__temp_0p00 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |  | <think>\n\n</think>:20 |
| no_think_prefix__max_20__temp_0p00 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 5:1.0000 | <think>\n\n</think>\n\n5:20 |
