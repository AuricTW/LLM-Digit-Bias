# Cross-Model Comparison

Mechanism labels are based on the pooled position-analysis heuristic over the five fixed ordering conditions.

| family | scale | model | overall | temp0 | temp0.2 | temp0 top digit | temp0 top pos | temp0 unique top digits | temp0.2 top digit | temp0.2 top pos | temp0.2 unique top digits |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: |
| Gemma | 1B | google/gemma-3-1b-it | position-leaning | position-leaning | position-leaning | 3:0.4000 | 1:0.8000 | 4 | 3:0.4000 | 1:0.7975 | 4 |
| pattern | pattern | pattern |  | ascending=3, descending=9, random_001=6, random_002=8, random_003=3 | ascending=3, descending=9, random_001=6, random_002=8, random_003=3 |  |  |  |  |  |  |
| Llama | 1B | meta-llama/Llama-3.2-1B-Instruct | digit-leaning | digit-leaning | digit-leaning | 8:0.6000 | 1:0.2000 | 2 | 7:0.4800 | 5:0.2950 | 2 |
| pattern | pattern | pattern |  | ascending=7, descending=8, random_001=8, random_002=8, random_003=7 | ascending=7, descending=8, random_001=8, random_002=8, random_003=7 |  |  |  |  |  |  |
| Llama | 3B | meta-llama/Llama-3.2-3B-Instruct | digit-leaning | digit-leaning | digit-leaning | 9:0.6000 | 1:0.2000 | 3 | 9:0.5275 | 6:0.1900 | 3 |
| pattern | pattern | pattern |  | ascending=5, descending=9, random_001=3, random_002=9, random_003=9 | ascending=5, descending=9, random_001=3, random_002=9, random_003=9 |  |  |  |  |  |  |
| Qwen2.5 | 0.5B | Qwen/Qwen2.5-0.5B-Instruct | digit-leaning | digit-leaning | digit-leaning | 1:0.6000 | 7:0.4000 | 2 | 1:0.5900 | 9:0.3675 | 2 |
| pattern | pattern | pattern |  | ascending=7, descending=1, random_001=1, random_002=1, random_003=7 | ascending=7, descending=1, random_001=1, random_002=1, random_003=7 |  |  |  |  |  |  |
| Qwen2.5 | 1.5B | Qwen/Qwen2.5-1.5B-Instruct | digit-leaning | digit-leaning | digit-leaning | 1:0.8000 | 9:0.4000 | 2 | 1:0.7925 | 9:0.4000 | 2 |
| pattern | pattern | pattern |  | ascending=3, descending=1, random_001=1, random_002=1, random_003=1 | ascending=3, descending=1, random_001=1, random_002=1, random_003=1 |  |  |  |  |  |  |
| Qwen3 | 4B | Qwen/Qwen3-4B-Instruct-2507 | mixed | mixed | mixed | 5:0.6000 | 1:0.4000 | 3 | 5:0.5975 | 1:0.4025 | 3 |
| pattern | pattern | pattern |  | ascending=5, descending=5, random_001=6, random_002=5, random_003=3 | ascending=5, descending=5, random_001=6, random_002=5, random_003=3 |  |  |  |  |  |  |
| SmolLM | 135M | HuggingFaceTB/SmolLM2-135M-Instruct | digit-leaning | digit-leaning | digit-leaning | 1:0.8000 | 1:0.2000 | 1 | 1:0.9913 | 9:0.3913 | 1 |
| pattern | pattern | pattern |  | ascending=1, descending=1, random_001=1, random_002=1, random_003=1 | ascending=1, descending=1, random_001=1, random_002=1, random_003=1 |  |  |  |  |  |  |
| SmolLM | 1.7B | HuggingFaceTB/SmolLM2-1.7B-Instruct | hybrid | mixed | digit-leaning | 7:0.4000 | 1:0.4000 | 4 | 7:0.4550 | 1:0.3025 | 4 |
| pattern | pattern | pattern |  | ascending=3, descending=9, random_001=7, random_002=8, random_003=7 | ascending=3, descending=9, random_001=7, random_002=8, random_003=7 |  |  |  |  |  |  |
