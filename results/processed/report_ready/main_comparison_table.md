# Main Comparison Table

This table covers the cleanly comparable local ordering-study runs that support the strict parser and the digit-vs-position mechanism analysis.

| family | scale | model | overall | temp0 | temp0.2 | top digit temp0 | top digit temp0.2 | max invalid | note |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| Gemma | 1B | gemma-3-1b-it | position-leaning | position-leaning | position-leaning | 3:0.400 | 3:0.400 | 0.000 | position concentration exceeds digit concentration |
| Llama | 1B | Llama-3.2-1B-Instruct | digit-leaning | digit-leaning | digit-leaning | 8:0.600 | 7:0.480 | 0.000 | digit identity dominates position |
| Llama | 3B | Llama-3.2-3B-Instruct | digit-leaning | digit-leaning | digit-leaning | 9:0.600 | 9:0.528 | 0.000 | digit identity dominates position |
| Qwen2.5 | 0.5B | Qwen2.5-0.5B-Instruct | digit-leaning | digit-leaning | digit-leaning | 1:0.600 | 1:0.590 | 0.000 | digit identity dominates position |
| Qwen2.5 | 1.5B | Qwen2.5-1.5B-Instruct | digit-leaning | digit-leaning | digit-leaning | 1:0.800 | 1:0.792 | 0.000 | digit identity dominates position |
| Qwen3 | 4B | Qwen3-4B-Instruct-2507 | mixed | mixed | mixed | 5:0.600 | 5:0.597 | 0.000 | digit and position concentration are comparable |
| SmolLM | 135M | SmolLM2-135M-Instruct | digit-leaning | digit-leaning | digit-leaning | 1:0.800 | 1:0.991 | 1.000 | digit-leaning, but strict-format compliance is fragile |
| SmolLM | 1.7B | SmolLM2-1.7B-Instruct | hybrid | mixed | digit-leaning | 7:0.400 | 7:0.455 | 0.000 | mechanism changes with temperature |
