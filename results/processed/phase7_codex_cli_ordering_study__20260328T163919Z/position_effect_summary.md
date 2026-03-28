# Position Effect Summary

Higher Jensen-Shannon divergence means the distribution is more concentrated away from uniform. If position JS exceeds digit JS, the model is more position-leaning under this heuristic.

| model | prompt | temp | top obs pos | top obs digit | obs pos JS | obs digit JS | policy pos JS | policy digit JS | obs heuristic | policy heuristic |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| gpt-5.4 | phase2_qwen_list_answer | 0.0 | 6:0.3950 | 4:0.4000 | 0.3500 | 0.3823 | n/a | n/a | digit-leaning | n/a |
| support | support |  | 6:0.3950, 3:0.3150, 7:0.2000, 1:0.0700, 2:0.0150, 8:0.0050 | 4:0.4000, 7:0.3850, 2:0.1350, 5:0.0700, 6:0.0100 |  |  | n/a | n/a |  |  |
