# Research Report Skeleton

## Title

Bias Analysis of LLMs in a Discrete Random Choice Task: The Case of Choosing a Number from 1 to 9

## 1. Research Question

- When a model is asked to "choose a random number from 1 to 9 and output exactly one Arabic numeral," is the final output distribution close to uniform?
- When token-level logprobs are available, is the conditional probability distribution over candidate digits already non-uniform before the final token is emitted?

## 2. Experimental Setup

- Model / endpoint:
- Provider / framework:
- Model version:
- Date and time:
- Hardware / runtime:
- Temperature values:
- Top-p values:
- Prompt templates:
- Number ordering conditions:
- Repetitions per condition:
- Output validation rule:
- Logprob availability:

## 3. Methods

### 3.1 Behavioral Layer

- Repeatedly sample each prompt-template x ordering x temperature condition.
- Accept only a single valid digit from `1` to `9` as a valid response.
- Mark every other response as invalid without manual correction.

### 3.2 Probability Layer

- When supported by the backend, record first-step logprobs or top-logprobs.
- Distinguish full candidate distributions from top-k visible distributions.
- Audit tokenizer behavior for digit forms such as `"7"`, `" 7"`, and `"\n7"`.

### 3.3 Statistics

- digit count / proportion
- invalid response rate
- chi-square goodness-of-fit against uniform
- entropy
- KL divergence to uniform
- Jensen-Shannon divergence to uniform
- optional: mean logprob distribution vs. empirical output frequency

## 4. Results

### 4.1 Output Distribution

- Main observation:
- Most frequent digit:
- Least frequent digit:
- Whether the distribution significantly departs from uniform:

### 4.2 Prompt And Ordering Effects

- Which prompt produces the strongest bias:
- Whether ascending / descending / random ordering changes the distribution:
- Whether temperature changes the bias magnitude:

### 4.3 Invalid Responses

- invalid rate:
- invalid pattern examples:

### 4.4 Logprob Findings

- Whether pre-output preference is visible:
- top-k visibility limitations:
- tokenization limitations:

## 5. Discussion

- Possible sources of the observed bias:
- Effects of prompt wording and digit ordering:
- Effects of decoding strategy:
- Whether output-level bias and token-level bias are aligned:

## 6. Limitations

- top-logprobs are not the same as a full candidate distribution
- tokenizer differences affect visible candidate probabilities
- some providers do not guarantee stable logprobs
- even `temperature=0` may still reflect implementation-specific behavior

## 7. Conclusion

- Whether the results support the hypothesis of approximately uniform random choice:
- Under which conditions the strongest bias appears:
- Recommended next steps for Phase 2 / Phase 3:
