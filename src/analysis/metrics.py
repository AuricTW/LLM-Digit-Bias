from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import chisquare


def frequency_distribution(values: Iterable[int], digits: list[int]) -> tuple[dict[str, int], dict[str, float]]:
    value_list = list(values)
    total = len(value_list)
    counts = {str(digit): 0 for digit in digits}
    for value in value_list:
        counts[str(value)] += 1
    proportions = {
        digit: (count / total if total > 0 else 0.0)
        for digit, count in counts.items()
    }
    return counts, proportions


def entropy_bits(proportions: dict[str, float]) -> float:
    return -sum(prob * math.log2(prob) for prob in proportions.values() if prob > 0)


def kl_divergence_to_uniform(proportions: dict[str, float]) -> float:
    if not proportions:
        return float("nan")
    uniform = 1.0 / len(proportions)
    return sum(prob * math.log2(prob / uniform) for prob in proportions.values() if prob > 0)


def jensen_shannon_divergence_to_uniform(proportions: dict[str, float]) -> float:
    if not proportions:
        return float("nan")
    uniform = {digit: 1.0 / len(proportions) for digit in proportions}
    midpoint = {digit: 0.5 * (proportions[digit] + uniform[digit]) for digit in proportions}
    return 0.5 * _kl(proportions, midpoint) + 0.5 * _kl(uniform, midpoint)


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(
        value * math.log2(value / q[key])
        for key, value in p.items()
        if value > 0 and q[key] > 0
    )


def chi_square_against_uniform(counts: dict[str, int]) -> tuple[float, float]:
    observed = np.array(list(counts.values()), dtype=float)
    total = observed.sum()
    if total <= 0:
        return float("nan"), float("nan")
    expected = np.full_like(observed, fill_value=total / len(observed))
    result = chisquare(f_obs=observed, f_exp=expected)
    return float(result.statistic), float(result.pvalue)


def pearson_correlation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = [key for key in left if key in right]
    if len(keys) < 2:
        return float("nan")
    x = np.array([left[key] for key in keys], dtype=float)
    y = np.array([right[key] for key in keys], dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
