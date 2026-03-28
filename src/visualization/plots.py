from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_digit_distribution(
    counts: dict[str, int],
    proportions: dict[str, float],
    title: str,
    output_path: str | Path,
) -> None:
    digits = list(counts.keys())
    values = [proportions[digit] for digit in digits]
    uniform = [1.0 / len(digits)] * len(digits)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(digits, values, color="#2364aa", alpha=0.85, label="Observed")
    ax.plot(digits, uniform, color="#d1495b", linestyle="--", linewidth=2, label="Uniform")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, max(max(values, default=0.0), max(uniform)) * 1.25 if digits else 1.0)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
