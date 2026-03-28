from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.io_utils import ensure_directory


FAMILY_COLORS = {
    "Gemma": "#c0392b",
    "Llama": "#1f77b4",
    "Qwen2.5": "#2ca02c",
    "Qwen3": "#9467bd",
    "SmolLM": "#ff7f0e",
    "Other": "#7f7f7f",
}


def _short_label(row: pd.Series) -> str:
    family = str(row["family"])
    scale = str(row["scale"])
    if family == "Qwen2.5":
        return f"Q2.5 {scale}"
    if family == "Qwen3":
        return f"Q3 {scale}"
    return f"{family} {scale}"


def _plot_temperature_slice(frame: pd.DataFrame, temperature: float, output_path: Path) -> None:
    subset = frame.loc[pd.to_numeric(frame["temperature"], errors="coerce") == temperature].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for _, row in subset.iterrows():
        x = float(row["observed_digit_js_to_uniform_bits"])
        y = float(row["observed_position_js_to_uniform_bits"])
        family = str(row["family"])
        color = FAMILY_COLORS.get(family, FAMILY_COLORS["Other"])
        size = 80 + 60 * float(row.get("ordering_diversity_count", 1))
        ax.scatter(x, y, s=size, color=color, alpha=0.9, edgecolors="white", linewidths=0.8)
        ax.annotate(
            _short_label(row),
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    max_value = max(
        subset["observed_digit_js_to_uniform_bits"].max(),
        subset["observed_position_js_to_uniform_bits"].max(),
    )
    limit = max(0.75, float(max_value) * 1.1)
    ax.plot([0, limit], [0, limit], linestyle="--", color="#555555", linewidth=1.5, label="position = digit")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Digit JS Divergence To Uniform (bits)")
    ax.set_ylabel("Position JS Divergence To Uniform (bits)")
    ax.set_title(f"Cross-Model Bias Mechanisms at temperature={temperature:.1f}")

    legend_handles = []
    for family, color in FAMILY_COLORS.items():
        if family not in set(subset["family"].astype(str)):
            continue
        legend_handles.append(plt.Line2D([0], [0], marker="o", color="w", label=family, markerfacecolor=color, markersize=8))
    legend_handles.append(plt.Line2D([0], [0], linestyle="--", color="#555555", label="position = digit"))
    ax.legend(handles=legend_handles, loc="best", fontsize=9)

    ax.text(limit * 0.72, limit * 0.12, "digit-leaning", fontsize=10, color="#333333")
    ax.text(limit * 0.12, limit * 0.72, "position-leaning", fontsize=10, color="#333333")
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_cross_model_plots(
    details_csv: str | Path = "results/processed/cross_model_temperature_details.csv",
    output_dir: str | Path = "results/processed/cross_model_figures",
) -> dict[str, Any]:
    details_path = Path(details_csv)
    frame = pd.read_csv(details_path)
    output_root = ensure_directory(Path(output_dir))

    outputs: list[str] = []
    for temperature, suffix in ((0.0, "0p0"), (0.2, "0p2")):
        output_path = output_root / f"cross_model_js_temp_{suffix}.png"
        _plot_temperature_slice(frame, temperature=temperature, output_path=output_path)
        if output_path.exists():
            outputs.append(str(output_path))

    return {
        "details_csv": str(details_path),
        "output_dir": str(output_root),
        "figure_paths": outputs,
        "figure_count": len(outputs),
    }
