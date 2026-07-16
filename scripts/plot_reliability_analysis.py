"""Stage 4D: Reliability analysis plots."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "reliability_analysis"
FIGURES_DIR = OUT_DIR / "figures"


def plot_reliability_by_trait_layer(results_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: x=projected_trait, bars=layers, y=reliability_1."""
    traits = results_df["projected_trait"].unique().tolist()
    layers = sorted(results_df["layer"].unique().tolist())
    n_layers = len(layers)
    x = range(len(traits))
    width = 0.7 / n_layers

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, layer in enumerate(layers):
        vals = []
        for trait in traits:
            row = results_df[
                (results_df["layer"] == layer) & (results_df["projected_trait"] == trait)
            ]
            vals.append(row["reliability_1"].values[0] if not row.empty else 0.0)
        offsets = [xi + (i - n_layers / 2 + 0.5) * width for xi in x]
        ax.bar(offsets, vals, width=width * 0.9, label=f"Layer {layer}")

    ax.set_xticks(list(x))
    ax.set_xticklabels(traits, rotation=15, ha="right")
    ax.set_ylabel("Reliability (k=1)")
    ax.set_title("Reliability by Trait and Layer (k=1 variant)")
    ax.set_ylim(0, 1)
    ax.axhline(0.70, color="gray", linestyle="--", linewidth=0.8, label="0.70 threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_d_study_curves(d_study_df: pd.DataFrame, out_path: Path) -> None:
    """Line plot per projected_trait, x=n_paraphrases, y=g_coefficient, subplots per layer."""
    layers = sorted(d_study_df["layer"].unique().tolist())
    traits = d_study_df["projected_trait"].unique().tolist()
    n_layers = len(layers)

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        layer_df = d_study_df[d_study_df["layer"] == layer]
        for trait in traits:
            tr = layer_df[layer_df["projected_trait"] == trait]
            ax.plot(tr["n_paraphrases"], tr["g_coefficient"], marker="o", label=trait)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("# Paraphrases (k)")
        ax.set_ylabel("G-coefficient")
        ax.set_ylim(0, 1)
        ax.axhline(0.70, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=8)

    fig.suptitle("D-Study: G-Coefficient vs. Number of Paraphrases")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_variance_decomposition(results_df: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar chart: between vs within variance per (projected_trait, layer)."""
    labels = [
        f"{row['projected_trait']}\nL{row['layer']}"
        for _, row in results_df.iterrows()
    ]
    between = results_df["between_item_var"].values
    within = results_df["within_item_var"].values

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    x = range(len(labels))
    ax.bar(x, between, label="Between-item variance")
    ax.bar(x, within, bottom=between, label="Within-item variance")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Variance")
    ax.set_title("Variance Decomposition: Between vs Within Item")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main() -> None:
    if not HAS_MATPLOTLIB:
        print("[plot_reliability_analysis] matplotlib not installed; skipping plots.")
        return

    summary_path = OUT_DIR / "reliability_summary.csv"
    d_study_path = OUT_DIR / "d_study_results.csv"

    if not summary_path.exists():
        print(f"[plot] {summary_path} not found. Run run_reliability_analysis.py first.")
        return
    if not d_study_path.exists():
        print(f"[plot] {d_study_path} not found. Run run_reliability_analysis.py first.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(summary_path)
    d_study_df = pd.read_csv(d_study_path)

    plot_reliability_by_trait_layer(results_df, FIGURES_DIR / "reliability_by_trait_layer.png")
    plot_d_study_curves(d_study_df, FIGURES_DIR / "d_study_curves.png")
    plot_variance_decomposition(results_df, FIGURES_DIR / "variance_decomposition.png")


if __name__ == "__main__":
    main()
