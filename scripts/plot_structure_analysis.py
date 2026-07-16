"""
Stage 4A — Visualization of projection structure analysis.

Generates per-layer figures:
  1. Correlation heatmap
  2. PCA scree plot (observed vs parallel-analysis random baseline)
  3. 2D PCA scatter colored by primary_trait

Requires matplotlib. If not installed, the script exits with a clear message.

Output: outputs/structure_analysis/figures/

Usage:
    python scripts/plot_structure_analysis.py
    python scripts/plot_structure_analysis.py --layers 32 40
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_LONG_PATH = "outputs/ethics_projection/ethics_trait_projections_centered_long.parquet"
_WIDE_PATH = "outputs/ethics_projection/ethics_trait_projections_centered_wide.parquet"
_ANALYSIS_DIR = "outputs/structure_analysis"
_DEFAULT_LAYERS = [32, 40, 47]
_PRIMARY_LAYER = 32
_DOWNSTREAM_BEST = 40

_TRAIT_COLORS = {
    "honesty": "#2166ac",
    "harmlessness": "#d6604d",
    "fairness": "#4dac26",
    "compassion": "#8e44ad",
}


def _check_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "\n  matplotlib is not installed.\n"
            "  Install with: pip install matplotlib\n"
            "  Skipping visualization.\n"
        )
        sys.exit(0)


def _save_correlation_heatmap(
    corr_df: "pd.DataFrame", layer: int, out_dir: Path
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(corr_df.columns)
    mat = corr_df.to_numpy()
    n = len(labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(mat[i, j]) < 0.7 else "white")
    ax.set_title(f"Trait projection correlations — Layer {layer}", fontsize=11)
    fig.tight_layout()
    path = out_dir / f"correlation_heatmap_layer{layer}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_scree_plot(summary: object, layer: int, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    pca = summary.pca
    pa = summary.parallel
    p = pca.n_variables
    xs = list(range(1, p + 1))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, pca.eigenvalues, "bo-", label="Observed", zorder=3)
    ax.plot(xs, pa.random_eigenvalue_95th, "r--", label="Random 95th pct")
    ax.plot(xs, pa.random_eigenvalue_mean, "r:", alpha=0.5, label="Random mean")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="Kaiser threshold")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"PC{k}" for k in xs])
    role = ""
    if layer == _PRIMARY_LAYER:
        role = " [contrast-selected]"
    elif layer == _DOWNSTREAM_BEST:
        role = " [downstream best]"
    ax.set_title(f"Scree plot — Layer {layer}{role}", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / f"scree_plot_layer{layer}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_pca_scatter(
    summary: object, wide_df: "pd.DataFrame", layer: int, out_dir: Path
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    scores = summary.pca.scores
    pca = summary.pca
    traits = wide_df["primary_trait"].tolist() if "primary_trait" in wide_df.columns else []

    fig, ax = plt.subplots(figsize=(6, 5))
    unique_traits = sorted(set(traits)) if traits else []
    if traits:
        for trait in unique_traits:
            idx = [i for i, t in enumerate(traits) if t == trait]
            color = _TRAIT_COLORS.get(trait, "#888888")
            ax.scatter(
                scores[idx, 0], scores[idx, 1],
                c=color, label=trait, alpha=0.65, s=30, edgecolors="none",
            )
        ax.legend(fontsize=8, markerscale=1.2, loc="best")
    else:
        ax.scatter(scores[:, 0], scores[:, 1], alpha=0.5, s=30)

    ev1 = pca.explained_variance_ratio[0]
    ev2 = pca.explained_variance_ratio[1]
    ax.set_xlabel(f"PC1 ({ev1:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({ev2:.1%})", fontsize=10)
    role = ""
    if layer == _PRIMARY_LAYER:
        role = " [contrast-selected]"
    elif layer == _DOWNSTREAM_BEST:
        role = " [downstream best]"
    ax.set_title(f"PCA item scores — Layer {layer}{role}", fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    path = out_dir / f"pca_scatter_layer{layer}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    _check_matplotlib()

    parser = argparse.ArgumentParser(
        description="Stage 4A: Plot projection structure analysis."
    )
    parser.add_argument("--long-path", default=_LONG_PATH)
    parser.add_argument("--wide-path", default=_WIDE_PATH)
    parser.add_argument("--analysis-dir", default=_ANALYSIS_DIR)
    parser.add_argument("--layers", nargs="+", type=int, default=_DEFAULT_LAYERS)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    from src.analysis.structure_analysis import (
        build_layer_wide_tables,
        load_centered_long,
        load_centered_wide,
        run_structure_analysis,
    )

    long_path = _ROOT / args.long_path
    wide_path = _ROOT / args.wide_path
    fig_dir = _ROOT / args.analysis_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not long_path.exists():
        print(f"  ERROR: {long_path} not found. Run run_structure_analysis.py first.")
        sys.exit(1)

    print("\n  ── Stage 4A: Plotting structure analysis ────────────────────────────\n")

    summaries = run_structure_analysis(
        long_path=long_path,
        wide_path_layer32=wide_path,
        layers=args.layers,
        run_pa=True,
        run_fa=False,
        random_seed=args.random_seed,
    )

    long_df = load_centered_long(long_path)
    layer_wide_tables = build_layer_wide_tables(long_df, args.layers)
    try:
        wide32 = load_centered_wide(wide_path)
        layer_wide_tables[32] = wide32
    except Exception:
        pass

    saved: list[Path] = []
    for layer in args.layers:
        if layer not in summaries:
            continue
        s = summaries[layer]
        wide_df = layer_wide_tables.get(layer)

        saved.append(_save_correlation_heatmap(s.corr_df, layer, fig_dir))
        saved.append(_save_scree_plot(s, layer, fig_dir))
        if wide_df is not None:
            saved.append(_save_pca_scatter(s, wide_df, layer, fig_dir))

    print(f"  Figures saved to {fig_dir.relative_to(_ROOT)}/")
    for p in saved:
        print(f"    {p.name}")


if __name__ == "__main__":
    main()
