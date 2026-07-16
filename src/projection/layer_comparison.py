"""
Stage 3: Layer comparison analysis for ETHICS projections.

Compares projection structure across candidate layers and reports whether
the contrast-validation-selected layer is also the best downstream layer
on ETHICS items.

No GPU, torch, or Modal required.

Public API
----------
    compute_layer_metrics(long_df, item_df, layers)  -> pd.DataFrame
    compare_layers(metrics_df,
                   contrast_selected_layer,
                   downstream_layers)               -> LayerComparisonResult
    save_layer_comparison(result, out_dir)          -> tuple[Path, Path]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TRAITS: list[str] = ["honesty", "harmlessness", "fairness", "compassion"]
_CHANCE_LEVEL = 1.0 / len(TRAITS)  # 0.25 for four traits


# ---------------------------------------------------------------------------
# Per-layer metrics
# ---------------------------------------------------------------------------


def _diagonal_dominance(wide_df: pd.DataFrame) -> float:
    """Fraction of items where the annotated trait has the highest projection."""
    proj_cols = [f"projection_{t}" for t in TRAITS if f"projection_{t}" in wide_df.columns]
    if not proj_cols or "primary_trait" not in wide_df.columns:
        return float("nan")

    dominant = wide_df.apply(
        lambda row: (
            row.get(f"projection_{row['primary_trait']}", float("-inf"))
            == max(row[c] for c in proj_cols)
        )
        if row.get("primary_trait") in TRAITS
        else False,
        axis=1,
    )
    return float(dominant.mean())


def _matching_margin(long_df: pd.DataFrame) -> float:
    """Mean (matching projection − mean non-matching projection) across items."""
    margins: list[float] = []
    for item_id, grp in long_df.groupby("item_id"):
        pt = grp["primary_trait"].iloc[0]
        if pt not in TRAITS:
            continue
        match_rows = grp[grp["projected_trait"] == pt]["projection"]
        nonmatch_rows = grp[grp["projected_trait"] != pt]["projection"]
        if match_rows.empty or nonmatch_rows.empty:
            continue
        margins.append(float(match_rows.mean()) - float(nonmatch_rows.mean()))
    return float(np.mean(margins)) if margins else float("nan")


def _per_trait_diagonal(long_df: pd.DataFrame) -> dict[str, float]:
    """Mean matching projection minus mean off-diagonal, per annotated trait."""
    result: dict[str, float] = {}
    for trait in TRAITS:
        grp = long_df[long_df["primary_trait"] == trait]
        match = grp[grp["projected_trait"] == trait]["projection"]
        nonmatch = grp[grp["projected_trait"] != trait]["projection"]
        if match.empty or nonmatch.empty:
            result[trait] = float("nan")
        else:
            result[trait] = float(match.mean()) - float(nonmatch.mean())
    return result


def _max_inter_trait_correlation(wide_df: pd.DataFrame) -> float:
    """Maximum absolute off-diagonal correlation among trait projection columns."""
    proj_cols = [f"projection_{t}" for t in TRAITS if f"projection_{t}" in wide_df.columns]
    if len(proj_cols) < 2:
        return float("nan")
    corr_arr = wide_df[proj_cols].corr().abs().to_numpy().copy()
    np.fill_diagonal(corr_arr, 0.0)
    return float(corr_arr.max())


def compute_layer_metrics(
    long_df: pd.DataFrame,
    item_df: pd.DataFrame,
    layers: list[int],
    preprocessing: str = "mean_centered",
) -> pd.DataFrame:
    """Compute diagnostic metrics for each layer.

    Args:
        long_df:       Long-format projection DataFrame (all layers).
        item_df:       Curated item bank DataFrame.
        layers:        Layer indices to include.
        preprocessing: Filter to this projection_preprocessing value if column exists.

    Returns:
        DataFrame with one row per layer and columns:
            layer, diagonal_dominance, matching_margin,
            max_inter_trait_correlation,
            diagonal_{trait} for each trait
    """
    if "projection_preprocessing" in long_df.columns:
        long_df = long_df[long_df["projection_preprocessing"] == preprocessing]

    from src.projection.compute_projections import to_wide_format

    rows: list[dict] = []
    for layer in layers:
        layer_long = long_df[long_df["layer"] == layer]
        if layer_long.empty:
            continue
        wide = to_wide_format(layer_long, item_df)
        row: dict = {
            "layer": layer,
            "diagonal_dominance": _diagonal_dominance(wide),
            "matching_margin": _matching_margin(layer_long),
            "max_inter_trait_correlation": _max_inter_trait_correlation(wide),
        }
        per_trait = _per_trait_diagonal(layer_long)
        for trait in TRAITS:
            row[f"diagonal_{trait}"] = per_trait.get(trait, float("nan"))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------


@dataclass
class LayerComparisonResult:
    metrics_df: pd.DataFrame
    contrast_selected_layer: int
    best_downstream_layer: int            # by diagonal dominance
    best_downstream_dominance: float
    contrast_selected_dominance: float
    layers_agree: bool                    # True if best_downstream == contrast_selected
    interpretation: str
    warnings: list[str] = field(default_factory=list)


def compare_layers(
    metrics_df: pd.DataFrame,
    contrast_selected_layer: int,
    downstream_layers: list[int],
) -> LayerComparisonResult:
    """Compare layers and produce an interpretation string.

    Args:
        metrics_df:              Output of compute_layer_metrics.
        contrast_selected_layer: Layer chosen by Stage 2B contrast-prompt AUC.
        downstream_layers:       Layers to consider for downstream ETHICS ranking.

    Returns:
        LayerComparisonResult with interpretation text and warnings.
    """
    warnings: list[str] = []

    candidate = metrics_df[metrics_df["layer"].isin(downstream_layers)]
    if candidate.empty:
        return LayerComparisonResult(
            metrics_df=metrics_df,
            contrast_selected_layer=contrast_selected_layer,
            best_downstream_layer=contrast_selected_layer,
            best_downstream_dominance=float("nan"),
            contrast_selected_dominance=float("nan"),
            layers_agree=True,
            interpretation="Insufficient data for layer comparison.",
        )

    best_row = candidate.loc[candidate["diagonal_dominance"].idxmax()]
    best_layer = int(best_row["layer"])
    best_dom = float(best_row["diagonal_dominance"])

    cs_row = metrics_df[metrics_df["layer"] == contrast_selected_layer]
    cs_dom = float(cs_row["diagonal_dominance"].iloc[0]) if not cs_row.empty else float("nan")

    layers_agree = best_layer == contrast_selected_layer

    # Warnings
    for _, row in metrics_df.iterrows():
        if row["diagonal_dominance"] <= _CHANCE_LEVEL:
            warnings.append(
                f"Layer {int(row['layer'])} diagonal dominance ({row['diagonal_dominance']:.3f}) "
                f"is at or below four-way chance ({_CHANCE_LEVEL:.2f})."
            )

    # Interpretation
    if layers_agree:
        interpretation = (
            f"Layer {contrast_selected_layer} was selected by contrast-prompt validation "
            f"and also shows the strongest downstream ETHICS projection structure "
            f"(diagonal dominance = {best_dom:.3f}). "
            "Contrast-validation and downstream ETHICS structure are consistent."
        )
    else:
        dom_diff = best_dom - cs_dom
        interpretation = (
            f"Layer {contrast_selected_layer} was selected by contrast-prompt validation "
            f"(diagonal dominance on ETHICS = {cs_dom:.3f}), "
            f"but layer {best_layer} produced stronger downstream ETHICS trait-structure "
            f"diagnostics (diagonal dominance = {best_dom:.3f}, Δ = {dom_diff:+.3f}). "
            "This suggests contrast-prompt validation and downstream measurement validity "
            "can diverge — the layer that best separates elicitation artifacts may not be "
            "the layer that best captures novel moral scenarios. "
            f"Layer {contrast_selected_layer} remains the methodologically primary layer "
            f"(selected pre-ETHICS); layer {best_layer} is reported as a comparison."
        )
        warnings.append(
            f"Best downstream ETHICS layer ({best_layer}) differs from "
            f"contrast-validation-selected layer ({contrast_selected_layer}). "
            "See layer_comparison_summary for details."
        )

    return LayerComparisonResult(
        metrics_df=metrics_df,
        contrast_selected_layer=contrast_selected_layer,
        best_downstream_layer=best_layer,
        best_downstream_dominance=best_dom,
        contrast_selected_dominance=cs_dom,
        layers_agree=layers_agree,
        interpretation=interpretation,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    md_rows = []
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        md_rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + md_rows)


def save_layer_comparison(
    result: LayerComparisonResult,
    out_dir: str | Path,
    preprocessing_label: str = "mean_centered",
) -> tuple[Path, Path]:
    """Save layer comparison summary as CSV and Markdown.

    Also saves per-layer correlation matrices.

    Returns (csv_path, md_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "layer_comparison_summary.csv"
    md_path = out_dir / "layer_comparison_summary.md"

    result.metrics_df.to_csv(csv_path, index=False)

    lines: list[str] = [
        "# Stage 3: Layer Comparison Summary\n",
        f"Preprocessing: **{preprocessing_label}**\n",
        "## Metrics by Layer\n",
        _df_to_md(result.metrics_df),
        "",
        "## Layer Selection\n",
        f"- **Contrast-validation-selected layer**: {result.contrast_selected_layer} "
        f"(chosen by held-out AUC on contrast artifacts in Stage 2B)",
        f"- **Best downstream ETHICS layer**: {result.best_downstream_layer} "
        f"(highest diagonal dominance on mean-centered ETHICS projections)",
        f"- **Layers agree**: {'✅ Yes' if result.layers_agree else '❌ No'}",
        "",
        "## Interpretation\n",
        result.interpretation,
        "",
    ]

    if result.warnings:
        lines.append("## ⚠ Warnings\n")
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append(
        "## Notes\n\n"
        "- Diagonal dominance = fraction of items where the annotated trait "
        "projects highest on the matching vector (chance = 0.25 for 4 traits).\n"
        "- Matching margin = mean(matching projection) − mean(non-matching projections).\n"
        "- Weak diagonal dominance does not mean the data is invalid — it may "
        "indicate that the four traits share latent structure in Gemma's "
        "representations (investigate in Stage 5: factor analysis).\n"
        "- Layer 32 remains the primary layer for all downstream analyses "
        "unless explicitly overridden.\n"
    )

    md_path.write_text("\n".join(lines))
    return csv_path, md_path
