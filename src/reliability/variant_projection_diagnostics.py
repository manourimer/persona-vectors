"""
Stage 4C: Diagnostics for reliability variant projections.

Pure Python (pandas + numpy).  No GPU, torch, Modal, or transformers.

Public API
----------
    compute_diagnostics(long_df, wide_df)   -> dict
    generate_report(diag_dict)              -> str (Markdown)
    save_diagnostics(diag_dict, wide_df, out_dir) -> None
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRAITS: list[str] = ["honesty", "harmlessness", "fairness", "compassion"]
PROJ_COLS = [f"projection_{t}" for t in TRAITS]


# ---------------------------------------------------------------------------
# Core diagnostics
# ---------------------------------------------------------------------------


def compute_diagnostics(long_df: pd.DataFrame, wide_df: pd.DataFrame) -> dict:
    """Compute comprehensive diagnostics for reliability variant projections.

    Args:
        long_df: Long-format projection DataFrame (centered preferred).
        wide_df: Wide-format projection DataFrame (one row per variant × layer).

    Returns:
        dict with diagnostic fields and a "warnings" list of triggered strings.
    """
    warnings: list[str] = []

    # Basic counts
    n_items = long_df["item_id"].nunique() if "item_id" in long_df.columns else 0
    n_variants = long_df["variant_id"].nunique() if "variant_id" in long_df.columns else 0

    variant_types = long_df["variant_type"].unique().tolist() if "variant_type" in long_df.columns else []
    n_originals = int((long_df["variant_type"] == "original").sum() // max(long_df["projected_trait"].nunique(), 1)) if "variant_type" in long_df.columns else 0
    n_paraphrases = int((long_df["variant_type"] == "paraphrase").sum() // max(long_df["projected_trait"].nunique(), 1)) if "variant_type" in long_df.columns else 0

    # Variants per item
    variants_per_item = (
        long_df.drop_duplicates(["item_id", "variant_id"])
        .groupby("item_id")["variant_id"]
        .count()
    )
    expected_per_item = variants_per_item.median()
    missing_variants = variants_per_item[
        variants_per_item < expected_per_item
    ].index.tolist()

    # Count breakdowns (from long_df unique variants)
    unique_variants = long_df.drop_duplicates(["item_id", "variant_id"])
    counts_by_variant_type = unique_variants["variant_type"].value_counts().to_dict() if "variant_type" in unique_variants.columns else {}
    counts_by_paraphrase_id = unique_variants["paraphrase_id"].value_counts().to_dict() if "paraphrase_id" in unique_variants.columns else {}
    counts_by_framing = unique_variants["framing"].value_counts().to_dict() if "framing" in unique_variants.columns else {}
    counts_by_primary_trait = unique_variants["primary_trait"].value_counts().to_dict() if "primary_trait" in unique_variants.columns else {}

    # Projection statistics (mean/std/min/max) per projected_trait × layer
    proj_stats = (
        long_df.groupby(["projected_trait", "layer"])["projection"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .to_dict(orient="records")
    )

    # Correlation matrices per layer
    correlation_matrices: dict[int, pd.DataFrame] = {}
    proj_cols_present = [c for c in PROJ_COLS if c in wide_df.columns]
    if proj_cols_present and "layer" in wide_df.columns:
        for layer in sorted(wide_df["layer"].unique()):
            sub = wide_df[wide_df["layer"] == layer][proj_cols_present].dropna()
            if len(sub) >= 3:
                correlation_matrices[int(layer)] = sub.corr()

    # Mean projection by primary_trait × projected_trait × layer
    if "primary_trait" in long_df.columns:
        mean_proj = (
            long_df.groupby(["primary_trait", "projected_trait", "layer"])["projection"]
            .mean()
            .reset_index()
            .rename(columns={"projection": "mean_projection"})
        )
        mean_projection_by_trait_x_layer = mean_proj
    else:
        mean_projection_by_trait_x_layer = pd.DataFrame()

    # Within-item std across variants, per projected_trait × layer
    within_item_std = (
        long_df.groupby(["item_id", "projected_trait", "layer"])["projection"]
        .std()
        .reset_index()
        .rename(columns={"projection": "within_item_std"})
    )

    # --- Warnings ---

    # 1. Mean within-item std of paraphrases vs std of item means
    paraphrase_long = long_df[long_df["variant_type"] == "paraphrase"] if "variant_type" in long_df.columns else pd.DataFrame()
    item_means = long_df.groupby(["item_id", "projected_trait", "layer"])["projection"].mean()
    within_std_mean = within_item_std["within_item_std"].mean()
    item_means_std = item_means.std()
    if within_std_mean > item_means_std:
        warnings.append(
            f"Mean within-item paraphrase std ({within_std_mean:.4f}) > "
            f"std of item means ({item_means_std:.4f}): paraphrases shift "
            "projections as much as items differ."
        )

    # 2. Any trait has per-item std > 2× median per-item std
    for trait in TRAITS:
        sub = within_item_std[within_item_std["projected_trait"] == trait]
        if sub.empty:
            continue
        median_std = sub["within_item_std"].median()
        if median_std == 0:
            continue
        unstable = sub[sub["within_item_std"] > 2 * median_std]
        if not unstable.empty:
            warnings.append(
                f"Trait '{trait}' has {len(unstable)} item(s) with within-item std "
                f"> 2× median ({2*median_std:.4f}) — potentially unstable."
            )

    # 3. Near-constant centered projection column
    for col in proj_cols_present:
        if wide_df[col].std(skipna=True) < 0.01:
            warnings.append(
                f"Column {col} has std < 0.01 (near-constant) — "
                "possible centering or vector issue."
            )

    # 4. Single trait captures >60% of total projection variance
    total_var = sum(wide_df[c].var(skipna=True) for c in proj_cols_present)
    if total_var > 0:
        for col in proj_cols_present:
            var_frac = wide_df[col].var(skipna=True) / total_var
            if var_frac > 0.60:
                warnings.append(
                    f"Column {col} captures {var_frac:.1%} of total projection "
                    "variance (>60%) — dominant vector concern."
                )

    return {
        "n_items": n_items,
        "n_variants": n_variants,
        "n_originals": n_originals,
        "n_paraphrases": n_paraphrases,
        "variants_per_item": variants_per_item,
        "missing_variants": missing_variants,
        "counts_by_variant_type": counts_by_variant_type,
        "counts_by_paraphrase_id": counts_by_paraphrase_id,
        "counts_by_framing": counts_by_framing,
        "counts_by_primary_trait": counts_by_primary_trait,
        "projection_stats": proj_stats,
        "correlation_matrices": correlation_matrices,
        "mean_projection_by_trait_x_layer": mean_projection_by_trait_x_layer,
        "within_item_std": within_item_std,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def generate_report(diag_dict: dict) -> str:
    """Generate a Markdown diagnostic report from compute_diagnostics output."""
    lines: list[str] = [
        "# Stage 4C: Reliability Variant Projection Diagnostics\n",
        "## Summary\n",
        f"- Items: {diag_dict['n_items']}",
        f"- Total variants: {diag_dict['n_variants']}",
        f"- Originals: {diag_dict['n_originals']}",
        f"- Paraphrases: {diag_dict['n_paraphrases']}",
        f"- Missing variants (below median): {len(diag_dict['missing_variants'])}",
        "",
        "## Count Breakdowns\n",
        "**By variant type:**",
        *[f"  - {k}: {v}" for k, v in diag_dict["counts_by_variant_type"].items()],
        "",
        "**By paraphrase_id:**",
        *[f"  - {k}: {v}" for k, v in diag_dict["counts_by_paraphrase_id"].items()],
        "",
        "**By framing:**",
        *[f"  - {k}: {v}" for k, v in diag_dict["counts_by_framing"].items()],
        "",
        "**By primary_trait:**",
        *[f"  - {k}: {v}" for k, v in diag_dict["counts_by_primary_trait"].items()],
        "",
        "## Projection Statistics (mean/std/min/max by trait × layer)\n",
    ]

    for stat_row in diag_dict["projection_stats"]:
        lines.append(
            f"  - {stat_row['projected_trait']} layer {stat_row['layer']}: "
            f"mean={stat_row['mean']:.4f}, std={stat_row['std']:.4f}, "
            f"min={stat_row['min']:.4f}, max={stat_row['max']:.4f}"
        )

    lines += ["", "## Correlation Matrices\n"]
    for layer, corr_df in diag_dict["correlation_matrices"].items():
        lines.append(f"### Layer {layer}\n")
        lines.append(corr_df.round(3).to_markdown() if hasattr(corr_df, "to_markdown") else str(corr_df.round(3)))
        lines.append("")

    lines += ["", "## Warnings\n"]
    warnings = diag_dict["warnings"]
    if not warnings:
        lines.append("No warnings triggered.")
    else:
        for w in warnings:
            lines.append(f"- {w}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_diagnostics(
    diag_dict: dict,
    wide_df: pd.DataFrame,
    out_dir: str | Path,
) -> None:
    """Save diagnostics report, summary CSV, and per-layer correlation matrices.

    Saves:
        out_dir/reliability_projection_diagnostics.md
        out_dir/reliability_projection_summary.csv
        out_dir/reliability_projection_corr_layer{N}.csv  (one per layer)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    report_md = generate_report(diag_dict)
    md_path = out_dir / "reliability_projection_diagnostics.md"
    md_path.write_text(report_md, encoding="utf-8")

    # Summary CSV from projection_stats
    if diag_dict["projection_stats"]:
        summary_df = pd.DataFrame(diag_dict["projection_stats"])
        summary_df.to_csv(out_dir / "reliability_projection_summary.csv", index=False)

    # Per-layer correlation matrix CSVs
    for layer, corr_df in diag_dict["correlation_matrices"].items():
        corr_path = out_dir / f"reliability_projection_corr_layer{layer}.csv"
        corr_df.to_csv(corr_path)
