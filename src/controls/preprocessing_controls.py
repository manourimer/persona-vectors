"""
Preprocessing and layer-robustness controls.

Checks whether conclusions depend on centering choice (raw vs centered)
and whether structure/reliability metrics are stable across layers.

Two scopes are analysed separately:

1. **Original ETHICS projections** (204 items, layer 32 only)
   Files: ethics_trait_projections_raw_wide.parquet
          ethics_trait_projections_centered_wide.parquet
   These are the projections used in Stage 4A structure analysis.

2. **Reliability-variant projections** (761 variants × 3 layers = 2283 rows)
   Files: reliability_trait_projections_wide_raw.parquet
          reliability_trait_projections_wide_centered.parquet
   Used for layer-robustness and variant-preprocessing comparisons.

Metrics are always computed **per-layer** (never pooled across layers).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.structure_analysis import (
    correlation_df as _corr_df,
    run_pca as _run_pca,
    PROJECTION_COLS,
    TRAIT_LABELS,
)

TRAITS = TRAIT_LABELS
PROJ_COLS = PROJECTION_COLS


# ---------------------------------------------------------------------------
# Internal helpers — delegate to structure_analysis for consistency
# ---------------------------------------------------------------------------


def _proj_matrix(df: pd.DataFrame) -> np.ndarray | None:
    available = [c for c in PROJ_COLS if c in df.columns]
    if len(available) < 2:
        return None
    return df[available].dropna().values


def _struct_metrics(df: pd.DataFrame) -> dict:
    mat = _proj_matrix(df)
    if mat is None or len(mat) < 4:
        return {
            "effective_dimensionality": float("nan"),
            "pc1_variance": float("nan"),
            "mean_abs_off_diag_corr": float("nan"),
            "max_abs_corr": float("nan"),
            "reliability_g1_proxy": float("nan"),
        }

    # Use structure_analysis PCA (eigendecomposition of correlation matrix)
    pca = _run_pca(mat, TRAITS[:mat.shape[1]], standardize_first=True)
    ed = pca.effective_dimensionality
    pc1 = float(pca.explained_variance_ratio[0])

    # Off-diagonal correlations via structure_analysis correlation_df
    corr = _corr_df(mat, TRAITS[:mat.shape[1]]).to_numpy()
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_abs = float(np.abs(corr[mask]).mean())
    max_abs = float(np.abs(corr[mask]).max())

    # Reliability proxy: between-column / total variance ratio
    col_means = mat.mean(axis=0)
    between = float(np.var(col_means, ddof=1)) if len(col_means) > 1 else 0.0
    total = float(np.var(mat, ddof=1)) if mat.size > 1 else 1.0
    g1_proxy = float(between / total) if total > 0 else 0.0

    return {
        "effective_dimensionality": ed,
        "pc1_variance": pc1,
        "mean_abs_off_diag_corr": mean_abs,
        "max_abs_corr": max_abs,
        "reliability_g1_proxy": g1_proxy,
    }


def _subset_by_layer(df: pd.DataFrame, layer: int) -> pd.DataFrame:
    """Return rows for a specific layer, or the whole df if no layer column."""
    if "layer" in df.columns:
        return df[df["layer"] == layer]
    return df


# ---------------------------------------------------------------------------
# Preprocessing comparison
# ---------------------------------------------------------------------------


def run_preprocessing_comparison(
    raw_wide_df: pd.DataFrame,
    centered_wide_df: pd.DataFrame,
    layers: list[int],
    traits: list[str],
    source_dataset: str = "ethics_original",
) -> pd.DataFrame:
    """
    For each layer: compute structure metrics on both raw and centered projections.

    Metrics are computed **per-layer** (never pooled).

    Parameters
    ----------
    raw_wide_df:
        Wide-format DataFrame with projection columns for raw projections.
        May or may not have a 'layer' column.
    centered_wide_df:
        Wide-format DataFrame with projection columns for centered projections.
        Must be a different object from raw_wide_df (checked by assertion).
    layers:
        List of layer IDs to iterate over.
    traits:
        Trait names (used for per-trait breakdown via 'primary_trait' column).
    source_dataset:
        Label written into the 'source_dataset' column of the output.
        Use 'ethics_original' for original ETHICS projections,
        'reliability_variants' for reliability-variant projections.

    Returns
    -------
    DataFrame with columns: source_dataset, layer, projected_trait, preprocessing,
    effective_dimensionality, pc1_variance, mean_abs_off_diag_corr, max_abs_corr,
    reliability_g1_proxy
    """
    assert raw_wide_df is not centered_wide_df, (
        "raw_wide_df and centered_wide_df must be different objects. "
        "Pass distinct DataFrames loaded from the raw and centered files."
    )

    rows = []
    for preprocessing, df in [("raw", raw_wide_df), ("centered", centered_wide_df)]:
        for layer in layers:
            sub = _subset_by_layer(df, layer)
            if sub.empty:
                continue
            metrics = _struct_metrics(sub)
            n_items = sub["item_id"].nunique() if "item_id" in sub.columns else len(sub)
            n_variants = (
                sub["variant_id"].nunique() if "variant_id" in sub.columns else None
            )
            rows.append({
                "source_dataset": source_dataset,
                "layer": layer,
                "projected_trait": "all",
                "preprocessing": preprocessing,
                "n_rows": len(sub),
                "n_items": n_items,
                "n_variants": n_variants,
                **metrics,
            })

            # Per-trait breakdown
            for trait in traits:
                pt_col = "primary_trait"
                if pt_col in sub.columns:
                    sub_trait = sub[sub[pt_col] == trait]
                else:
                    sub_trait = sub
                if sub_trait.empty:
                    continue
                t_metrics = _struct_metrics(sub_trait)
                rows.append({
                    "source_dataset": source_dataset,
                    "layer": layer,
                    "projected_trait": trait,
                    "preprocessing": preprocessing,
                    "n_rows": len(sub_trait),
                    "n_items": (
                        sub_trait["item_id"].nunique()
                        if "item_id" in sub_trait.columns else len(sub_trait)
                    ),
                    "n_variants": (
                        sub_trait["variant_id"].nunique()
                        if "variant_id" in sub_trait.columns else None
                    ),
                    **t_metrics,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer robustness
# ---------------------------------------------------------------------------


def run_layer_robustness(
    centered_wide_df: pd.DataFrame,
    layers: list[int],
    traits: list[str],
    source_dataset: str = "reliability_variants",
) -> pd.DataFrame:
    """
    Compare structure and reliability metrics across layers 32, 40, 47.

    Metrics are computed **per-layer** (never pooled).

    Parameters
    ----------
    centered_wide_df:
        Wide-format centered projections.  Must have a 'layer' column for
        multi-layer analysis (reliability variants file has this).
    layers:
        Layer IDs to iterate over.
    traits:
        Trait names for per-trait breakdown.
    source_dataset:
        Label written into the 'source_dataset' column of the output.

    Returns
    -------
    DataFrame with columns: source_dataset, layer, projected_trait,
    effective_dimensionality, pc1_variance, mean_abs_off_diag_corr,
    max_abs_corr, reliability_g1_proxy
    """
    rows = []
    for layer in layers:
        sub = _subset_by_layer(centered_wide_df, layer)
        if sub.empty:
            continue
        metrics = _struct_metrics(sub)
        n_items = sub["item_id"].nunique() if "item_id" in sub.columns else len(sub)
        rows.append({
            "source_dataset": source_dataset,
            "layer": layer,
            "projected_trait": "all",
            "n_rows": len(sub),
            "n_items": n_items,
            **metrics,
        })

        for trait in traits:
            pt_col = "primary_trait"
            if pt_col in sub.columns:
                sub_trait = sub[sub[pt_col] == trait]
            else:
                sub_trait = sub
            if sub_trait.empty:
                continue
            t_metrics = _struct_metrics(sub_trait)
            rows.append({
                "source_dataset": source_dataset,
                "layer": layer,
                "projected_trait": trait,
                "n_rows": len(sub_trait),
                "n_items": (
                    sub_trait["item_id"].nunique()
                    if "item_id" in sub_trait.columns else len(sub_trait)
                ),
                **t_metrics,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_preprocessing_controls(results_dict: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "preprocessing_df" in results_dict:
        results_dict["preprocessing_df"].to_csv(
            out_dir / "preprocessing_robustness_summary.csv", index=False
        )
    if "layer_robustness_df" in results_dict:
        results_dict["layer_robustness_df"].to_csv(
            out_dir / "layer_robustness_summary.csv", index=False
        )
    print(f"[preprocessing_controls] Saved to {out_dir}")
