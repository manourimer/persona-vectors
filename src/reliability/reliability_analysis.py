"""
Stage 4D: Reliability / Generalizability Analysis.

Estimates how stable persona-vector projections are across paraphrase variants
of the same item, using one-way ANOVA variance decomposition.

object of measurement : item_id
incidental facet      : paraphrase_id (different wordings of the same scenario)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "item_id",
    "variant_id",
    "paraphrase_id",
    "framing",
    "primary_trait",
    "projected_trait",
    "layer",
    "projection",
]


def load_projection_table(path: str | Path) -> pd.DataFrame:
    """Load parquet projection table and validate required columns."""
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Projection table is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def filter_table(
    df: pd.DataFrame,
    layers: list[int],
    projected_traits: list[str],
    min_variants_per_item: int = 2,
) -> pd.DataFrame:
    """Filter to specified layers and traits; drop items with too few variants."""
    df = df[df["layer"].isin(layers) & df["projected_trait"].isin(projected_traits)].copy()

    # Count variants per (layer, projected_trait, item_id)
    counts = (
        df.groupby(["layer", "projected_trait", "item_id"])["variant_id"]
        .nunique()
        .reset_index(name="n_variants")
    )
    # Keep item_ids that meet min threshold in ALL (layer, projected_trait) groups
    too_few = counts[counts["n_variants"] < min_variants_per_item]["item_id"].unique()
    n_dropped = len(too_few)
    if n_dropped > 0:
        print(f"[filter_table] Dropping {n_dropped} items with fewer than "
              f"{min_variants_per_item} variants in at least one layer/trait group.")
    else:
        print(f"[filter_table] All items meet the minimum {min_variants_per_item} variants threshold.")

    df = df[~df["item_id"].isin(too_few)].copy()
    return df


@dataclass
class VarianceComponents:
    between_item_var: float
    within_item_var: float
    total_var: float
    n_items: int
    n_variants_per_item_mean: float
    n_variants_per_item_min: int
    clamped_negative: bool = field(default=False)


def estimate_variance_components(df_group: pd.DataFrame) -> VarianceComponents:
    """
    One-way ANOVA variance decomposition for a single (layer, projected_trait) group.

    df_group must have columns: item_id, projection
    """
    items = df_group["item_id"].values
    projections = df_group["projection"].values

    unique_items = np.unique(items)
    I = len(unique_items)
    N = len(projections)

    if I < 2:
        warnings.warn("Cannot estimate variance components with fewer than 2 items.")
        return VarianceComponents(
            between_item_var=0.0,
            within_item_var=0.0,
            total_var=0.0,
            n_items=I,
            n_variants_per_item_mean=float(N),
            n_variants_per_item_min=N,
            clamped_negative=False,
        )

    grand_mean = projections.mean()

    ss_between = 0.0
    ss_within = 0.0
    n_per_item = []
    sum_ni_sq = 0.0

    for item in unique_items:
        mask = items == item
        yi = projections[mask]
        ni = len(yi)
        yi_mean = yi.mean()
        n_per_item.append(ni)
        ss_between += ni * (yi_mean - grand_mean) ** 2
        ss_within += np.sum((yi - yi_mean) ** 2)
        sum_ni_sq += ni ** 2

    df_between = I - 1
    df_within = N - I

    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    # Effective n per item for unbalanced designs
    n0 = (N - sum_ni_sq / N) / (I - 1) if I > 1 else float(N)

    raw_between = (ms_between - ms_within) / n0 if n0 > 0 else 0.0
    clamped = raw_between < 0
    if clamped:
        warnings.warn(
            "[estimate_variance_components] Negative between-item variance estimate "
            f"({raw_between:.4f}); clamping to 0. This can happen when within-item "
            "noise dominates or when the design is severely unbalanced."
        )
    between_item_var = max(0.0, raw_between)
    within_item_var = ms_within
    total_var = between_item_var + within_item_var

    return VarianceComponents(
        between_item_var=float(between_item_var),
        within_item_var=float(within_item_var),
        total_var=float(total_var),
        n_items=int(I),
        n_variants_per_item_mean=float(np.mean(n_per_item)),
        n_variants_per_item_min=int(np.min(n_per_item)),
        clamped_negative=bool(clamped),
    )


@dataclass
class ReliabilityResult:
    layer: int
    projected_trait: str
    variance_components: VarianceComponents
    reliability_single: float
    reliability_k: dict  # int -> float
    within_item_sd: float
    between_item_sd: float
    variance_ratio: float
    item_means: pd.Series
    item_sds: pd.Series
    n_items_used: int


def compute_reliability(vc: VarianceComponents, k_values: list[int]) -> dict:
    """
    Compute reliability (ICC/G) for a range of k values.

    reliability_single = between / (between + within)  [k=1]
    reliability_k      = between / (between + within/k)
    """
    total = vc.between_item_var + vc.within_item_var
    if total == 0.0:
        warnings.warn("[compute_reliability] Both variance components are zero; returning 0.0.")
        return {k: 0.0 for k in k_values}

    result = {}
    for k in k_values:
        denom = vc.between_item_var + vc.within_item_var / k
        rel = vc.between_item_var / denom if denom > 0 else 0.0
        rel = float(np.clip(rel, 0.0, 1.0))
        result[k] = rel
    return result


def run_reliability_analysis(
    long_df: pd.DataFrame,
    layers: list[int],
    projected_traits: list[str],
    min_variants_per_item: int = 2,
    k_values: list[int] = None,
) -> list[ReliabilityResult]:
    """Group by (layer, projected_trait) and compute reliability for each."""
    if k_values is None:
        k_values = [1, 2, 3, 4, 5]

    results = []
    for layer in layers:
        for trait in projected_traits:
            subset = long_df[
                (long_df["layer"] == layer) & (long_df["projected_trait"] == trait)
            ][["item_id", "projection"]].dropna()

            if subset.empty:
                print(f"[run_reliability_analysis] No data for layer={layer}, trait={trait}; skipping.")
                continue

            vc = estimate_variance_components(subset)
            rel_dict = compute_reliability(vc, k_values)

            item_stats = subset.groupby("item_id")["projection"]
            item_means = item_stats.mean()
            item_sds = item_stats.std(ddof=1).fillna(0.0)

            variance_ratio = (
                vc.between_item_var / vc.within_item_var
                if vc.within_item_var > 0
                else np.inf
            )

            results.append(ReliabilityResult(
                layer=layer,
                projected_trait=trait,
                variance_components=vc,
                reliability_single=rel_dict.get(1, 0.0),
                reliability_k=rel_dict,
                within_item_sd=float(np.sqrt(vc.within_item_var)),
                between_item_sd=float(np.sqrt(vc.between_item_var)),
                variance_ratio=float(variance_ratio),
                item_means=item_means,
                item_sds=item_sds,
                n_items_used=vc.n_items,
            ))

    return results


def results_to_dataframe(results: list[ReliabilityResult]) -> pd.DataFrame:
    """Convert list of ReliabilityResult to a tidy summary DataFrame."""
    rows = []
    for r in results:
        vc = r.variance_components
        row = {
            "layer": r.layer,
            "projected_trait": r.projected_trait,
            "between_item_var": vc.between_item_var,
            "within_item_var": vc.within_item_var,
            "total_var": vc.total_var,
            "within_item_sd": r.within_item_sd,
            "between_item_sd": r.between_item_sd,
            "variance_ratio": r.variance_ratio,
            "reliability_1": r.reliability_k.get(1, float("nan")),
            "reliability_2": r.reliability_k.get(2, float("nan")),
            "reliability_3": r.reliability_k.get(3, float("nan")),
            "reliability_4": r.reliability_k.get(4, float("nan")),
            "reliability_5": r.reliability_k.get(5, float("nan")),
            "n_items_used": r.n_items_used,
            "clamped_negative": vc.clamped_negative,
        }
        rows.append(row)
    return pd.DataFrame(rows)
