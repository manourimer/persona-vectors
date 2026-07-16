"""
Shuffled-label negative control.

Question: Is the observed trait-label alignment better than chance?
Expected: Real label alignment should exceed shuffled-label null.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]


def compute_diagonal_dominance(wide_df: pd.DataFrame, trait_cols: list[str] | None = None) -> float:
    """Fraction of items where the correct-trait projection column is the maximum."""
    if trait_cols is None:
        trait_cols = [f"projection_{t}" for t in TRAITS]
    available = [c for c in trait_cols if c in wide_df.columns]
    if not available or "primary_trait" not in wide_df.columns:
        return float("nan")

    df = wide_df.dropna(subset=["primary_trait"] + available)
    correct = df.apply(
        lambda row: (
            row.get(f"projection_{row['primary_trait']}", float("-inf"))
            == row[available].max()
        )
        if row.get("primary_trait") in TRAITS
        else False,
        axis=1,
    )
    return float(correct.mean())


def compute_matching_margins(wide_df: pd.DataFrame, trait_cols: list[str] | None = None) -> float:
    """Mean margin: (correct-trait projection - max(other projections))."""
    if trait_cols is None:
        trait_cols = [f"projection_{t}" for t in TRAITS]
    available = [c for c in trait_cols if c in wide_df.columns]
    if not available or "primary_trait" not in wide_df.columns:
        return float("nan")

    margins = []
    for _, row in wide_df.dropna(subset=["primary_trait"] + available).iterrows():
        pt = row.get("primary_trait")
        correct_col = f"projection_{pt}"
        if pt not in TRAITS or correct_col not in available:
            continue
        correct_val = row[correct_col]
        other_vals = [row[c] for c in available if c != correct_col]
        if not other_vals:
            continue
        margins.append(float(correct_val) - float(max(other_vals)))
    return float(np.mean(margins)) if margins else float("nan")


def run_shuffled_label_control(
    wide_df: pd.DataFrame,
    n_permutations: int = 1000,
    random_seed: int = 42,
) -> dict:
    """
    For each permutation: shuffle primary_trait column, recompute diagonal dominance
    and matching margin.

    wide_df must have primary_trait column and projection_* columns.

    Returns dict with keys:
        null_dist_df    — permutation results (n_permutations rows)
        real_metrics    — dict with real diagonal_dominance and matching_margin
        p_values        — {metric: p_value}
        percentiles     — {metric: percentile of real value in null distribution}
    """
    trait_cols = [f"projection_{t}" for t in TRAITS if f"projection_{t}" in wide_df.columns]
    real_dd = compute_diagonal_dominance(wide_df, trait_cols)
    real_mm = compute_matching_margins(wide_df, trait_cols)
    real_metrics = {"diagonal_dominance": real_dd, "matching_margin": real_mm}

    rng = np.random.default_rng(random_seed)
    df_work = wide_df.copy()
    valid_mask = df_work["primary_trait"].isin(TRAITS)
    valid_labels = df_work.loc[valid_mask, "primary_trait"].values

    null_rows = []
    for perm_idx in range(n_permutations):
        shuffled = rng.permutation(valid_labels)
        df_perm = df_work.copy()
        df_perm.loc[valid_mask, "primary_trait"] = shuffled
        null_rows.append({
            "permutation": perm_idx,
            "diagonal_dominance": compute_diagonal_dominance(df_perm, trait_cols),
            "matching_margin": compute_matching_margins(df_perm, trait_cols),
        })
    null_dist_df = pd.DataFrame(null_rows)

    p_values: dict = {}
    percentiles: dict = {}
    for metric in ["diagonal_dominance", "matching_margin"]:
        null_vals = null_dist_df[metric].dropna().values
        real_val = real_metrics[metric]
        if len(null_vals) == 0 or np.isnan(real_val):
            p_values[metric] = float("nan")
            percentiles[metric] = float("nan")
        else:
            p_values[metric] = float(np.mean(null_vals >= real_val))
            percentiles[metric] = float(np.mean(null_vals <= real_val)) * 100

    return {
        "null_dist_df": null_dist_df,
        "real_metrics": real_metrics,
        "p_values": p_values,
        "percentiles": percentiles,
    }


def save_shuffled_label_control(results_dict: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{
        "metric": m,
        "real_value": results_dict["real_metrics"][m],
        "p_value": results_dict["p_values"][m],
        "percentile_of_real": results_dict["percentiles"][m],
        "null_mean": results_dict["null_dist_df"][m].mean(),
        "null_std": results_dict["null_dist_df"][m].std(),
        "null_p95": results_dict["null_dist_df"][m].quantile(0.95),
    } for m in ["diagonal_dominance", "matching_margin"]])
    summary.to_csv(out_dir / "shuffled_label_control_summary.csv", index=False)
    results_dict["null_dist_df"].to_csv(out_dir / "shuffled_label_null_distribution.csv", index=False)
    print(f"[shuffled_label_control] Saved to {out_dir}")
