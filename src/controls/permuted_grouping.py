"""
Permuted item-variant grouping negative control.

Question: Is reliability driven by stable item identity, not coincidental grouping?
Expected: True grouping produces higher G-coefficient than permuted grouping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.reliability.reliability_analysis import estimate_variance_components, compute_reliability

TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]


def permute_item_grouping(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly reassign variant_id rows to different item_id groups while preserving
    variant counts per item.

    Returns a modified copy of long_df with shuffled item_id assignments.
    """
    df = long_df.copy()
    # For each unique (layer, projected_trait) group, permute the item_id assignments
    # while preserving how many variants each item_id has.
    group_keys = []
    if "layer" in df.columns:
        group_keys.append("layer")
    if "projected_trait" in df.columns:
        group_keys.append("projected_trait")

    if not group_keys:
        # Just permute item_id globally
        df["item_id"] = np.random.permutation(df["item_id"].values)
        return df

    result_parts = []
    for keys, grp in df.groupby(group_keys):
        grp = grp.copy()
        # Count variants per item_id in this group
        item_ids = grp["item_id"].values
        unique_items = np.unique(item_ids)
        counts = {item: int((item_ids == item).sum()) for item in unique_items}

        # Shuffle the assignment: re-assign item labels based on original counts
        all_rows = list(range(len(grp)))
        np.random.shuffle(all_rows)

        # Use object dtype to avoid pandas StringDtype compatibility issues
        new_item_ids = np.empty(len(grp), dtype=object)
        idx = 0
        for item_label, count in counts.items():
            row_indices = all_rows[idx: idx + count]
            new_item_ids[row_indices] = item_label
            idx += count

        grp = grp.reset_index(drop=True)
        grp["item_id"] = new_item_ids
        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True)


def _compute_g_for_group(df_group: pd.DataFrame, k_values: list[int]) -> dict:
    """Run variance decomposition and return G(k) for each k."""
    if "projection" not in df_group.columns:
        return {k: float("nan") for k in k_values}
    vc = estimate_variance_components(df_group[["item_id", "projection"]])
    return compute_reliability(vc, k_values)


def run_permuted_grouping_control(
    reliability_long_df: pd.DataFrame,
    n_permutations: int = 1000,
    random_seed: int = 42,
    k_values: list[int] = None,
) -> dict:
    """
    For each permutation × layer × projected_trait: permute grouping, compute G(k=1) and G(k=3).

    Returns dict with keys: null_dist_df, real_g_coefficients, p_values, percentiles
    """
    if k_values is None:
        k_values = [1, 3]

    layers = reliability_long_df["layer"].unique().tolist() if "layer" in reliability_long_df.columns else [32]
    projected_traits = reliability_long_df["projected_trait"].unique().tolist() if "projected_trait" in reliability_long_df.columns else TRAITS

    # Compute real G-coefficients
    real_rows = []
    for layer in layers:
        for trait in projected_traits:
            sub = reliability_long_df[
                (reliability_long_df["layer"] == layer) & (reliability_long_df["projected_trait"] == trait)
            ]
            if sub.empty:
                continue
            g_dict = _compute_g_for_group(sub, k_values)
            row = {"layer": layer, "projected_trait": trait}
            for k in k_values:
                row[f"g_{k}"] = g_dict.get(k, float("nan"))
            real_rows.append(row)
    real_g_df = pd.DataFrame(real_rows)

    # Permutation null
    rng = np.random.default_rng(random_seed)
    null_rows = []
    for perm_idx in range(n_permutations):
        # set global seed for each permutation
        np.random.seed(rng.integers(0, 2**31))
        perm_df = permute_item_grouping(reliability_long_df)
        for layer in layers:
            for trait in projected_traits:
                sub = perm_df[
                    (perm_df["layer"] == layer) & (perm_df["projected_trait"] == trait)
                ]
                if sub.empty:
                    continue
                g_dict = _compute_g_for_group(sub, k_values)
                row = {"permutation": perm_idx, "layer": layer, "projected_trait": trait}
                for k in k_values:
                    row[f"g_{k}"] = g_dict.get(k, float("nan"))
                null_rows.append(row)
    null_dist_df = pd.DataFrame(null_rows)

    # P-values and percentiles
    p_values: dict = {}
    percentiles: dict = {}
    for k in k_values:
        col = f"g_{k}"
        p_values[col] = {}
        percentiles[col] = {}
        for layer in layers:
            for trait in projected_traits:
                real_row = real_g_df[
                    (real_g_df["layer"] == layer) & (real_g_df["projected_trait"] == trait)
                ]
                null_sub = null_dist_df[
                    (null_dist_df["layer"] == layer) & (null_dist_df["projected_trait"] == trait)
                ]
                if real_row.empty or null_sub.empty:
                    continue
                real_val = real_row[col].iloc[0]
                null_vals = null_sub[col].dropna().values
                key = f"{layer}_{trait}"
                p_values[col][key] = float(np.mean(null_vals >= real_val))
                percentiles[col][key] = float(np.mean(null_vals <= real_val)) * 100

    return {
        "null_dist_df": null_dist_df,
        "real_g_coefficients": real_g_df,
        "p_values": p_values,
        "percentiles": percentiles,
    }


def save_permuted_grouping_control(results_dict: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dict["real_g_coefficients"].to_csv(
        out_dir / "permuted_grouping_control_summary.csv", index=False
    )
    results_dict["null_dist_df"].to_csv(
        out_dir / "permuted_grouping_null_distribution.csv", index=False
    )
    print(f"[permuted_grouping_control] Saved to {out_dir}")
