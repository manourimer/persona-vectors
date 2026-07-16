"""
Run the permuted item-variant grouping negative control.

Loads reliability long-format centered projections and tests whether
reliability is driven by stable item identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls.permuted_grouping import run_permuted_grouping_control, save_permuted_grouping_control

RELIABILITY_LONG = "outputs/reliability_projection/reliability_trait_projections_long_centered.parquet"
OUT_DIR = "outputs/controls/"


def main():
    print("[permuted_grouping_control] Loading reliability long projections...")
    try:
        long_df = pd.read_parquet(RELIABILITY_LONG)
    except FileNotFoundError:
        print(f"ERROR: {RELIABILITY_LONG} not found.")
        print("Run Stage 4C (reliability variant projection) first.")
        return

    print(f"  Shape: {long_df.shape}")
    print(f"  Columns: {long_df.columns.tolist()}")

    print("[permuted_grouping_control] Running permuted grouping control (1000 permutations)...")
    print("  Note: This may take several minutes.")
    results = run_permuted_grouping_control(
        long_df,
        n_permutations=1000,
        random_seed=42,
        k_values=[1, 3],
    )

    save_permuted_grouping_control(results, OUT_DIR)

    print("\n=== Permuted Grouping Control Results ===")
    print("\nReal G-coefficients:")
    print(results["real_g_coefficients"].to_string())

    null_df = results["null_dist_df"]
    for k in [1, 3]:
        col = f"g_{k}"
        if col in null_df.columns:
            null_mean = null_df[col].mean()
            null_p95 = null_df[col].quantile(0.95)
            print(f"\nNull distribution G(k={k}): mean={null_mean:.4f}, 95th pct={null_p95:.4f}")

    pv = results["p_values"]
    print("\np-values (real G > permuted null):")
    for k in [1, 3]:
        col = f"g_{k}"
        if col in pv:
            for key, p in pv[col].items():
                print(f"  G(k={k}) {key}: p={p:.4f}")


if __name__ == "__main__":
    main()
