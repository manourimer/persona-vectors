"""
Run the shuffled-label negative control.

Loads centered ETHICS wide projections and tests whether trait-label alignment
exceeds chance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls.shuffled_labels import run_shuffled_label_control, save_shuffled_label_control

ETHICS_WIDE = "outputs/ethics_projection/ethics_trait_projections_centered_wide.csv"
OUT_DIR = "outputs/controls/"


def main():
    print("[shuffled_label_control] Loading ETHICS wide projections...")
    ethics_df = pd.read_csv(ETHICS_WIDE)

    print(f"  Shape: {ethics_df.shape}")
    print(f"  primary_trait counts:\n{ethics_df['primary_trait'].value_counts()}")

    print("[shuffled_label_control] Running shuffled label control (1000 permutations)...")
    results = run_shuffled_label_control(ethics_df, n_permutations=10000, random_seed=42)

    save_shuffled_label_control(results, OUT_DIR)

    print("\n=== Shuffled Label Control Results ===")
    for metric in ["diagonal_dominance", "matching_margin"]:
        real_val = results["real_metrics"][metric]
        p = results["p_values"][metric]
        pct = results["percentiles"][metric]
        null_p95 = results["null_dist_df"][metric].quantile(0.95)
        null_mean = results["null_dist_df"][metric].mean()
        print(f"\n{metric}:")
        print(f"  Real value:     {real_val:.4f}")
        print(f"  Null mean:      {null_mean:.4f}")
        print(f"  Null 95th pct:  {null_p95:.4f}")
        print(f"  p-value:        {p:.4f}")
        print(f"  Percentile:     {pct:.1f}%")

    print("\nConclusion: "
          + ("SIGNIFICANT — label alignment exceeds chance (p < 0.05)"
             if results["p_values"]["diagonal_dominance"] < 0.05
             else "NOT SIGNIFICANT — label alignment does not clearly exceed chance"))


if __name__ == "__main__":
    main()
