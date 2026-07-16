"""
Run the random-vector negative control.

Loads centered ETHICS wide projections and reliability wide projections.
Tries to load raw activations for precise projections; falls back to
structure on projected data if activations are unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls.random_vectors import (
    run_random_vector_control,
    save_random_vector_control,
    compare_to_real,
)
from src.controls.shuffled_labels import compute_diagonal_dominance
from src.controls.preprocessing_controls import _struct_metrics

ETHICS_WIDE = "outputs/ethics_projection/ethics_trait_projections_centered_wide.csv"
RELIABILITY_WIDE = "outputs/reliability_projection/reliability_trait_projections_wide_centered.parquet"
OUT_DIR = "outputs/controls/"
LAYERS = [32, 40, 47]


def main():
    print("[random_vector_control] Loading data...")
    ethics_df = pd.read_csv(ETHICS_WIDE)
    rel_df = pd.read_parquet(RELIABILITY_WIDE)

    # Compute real metrics for comparison
    real_metrics = {}
    for layer in LAYERS:
        if "layer" in rel_df.columns:
            sub = rel_df[rel_df["layer"] == layer]
        else:
            sub = ethics_df
        m = _struct_metrics(sub)
        real_metrics[layer] = {k: v for k, v in m.items() if "reliability" not in k}

    print("[random_vector_control] Running random vector control (100 repeats)...")
    results = run_random_vector_control(
        ethics_wide_df=ethics_df,
        reliability_wide_df=rel_df,
        activation_paths_by_layer=None,  # no raw activations available by default
        n_repeats=100,
        random_seed=42,
    )

    # Compare to real
    compare_df = compare_to_real(results["distributions_df"], real_metrics)
    results["compare_df"] = compare_df

    save_random_vector_control(results, OUT_DIR)

    print("\n=== Random Vector Control Summary ===")
    print(results["summary_df"].to_string())
    print("\n=== Comparison to Real Metrics ===")
    print(compare_df.to_string())
    print(f"\nMethod used: {results['method']}")


if __name__ == "__main__":
    main()
