"""
Exact-duplicate positive control.

Question: Does the reliability pipeline behave correctly when wording variation is zero?
Expected: Reliability should be near 1.0 for exact duplicates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.reliability.reliability_analysis import run_reliability_analysis, results_to_dataframe

TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]
PROJ_COLS = [f"projection_{t}" for t in TRAITS]


def build_exact_duplicate_projections(wide_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Takes a projection wide DataFrame (one row per item × layer).
    For each item, creates k identical rows with paraphrase_id = p1..pk and
    variant_type = 'paraphrase', plus the original row as paraphrase_id = 'original'.

    Returns long-format DataFrame with columns compatible with run_reliability_analysis:
        item_id, variant_id, paraphrase_id, framing, primary_trait,
        projected_trait, layer, projection
    """
    if "layer" in wide_df.columns:
        layers = wide_df["layer"].unique().tolist()
    else:
        layers = [32]

    long_rows = []
    for _, row in wide_df.iterrows():
        item_id = row.get("item_id", f"item_{_}")
        primary_trait = row.get("primary_trait", "unknown")

        # Determine layer
        if "layer" in row.index:
            row_layers = [int(row["layer"])]
        else:
            row_layers = layers

        for trait in TRAITS:
            proj_col = f"projection_{trait}"
            proj_val = row.get(proj_col, float("nan"))

            for layer_val in row_layers:
                # Original
                long_rows.append({
                    "item_id": item_id,
                    "variant_id": f"{item_id}_original",
                    "paraphrase_id": "original",
                    "variant_type": "original",
                    "framing": "neutral",
                    "primary_trait": primary_trait,
                    "projected_trait": trait,
                    "layer": layer_val,
                    "projection": proj_val,
                })
                # k duplicates
                for i in range(1, k + 1):
                    long_rows.append({
                        "item_id": item_id,
                        "variant_id": f"{item_id}_p{i}",
                        "paraphrase_id": f"p{i}",
                        "variant_type": "paraphrase",
                        "framing": "neutral",
                        "primary_trait": primary_trait,
                        "projected_trait": trait,
                        "layer": layer_val,
                        "projection": proj_val,  # identical — exact duplicate
                    })

    return pd.DataFrame(long_rows)


def run_exact_duplicate_control(ethics_wide_df: pd.DataFrame, k: int = 3) -> dict:
    """
    Builds exact-duplicate projections, converts to long format, runs reliability analysis.
    G(k=1) should be ~1.0 for exact duplicates.
    """
    long_df = build_exact_duplicate_projections(ethics_wide_df, k=k)

    layers = long_df["layer"].unique().tolist()
    projected_traits = TRAITS

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_reliability_analysis(
            long_df=long_df,
            layers=layers,
            projected_traits=projected_traits,
            min_variants_per_item=2,
            k_values=[1, 3],
        )

    summary_df = results_to_dataframe(results)
    return {
        "reliability_results": results,
        "summary_df": summary_df,
        "long_df": long_df,
    }


def save_exact_duplicate_control(results_dict: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dict["summary_df"].to_csv(out_dir / "exact_duplicate_control_summary.csv", index=False)
    print(f"[exact_duplicate_control] Saved to {out_dir}")
