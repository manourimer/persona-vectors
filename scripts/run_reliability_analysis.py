"""Stage 4D: Reliability / Generalizability Analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reliability.reliability_analysis import (
    load_projection_table,
    filter_table,
    run_reliability_analysis,
    results_to_dataframe,
)
from src.reliability.g_theory import run_d_study_for_all
from src.reliability.reliability_reports import save_all


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "mvp_experiment.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ra_cfg = config["reliability_analysis"]

    long_path = PROJECT_ROOT / ra_cfg["input_long_path"]
    out_dir = PROJECT_ROOT / ra_cfg["output_dir"]
    layers = ra_cfg["layers"]
    projected_traits = ra_cfg["projected_traits"]
    min_variants = ra_cfg.get("minimum_variants_per_item", 2)
    k_values = ra_cfg["d_study"]["n_paraphrases_to_evaluate"]
    primary_layer = ra_cfg["primary_layer"]
    downstream_layer = ra_cfg["downstream_best_layer"]

    print("── Stage 4D: Reliability Analysis ──────────────────────────────")

    # Load and filter
    long_df = load_projection_table(long_path)
    n_items_total = long_df["item_id"].nunique()
    n_variants_total = len(long_df)

    long_df_filtered = filter_table(long_df, layers, projected_traits, min_variants)
    n_items_used = long_df_filtered["item_id"].nunique()
    n_items_missing = n_items_total - n_items_used

    # Run reliability analysis
    results = run_reliability_analysis(
        long_df_filtered,
        layers=layers,
        projected_traits=projected_traits,
        min_variants_per_item=min_variants,
        k_values=k_values,
    )

    results_df = results_to_dataframe(results)

    # D-study
    d_study_df = run_d_study_for_all(results, k_values)

    # Meta for report
    meta = {
        "n_items_total": n_items_total,
        "n_items_used": n_items_used,
        "n_variants_total": n_variants_total,
        "n_items_missing_paraphrases": n_items_missing,
        "layers": layers,
        "projected_traits": projected_traits,
        "primary_layer": primary_layer,
        "downstream_best_layer": downstream_layer,
    }

    # Save all outputs
    save_all(results, results_df, d_study_df, meta, out_dir)

    # ── Terminal summary ──────────────────────────────────────────────────
    print(f"\nItems used       : {n_items_used}")
    print(f"Layers analyzed  : {', '.join(str(l) for l in layers)}")
    print()
    print("Reliability (single variant) by layer × projected trait:")
    if not results_df.empty:
        display_cols = ["layer", "projected_trait", "reliability_1", "reliability_3"]
        tbl = results_df[display_cols].copy()
        tbl["reliability_1"] = tbl["reliability_1"].map("{:.3f}".format)
        tbl["reliability_3"] = tbl["reliability_3"].map("{:.3f}".format)
        print(tbl.to_string(index=False))
    print()

    # Best/worst at primary layer
    if not results_df.empty:
        primary = results_df[results_df["layer"] == primary_layer]
        if not primary.empty:
            best = primary.loc[primary["reliability_1"].idxmax()]
            worst = primary.loc[primary["reliability_1"].idxmin()]
            print(f"Best overall (layer {primary_layer}): {best['projected_trait']} = {best['reliability_1']:.3f}")
            print(f"Worst overall (layer {primary_layer}): {worst['projected_trait']} = {worst['reliability_1']:.3f}")
            print()

    # D-study summary
    print(f"D-study (layer {primary_layer}) — reliability improves with k paraphrases:")
    primary_d = d_study_df[d_study_df["layer"] == primary_layer]
    for k in [1, 3]:
        k_row = primary_d[primary_d["n_paraphrases"] == k]
        parts = []
        for trait in projected_traits:
            tr = k_row[k_row["projected_trait"] == trait]
            if not tr.empty:
                val = tr["g_coefficient"].values[0]
                parts.append(f"{trait}={val:.3f}")
        print(f"  k={k}: {', '.join(parts)}")

    print()
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
