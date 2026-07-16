"""
Run preprocessing and layer robustness controls.

Two scopes:

1. Original ETHICS projections (204 items, layer 32 only)
   - raw:     ethics_trait_projections_raw_wide.parquet
   - centered: ethics_trait_projections_centered_wide.parquet
   These are the same projections used in Stage 4A structure analysis.
   The centered metrics should match structure_summary.csv exactly.

2. Reliability-variant projections (761 variants × 3 layers)
   - raw:     reliability_trait_projections_wide_raw.parquet
   - centered: reliability_trait_projections_wide_centered.parquet
   Used to verify that structure metrics are stable across layers 32/40/47
   and robust to the centering choice.

Metrics are always computed per-layer (never pooled across layers).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls.preprocessing_controls import (
    run_preprocessing_comparison,
    run_layer_robustness,
    save_preprocessing_controls,
)

ROOT = Path(__file__).resolve().parent.parent

# --- Original ETHICS projection files (204 items, layer 32 only) ---
ETHICS_RAW = ROOT / "outputs/ethics_projection/ethics_trait_projections_raw_wide.parquet"
ETHICS_CENTERED = ROOT / "outputs/ethics_projection/ethics_trait_projections_centered_wide.parquet"

# --- Reliability variant projection files (761 variants × 3 layers) ---
REL_RAW = ROOT / "outputs/reliability_projection/reliability_trait_projections_wide_raw.parquet"
REL_CENTERED = ROOT / "outputs/reliability_projection/reliability_trait_projections_wide_centered.parquet"

OUT_DIR = ROOT / "outputs/controls"
TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]


def main():
    # ------------------------------------------------------------------
    # 1. Original ETHICS preprocessing comparison (layer 32 only)
    # ------------------------------------------------------------------
    print("[preprocessing_controls] Loading original ETHICS projection files (layer 32)...")
    ethics_raw_df = pd.read_parquet(ETHICS_RAW)
    ethics_centered_df = pd.read_parquet(ETHICS_CENTERED)
    print(f"  Raw shape:      {ethics_raw_df.shape}")
    print(f"  Centered shape: {ethics_centered_df.shape}")

    print("[preprocessing_controls] Computing ETHICS preprocessing comparison (layer 32)...")
    ethics_preprocessing_df = run_preprocessing_comparison(
        raw_wide_df=ethics_raw_df,
        centered_wide_df=ethics_centered_df,
        layers=[32],
        traits=TRAITS,
        source_dataset="ethics_original",
    )

    # ------------------------------------------------------------------
    # 2. Reliability-variant preprocessing comparison (layers 32, 40, 47)
    # ------------------------------------------------------------------
    print("[preprocessing_controls] Loading reliability-variant projection files (layers 32/40/47)...")
    rel_raw_df = pd.read_parquet(REL_RAW)
    rel_centered_df = pd.read_parquet(REL_CENTERED)
    print(f"  Raw shape:      {rel_raw_df.shape}, layers: {sorted(rel_raw_df['layer'].unique())}")
    print(f"  Centered shape: {rel_centered_df.shape}, layers: {sorted(rel_centered_df['layer'].unique())}")

    print("[preprocessing_controls] Computing reliability-variant preprocessing comparison...")
    rel_preprocessing_df = run_preprocessing_comparison(
        raw_wide_df=rel_raw_df,
        centered_wide_df=rel_centered_df,
        layers=[32, 40, 47],
        traits=TRAITS,
        source_dataset="reliability_variants",
    )

    # ------------------------------------------------------------------
    # 3. Layer robustness (reliability variants, centered)
    # ------------------------------------------------------------------
    print("[preprocessing_controls] Computing layer robustness (reliability variants, centered)...")
    layer_df = run_layer_robustness(
        centered_wide_df=rel_centered_df,
        layers=[32, 40, 47],
        traits=TRAITS,
        source_dataset="reliability_variants",
    )

    # ------------------------------------------------------------------
    # 4. Combined preprocessing summary and save
    # ------------------------------------------------------------------
    combined_preprocessing_df = pd.concat(
        [ethics_preprocessing_df, rel_preprocessing_df], ignore_index=True
    )

    results = {
        "preprocessing_df": combined_preprocessing_df,
        "layer_robustness_df": layer_df,
    }
    save_preprocessing_controls(results, OUT_DIR)

    # ------------------------------------------------------------------
    # 5. Verify ETHICS centered layer 32 matches Stage 4A
    # ------------------------------------------------------------------
    struct_path = ROOT / "outputs/structure_analysis/structure_summary.csv"
    if struct_path.exists():
        struct_df = pd.read_csv(struct_path)
        ref_row = struct_df[struct_df["layer"] == 32].iloc[0]
        ctrl_row = ethics_preprocessing_df[
            (ethics_preprocessing_df["preprocessing"] == "centered") &
            (ethics_preprocessing_df["projected_trait"] == "all") &
            (ethics_preprocessing_df["layer"] == 32)
        ]
        if not ctrl_row.empty:
            ctrl_row = ctrl_row.iloc[0]
            print("\n=== Stage 4A vs recomputed (ETHICS centered, layer 32) ===")
            for metric, stage4a_col in [
                ("effective_dimensionality", "effective_dimensionality"),
                ("pc1_variance", "first_pc_variance"),
                ("mean_abs_off_diag_corr", "mean_abs_off_diag_corr"),
            ]:
                ref_val = ref_row[stage4a_col]
                ctrl_val = ctrl_row[metric]
                match = abs(ref_val - ctrl_val) < 0.01
                print(f"  {metric}: Stage4A={ref_val:.4f}, Control={ctrl_val:.4f}, match={match}")

    print("\n=== Combined Preprocessing Comparison ===")
    print(combined_preprocessing_df[
        combined_preprocessing_df["projected_trait"] == "all"
    ].to_string())

    print("\n=== Layer Robustness (reliability variants, all traits) ===")
    print(layer_df[layer_df["projected_trait"] == "all"].to_string())


if __name__ == "__main__":
    main()
