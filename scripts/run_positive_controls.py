"""
Run positive controls:
1. Contrast validation (Stage 2B AUC check).
2. Create synthetic obvious-scenario scaffold if it doesn't exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls.positive_controls import (
    run_contrast_validation_control,
    save_contrast_validation_control,
    build_synthetic_scenario_scaffold,
    save_synthetic_scenario_scaffold,
)

VALIDATION_RESULTS = "outputs/vector_construction/vector_validation_results.csv"
VECTOR_METADATA = "outputs/vector_construction/persona_vector_metadata.csv"
OUT_DIR = "outputs/controls/"
DATA_DIR = "data/processed/"
SYNTHETIC_PATH = "data/processed/synthetic_moral_scenarios.csv"


def main():
    # --- Contrast validation ---
    print("[contrast_validation] Running contrast validation positive control...")
    cv_results = run_contrast_validation_control(VALIDATION_RESULTS, VECTOR_METADATA)
    save_contrast_validation_control(cv_results, OUT_DIR)

    print(f"\n=== Contrast Validation Results ===")
    print(f"All pass: {cv_results['all_pass']}")
    print(f"Warnings: {cv_results['n_warnings']}")
    print(cv_results["summary_df"].to_string())
    if not cv_results["warnings_df"].empty:
        print("\nWarnings:")
        print(cv_results["warnings_df"].to_string())

    # --- Synthetic scenarios ---
    synth_path = Path(SYNTHETIC_PATH)
    if synth_path.exists():
        print(f"\n[synthetic_scenarios] Loading existing scaffold from {synth_path}")
        df = pd.read_csv(synth_path)
    else:
        print("\n[synthetic_scenarios] Building synthetic scenario scaffold...")
        df = build_synthetic_scenario_scaffold(n_per_trait=25)

    save_synthetic_scenario_scaffold(df, OUT_DIR, DATA_DIR)
    n_reviewed = int(df["reviewed"].sum())
    n_total = len(df)
    n_filled = int((df["scenario_text"] != "").sum())
    print(f"\nSynthetic scenarios: {n_total} total, {n_filled} filled, {n_reviewed} reviewed")
    print(f"Next step: Fill in blank rows in {SYNTHETIC_PATH} and set reviewed=True.")


if __name__ == "__main__":
    main()
