"""
Stage 4C: Run diagnostics on reliability variant projections.

Loads the centered wide projections, computes diagnostics, saves reports
and correlation matrices, and prints key findings.

Usage:
    python scripts/diagnose_reliability_variant_projections.py

    python scripts/diagnose_reliability_variant_projections.py \\
        --wide-path outputs/reliability_projection/reliability_trait_projections_wide.parquet \\
        --long-path outputs/reliability_projection/reliability_trait_projections_long.parquet \\
        --output-dir outputs/reliability_projection/
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.reliability.variant_projection_diagnostics import (
    compute_diagnostics,
    generate_report,
    save_diagnostics,
)

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose reliability variant projections."
    )
    parser.add_argument(
        "--wide-path",
        default="outputs/reliability_projection/reliability_trait_projections_wide.parquet",
        help="Path to centered wide projections parquet.",
    )
    parser.add_argument(
        "--long-path",
        default="outputs/reliability_projection/reliability_trait_projections_long.parquet",
        help="Path to centered long projections parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reliability_projection/",
        help="Directory to save diagnostic outputs.",
    )
    args = parser.parse_args()

    wide_path = _ROOT / args.wide_path
    long_path = _ROOT / args.long_path
    out_dir = _ROOT / args.output_dir

    print("\n  Stage 4C: Diagnosing reliability variant projections")

    for p in [wide_path, long_path]:
        if not p.exists():
            print(
                f"ERROR: File not found: {p}\n"
                "Run: python scripts/compute_reliability_variant_projections.py --preprocessing both"
            )
            sys.exit(1)

    wide_df = pd.read_parquet(wide_path)
    long_df = pd.read_parquet(long_path)

    print(f"  Wide shape: {wide_df.shape}")
    print(f"  Long shape: {long_df.shape}")

    diag = compute_diagnostics(long_df, wide_df)

    print(f"\n  --- Key Findings ---")
    print(f"  Items:       {diag['n_items']}")
    print(f"  Variants:    {diag['n_variants']}")
    print(f"  Originals:   {diag['n_originals']}")
    print(f"  Paraphrases: {diag['n_paraphrases']}")
    print(f"  Missing variants: {len(diag['missing_variants'])}")

    if diag["warnings"]:
        print(f"\n  WARNINGS ({len(diag['warnings'])}):")
        for w in diag["warnings"]:
            print(f"    - {w}")
    else:
        print("\n  No warnings triggered.")

    # Print within-item std summary
    within_std = diag["within_item_std"]
    if not within_std.empty:
        mean_std = within_std["within_item_std"].mean()
        print(f"\n  Mean within-item std across variants: {mean_std:.4f}")

    save_diagnostics(diag, wide_df, out_dir)
    print(f"\n  Diagnostics saved to: {out_dir}")
    print(f"    - reliability_projection_diagnostics.md")
    print(f"    - reliability_projection_summary.csv")
    print(f"    - reliability_projection_corr_layer{{N}}.csv (one per layer)")


if __name__ == "__main__":
    main()
