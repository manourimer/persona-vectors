"""
Stage 3 — Step 3: Run diagnostics on ETHICS trait projections.

Defaults to mean-centered projections.  Pass --raw to diagnose raw projections.

Usage:
    python scripts/diagnose_ethics_projections.py
    python scripts/diagnose_ethics_projections.py --raw
    python scripts/diagnose_ethics_projections.py --target-layer 40
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_OUT_DIR = "outputs/ethics_projection"
_ITEM_BANK = "data/processed/ethics_curated_mvp.parquet"
_DEFAULT_TARGET_LAYER = 32


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diagnostics on ETHICS trait projections."
    )
    parser.add_argument("--out-dir", default=_OUT_DIR)
    parser.add_argument("--item-bank", default=_ITEM_BANK)
    parser.add_argument("--target-layer", type=int, default=_DEFAULT_TARGET_LAYER)
    parser.add_argument(
        "--raw", action="store_true",
        help="Run diagnostics on raw projections instead of centered.",
    )
    args = parser.parse_args()

    import pandas as pd
    from src.projection.ethics_projection import load_item_bank
    from src.projection.projection_diagnostics import run_diagnostics, save_diagnostics

    preprocessing = "raw" if args.raw else "mean_centered"
    suffix = "raw" if args.raw else "centered"
    out_path = _ROOT / args.out_dir

    long_file = out_path / f"ethics_trait_projections_{suffix}_long.parquet"
    wide_file = out_path / f"ethics_trait_projections_{suffix}_wide.parquet"

    # Fall back to default (centered) files
    if not long_file.exists():
        long_file = out_path / "ethics_trait_projections_long.parquet"
        wide_file = out_path / "ethics_trait_projections_wide.parquet"

    if not long_file.exists():
        print(f"  ERROR: Projection files not found in {out_path}")
        print("         Run scripts/compute_ethics_projections.py first.")
        sys.exit(1)

    long_df = pd.read_parquet(long_file)
    wide_df = pd.read_parquet(wide_file)
    item_df = load_item_bank(_ROOT / args.item_bank)

    print(f"\n  ── Stage 3: Projection Diagnostics ({preprocessing}) ───────────────\n")
    print(f"  Long-format rows: {len(long_df)}")
    print(f"  Wide-format rows: {len(wide_df)}")

    result = run_diagnostics(
        long_df, wide_df, item_df,
        target_layer=args.target_layer,
        preprocessing=preprocessing,
    )

    print(f"\n  Items projected    : {result.n_items}")
    print(f"  Trait vectors      : {result.n_traits}")
    print(f"  Layers             : {result.n_layers}")
    print(f"  Missing activations: {result.n_missing_activations}")

    print(f"\n  Projection distribution (layer {args.target_layer}, {preprocessing}):")
    if not result.projection_stats.empty:
        for _, row in result.projection_stats.iterrows():
            print(
                f"    {row['projected_trait']:<16} "
                f"mean={row['mean']:>9.2f}  std={row['std']:>8.2f}  "
                f"[{row['min']:.2f}, {row['max']:.2f}]"
            )

    print(f"\n  Mean projection by primary_trait × projected_trait (layer {args.target_layer}):")
    if not result.matching_table.empty:
        print(result.matching_table.to_string(float_format="{:.1f}".format))

    dom = result.diagonal_dominance_rate
    print(
        f"\n  Diagonal dominance: {dom:.3f} "
        f"({'✅' if result.diagonal_dominance else '❌'} "
        f"matching trait is top projection for {dom:.1%} of items, "
        f"chance = 25.0%)"
    )

    if result.warnings:
        print(f"\n  ⚠  {len(result.warnings)} warning(s):")
        for w in result.warnings:
            print(f"     - {w}")
    else:
        print("\n  ✅ No diagnostic warnings.")

    md_path, summary_path, corr_path = save_diagnostics(result, out_path, preprocessing)
    print(f"\n  Saved:")
    print(f"    Diagnostics MD : {md_path}")
    print(f"    Summary CSV    : {summary_path}")
    print(f"    Correlation CSV: {corr_path}")


if __name__ == "__main__":
    main()
