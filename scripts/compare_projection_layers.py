"""
Stage 3 — Step 4: Compare projection structure across candidate layers.

Reads centered projections, computes per-layer diagnostic metrics, and
reports whether the contrast-validation-selected layer is also the best
downstream ETHICS layer.

Usage:
    python scripts/compare_projection_layers.py
    python scripts/compare_projection_layers.py --layers 32 40 47
    python scripts/compare_projection_layers.py --contrast-selected-layer 32
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_OUT_DIR = "outputs/ethics_projection"
_ITEM_BANK = "data/processed/ethics_curated_mvp.parquet"
_DEFAULT_LAYERS = [32, 40, 47]
_CONTRAST_SELECTED = 32


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ETHICS projection structure across candidate layers."
    )
    parser.add_argument("--out-dir", default=_OUT_DIR)
    parser.add_argument("--item-bank", default=_ITEM_BANK)
    parser.add_argument(
        "--layers", nargs="+", type=int, default=_DEFAULT_LAYERS,
    )
    parser.add_argument(
        "--contrast-selected-layer", type=int, default=_CONTRAST_SELECTED,
        help="Layer chosen by Stage 2B contrast-prompt validation AUC.",
    )
    parser.add_argument(
        "--preprocessing",
        choices=["raw", "mean_centered"],
        default="mean_centered",
    )
    args = parser.parse_args()

    import pandas as pd
    from src.projection.ethics_projection import load_item_bank
    from src.projection.layer_comparison import (
        compare_layers,
        compute_layer_metrics,
        save_layer_comparison,
    )

    out_path = _ROOT / args.out_dir
    preprocessing = args.preprocessing
    suffix = "centered" if preprocessing == "mean_centered" else "raw"

    long_file = out_path / f"ethics_trait_projections_{suffix}_long.parquet"
    if not long_file.exists():
        long_file = out_path / "ethics_trait_projections_long.parquet"
    if not long_file.exists():
        print(f"  ERROR: Projection files not found in {out_path}")
        print("         Run scripts/compute_ethics_projections.py first.")
        sys.exit(1)

    long_df = pd.read_parquet(long_file)
    item_df = load_item_bank(_ROOT / args.item_bank)

    print(f"\n  ── Stage 3: Layer Comparison ({preprocessing}) ─────────────────────\n")
    print(f"  Layers          : {args.layers}")
    print(f"  Contrast-selected: {args.contrast_selected_layer}")

    metrics_df = compute_layer_metrics(
        long_df, item_df, args.layers, preprocessing=preprocessing
    )

    print(f"\n  Per-layer metrics:")
    for _, row in metrics_df.iterrows():
        print(
            f"    Layer {int(row['layer']):>2}  "
            f"diag_dom={row['diagonal_dominance']:.3f}  "
            f"margin={row['matching_margin']:>7.1f}  "
            f"max_corr={row['max_inter_trait_correlation']:.3f}"
        )

    result = compare_layers(
        metrics_df,
        contrast_selected_layer=args.contrast_selected_layer,
        downstream_layers=args.layers,
    )

    print(f"\n  Best downstream ETHICS layer: {result.best_downstream_layer} "
          f"(diagonal dominance = {result.best_downstream_dominance:.3f})")
    print(f"  Contrast-selected layer {result.contrast_selected_layer}: "
          f"diagonal dominance = {result.contrast_selected_dominance:.3f}")
    print(f"  Layers agree: {'✅ Yes' if result.layers_agree else '❌ No'}")

    print(f"\n  Interpretation:")
    for line in result.interpretation.split(". "):
        if line.strip():
            print(f"    {line.strip()}.")

    if result.warnings:
        print(f"\n  ⚠  Warnings:")
        for w in result.warnings:
            print(f"     - {w}")

    csv_path, md_path = save_layer_comparison(result, out_path, preprocessing_label=preprocessing)
    print(f"\n  Saved:")
    print(f"    Layer comparison CSV: {csv_path}")
    print(f"    Layer comparison MD : {md_path}")

    # Save per-layer correlation matrices
    from src.projection.compute_projections import to_wide_format
    for layer in args.layers:
        layer_long = long_df[long_df["layer"] == layer]
        if "projection_preprocessing" in layer_long.columns:
            layer_long = layer_long[layer_long["projection_preprocessing"] == preprocessing]
        if layer_long.empty:
            continue
        wide = to_wide_format(layer_long, item_df)
        proj_cols = [f"projection_{t}" for t in ["honesty","harmlessness","fairness","compassion"]
                     if f"projection_{t}" in wide.columns]
        if len(proj_cols) >= 2:
            corr = wide[proj_cols].corr()
            corr.index = [c.replace("projection_", "") for c in corr.index]
            corr.columns = [c.replace("projection_", "") for c in corr.columns]
            corr_path = out_path / f"projection_correlation_matrix_layer{layer}.csv"
            corr.to_csv(corr_path)
            print(f"    Correlation layer {layer}: {corr_path}")


if __name__ == "__main__":
    main()
