"""
Stage 3 — Step 2: Project ETHICS item activations onto persona vectors.

Produces raw and/or mean-centered projection tables.  Mean-centered is the
default for downstream diagnostics; raw is preserved for audit.

Usage:
    # Default: both raw and centered, all three layers
    python scripts/compute_ethics_projections.py

    # Centered only
    python scripts/compute_ethics_projections.py --preprocessing mean_centered

    # Specific layers
    python scripts/compute_ethics_projections.py --layers 32 40 47

    # Mock mode (no GPU, for smoke testing)
    python scripts/compute_ethics_projections.py --mock
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_TARGET_LAYER = 32
_DEFAULT_COMPARISON_LAYERS = [40, 47]
_OUT_DIR = "outputs/ethics_projection"
_VEC_META = "outputs/vector_construction/persona_vector_metadata.csv"
_VEC_VAL = "outputs/vector_construction/vector_validation_results.csv"
_ITEM_BANK = "data/processed/ethics_curated_mvp.parquet"
_MIN_AUC = 0.75


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project ETHICS item activations onto persona vectors."
    )
    parser.add_argument(
        "--preprocessing",
        choices=["raw", "mean_centered", "both"],
        default="both",
        help="Which projection variant(s) to produce (default: both).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layer indices to project at (default: 32 40 47).",
    )
    parser.add_argument("--target-layer", type=int, default=_DEFAULT_TARGET_LAYER)
    parser.add_argument("--out-dir", default=_OUT_DIR)
    parser.add_argument("--vec-meta", default=_VEC_META)
    parser.add_argument("--item-bank", default=_ITEM_BANK)
    parser.add_argument("--no-auc-check", action="store_true")
    parser.add_argument("--min-auc", type=float, default=_MIN_AUC)
    parser.add_argument("--mock", action="store_true",
                        help="Generate synthetic projections (no GPU required).")
    parser.add_argument("--mock-hidden-dim", type=int, default=64)
    args = parser.parse_args()

    from src.projection.compute_projections import (
        load_ethics_activation_metadata,
        mock_project,
        project_activations,
        save_centering_metadata,
        save_projection_set,
        to_wide_format,
    )
    from src.projection.ethics_projection import (
        TRAITS,
        enforce_auc_threshold,
        load_item_bank,
        load_vector_metadata,
        load_vector_validation_results,
        select_vectors,
    )

    all_layers = args.layers if args.layers else [args.target_layer] + _DEFAULT_COMPARISON_LAYERS
    target_layer = args.target_layer
    out_path = _ROOT / args.out_dir

    print("\n  ── Stage 3: Compute ETHICS Projections ────────────────────────────\n")
    print(f"  Preprocessing  : {args.preprocessing}")
    print(f"  Layers         : {all_layers}")

    item_df = load_item_bank(_ROOT / args.item_bank)

    if args.mock:
        print("  [mock] Generating synthetic projections ...")
        results, layer_means, n_items_by_layer = mock_project(
            item_df,
            layers=all_layers,
            hidden_dim=args.mock_hidden_dim,
            out_dir=str(out_path),
            preprocessing=args.preprocessing,
        )
    else:
        if not args.no_auc_check:
            val_df = load_vector_validation_results(_ROOT / _VEC_VAL)
            enforce_auc_threshold(val_df, all_layers, TRAITS, args.min_auc)
            print(f"  ✅ All vectors pass AUC ≥ {args.min_auc}")

        vec_meta_df = load_vector_metadata(_ROOT / args.vec_meta)
        comparison = [l for l in all_layers if l != target_layer]
        selected_vecs = select_vectors(vec_meta_df, target_layer, comparison)
        print(f"  Loaded {len(selected_vecs)} vector metadata records")

        meta_path = out_path / "ethics_activation_metadata.parquet"
        act_meta_df = load_ethics_activation_metadata(meta_path)
        print(f"  Loaded {len(act_meta_df)} activation records")

        print(f"  Projecting ...")
        results, layer_means, n_items_by_layer = project_activations(
            act_meta_df=act_meta_df,
            vec_meta_df=selected_vecs,
            act_dir=out_path,
            vec_dir=_ROOT,
            preprocessing=args.preprocessing,
        )

    # Save centering metadata
    if layer_means:
        cen_path = save_centering_metadata(layer_means, out_path, n_items_by_layer)
        print(f"  Centering metadata: {cen_path}")

    saved_files: list[Path] = []

    # Save raw projections
    if "raw" in results:
        raw_long = results["raw"]
        raw_wide = to_wide_format(raw_long[raw_long["layer"] == target_layer], item_df)
        paths = save_projection_set(raw_long, raw_wide, out_path, "ethics_trait_projections_raw")
        saved_files.extend(paths)
        print(f"  Raw long rows   : {len(raw_long)}")

    # Save centered projections (and as default alias)
    if "mean_centered" in results:
        cen_long = results["mean_centered"]
        cen_wide = to_wide_format(cen_long[cen_long["layer"] == target_layer], item_df)
        paths = save_projection_set(cen_long, cen_wide, out_path, "ethics_trait_projections_centered")
        saved_files.extend(paths)
        # Also write backwards-compatible default filenames pointing to centered output
        _compat = save_projection_set(cen_long, cen_wide, out_path, "ethics_trait_projections")
        print(f"  Centered long rows: {len(cen_long)}")

    print(f"\n  Output directory: {out_path}")
    print(f"  Next: python scripts/diagnose_ethics_projections.py")
    print(f"        python scripts/compare_projection_layers.py")


if __name__ == "__main__":
    main()
