"""
Stage 3 — Step 0: Build and inspect projection jobs for ETHICS items.

Loads the curated item bank, validates schema, enforces vector AUC thresholds,
and prints a summary of the jobs that will be run.

This script does not run any GPU inference — it is a pre-flight check.

Usage:
    python scripts/build_ethics_projection_jobs.py
    python scripts/build_ethics_projection_jobs.py --no-auc-check
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_ITEM_BANK = _ROOT / "data/processed/ethics_curated_mvp.parquet"
_VAL_RESULTS = _ROOT / "outputs/vector_construction/vector_validation_results.csv"
_VEC_META = _ROOT / "outputs/vector_construction/persona_vector_metadata.csv"
_TARGET_LAYER = 32
_COMPARISON_LAYERS = [40, 47]
_MIN_AUC = 0.75


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight check: build Stage 3 ETHICS projection jobs."
    )
    parser.add_argument(
        "--item-bank", default=str(_ITEM_BANK),
        help="Path to curated item bank parquet.",
    )
    parser.add_argument(
        "--target-layer", type=int, default=_TARGET_LAYER,
    )
    parser.add_argument(
        "--comparison-layers", default=",".join(str(l) for l in _COMPARISON_LAYERS),
    )
    parser.add_argument(
        "--min-auc", type=float, default=_MIN_AUC,
    )
    parser.add_argument(
        "--no-auc-check", action="store_true",
        help="Skip vector AUC threshold enforcement.",
    )
    args = parser.parse_args()

    from src.projection.ethics_projection import (
        build_projection_jobs,
        enforce_auc_threshold,
        load_item_bank,
        load_vector_metadata,
        load_vector_validation_results,
        select_vectors,
        validate_item_bank,
        TRAITS,
    )

    comparison = [int(l) for l in args.comparison_layers.split(",") if l.strip()]
    all_layers = [args.target_layer] + comparison

    print("\n  ── Stage 3: ETHICS Projection Pre-flight ─────────────────────────\n")

    # Load and validate item bank
    print(f"  Loading item bank: {args.item_bank}")
    item_df = load_item_bank(args.item_bank)
    validate_item_bank(item_df)
    print(f"  Items loaded: {len(item_df)}")
    print(f"  Trait distribution:")
    for trait, count in item_df["primary_trait"].value_counts().items():
        print(f"    {trait:<16} {count}")

    # Enforce AUC threshold
    if not args.no_auc_check:
        print(f"\n  Checking vector AUC ≥ {args.min_auc} at layers {all_layers} ...")
        val_df = load_vector_validation_results(_VAL_RESULTS)
        try:
            enforce_auc_threshold(val_df, all_layers, TRAITS, args.min_auc)
            print("  ✅ All required vectors pass AUC threshold.")
        except ValueError as e:
            print(f"\n  ERROR: {e}")
            sys.exit(1)
    else:
        print("\n  ⚠  AUC check skipped (--no-auc-check).")

    # Load and select vectors
    print(f"\n  Loading vector metadata: {_VEC_META}")
    vec_meta = load_vector_metadata(_VEC_META)
    selected = select_vectors(vec_meta, args.target_layer, comparison)
    print(f"  Selected vectors: {len(selected)} ({len(TRAITS)} traits × {len(all_layers)} layers)")

    # Build jobs
    jobs_df = build_projection_jobs(item_df, all_layers)
    print(f"\n  Total forward-pass jobs: {len(jobs_df)}")
    print(f"    = {len(item_df)} items × {len(all_layers)} layers")
    print(f"  Token scope: last_prompt_token")
    print(f"  Target layer: {args.target_layer}")
    print(f"  Comparison layers: {comparison}")

    print("\n  ── Ready ──────────────────────────────────────────────────────────")
    print("  Next: python scripts/extract_ethics_activations.py --limit 10")
    print("        python scripts/extract_ethics_activations.py")


if __name__ == "__main__":
    main()
