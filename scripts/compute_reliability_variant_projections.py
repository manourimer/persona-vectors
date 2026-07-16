"""
Stage 4C: Compute dot-product projections for reliability variants.

Loads cached activation .npy files and validated persona vectors, computes
projections (raw and/or mean-centered), and saves long + wide output tables.

No GPU, torch, or Modal required — pure NumPy + pandas.

Usage:
    python scripts/compute_reliability_variant_projections.py --preprocessing both --layers 32 40 47

    python scripts/compute_reliability_variant_projections.py \\
        --preprocessing centered \\
        --layers 32 \\
        --activation-metadata outputs/reliability_projection/reliability_activation_metadata.parquet \\
        --vector-dir outputs/vector_construction/persona_vectors/ \\
        --output-dir outputs/reliability_projection/
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.reliability.compute_variant_projections import (
    compute_projections,
    load_activation_metadata,
    load_persona_vectors,
    mean_center_projections,
    save_projections,
    to_wide_format,
    PREPROCESSING_RAW,
    PREPROCESSING_CENTERED,
)

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute reliability variant projections onto persona vectors."
    )
    parser.add_argument(
        "--preprocessing",
        choices=["raw", "centered", "both"],
        default="both",
        help="Which projection variant(s) to compute (default: both).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[32, 40, 47],
        help="Layer indices to project at (default: 32 40 47).",
    )
    parser.add_argument(
        "--activation-metadata",
        default="outputs/reliability_projection/reliability_activation_metadata.parquet",
        help="Path to reliability_activation_metadata.parquet.",
    )
    parser.add_argument(
        "--vector-dir",
        default="outputs/vector_construction/persona_vectors/",
        help="Directory containing persona vector .npy files.",
    )
    parser.add_argument(
        "--vector-metadata",
        default="outputs/vector_construction/persona_vector_metadata.csv",
        help="Path to persona_vector_metadata.csv.",
    )
    parser.add_argument(
        "--vector-validation",
        default="outputs/vector_construction/vector_validation_results.csv",
        help="Path to vector_validation_results.csv (optional; used for AUC gate).",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.75,
        help="Minimum vector AUC to include (default: 0.75).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reliability_projection/",
        help="Directory to save projection outputs.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        default=True,
        help="Include scenario text columns in wide-format output (default: True).",
    )
    args = parser.parse_args()

    act_meta_path = _ROOT / args.activation_metadata
    vector_dir = _ROOT / args.vector_dir
    vector_meta_path = _ROOT / args.vector_metadata
    val_path = _ROOT / args.vector_validation
    out_dir = _ROOT / args.output_dir
    layers = args.layers

    print("\n  Stage 4C: Computing reliability variant projections")
    print(f"  Activation metadata: {act_meta_path}")
    print(f"  Vector dir:          {vector_dir}")
    print(f"  Layers:              {layers}")
    print(f"  Preprocessing:       {args.preprocessing}")
    print(f"  Output dir:          {out_dir}")

    # Load persona vectors
    val_path_arg = val_path if val_path.exists() else None
    vector_dict = load_persona_vectors(
        vector_dir=vector_dir,
        metadata_path=vector_meta_path,
        validation_path=val_path_arg,
        min_auc=args.min_auc,
    )
    print(f"\n  Loaded {len(vector_dict)} persona vectors ({', '.join(sorted(vector_dict.keys())[:6])} ...)")

    # Load activation metadata
    meta_df = load_activation_metadata(act_meta_path)
    print(f"  Loaded activation metadata: {len(meta_df)} rows")

    # Compute raw projections
    raw_long_df = compute_projections(meta_df, vector_dict, layers)
    print(f"  Raw projections computed: {len(raw_long_df)} rows")

    # Compute centered projections
    if args.preprocessing in ("centered", "both"):
        centered_long_df = mean_center_projections(raw_long_df)
        print(f"  Centered projections computed: {len(centered_long_df)} rows")

    # Build combined long df for saving
    if args.preprocessing == "raw":
        long_df = raw_long_df
    elif args.preprocessing == "centered":
        long_df = centered_long_df
    else:
        long_df = pd.concat([raw_long_df, centered_long_df], ignore_index=True)

    # Wide format (per preprocessing type)
    wide_dfs = []
    for pp_label, ldf in [
        (PREPROCESSING_RAW, raw_long_df),
        (PREPROCESSING_CENTERED, centered_long_df if args.preprocessing in ("centered", "both") else None),
    ]:
        if ldf is None:
            continue
        if args.preprocessing == "raw" and pp_label == PREPROCESSING_CENTERED:
            continue
        if args.preprocessing == "centered" and pp_label == PREPROCESSING_RAW:
            continue
        wdf = to_wide_format(ldf, include_text=args.include_text)
        wide_dfs.append(wdf)

    wide_df = pd.concat(wide_dfs, ignore_index=True) if len(wide_dfs) > 1 else wide_dfs[0]

    # Save
    saved = save_projections(long_df, wide_df, out_dir, preprocessing_label=args.preprocessing)
    print(f"\n  Saved outputs:")
    for name, path in saved.items():
        print(f"    {path}")

    print(
        "\n  Next: python scripts/diagnose_reliability_variant_projections.py"
    )


if __name__ == "__main__":
    main()
