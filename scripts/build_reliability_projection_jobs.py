"""
Stage 4C: Build reliability variant projection jobs parquet.

Loads the accepted variant bank from Stage 4B, constructs forward-pass job
rows (one per variant × layer), and saves to
outputs/reliability_projection/reliability_projection_jobs.parquet.

Usage:
    python scripts/build_reliability_projection_jobs.py
    python scripts/build_reliability_projection_jobs.py \\
        --variant-bank data/processed/reliability_variants/ethics_reliability_variants.parquet \\
        --layers 32 40 47 \\
        --out-dir outputs/reliability_projection
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.reliability.variant_projection import (
    build_projection_jobs,
    load_variant_bank,
    save_projection_jobs,
    validate_projection_jobs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reliability variant projection jobs parquet."
    )
    parser.add_argument(
        "--variant-bank",
        default="data/processed/reliability_variants/ethics_reliability_variants.parquet",
        help="Path to the accepted reliability variant bank parquet.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[32, 40, 47],
        help="Transformer layer indices to extract at (default: 32 40 47).",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/reliability_projection",
        help="Output directory for jobs files.",
    )
    parser.add_argument(
        "--token-position",
        default="last_prompt_token",
        help="Token position label (default: last_prompt_token).",
    )
    args = parser.parse_args()

    variant_bank_path = _ROOT / args.variant_bank
    out_dir = _ROOT / args.out_dir
    layers = args.layers

    print("\n  Stage 4C: Building reliability variant projection jobs")
    print(f"  Variant bank: {variant_bank_path}")
    print(f"  Layers: {layers}")
    print(f"  Token position: {args.token_position}")
    print(f"  Output dir: {out_dir}")

    # Load and filter variant bank
    variant_df = load_variant_bank(variant_bank_path)
    n_items = variant_df["item_id"].nunique()
    n_variants = len(variant_df)
    print(f"\n  Loaded variant bank: {n_variants} accepted variants across {n_items} items")

    # Build jobs
    jobs_df = build_projection_jobs(variant_df, layers, token_position=args.token_position)
    n_jobs = len(jobs_df)
    print(f"  Built {n_jobs} projection jobs ({n_variants} variants × {len(layers)} layers)")

    # Validate
    summary = validate_projection_jobs(jobs_df)
    if summary["warnings"]:
        print(f"\n  WARNINGS:")
        for w in summary["warnings"]:
            print(f"    - {w}")
    else:
        print("  Validation passed: no warnings.")

    # Save
    pq_path, csv_path = save_projection_jobs(jobs_df, out_dir, stem="reliability_projection_jobs")
    print(f"\n  Saved jobs parquet: {pq_path}")
    print(f"  Saved jobs CSV:     {csv_path}")

    print(f"\n  Summary:")
    print(f"    Items:      {summary['n_items']}")
    print(f"    Variants:   {summary['n_variants']}")
    print(f"    Layers:     {summary['n_layers']}")
    print(f"    Total jobs: {summary['n_jobs']}")
    print(f"    Token pos:  {args.token_position}")
    print(
        "\n  Next: python scripts/extract_reliability_variant_activations.py --limit 20  # smoke test"
    )


if __name__ == "__main__":
    main()
