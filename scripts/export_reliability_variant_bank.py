"""
Stage 4B — Export accepted reliability variants for Stage 4C projection.

Merges raw generated variants with any human keep_variant overrides from
the review CSV, then exports keep_variant=True rows to the final variant bank.

Usage:
    python scripts/export_reliability_variant_bank.py
    python scripts/export_reliability_variant_bank.py --no-review-override
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_RAW = "data/processed/reliability_variants/ethics_reliability_variants_raw.parquet"
_DEFAULT_REVIEW = "data/processed/reliability_variants/reliability_variants_review.csv"
_DEFAULT_OUT_DIR = "data/processed/reliability_variants"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export accepted reliability variants for Stage 4C."
    )
    parser.add_argument("--raw-input", default=_DEFAULT_RAW)
    parser.add_argument("--review-csv", default=_DEFAULT_REVIEW)
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    parser.add_argument(
        "--no-review-override", action="store_true",
        help="Use keep_variant from generated file only; ignore review CSV.",
    )
    args = parser.parse_args()

    import pandas as pd
    from src.reliability.variant_generation import (
        VARIANT_COLUMNS,
        load_variants,
        save_variants,
    )
    from src.reliability.variant_validation import validate_variant_bank

    raw_path = _ROOT / args.raw_input
    if not raw_path.exists():
        print(f"\n  ERROR: Raw variant bank not found: {raw_path}")
        print("  Run: python scripts/generate_reliability_variants.py")
        sys.exit(1)

    df = load_variants(raw_path)

    # Apply human overrides from review CSV
    review_path = _ROOT / args.review_csv
    if not args.no_review_override and review_path.exists():
        review_df = pd.read_csv(review_path, usecols=["variant_id", "keep_variant"])
        overrides = review_df.set_index("variant_id")["keep_variant"].to_dict()
        n_overrides = 0
        for idx, row in df.iterrows():
            vid = row["variant_id"]
            if vid in overrides:
                new_val = str(overrides[vid]).strip().lower() in {"true", "1", "yes"}
                if new_val != bool(row["keep_variant"]):
                    df.at[idx, "keep_variant"] = new_val
                    n_overrides += 1
        if n_overrides:
            print(f"  Applied {n_overrides} keep_variant override(s) from review CSV.")
    else:
        if not args.no_review_override:
            print(f"  No review CSV found at {review_path}; using generated keep_variant values.")

    # Export only kept variants
    final_df = df[df["keep_variant"].astype(bool)].copy().reset_index(drop=True)
    out_dir = _ROOT / args.out_dir
    csv_path, parquet_path = save_variants(final_df, out_dir, "ethics_reliability_variants")

    # Validation summary
    result = validate_variant_bank(final_df)

    print(f"\n  ── Stage 4B: Export Complete ───────────────────────────────────")
    print(f"  Total kept rows  : {len(final_df)}")
    print(f"  Originals        : {result['n_originals']}")
    print(f"  Paraphrases      : {result['n_paraphrases']}")
    print(f"  Items            : {result['n_items']}")

    print(f"\n  Status distribution (kept only):")
    for status, count in sorted(result["status_counts"].items()):
        print(f"    {status:<40} {count:>4}")

    print(f"\n  Output files:")
    print(f"    {parquet_path}")
    print(f"    {csv_path}")
    print(f"\n  Next: Stage 4C — project all accepted variants using Stage 3 machinery")
    print(f"    python scripts/extract_variant_activations.py  (to be implemented)")


if __name__ == "__main__":
    main()
