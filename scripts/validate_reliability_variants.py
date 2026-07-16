"""
Stage 4B — Validate the reliability variant bank.

Checks schema, counts, duplicates, and semantic-equivalence status distribution.

Usage:
    python scripts/validate_reliability_variants.py
    python scripts/validate_reliability_variants.py --input data/processed/reliability_variants/ethics_reliability_variants_raw.parquet
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_INPUT = "data/processed/reliability_variants/ethics_reliability_variants_raw.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate reliability variant bank.")
    parser.add_argument("--input", default=_DEFAULT_INPUT)
    parser.add_argument("--n-paraphrases", type=int, default=3)
    args = parser.parse_args()

    import pandas as pd
    from src.reliability.variant_generation import load_variants
    from src.reliability.variant_validation import validate_variant_bank

    input_path = _ROOT / args.input
    if not input_path.exists():
        print(f"\n  ERROR: Variant bank not found: {input_path}")
        print("  Run: python scripts/generate_reliability_variants.py")
        sys.exit(1)

    df = load_variants(input_path)

    print(f"\n  ── Stage 4B: Variant Bank Validation ──────────────────────────")
    print(f"  File: {input_path}")

    result = validate_variant_bank(df, expected_n_paraphrases=args.n_paraphrases)

    print(f"\n  Counts")
    print(f"    Items              : {result['n_items']}")
    print(f"    Originals          : {result['n_originals']}")
    print(f"    Paraphrases        : {result['n_paraphrases']}")
    print(f"    Total rows         : {result['n_total']}")
    print(f"    keep_variant=True  : {result['n_keep']}")

    print(f"\n  Semantic equivalence status")
    for status, count in sorted(result["status_counts"].items()):
        marker = "⚠ " if "flagged" in status or "failed" in status else "  "
        print(f"    {marker}{status:<40} {count:>4}")

    print(f"\n  Completeness")
    print(f"    Items with {args.n_paraphrases} paraphrases : {result['items_with_complete_paraphrases']}")
    incomplete = result["items_with_incomplete_paraphrases"]
    if incomplete:
        print(f"    Items with fewer paraphrases : {len(incomplete)}")
        for item_id in incomplete[:10]:
            print(f"      - {item_id}")
        if len(incomplete) > 10:
            print(f"      … and {len(incomplete)-10} more")

    print(f"\n  Schema")
    if result["missing_columns"]:
        print(f"    ❌ Missing columns: {result['missing_columns']}")
    else:
        print(f"    ✅ All required columns present")

    if result["duplicate_variant_ids"]:
        print(f"    ❌ Duplicate variant_ids: {len(result['duplicate_variant_ids'])}")
    else:
        print(f"    ✅ All variant_ids unique")

    if result["items_with_empty_kept_text"]:
        print(f"    ❌ keep_variant=True rows with empty text: {len(result['items_with_empty_kept_text'])}")
    else:
        print(f"    ✅ No empty text in kept variants")

    print(f"\n  By primary_trait (paraphrases only)")
    para_df = df[df["variant_type"] == "paraphrase"]
    for trait, count in para_df["primary_trait"].value_counts().items():
        print(f"    {trait:<20} {count:>4}")

    print(f"\n  By source_split (paraphrases only)")
    for split, count in para_df["source_split"].value_counts().items():
        print(f"    {split:<20} {count:>4}")

    overall = "✅ Valid" if result["is_valid"] else "❌ Issues found"
    print(f"\n  Overall: {overall}")
    if not result["is_valid"]:
        print("  Run scripts/review_reliability_variants.py for flagged items.")
        sys.exit(1)


if __name__ == "__main__":
    main()
