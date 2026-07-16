"""
Stage 4B — Review flagged reliability variants.

Prints all flagged or failed variants and exports a review CSV for
manual inspection before exporting the final variant bank.

Usage:
    python scripts/review_reliability_variants.py
    python scripts/review_reliability_variants.py --show-passed
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_INPUT = "data/processed/reliability_variants/ethics_reliability_variants_raw.parquet"
_DEFAULT_REVIEW_OUT = "data/processed/reliability_variants/reliability_variants_review.csv"

_FLAGGED_STATUSES = {
    "flagged_length",
    "flagged_duplicate",
    "flagged_possible_meaning_shift",
    "failed_parse",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Review flagged reliability variants.")
    parser.add_argument("--input", default=_DEFAULT_INPUT)
    parser.add_argument("--review-output", default=_DEFAULT_REVIEW_OUT)
    parser.add_argument("--show-passed", action="store_true", help="Also print passed variants")
    parser.add_argument("--max-print", type=int, default=30, help="Max flagged rows to print")
    args = parser.parse_args()

    import pandas as pd
    from src.reliability.variant_generation import load_variants

    input_path = _ROOT / args.input
    if not input_path.exists():
        print(f"\n  ERROR: Variant bank not found: {input_path}")
        print("  Run: python scripts/generate_reliability_variants.py")
        sys.exit(1)

    df = load_variants(input_path)
    flagged = df[df["semantic_equivalence_status"].isin(_FLAGGED_STATUSES)]

    print(f"\n  ── Stage 4B: Flagged Variant Review ────────────────────────────")
    print(f"  Total variants  : {len(df)}")
    print(f"  Flagged/failed  : {len(flagged)}")
    print(f"  Passed          : {int((df['semantic_equivalence_status']=='passed').sum())}")

    if len(flagged) == 0:
        print("\n  ✅ No flagged variants. Proceed to export.")
    else:
        print(f"\n  Status breakdown:")
        for status, count in flagged["semantic_equivalence_status"].value_counts().items():
            print(f"    {status:<45} {count:>4}")

        print(f"\n  First {min(args.max_print, len(flagged))} flagged variants:\n")
        for i, (_, row) in enumerate(flagged.head(args.max_print).iterrows()):
            print(f"  [{i+1}] {row['variant_id']}  |  {row['semantic_equivalence_status']}")
            print(f"       trait: {row['primary_trait']}  split: {row['source_split']}")
            print(f"       ORIGINAL : {row['scenario_text_original'][:120]}")
            print(f"       VARIANT  : {row['scenario_text_variant'][:120]}")
            if row["generation_notes"]:
                print(f"       notes    : {row['generation_notes'][:80]}")
            print()

        if len(flagged) > args.max_print:
            print(f"  … {len(flagged) - args.max_print} more flagged variants not shown.")

    # Export review CSV
    review_cols = [
        "variant_id", "item_id", "primary_trait", "source_split",
        "semantic_equivalence_status", "keep_variant",
        "scenario_text_original", "scenario_text_variant",
        "generation_notes",
    ]
    review_df = df[df["variant_type"] == "paraphrase"][review_cols].copy()
    review_path = _ROOT / args.review_output
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(review_path, index=False)
    print(f"  Review CSV exported: {review_path}")
    print(f"\n  To accept/reject variants, edit keep_variant in the review CSV,")
    print(f"  then run: python scripts/export_reliability_variant_bank.py")


if __name__ == "__main__":
    main()
