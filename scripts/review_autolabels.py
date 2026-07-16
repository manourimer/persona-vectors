#!/usr/bin/env python3
"""
Review the autolabeled annotation sheet before human correction.

Prints:
  - Counts by primary_trait, annotation_confidence, keep_for_mvp
  - source_split × primary_trait table
  - All low-confidence rows (for priority review)
  - All unclear / not_applicable rows
  - keep_for_mvp=true rows with medium/low confidence (highest-risk rows)
  - Trait imbalance warnings

Run from the project root:
  python scripts/review_autolabels.py
  python scripts/review_autolabels.py --input data/processed/ethics_annotation_pilot_autolabeled.csv
  python scripts/review_autolabels.py --show-text   # include scenario text in row previews
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_INPUT = "data/processed/ethics_annotation_sheet_autolabeled.csv"
CONSTRUCT_TRAITS = {"honesty", "harmlessness", "fairness", "compassion"}
MIN_PER_TRAIT = 20  # warn if fewer than this in the keep_for_mvp=true set


def _bool_mask(series: pd.Series, truthy: set[str]) -> pd.Series:
    return series.str.strip().str.lower().isin(truthy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review autolabeled annotation sheet")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT)
    parser.add_argument(
        "--show-text", action="store_true",
        help="Include (truncated) scenario text in row previews",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path

    if not input_path.exists():
        print(
            f"ERROR: File not found: {input_path}\n"
            "Run: python scripts/auto_annotate_items.py first."
        )
        sys.exit(1)

    df = pd.read_csv(input_path, dtype=str).fillna("")

    print("=" * 66)
    print("  Autolabel Review Report")
    print(f"  File: {input_path.name}")
    print("=" * 66)
    print(f"\n  Total rows: {len(df)}")

    # 1. primary_trait distribution
    print("\n[1] Primary trait distribution")
    pt_counts = df["primary_trait"].value_counts()
    for trait, count in pt_counts.items():
        bar = "█" * min(count, 40)
        print(f"  {trait:<18} {count:>4}  {bar}")

    # 2. annotation_confidence
    print("\n[2] Annotation confidence")
    for conf, count in df["annotation_confidence"].value_counts().items():
        print(f"  {conf:<10} {count:>4}")

    # 3. keep_for_mvp
    keep_mask = _bool_mask(df["keep_for_mvp"], {"true", "yes", "1"})
    print(f"\n[3] keep_for_mvp")
    print(f"  true   {keep_mask.sum():>4}")
    print(f"  false  {(~keep_mask).sum():>4}")

    # 4. source_split × primary_trait
    print("\n[4] Source split × primary trait")
    try:
        crosstab = (
            df.groupby(["source_split", "primary_trait"])
            .size()
            .unstack(fill_value=0)
        )
        print("  " + crosstab.to_string().replace("\n", "\n  "))
    except Exception:
        print("  (could not build cross-tab)")

    # 5. Low-confidence rows — REVIEW THESE FIRST
    low_conf = df[df["annotation_confidence"].str.strip() == "low"]
    print(f"\n[5] Low-confidence rows ({len(low_conf)}) — REVIEW FIRST")
    _print_rows(low_conf, args.show_text)

    # 6. unclear / not_applicable
    special = df[df["primary_trait"].str.strip().isin({"unclear", "not_applicable"})]
    print(f"\n[6] Special labels: unclear / not_applicable ({len(special)})")
    _print_rows(special, args.show_text, max_rows=20)

    # 7. keep_for_mvp=true with medium/low confidence — highest risk
    risky = df[
        keep_mask
        & df["annotation_confidence"].str.strip().isin({"medium", "low"})
    ]
    print(f"\n[7] keep_for_mvp=true with medium/low confidence ({len(risky)}) — HIGHEST RISK")
    _print_rows(risky, args.show_text)

    # 8. Trait imbalance warnings
    kept_traits = df[keep_mask & df["primary_trait"].str.strip().isin(CONSTRUCT_TRAITS)]
    print(f"\n[8] Trait coverage in keep_for_mvp=true set")
    warnings: list[str] = []
    for trait in sorted(CONSTRUCT_TRAITS):
        count = int((kept_traits["primary_trait"].str.strip() == trait).sum())
        flag = ""
        if count < MIN_PER_TRAIT:
            flag = f"  ⚠ LOW (target ≥ {MIN_PER_TRAIT})"
            warnings.append(f"'{trait}' has only {count} retained items.")
        print(f"  {trait:<18} {count:>4}{flag}")

    if warnings:
        print("\n  ⚠  Trait imbalance warnings:")
        for w in warnings:
            print(f"    • {w}")
        print(
            "  Consider reviewing 'not_applicable' and 'unclear' items to see "
            "if any should be reclassified to an underrepresented trait."
        )

    print("\n" + "=" * 66)
    print("  After reviewing and correcting the CSV, validate with:")
    print(f"    python scripts/validate_annotations.py --input {input_path.name}")
    print("  Then export the curated item bank:")
    print("    python scripts/export_curated_item_bank.py \\")
    print(f"      --input {input_path.name}")
    print("=" * 66)


def _print_rows(
    df: pd.DataFrame,
    show_text: bool,
    max_rows: int = 30,
) -> None:
    if df.empty:
        print("  (none)")
        return
    shown = df.head(max_rows)
    for _, row in shown.iterrows():
        pt = row.get("primary_trait", "")
        conf = row.get("annotation_confidence", "")
        keep = row.get("keep_for_mvp", "")
        notes = str(row.get("annotation_notes", ""))[:80]
        print(
            f"  {row.get('item_id', ''):<36}  "
            f"trait={pt:<16}  conf={conf:<6}  keep={keep}"
        )
        if notes:
            print(f"    note: {notes}")
        if show_text:
            text = str(row.get("scenario_text", ""))[:100]
            print(f"    text: {text}")
    if len(df) > max_rows:
        print(f"  … ({len(df) - max_rows} more rows not shown)")


if __name__ == "__main__":
    main()
