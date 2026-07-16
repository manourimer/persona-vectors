#!/usr/bin/env python3
"""
Diagnostic: commonsense-only MVP comparison.

Reports what the curated item bank would look like if restricted to
source_split == 'commonsense'.  This is a fallback diagnostic, not the
default behaviour.  Use it when the multi-split sample produces poor
per-trait coverage after annotation.

Run from the project root:
    python scripts/compare_commonsense_only.py
    python scripts/compare_commonsense_only.py --input data/processed/ethics_annotation_sheet.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.annotation import load_annotation_sheet
from src.data.curation import (
    commonsense_only_summary,
    filter_keep_for_mvp,
    format_curation_report,
    produce_curation_report,
)

DEFAULT_INPUT = "data/processed/ethics_annotation_sheet.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Commonsense-only MVP diagnostic"
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Annotation CSV (default: {DEFAULT_INPUT})",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path

    try:
        df = load_annotation_sheet(input_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("=" * 64)
    print("  Commonsense-Only MVP Diagnostic")
    print("  (This is a diagnostic view — NOT the default behaviour)")
    print("=" * 64)
    print(f"  Input: {input_path}")
    print()

    # Full-sample summary
    curated_all = filter_keep_for_mvp(df)
    report_all = produce_curation_report(df)
    print("[1] Full sample (all source splits)")
    print(format_curation_report(report_all))

    # Commonsense-only summary
    print("\n[2] Commonsense-only comparison")
    print(commonsense_only_summary(df))

    # Side-by-side trait counts
    cs_only = df[df["source_split"] == "commonsense"]
    curated_cs = filter_keep_for_mvp(cs_only)
    CONSTRUCT_TRAITS = {"honesty", "harmlessness", "fairness", "compassion"}

    print("\n[3] Side-by-side: retained items per trait")
    print(f"  {'Trait':<18} {'All splits':>12} {'Commonsense only':>18}")
    print(f"  {'-'*18} {'-'*12} {'-'*18}")
    for trait in sorted(CONSTRUCT_TRAITS):
        all_count = int((curated_all["primary_trait"] == trait).sum()) if len(curated_all) else 0
        cs_count = int((curated_cs["primary_trait"] == trait).sum()) if len(curated_cs) else 0
        print(f"  {trait:<18} {all_count:>12} {cs_count:>18}")
    total_all = len(curated_all)
    total_cs = len(curated_cs)
    print(f"  {'TOTAL':<18} {total_all:>12} {total_cs:>18}")

    print("\n" + "=" * 64)
    print("  To use commonsense-only as the MVP item bank:")
    print("  Set dataset.ethics_splits: [commonsense] in")
    print("  configs/mvp_experiment.yaml and re-run build_ethics_sample.py.")
    print()
    print("  Only do this if the multi-split curated bank has poor trait")
    print("  coverage after full annotation.")
    print("=" * 64)


if __name__ == "__main__":
    main()
