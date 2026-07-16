#!/usr/bin/env python3
"""
Stage 1c — Export the curated MVP item bank.

Reads a validated annotation sheet, filters to keep_for_mvp == true items
whose primary_trait is one of the four construct traits, and saves the
curated set to data/processed/ethics_curated_mvp.{csv,parquet}.

Run from the project root:
    python scripts/export_curated_item_bank.py
    python scripts/export_curated_item_bank.py --input data/processed/ethics_annotation_sheet.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_mvp_config
from src.data.annotation import load_annotation_sheet, validate_annotations
from src.data.curation import (
    export_curated_item_bank,
    filter_keep_for_mvp,
    format_curation_report,
    produce_curation_report,
)

DEFAULT_INPUT = "data/processed/ethics_annotation_sheet.csv"
OUTPUT_CSV = "data/processed/ethics_curated_mvp.csv"
OUTPUT_PARQUET = "data/processed/ethics_curated_mvp.parquet"
CONSTRUCT_TRAITS = {"honesty", "harmlessness", "fairness", "compassion"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export curated MVP item bank")
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Annotation CSV to read (default: {DEFAULT_INPUT})",
    )
    parser.add_argument("--config", default="configs/mvp_experiment.yaml")
    parser.add_argument(
        "--min-per-trait", type=int, default=None,
        help="Override minimum items per trait for coverage warnings",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_mvp_config(root / args.config)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path

    output_csv = root / OUTPUT_CSV
    output_parquet = root / OUTPUT_PARQUET

    min_per_trait = (
        args.min_per_trait
        if args.min_per_trait is not None
        else cfg.annotation_pilot.minimum_items_per_trait_target
    )

    print("=" * 64)
    print("  Export Curated MVP Item Bank")
    print("=" * 64)
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_csv}")
    print()

    # Load
    try:
        df = load_annotation_sheet(input_path)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Validate first
    result = validate_annotations(df, annotation_cfg=cfg.annotation)
    if not result.is_valid:
        print(
            f"  ERROR: {len(result.invalid_rows)} invalid annotation label(s) found.\n"
            "  Run validate_annotations.py --input <file> and fix errors first."
        )
        for _, row in result.invalid_rows.iterrows():
            print(f"    {row['item_id']}: {row['errors']}")
        sys.exit(1)

    if result.annotated == 0:
        print("  ERROR: No items have been annotated yet.")
        print(f"  Fill in {input_path.name} and re-run.")
        sys.exit(1)

    # Filter
    curated = filter_keep_for_mvp(df)

    if len(curated) == 0:
        print(
            "  WARNING: No items have keep_for_mvp=true with a construct trait.\n"
            "  Annotate items and mark keep_for_mvp=true before exporting."
        )
        sys.exit(0)

    # Report on the full sheet (shows not_applicable / unclear counts too)
    report = produce_curation_report(df, min_per_trait=min_per_trait)

    print("[1] Annotation sheet summary")
    print(format_curation_report(report))

    # Export
    export_curated_item_bank(curated, output_csv, output_parquet)

    print(f"\n[2] Curated item bank exported")
    print(f"  {len(curated)} items retained")
    print(f"  CSV     : {output_csv}")
    print(f"  Parquet : {output_parquet}")

    # Per-trait counts in curated set
    print("\n[3] Curated set — trait distribution")
    for trait in sorted(CONSTRUCT_TRAITS):
        count = int((curated["primary_trait"] == trait).sum())
        flag = "  ⚠ LOW" if count < min_per_trait else ""
        print(f"  {trait:<18} {count:>4}{flag}")

    print("\n[4] Curated set — source_split distribution")
    for split, count in curated.groupby("source_split").size().items():
        print(f"  {split:<20} {count:>4}")

    # Warnings
    if report.has_warnings:
        print("\n[5] Warnings")
        for w in report.low_coverage_warnings + report.split_dominance_warnings:
            print(f"  ⚠  {w}")

    print("\n" + "=" * 64)
    if report.has_warnings:
        print("  STATUS: Curated bank exported with warnings (see above).")
        print("  Consider annotating more items before activation extraction.")
    else:
        print("  STATUS: Curated bank looks healthy.")
        print("  Next step: implement activation extraction (Stage 2).")
    print()
    print("  REMINDER: Do not begin activation extraction until persona vectors")
    print("  have been constructed and validated on held-out descriptions.")
    print("=" * 64)


if __name__ == "__main__":
    main()
