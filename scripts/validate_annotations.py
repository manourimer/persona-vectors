#!/usr/bin/env python3
"""
Validate an annotation sheet (full or pilot).

Loads the specified annotation CSV, validates all labels, and prints a
coverage + distribution report.

Run from the project root:
    # Full annotation sheet (default)
    python scripts/validate_annotations.py

    # Pilot sheet
    python scripts/validate_annotations.py --input data/processed/ethics_annotation_pilot.csv

    # Any manually edited file
    python scripts/validate_annotations.py --input path/to/my_annotations.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_mvp_config
from src.data.annotation import (
    annotation_coverage_report,
    load_annotation_sheet,
    validate_annotations,
)

DEFAULT_INPUT = "data/processed/ethics_annotation_sheet.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an annotation sheet")
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=(
            "Path to annotation CSV to validate. "
            f"Defaults to {DEFAULT_INPUT}"
        ),
    )
    # Keep --sheet as a hidden alias for backward compatibility
    parser.add_argument("--sheet", dest="input", help=argparse.SUPPRESS)
    parser.add_argument("--config", default="configs/mvp_experiment.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_mvp_config(root / args.config)

    sheet_path = Path(args.input)
    if not sheet_path.is_absolute():
        sheet_path = root / sheet_path

    print("=" * 64)
    print("  Annotation Validation Report")
    print("=" * 64)
    print(f"  File   : {sheet_path}")
    print()

    try:
        df = load_annotation_sheet(sheet_path)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    result = validate_annotations(df, annotation_cfg=cfg.annotation)

    # --- Coverage ---
    print("[1] Coverage")
    print(annotation_coverage_report(result))

    # --- Invalid rows ---
    print(f"\n[2] Invalid annotation labels  ({len(result.invalid_rows)} rows)")
    if result.invalid_rows.empty:
        print("  None — all annotated labels are valid.")
    else:
        for _, row in result.invalid_rows.iterrows():
            print(f"  {row['item_id']}")
            print(f"    {row['errors']}")

    # --- Special labels ---
    print(f"\n[3] Special labels")
    print(f"  not_applicable : {result.not_applicable_count}")
    print(f"  unclear        : {result.unclear_count}")
    print(f"  unannotated    : {result.unannotated}")

    # --- Summary verdict ---
    print("\n" + "=" * 64)
    if result.unannotated == result.total:
        print("  STATUS: No items annotated yet.")
        print(f"  Fill in {sheet_path.name} and re-run this script.")
    elif not result.is_valid:
        print(f"  STATUS: {len(result.invalid_rows)} invalid label(s) found.")
        print("  Fix the errors above and re-run.")
    else:
        kept = result.kept_for_mvp
        total = result.total
        print("  STATUS: All annotated labels are valid.")
        print(
            f"  {result.annotated}/{total} items annotated, "
            f"{kept} marked keep_for_mvp."
        )
    print("=" * 64)


if __name__ == "__main__":
    main()
