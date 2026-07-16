#!/usr/bin/env python3
"""
Stage 1b — Create the manual annotation sheet.

Reads data/processed/ethics_mvp_sample.parquet (built by build_ethics_sample.py)
and writes data/processed/ethics_annotation_sheet.csv with blank annotation
columns ready to fill in.

Run from the project root:
    python scripts/create_annotation_sheet.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.annotation import create_annotation_sheet

SAMPLE_PARQUET = "data/processed/ethics_mvp_sample.parquet"
SAMPLE_CSV = "data/processed/ethics_mvp_sample.csv"
OUTPUT_PATH = "data/processed/ethics_annotation_sheet.csv"


def main() -> None:
    import pandas as pd

    root = Path(__file__).resolve().parent.parent

    # Load item bank
    parquet = root / SAMPLE_PARQUET
    csv = root / SAMPLE_CSV
    if parquet.exists():
        item_bank = pd.read_parquet(parquet)
    elif csv.exists():
        item_bank = pd.read_csv(csv)
    else:
        print(
            f"ERROR: No item bank found.\n"
            f"  Expected: {parquet}\n"
            f"  Run: python scripts/build_ethics_sample.py"
        )
        sys.exit(1)

    output_path = root / OUTPUT_PATH
    sheet = create_annotation_sheet(item_bank, output_path)

    print("=" * 64)
    print("  Annotation Sheet Created")
    print("=" * 64)
    print(f"  Items      : {len(sheet)}")
    print(f"  Output     : {output_path}")
    print()
    print("  Columns to fill in:")
    print("    primary_trait       — main construct trait engaged by this item")
    print("    secondary_traits    — optional, comma-separated secondary traits")
    print("    annotation_confidence — high / medium / low")
    print("    annotation_notes    — free-text rationale (optional)")
    print("    keep_for_mvp        — true / false")
    print()
    print("  Valid primary_trait values:")
    print("    honesty         — item engages truth-telling / non-deception")
    print("    harmlessness    — item engages harm avoidance / safety")
    print("    fairness        — item engages equal treatment / non-discrimination")
    print("    compassion      — item engages empathy / response to suffering")
    print("    not_applicable  — item does not engage any target trait")
    print("                      (leave secondary_traits BLANK)")
    print("    unclear         — annotator cannot confidently assign a trait")
    print()
    print("  Valid secondary_traits values (construct traits only):")
    print("    honesty | harmlessness | fairness | compassion")
    print("    (not_applicable and unclear are NOT valid secondary values)")
    print()
    print("  Annotation rubric: see configs/traits.yaml")
    print("    → item_tagging_guidance per trait")
    print("    → example_keywords_and_patterns per trait")
    print()
    print("  NOTE: ETHICS split names (commonsense, deontology, justice …)")
    print("  are NOT valid trait labels. They describe how items were collected,")
    print("  not which moral construct the item measures.")
    print()
    print("  After annotating, run:")
    print("    python scripts/validate_annotations.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
