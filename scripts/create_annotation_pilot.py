#!/usr/bin/env python3
"""
Stage 1c — Create the annotation pilot sheet.

Samples a small subset (default 50 items) from the full annotation sheet,
stratified by source_split, so you can annotate a manageable pilot first,
check trait coverage, then annotate the remainder.

Run from the project root:
    python scripts/create_annotation_pilot.py
    python scripts/create_annotation_pilot.py --n 30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_mvp_config
from src.data.curation import create_annotation_pilot_sheet

INPUT_SHEET = "data/processed/ethics_annotation_sheet.csv"
OUTPUT_PILOT = "data/processed/ethics_annotation_pilot.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create annotation pilot sheet")
    parser.add_argument("--config", default="configs/mvp_experiment.yaml")
    parser.add_argument("--n", type=int, default=None,
                        help="Override pilot_n_items from config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override pilot random seed")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_mvp_config(root / args.config)
    pilot_cfg = cfg.annotation_pilot

    n_items = args.n if args.n is not None else pilot_cfg.pilot_n_items
    seed = args.seed if args.seed is not None else pilot_cfg.effective_seed(cfg.random_seed)
    stratify = pilot_cfg.pilot_stratify_by_source_split

    input_path = root / INPUT_SHEET
    output_path = root / OUTPUT_PILOT

    if not input_path.exists():
        print(
            f"ERROR: Annotation sheet not found: {input_path}\n"
            "Run: python scripts/create_annotation_sheet.py"
        )
        sys.exit(1)

    print("=" * 64)
    print("  Annotation Pilot Sheet")
    print("=" * 64)
    print(f"  Input sheet  : {input_path}")
    print(f"  Output pilot : {output_path}")
    print(f"  Items        : {n_items}")
    print(f"  Seed         : {seed}")
    print(f"  Stratify     : {stratify}")
    print()

    pilot = create_annotation_pilot_sheet(
        input_sheet=input_path,
        output_path=output_path,
        n_items=n_items,
        seed=seed,
        stratify_by_source_split=stratify,
    )

    print(f"  Pilot written: {len(pilot)} items")
    from collections import Counter
    split_counts = Counter(pilot["source_split"])
    for split, count in sorted(split_counts.items()):
        print(f"    {split:<20} {count:>3} items")

    print()
    print("  Next steps:")
    print(f"  1. Open {output_path.name} in a spreadsheet editor")
    print("  2. Fill in primary_trait for each row using the rubric in")
    print("     configs/traits.yaml (item_tagging_guidance per trait)")
    print()
    print("  Valid primary_trait values:")
    print("    honesty | harmlessness | fairness | compassion")
    print("    not_applicable  (leave secondary_traits blank)")
    print("    unclear")
    print()
    print("  3. Validate the pilot with:")
    print("     python scripts/validate_annotations.py \\")
    print(f"       --input {OUTPUT_PILOT}")
    print()
    print("  4. Check that each of the 4 traits has reasonable coverage.")
    print(f"     Target: ≥ {pilot_cfg.minimum_items_per_trait_target} items per trait")
    print("     in the full annotation before starting activation extraction.")
    print()
    print("  5. If coverage looks good, annotate the full sheet:")
    print(f"     {INPUT_SHEET}")
    print("     Then export with:")
    print("     python scripts/export_curated_item_bank.py \\")
    print(f"       --input {INPUT_SHEET}")
    print("=" * 64)


if __name__ == "__main__":
    main()
