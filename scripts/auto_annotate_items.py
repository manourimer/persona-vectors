#!/usr/bin/env python3
"""
Stage 1c — LLM-assisted annotation of the ETHICS item bank.

Produces FIRST-PASS SUGGESTIONS for human review.
Auto-labels are NOT ground truth and should be inspected before export.

Run from the project root:

  # Inspect prompts without calling the API
  python scripts/auto_annotate_items.py --dry-run

  # Annotate first 10 items (test run)
  python scripts/auto_annotate_items.py --limit 10

  # Annotate everything, skipping already-annotated rows
  python scripts/auto_annotate_items.py --resume

  # Custom input / output
  python scripts/auto_annotate_items.py \\
      --input  data/processed/ethics_annotation_pilot.csv \\
      --output data/processed/ethics_annotation_pilot_autolabeled.csv

Requires:
  ANTHROPIC_API_KEY environment variable (ignored in --dry-run mode)
  pip install anthropic  (or: pip install -e ".[annotation]")
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "data/processed/ethics_annotation_sheet.csv"
DEFAULT_OUTPUT = "data/processed/ethics_annotation_sheet_autolabeled.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-assisted annotation of ETHICS items (suggestions for human review)"
    )
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT)
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Annotate only this many rows (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without calling the API",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip rows that already have a non-empty primary_trait",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override the annotation model (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds between API calls (default: 0.3)",
    )
    parser.add_argument("--config", default="configs/mvp_experiment.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent

    # ---- imports after path setup ----
    from src.config import load_mvp_config, load_trait_space
    from src.data.auto_annotation import (
        ANNOTATION_MODEL,
        annotate_dataframe,
        load_annotation_sheet_for_auto,
        save_autolabeled_sheet,
    )

    cfg = load_mvp_config(root / args.config)
    trait_space = load_trait_space(root / cfg.traits.path)
    model = args.model or ANNOTATION_MODEL

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    print("=" * 64)
    print("  LLM-Assisted Annotation")
    print("  ⚠  Auto-labels are SUGGESTIONS — human review required.")
    print("=" * 64)
    print(f"  Input   : {input_path}")
    print(f"  Output  : {output_path}")
    print(f"  Model   : {model}")
    print(f"  Limit   : {args.limit or 'all rows'}")
    print(f"  Resume  : {args.resume}")
    print(f"  Dry run : {args.dry_run}")
    print()

    # Load sheet
    try:
        df = load_annotation_sheet_for_auto(input_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if args.limit:
        df = df.head(args.limit).copy()
        print(f"  (Limiting to first {args.limit} rows)")

    # API client — only needed outside dry-run
    client = None
    if not args.dry_run:
        try:
            import anthropic
        except ImportError:
            print(
                "ERROR: 'anthropic' package not installed.\n"
                "  Run: pip install anthropic\n"
                "  Or:  pip install -e '.[annotation]'"
            )
            sys.exit(1)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "ERROR: ANTHROPIC_API_KEY environment variable not set.\n"
                "  Export it: export ANTHROPIC_API_KEY=sk-ant-..."
            )
            sys.exit(1)

        client = anthropic.Anthropic(api_key=api_key)

    # Annotate
    result = annotate_dataframe(
        df=df,
        trait_space=trait_space,
        client=client,
        model=model,
        resume=args.resume,
        rate_limit_delay=args.delay if not args.dry_run else 0.0,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\n[DRY RUN complete — no file written, no API calls made]")
        print("Remove --dry-run to run actual annotation.")
        return

    # Save
    save_autolabeled_sheet(result, output_path)

    # Summary
    print(f"\n  Saved: {output_path}")
    annotated = (result["primary_trait"].str.strip() != "").sum()
    print(f"  Rows with primary_trait filled: {annotated} / {len(result)}")

    pt_counts = result["primary_trait"].value_counts()
    print("\n  primary_trait distribution:")
    for trait, count in pt_counts.items():
        print(f"    {trait:<18} {count:>4}")

    print()
    print("  Next steps:")
    print("  1. Review the autolabels:")
    print(f"     python scripts/review_autolabels.py --input {output_path.name}")
    print("  2. Correct errors in the CSV (especially low-confidence rows).")
    print("  3. Validate corrected annotations:")
    print(f"     python scripts/validate_annotations.py --input {output_path.name}")
    print("  4. Export the curated item bank:")
    print("     python scripts/export_curated_item_bank.py \\")
    print(f"       --input {output_path.name}")
    print("=" * 64)


if __name__ == "__main__":
    main()
