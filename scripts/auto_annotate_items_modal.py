#!/usr/bin/env python3
"""
Run the Modal/vLLM annotation app on the ETHICS item bank.

This is the primary automated annotation path.  It invokes:
    modal run modal_apps/annotate_ethics_items.py [args]

Pre-requisites (one-time):
    pip install modal
    modal token new           # authenticate your Modal account
    # Qwen2.5 is public — no HuggingFace login required.
    # Model weights are downloaded to a Modal Volume on first run and cached.

Smoke test (10 items, ~2 min including cold-start):
    python scripts/auto_annotate_items_modal.py --limit 10

Full annotation run (300 items):
    python scripts/auto_annotate_items_modal.py

Resume after interruption:
    python scripts/auto_annotate_items_modal.py --resume

Review and validate output:
    python scripts/review_autolabels.py
    python scripts/validate_annotations.py \\
        --input data/processed/ethics_annotation_sheet_autolabeled.csv
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODAL_APP = ROOT / "modal_apps" / "annotate_ethics_items.py"


def _check_modal() -> None:
    if shutil.which("modal") is None:
        print(
            "ERROR: 'modal' CLI not found.\n"
            "\n"
            "Install it with:\n"
            "    pip install modal\n"
            "\n"
            "Then authenticate once:\n"
            "    modal token new\n"
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate the ETHICS item bank using an open-source model on Modal/vLLM. "
            "Outputs first-pass suggestions for human review."
        )
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed/ethics_annotation_sheet.csv",
        help="Annotation sheet to annotate (default: %(default)s)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/ethics_annotation_sheet_autolabeled.csv",
        help="Output path for autolabeled sheet (default: %(default)s)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Annotate only this many rows. Use 10 for a smoke test (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Items per Modal inference call (default: %(default)s)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip rows that already have a non-empty primary_trait",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID to use (default: %(default)s)",
    )
    parser.add_argument(
        "--non-strict", dest="strict", action="store_false", default=True,
        help=(
            "In non-strict mode, parse/validation failures become "
            "unclear/low/false instead of a warning"
        ),
    )
    args = parser.parse_args()

    _check_modal()

    # Build the modal run command.
    # Modal's local_entrypoint receives keyword args directly.
    cmd = [
        "modal", "run", str(MODAL_APP),
        "--input", args.input,
        "--output", args.output,
        "--batch-size", str(args.batch_size),
        "--model-name", args.model_name,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.resume:
        cmd += ["--resume"]
    if not args.strict:
        cmd += ["--strict=False"]

    print("Running:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
