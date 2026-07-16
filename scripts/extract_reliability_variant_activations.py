"""
Stage 4C: Thin wrapper to invoke the Modal reliability variant activation extractor.

Checks modal is installed, then invokes:
    modal run modal_apps/extract_reliability_variant_activations.py [flags]

Usage:
    # Smoke test (20 variants)
    python scripts/extract_reliability_variant_activations.py --limit 20

    # Full extraction
    python scripts/extract_reliability_variant_activations.py

    # Resume interrupted run
    python scripts/extract_reliability_variant_activations.py --resume
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract last-prompt-token activations for reliability variants via Modal."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to N variants for smoke testing (0 = all).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip variants whose activation files already exist.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Variants per Modal batch (default: 8).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reliability_projection",
        help="Output directory for activations and metadata.",
    )
    parser.add_argument(
        "--jobs-path",
        default="outputs/reliability_projection/reliability_projection_jobs.parquet",
        help="Path to projection jobs parquet from build_reliability_projection_jobs.py.",
    )
    parser.add_argument(
        "--candidate-layers",
        default="32,40,47",
        help="Comma-separated layer indices (default: 32,40,47).",
    )
    args = parser.parse_args()

    # Check modal is installed
    if shutil.which("modal") is None:
        print("ERROR: 'modal' CLI not found.  Install with: pip install modal")
        sys.exit(1)

    # Check jobs parquet exists
    jobs_path = _ROOT / args.jobs_path
    if not jobs_path.exists():
        print(
            f"ERROR: Jobs parquet not found: {jobs_path}\n"
            "Run first: python scripts/build_reliability_projection_jobs.py"
        )
        sys.exit(1)

    modal_app = _ROOT / "modal_apps" / "extract_reliability_variant_activations.py"

    cmd = [
        "modal", "run", str(modal_app),
        f"--jobs-path={args.jobs_path}",
        f"--candidate-layers={args.candidate_layers}",
        f"--batch-size={args.batch_size}",
        f"--out-dir={args.output_dir}",
    ]
    if args.limit:
        cmd.append(f"--limit={args.limit}")
    if args.resume:
        cmd.append("--resume")

    print(f"\n  Stage 4C: Extracting reliability variant activations via Modal")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
