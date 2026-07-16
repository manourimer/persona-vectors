"""
Stage 4C: Full pipeline smoke test (20 variants end to end).

Runs:
    1. build_reliability_projection_jobs.py
    2. extract_reliability_variant_activations.py --limit 20
    3. compute_reliability_variant_projections.py --preprocessing both --layers 32 40 47
    4. diagnose_reliability_variant_projections.py

Usage:
    python scripts/run_reliability_projection_smoke_test.py
"""

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run_step(step_name: str, cmd: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        print(f"\n  ERROR: Step '{step_name}' failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  DONE: {step_name}")


def main() -> None:
    print("\n  Stage 4C: Reliability Variant Projection Smoke Test")
    print("  Limit: 20 variants\n")

    # Step 1: Build jobs
    run_step(
        "1. Build projection jobs",
        [PYTHON, "scripts/build_reliability_projection_jobs.py"],
    )

    # Step 2: Extract activations (limit 20, requires Modal)
    run_step(
        "2. Extract variant activations (--limit 20)",
        [PYTHON, "scripts/extract_reliability_variant_activations.py", "--limit", "20"],
    )

    # Step 3: Compute projections
    run_step(
        "3. Compute projections (raw + centered, layers 32 40 47)",
        [
            PYTHON, "scripts/compute_reliability_variant_projections.py",
            "--preprocessing", "both",
            "--layers", "32", "40", "47",
        ],
    )

    # Step 4: Diagnostics
    run_step(
        "4. Run diagnostics",
        [PYTHON, "scripts/diagnose_reliability_variant_projections.py"],
    )

    print(f"\n{'='*60}")
    print("  Smoke test COMPLETE.")
    print(f"  Output dir: {_ROOT / 'outputs/reliability_projection/'}")
    print(
        "\n  Full run (all variants):\n"
        "    python scripts/extract_reliability_variant_activations.py\n"
        "    python scripts/compute_reliability_variant_projections.py --preprocessing both --layers 32 40 47\n"
        "    python scripts/diagnose_reliability_variant_projections.py"
    )


if __name__ == "__main__":
    main()
