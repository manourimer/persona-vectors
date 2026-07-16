"""
Stage 3 smoke test — runs the full projection pipeline in mock mode.

No GPU, Modal, or HuggingFace credentials required.

Steps:
  1. Build projection jobs (pre-flight check).
  2. Extract mock activations for 10 items.
  3. Compute projections (raw + centered).
  4. Run diagnostics (centered).
  5. Compare layers.

Usage:
    python scripts/run_ethics_projection_smoke_test.py
"""

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        print(f"\n  ERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    print("\n  ══════ Stage 3 Smoke Test ══════════════════════════════════════════\n")
    print("  Mode: mock (no GPU, synthetic activations)")
    print("  Items: 10\n")

    run(["python", "scripts/build_ethics_projection_jobs.py"])
    run(["python", "scripts/extract_ethics_activations.py", "--mock", "--limit", "10"])
    run(["python", "scripts/compute_ethics_projections.py", "--mock", "--preprocessing", "both"])
    run(["python", "scripts/diagnose_ethics_projections.py"])
    run(["python", "scripts/compare_projection_layers.py"])

    print("\n  ══════ Smoke Test Complete ══════════════════════════════════════════")
    print("  All five pipeline steps ran successfully in mock mode.\n")
    print("  Full run:")
    print("    python scripts/extract_ethics_activations.py --limit 10")
    print("    python scripts/extract_ethics_activations.py")
    print("    python scripts/compute_ethics_projections.py --preprocessing both --layers 32 40 47")
    print("    python scripts/diagnose_ethics_projections.py")
    print("    python scripts/compare_projection_layers.py")


if __name__ == "__main__":
    main()
