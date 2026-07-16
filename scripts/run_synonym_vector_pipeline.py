"""
Stage 2B (synonym variant) — Generate, score, extract, and compute synonym vectors.

Runs the full Stage 2B pipeline for the MVP synonym set:
  truthfulness  (parent: honesty)
  harm_avoidance (parent: harmlessness)
  impartiality   (parent: fairness)
  empathy        (parent: compassion)

Outputs go to outputs/controls/synonym_vectors/ to keep them separate from the
primary persona vectors in outputs/vector_construction/.

Usage:
    # Full GPU run
    python scripts/run_synonym_vector_pipeline.py

    # Smoke test (20 responses, no GPU)
    python scripts/run_synonym_vector_pipeline.py --mock --limit 20

    # Resume after interruption
    python scripts/run_synonym_vector_pipeline.py --resume

    # Extraction split only (then validation separately)
    python scripts/run_synonym_vector_pipeline.py --split extraction
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_ARTIFACTS = "configs/synonym_vector_artifacts.yaml"
_OUT_DIR = "outputs/controls/synonym_vectors"
_TRAITS = "truthfulness,harm_avoidance,impartiality,empathy"
_LAYERS = "16,24,28,32,40,47"


def _run(cmd: list[str], step: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {step}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        print(f"\n  ERROR: step '{step}' failed (exit {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full Stage 2B pipeline for synonym vectors."
    )
    parser.add_argument(
        "--split",
        choices=["extraction", "validation", "both"],
        default="both",
        help="Which question split(s) to process.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock generation/scoring/extraction (no GPU). For smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted runs (skip already-generated responses).",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip response generation (use existing generated_responses files).",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Skip response scoring (use existing scored_responses files).",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip activation extraction (use existing activation files).",
    )
    parser.add_argument(
        "--vectors-only",
        action="store_true",
        help="Only run compute + validate (assumes activations already exist).",
    )
    args = parser.parse_args()

    out_dir = _ROOT / _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Synonym Vector Pipeline")
    print(f"  Artifacts    : {_ARTIFACTS}")
    print(f"  Traits       : {_TRAITS}")
    print(f"  Output dir   : {_OUT_DIR}")
    print(f"  Split(s)     : {args.split}")
    print(f"  Mock mode    : {args.mock}")

    skip_generate = args.skip_generate or args.vectors_only
    skip_score    = args.skip_score    or args.vectors_only
    skip_extract  = args.skip_extract  or args.vectors_only

    # ── Step 1: Generate responses ──────────────────────────────────────────
    if not skip_generate:
        cmd = [
            "python", "scripts/generate_vector_responses.py",
            f"--split={args.split}",
            f"--artifacts={_ARTIFACTS}",
            f"--out-dir={_OUT_DIR}",
            f"--batch-size={args.batch_size}",
        ]
        if args.limit:
            cmd.append(f"--limit={args.limit}")
        if args.mock:
            cmd.append("--mock")
        _run(cmd, "Step 1: Generate responses")

    # ── Step 2: Score responses ─────────────────────────────────────────────
    if not skip_score:
        splits = ["extraction", "validation"] if args.split == "both" else [args.split]
        for split in splits:
            responses_path = str(out_dir / f"generated_responses_{split}.parquet")
            cmd = [
                "python", "scripts/score_vector_responses.py",
                f"--split={split}",
                f"--responses-path={responses_path}",
                f"--out-dir={_OUT_DIR}",
                f"--artifacts-path={_ARTIFACTS}",
            ]
            if args.limit:
                cmd.append(f"--limit={args.limit}")
            if args.mock:
                cmd.append("--mock")
            _run(cmd, f"Step 2: Score responses ({split})")

    # ── Step 3: Extract activations ─────────────────────────────────────────
    if not skip_extract:
        splits = ["extraction", "validation"] if args.split == "both" else [args.split]
        for split in splits:
            scored_path = str(out_dir / f"scored_responses_{split}.parquet")
            cmd = [
                "python", "scripts/extract_vector_activations.py",
                f"--split={split}",
                f"--scored-path={scored_path}",
                f"--out-dir={_OUT_DIR}",
                f"--candidate-layers={_LAYERS}",
            ]
            if args.limit:
                cmd.append(f"--limit={args.limit}")
            if args.mock:
                cmd.append("--mock")
            _run(cmd, f"Step 3: Extract activations ({split})")

    # ── Step 4: Compute vectors ─────────────────────────────────────────────
    activation_meta = str(out_dir / "activation_metadata_extraction.parquet")
    cmd = [
        "python", "scripts/compute_persona_vectors.py",
        f"--traits={_TRAITS}",
        f"--activation-metadata={activation_meta}",
        f"--out-dir={_OUT_DIR}",
        f"--candidate-layers={_LAYERS}",
    ]
    _run(cmd, "Step 4: Compute difference-of-means synonym vectors")

    # ── Step 5: Validate vectors ────────────────────────────────────────────
    val_meta = str(out_dir / "activation_metadata_validation.parquet")
    val_exists = (out_dir / "activation_metadata_validation.parquet").exists()
    if val_exists:
        cmd = [
            "python", "scripts/validate_persona_vectors.py",
            f"--traits={_TRAITS}",
            f"--validation-metadata={val_meta}",
            f"--vector-dir={_OUT_DIR}",
            f"--out-dir={_OUT_DIR}",
            f"--candidate-layers={_LAYERS}",
        ]
        _run(cmd, "Step 5: Validate synonym vectors")
    else:
        print("\n  Step 5 skipped: validation split activations not found.")
        print(f"  Re-run with --split validation to generate them.")

    # ── Step 6: Run synonym controls ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Step 6: Running synonym vector controls...")
    print(f"{'='*60}")
    cmd = [
        "python", "scripts/run_synonym_vector_controls.py",
        "--mvp-only",
    ]
    _run(cmd, "Step 6: Synonym controls (cosine similarity + projection agreement)")

    print(f"\n{'='*60}")
    print("  Synonym vector pipeline complete.")
    print(f"  Outputs: {_OUT_DIR}/")
    print(f"  Report:  {_OUT_DIR}/synonym_control_summary.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
