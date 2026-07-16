"""
Stage 2B — Step 1: Generate responses under contrastive system prompts.

Default mode: delegates to Modal/Gemma GPU app.
Mock mode  : generates synthetic responses locally (no GPU required).

Usage:
    # Smoke test — 8 jobs, mock mode
    python scripts/generate_vector_responses.py --limit 8 --mock

    # Full extraction split via Modal
    python scripts/generate_vector_responses.py --split extraction

    # Validation split via Modal
    python scripts/generate_vector_responses.py --split validation

Outputs:
    outputs/vector_construction/generated_responses_{split}.parquet
    outputs/vector_construction/generated_responses_{split}.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate contrastive responses for persona-vector construction."
    )
    parser.add_argument(
        "--split",
        choices=["extraction", "validation", "both"],
        default="extraction",
        help="Which question split to process.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap number of jobs (0=all).")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Jobs per Modal inference call."
    )
    parser.add_argument(
        "--model-name",
        default="google/gemma-3-12b-it",
        help="HuggingFace model ID for generation.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/vector_construction",
        help="Output directory.",
    )
    parser.add_argument(
        "--artifacts",
        default="configs/trait_vector_artifacts.yaml",
        help="Path to artifact bank YAML (default: trait_vector_artifacts.yaml). "
             "Pass configs/synonym_vector_artifacts.yaml for synonym controls.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock responses instead of calling Modal (for tests/offline).",
    )
    args = parser.parse_args()

    splits = ["extraction", "validation"] if args.split == "both" else [args.split]

    if args.mock:
        _run_mock(splits, args.limit, args.out_dir, args.artifacts)
    else:
        _run_modal(splits, args.limit, args.batch_size, args.model_name, args.out_dir, args.artifacts)


def _run_mock(splits: list[str], limit: int, out_dir: str, artifacts: str = "configs/trait_vector_artifacts.yaml") -> None:
    from src.vectors.artifact_bank import load_artifact_bank, load_artifact_bank_flexible
    from src.vectors.generate_responses import (
        build_generation_jobs,
        mock_generate,
        save_responses,
    )

    out_path = _ROOT / out_dir
    artifact_path = _ROOT / artifacts
    bank = (
        load_artifact_bank_flexible(artifact_path)
        if artifacts != "configs/trait_vector_artifacts.yaml"
        else load_artifact_bank(artifact_path)
    )

    for split in splits:
        jobs = build_generation_jobs(bank, split=split, limit=limit)
        print(f"\n  [mock] Generating {len(jobs)} {split}-split responses...")
        responses = mock_generate(jobs)
        parquet_path, csv_path = save_responses(
            responses, out_path, stem=f"generated_responses_{split}"
        )
        print(f"  Saved: {parquet_path}")
        print(f"         {csv_path}")


def _run_modal(
    splits: list[str],
    limit: int,
    batch_size: int,
    model_name: str,
    out_dir: str,
    artifacts: str = "configs/trait_vector_artifacts.yaml",
) -> None:
    for split in splits:
        cmd = [
            "modal", "run",
            "modal_apps/generate_gemma_responses.py",
            f"--split={split}",
            f"--limit={limit}",
            f"--batch-size={batch_size}",
            f"--model-name={model_name}",
            f"--out-dir={out_dir}",
            f"--artifacts-path={artifacts}",
        ]
        print(f"\n  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(_ROOT))
        if result.returncode != 0:
            print(f"  ERROR: Modal run failed for split={split}")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
