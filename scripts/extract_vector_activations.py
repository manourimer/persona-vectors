"""
Stage 2B — Step 3: Extract response-token activations at candidate layers.

Token scope: response_tokens (mean over generated response token positions).
This is the correct scope for VECTOR EXTRACTION.
Stage 3+ (ETHICS monitoring) will use last_prompt_token.

ETHICS items are NOT used here.

Usage:
    # Mock extraction (no GPU required)
    python scripts/extract_vector_activations.py --mock

    # Full run via Modal
    python scripts/extract_vector_activations.py --split extraction
    python scripts/extract_vector_activations.py --split validation

Outputs:
    outputs/vector_construction/activations/{trait}_{pole}/{response_id}_layer{N}.npy
    outputs/vector_construction/activation_metadata_{split}.parquet
"""

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_CANDIDATE_LAYERS = [16, 24, 28, 32, 40, 47]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract activations from retained responses at candidate layers."
    )
    parser.add_argument(
        "--split",
        choices=["extraction", "validation", "both"],
        default="extraction",
    )
    parser.add_argument(
        "--scored-path",
        default="",
        help="Override input parquet path.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/vector_construction",
    )
    parser.add_argument(
        "--candidate-layers",
        default=",".join(str(l) for l in _DEFAULT_CANDIDATE_LAYERS),
        help="Comma-separated layer indices.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate random mock activations (no GPU required).",
    )
    parser.add_argument(
        "--mock-hidden-dim",
        type=int,
        default=64,
        help="Hidden dim for mock activations.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    candidate_layers = [int(l) for l in args.candidate_layers.split(",")]
    splits = ["extraction", "validation"] if args.split == "both" else [args.split]
    out_path = _ROOT / args.out_dir

    if args.mock:
        _run_mock(splits, candidate_layers, out_path, args.mock_hidden_dim, args.limit)
    else:
        _run_modal(splits, args.scored_path, candidate_layers, args.batch_size, out_path)


def _run_mock(splits, candidate_layers, out_path, hidden_dim, limit) -> None:
    from src.vectors.extract_activations import (
        mock_extract,
        save_activation_metadata,
    )
    from src.vectors.score_responses import load_scored

    for split in splits:
        scored_file = out_path / f"scored_responses_{split}.parquet"
        if not scored_file.exists():
            print(f"  ERROR: Scored responses not found: {scored_file}")
            print(f"         Run score_vector_responses.py --split {split} --mock first.")
            sys.exit(1)

        scored = load_scored(scored_file)
        if limit:
            scored = scored[:limit]

        retained = [s for s in scored if s.keep_for_vector_extraction]
        print(f"\n  [mock] Extracting activations for {len(retained)} retained {split}-split responses ...")
        print(f"  Candidate layers: {candidate_layers}  hidden_dim: {hidden_dim}")

        records = mock_extract(scored, candidate_layers, out_path, hidden_dim=hidden_dim)
        meta_path = save_activation_metadata(
            records, out_path, stem=f"activation_metadata_{split}"
        )
        print(f"  Activation records: {len(records)}")
        print(f"  Metadata saved: {meta_path}")


def _run_modal(splits, scored_path_override, candidate_layers, batch_size, out_path) -> None:
    for split in splits:
        if scored_path_override:
            scored_path = scored_path_override
        else:
            scored_path = str(
                out_path.relative_to(_ROOT) / f"scored_responses_{split}.parquet"
            )
        layers_str = ",".join(str(l) for l in candidate_layers)
        out_dir_str = str(out_path.relative_to(_ROOT))
        cmd = [
            "modal", "run",
            "modal_apps/extract_gemma_activations.py",
            f"--split={split}",
            f"--candidate-layers={layers_str}",
            f"--batch-size={batch_size}",
            f"--scored-path={scored_path}",
            f"--out-dir={out_dir_str}",
        ]
        print(f"\n  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(_ROOT))
        if result.returncode != 0:
            print(f"  ERROR: Modal extraction failed for split={split}")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
