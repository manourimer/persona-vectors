"""
Stage 2B — Step 4: Compute difference-of-means persona vectors.

Loads extraction-split activation records, computes one vector per trait×layer,
and saves .npy files.  Pure NumPy — no GPU required.

Algorithm:
    vector = mean(positive_activations) − mean(negative_activations)
    if normalize: vector = vector / ||vector||₂

Usage:
    python scripts/compute_persona_vectors.py
    python scripts/compute_persona_vectors.py --no-normalize
    python scripts/compute_persona_vectors.py --candidate-layers 16,28,40

Outputs:
    outputs/vector_construction/persona_vectors/{trait}_layer{N}.npy
    outputs/vector_construction/persona_vector_metadata.csv
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]
_DEFAULT_CANDIDATE_LAYERS = [16, 24, 28, 32, 40, 47]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute difference-of-means persona vectors from cached activations."
    )
    parser.add_argument(
        "--activation-metadata",
        default="outputs/vector_construction/activation_metadata_extraction.parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/vector_construction",
    )
    parser.add_argument(
        "--candidate-layers",
        default=",".join(str(l) for l in _DEFAULT_CANDIDATE_LAYERS),
    )
    parser.add_argument(
        "--traits",
        default=",".join(_DEFAULT_TRAITS),
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not normalize vectors to unit norm.",
    )
    args = parser.parse_args()

    from src.vectors.compute_vectors import (
        compute_all_vectors,
        save_vector_metadata,
    )
    from src.vectors.extract_activations import load_activation_metadata

    candidate_layers = [int(l) for l in args.candidate_layers.split(",")]
    traits = [t.strip() for t in args.traits.split(",")]
    out_path = _ROOT / args.out_dir
    meta_path = _ROOT / args.activation_metadata

    if not meta_path.exists():
        print(f"  ERROR: Activation metadata not found: {meta_path}")
        print("         Run extract_vector_activations.py first.")
        sys.exit(1)

    records = load_activation_metadata(meta_path)
    print(f"  Loaded {len(records)} activation records.")
    print(f"  Traits          : {traits}")
    print(f"  Candidate layers: {candidate_layers}")

    vectors_dir = out_path / "persona_vectors"
    metas = compute_all_vectors(
        records=records,
        candidate_layers=candidate_layers,
        traits=traits,
        normalize=not args.no_normalize,
        out_dir=vectors_dir,
    )

    if not metas:
        print("  WARNING: No vectors were computed. Check that activation records exist")
        print("           for both positive and negative poles at the requested layers.")
        sys.exit(1)

    csv_path = save_vector_metadata(metas, out_path)
    print(f"\n  Computed {len(metas)} vectors ({len(traits)} traits × {len(candidate_layers)} layers).")
    print(f"  Vector files  : {vectors_dir}/")
    print(f"  Metadata CSV  : {csv_path}")

    print("\n  Summary:")
    for m in sorted(metas, key=lambda x: (x.trait, x.layer)):
        print(
            f"    {m.trait:<16} layer {m.layer:>2}  "
            f"n_pos={m.n_positive:>3}  n_neg={m.n_negative:>3}  "
            f"dim={m.hidden_dim}  norm={m.normalization}"
        )

    print("\n  Next: validate persona vectors.")
    print("    python scripts/validate_persona_vectors.py")


if __name__ == "__main__":
    main()
