"""
Stage 2B — Step 5: Held-out vector validation.

Projects validation-split activations onto each trait×layer persona vector
and computes ROC-AUC, accuracy, Cohen's d, and projection gap.

If a trait vector fails the minimum AUC target, that is a FINDING — it means
the contrast prompts may be confounded or the layer is wrong.
Do NOT proceed to ETHICS projection harvesting until all trait vectors pass.

Usage:
    python scripts/validate_persona_vectors.py
    python scripts/validate_persona_vectors.py --minimum-auc 0.70

Outputs:
    outputs/vector_construction/vector_validation_results.csv
    outputs/vector_construction/vector_validation_results.md
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate persona vectors against held-out validation split."
    )
    parser.add_argument(
        "--extraction-metadata",
        default="outputs/vector_construction/activation_metadata_extraction.parquet",
        help="Extraction-split activation metadata.",
    )
    parser.add_argument(
        "--validation-metadata",
        default="outputs/vector_construction/activation_metadata_validation.parquet",
        help="Validation-split activation metadata.",
    )
    parser.add_argument(
        "--vector-metadata",
        default="outputs/vector_construction/persona_vector_metadata.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/vector_construction",
    )
    parser.add_argument(
        "--minimum-auc",
        type=float,
        default=0.75,
        help="AUC threshold for passing validation.",
    )
    parser.add_argument(
        "--traits",
        default=",".join(_DEFAULT_TRAITS),
    )
    args = parser.parse_args()

    from src.vectors.compute_vectors import load_vector_metadata
    from src.vectors.extract_activations import load_activation_metadata
    from src.vectors.validate_vectors import (
        save_validation_results,
        select_best_layer,
        validate_all_vectors,
    )

    out_path = _ROOT / args.out_dir
    traits = [t.strip() for t in args.traits.split(",")]

    # Load activation records from both splits
    ext_meta_path = _ROOT / args.extraction_metadata
    val_meta_path = _ROOT / args.validation_metadata

    all_records = []
    for path, label in [(ext_meta_path, "extraction"), (val_meta_path, "validation")]:
        if not path.exists():
            print(f"  ERROR: {label} activation metadata not found: {path}")
            print(f"         Run extract_vector_activations.py --split {label} first.")
            sys.exit(1)
        all_records.extend(load_activation_metadata(path))

    vec_meta_path = _ROOT / args.vector_metadata
    if not vec_meta_path.exists():
        print(f"  ERROR: Vector metadata not found: {vec_meta_path}")
        print("         Run compute_persona_vectors.py first.")
        sys.exit(1)

    vec_metas = load_vector_metadata(vec_meta_path)
    print(f"  Loaded {len(all_records)} activation records.")
    print(f"  Loaded {len(vec_metas)} persona vectors.")

    results = validate_all_vectors(
        act_records=all_records,
        vec_metas=vec_metas,
        minimum_auc_target=args.minimum_auc,
        out_dir=out_path,
    )

    if not results:
        print("  WARNING: No validation results computed.")
        print("           Check that validation-split activations exist.")
        sys.exit(1)

    csv_path = out_path / "vector_validation_results.csv"
    md_path = out_path / "vector_validation_results.md"
    print(f"\n  Results saved:")
    print(f"    CSV : {csv_path}")
    print(f"    MD  : {md_path}")

    # Print summary table
    print("\n  ══════ Validation Results ══════════════════════════════════════════")
    cohens_d_hdr = "Cohen's d"
    print(f"  {'Trait':<16} {'Layer':>5} {'AUC':>6} {'Acc':>6} {cohens_d_hdr:>9} {'Pass?'}")
    print("  " + "─" * 60)
    for r in sorted(results, key=lambda x: (x.trait, x.layer)):
        flag = "✅" if r.passes_minimum_auc else "❌"
        print(
            f"  {r.trait:<16} {r.layer:>5} {r.auc:>6.3f} "
            f"{r.accuracy:>6.3f} {r.cohens_d:>9.3f}  {flag}"
        )
    print("  " + "─" * 60)

    # Layer selection
    try:
        best_layer = select_best_layer(results, traits)
        mean_auc = sum(
            r.auc for r in results if r.trait in traits and r.layer == best_layer
        ) / max(
            sum(1 for r in results if r.trait in traits and r.layer == best_layer), 1
        )
        print(f"\n  Recommended layer : {best_layer}  (mean AUC across traits = {mean_auc:.3f})")
        print(f"  Update configs/mvp_experiment.yaml → model.target_layer: {best_layer}")
    except ValueError as exc:
        print(f"\n  WARNING: {exc}")

    # Failure check
    failures = [r for r in results if not r.passes_minimum_auc]
    passing = [r for r in results if r.passes_minimum_auc]
    print(f"\n  {len(passing)} vectors pass AUC ≥ {args.minimum_auc}")

    if failures:
        print(f"\n  ⚠  {len(failures)} vector(s) FAILED minimum AUC target:")
        for r in failures:
            print(f"     [{r.trait}] layer {r.layer} — AUC = {r.auc:.3f}")
        print("\n  Do NOT proceed to ETHICS projection until all trait vectors pass.")
        print("  Investigate: check audit findings, review low-scoring responses,")
        print("  consider revising contrastive prompts or extending training set.")
        sys.exit(1)
    else:
        print("\n  ✓ All trait vectors pass the minimum AUC target.")
        print("  You may proceed to Stage 3: ETHICS item projection / monitoring.")


if __name__ == "__main__":
    main()
