"""
Stage 2B smoke test — end-to-end pipeline in mock mode.

Runs the full vector construction pipeline with:
  - Mock response generation (no GPU)
  - Mock scoring (deterministic)
  - Mock activation extraction (random numpy)
  - Real vector computation (pure numpy)
  - Real validation metrics (sklearn)

Usage:
    python scripts/run_vector_construction_smoke_test.py

For real Modal/GPU runs, use the individual scripts:
    python scripts/generate_vector_responses.py --split extraction
    python scripts/generate_vector_responses.py --split validation
    python scripts/score_vector_responses.py --split both
    python scripts/extract_vector_activations.py --split both
    python scripts/compute_persona_vectors.py
    python scripts/validate_persona_vectors.py
"""

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]
_CANDIDATE_LAYERS = [0, 1]    # tiny for smoke test
_MOCK_HIDDEN_DIM = 64
_LIMIT = 8                    # 8 jobs per split


def _sep(char: str = "─", w: int = 64) -> str:
    return char * w


def _build_bipolar_smoke_jobs(bank, split: str, n_prompts: int = 1, n_questions: int = 2) -> list:
    """Build a minimal set of jobs covering all traits × both poles."""
    sp_df = bank.system_prompts_df
    q_df = bank.questions_df[bank.questions_df["split"] == split]
    jobs: list[dict] = []
    from src.vectors.vector_data import GeneratedResponse as GR
    for trait in _TRAITS:
        for pole in ("positive", "negative"):
            prompts = sp_df[(sp_df["trait"] == trait) & (sp_df["pole"] == pole)].head(n_prompts)
            questions = q_df[q_df["trait"] == trait].head(n_questions)
            for _, sp_row in prompts.iterrows():
                for _, q_row in questions.iterrows():
                    jobs.append({
                        "response_id": GR.make_id(sp_row["prompt_id"], q_row["question_id"]),
                        "trait": trait,
                        "pole": pole,
                        "split": split,
                        "system_prompt_id": sp_row["prompt_id"],
                        "question_id": q_row["question_id"],
                        "system_prompt_text": sp_row["prompt_text"],
                        "question_text": q_row["question_text"],
                    })
    return jobs


def main() -> None:  # noqa: C901
    print("=" * 64)
    print("  Stage 2B Smoke Test — Mock Pipeline")
    print("=" * 64)
    print()
    print("  This runs the full vector construction pipeline using")
    print("  mock responses and synthetic activations.")
    print("  No GPU, Modal, or HuggingFace credentials required.")
    print()

    from src.vectors.artifact_bank import load_artifact_bank
    from src.vectors.compute_vectors import compute_all_vectors, save_vector_metadata
    from src.vectors.extract_activations import (
        mock_extract,
        save_activation_metadata,
    )
    from src.vectors.generate_responses import build_generation_jobs, mock_generate, save_responses
    from src.vectors.score_responses import filter_for_extraction, mock_score, save_scored
    from src.vectors.validate_vectors import (
        save_validation_results,
        select_best_layer,
        validate_all_vectors,
    )

    artifact_path = _ROOT / "configs" / "trait_vector_artifacts.yaml"
    bank = load_artifact_bank(artifact_path)

    with tempfile.TemporaryDirectory(prefix="stage2b_smoke_") as tmp:
        out_path = Path(tmp)

        # ── Step 1: Generate responses (both splits) ────────────────────────
        print(_sep())
        print("  Step 1 — Generate responses (mock)")
        print(_sep())
        all_scored = []

        for split in ("extraction", "validation"):
            # Build jobs that cover all 4 traits × both poles.
            # Each trait gets 1 prompt per pole × 1 question = 8 jobs/trait × 4 = 32 total.
            jobs = _build_bipolar_smoke_jobs(bank, split)
            responses = mock_generate(jobs)
            save_responses(responses, out_path, stem=f"generated_responses_{split}")
            print(f"  {split:<12}: {len(responses)} responses generated")

            # ── Step 2: Score responses ─────────────────────────────────────
            scored = mock_score(responses)
            retained = filter_for_extraction(scored)
            save_scored(scored, out_path, stem=f"scored_responses_{split}")
            print(f"  {split:<12}: {len(retained)}/{len(scored)} retained (score thresholds)")
            all_scored.extend(scored)

        print()

        # ── Step 3: Extract activations (both splits) ───────────────────────
        print(_sep())
        print("  Step 2 — Extract activations (mock)")
        print(_sep())
        all_records = []

        for split in ("extraction", "validation"):
            split_scored = [s for s in all_scored if s.split == split]
            records = mock_extract(
                split_scored,
                candidate_layers=_CANDIDATE_LAYERS,
                out_dir=out_path,
                hidden_dim=_MOCK_HIDDEN_DIM,
            )
            save_activation_metadata(
                records, out_path, stem=f"activation_metadata_{split}"
            )
            all_records.extend(records)
            n_retained = sum(1 for r in records if r.layer == _CANDIDATE_LAYERS[0])
            print(f"  {split:<12}: {n_retained} retained responses × {len(_CANDIDATE_LAYERS)} layers = {len(records)} records")

        print()

        # ── Step 4: Compute vectors ─────────────────────────────────────────
        print(_sep())
        print("  Step 3 — Compute persona vectors (numpy, no GPU)")
        print(_sep())
        vec_dir = out_path / "persona_vectors"
        ext_records = [r for r in all_records if r.split == "extraction"]
        vec_metas = compute_all_vectors(
            records=ext_records,
            candidate_layers=_CANDIDATE_LAYERS,
            traits=_TRAITS,
            normalize=True,
            out_dir=vec_dir,
        )
        save_vector_metadata(vec_metas, out_path)

        print(f"  Computed {len(vec_metas)} vectors "
              f"({len(_TRAITS)} traits × {len(_CANDIDATE_LAYERS)} layers)")
        for m in sorted(vec_metas, key=lambda x: (x.trait, x.layer)):
            print(
                f"    {m.trait:<16} layer {m.layer}  "
                f"n_pos={m.n_positive}  n_neg={m.n_negative}  "
                f"dim={m.hidden_dim}"
            )
        print()

        # ── Step 5: Validate vectors ────────────────────────────────────────
        print(_sep())
        print("  Step 4 — Validate persona vectors (held-out split)")
        print(_sep())
        results = validate_all_vectors(
            act_records=all_records,
            vec_metas=vec_metas,
            minimum_auc_target=0.50,   # lower threshold for mock data
            out_dir=out_path,
        )

        header = "  {:<16} {:>5} {:>6} {:>6} {:>9} {}".format("Trait", "Layer", "AUC", "Acc", "Cohen's d", "Pass?")
        print(f"\n{header}")
        print("  " + "─" * 55)
        for r in sorted(results, key=lambda x: (x.trait, x.layer)):
            flag = "✅" if r.passes_minimum_auc else "❌"
            print(
                f"  {r.trait:<16} {r.layer:>5}  {r.auc:>5.3f}  "
                f"{r.accuracy:>5.3f}  {r.cohens_d:>8.3f}  {flag}"
            )

        try:
            best = select_best_layer(results, _TRAITS)
            print(f"\n  Recommended layer for real run: {best}")
        except ValueError:
            pass

        failures = [r for r in results if not r.passes_minimum_auc]
        if failures:
            print(
                f"\n  ⚠  {len(failures)} vector(s) below AUC threshold "
                f"(expected with tiny mock data — not a bug)."
            )

    print()
    print("=" * 64)
    print("  Smoke test complete — all pipeline stages ran successfully.")
    print("=" * 64)
    print()
    print("  For the real Modal/GPU run:")
    print("    python scripts/generate_vector_responses.py --split extraction")
    print("    python scripts/generate_vector_responses.py --split validation")
    print("    python scripts/score_vector_responses.py --split extraction")
    print("    python scripts/score_vector_responses.py --split validation")
    print("    # (or use --mock for offline scoring)")
    print("    python scripts/extract_vector_activations.py --split extraction")
    print("    python scripts/extract_vector_activations.py --split validation")
    print("    python scripts/compute_persona_vectors.py")
    print("    python scripts/validate_persona_vectors.py")
    print()
    print("  Do NOT run ETHICS projection (Stage 3) until all trait")
    print("  vectors pass the minimum AUC target.")
    print("=" * 64)


if __name__ == "__main__":
    main()
