"""
Inspect the trait vector artifact bank and print a structured summary.

Usage:
    python scripts/inspect_vector_artifacts.py
    python scripts/inspect_vector_artifacts.py --path configs/trait_vector_artifacts.yaml
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.vectors.artifact_bank import ArtifactBank, load_artifact_bank  # noqa: E402


def _sep(char: str = "-", width: int = 64) -> None:
    print(char * width)


def _print_banner() -> None:
    _sep("=")
    print("  Trait Vector Artifact Bank — Inspection Report")
    _sep("=")
    print()
    print("  NOTE: These artifacts are for persona-vector CONSTRUCTION only.")
    print("  ETHICS items are NOT used here and are NOT referenced in this file.")
    print("  The ETHICS item bank is used separately for projection / reliability.")
    print()


def _print_system_prompts(bank: ArtifactBank) -> None:
    print("── System Prompts (contrastive pairs per trait) " + "─" * 17)
    df = bank.system_prompts_df
    for trait in sorted(df["trait"].unique()):
        sub = df[df["trait"] == trait]
        pos = sub[sub["pole"] == "positive"]
        neg = sub[sub["pole"] == "negative"]
        print(f"\n  {trait}")
        print(f"    positive : {len(pos):>2}  ids: {', '.join(pos['prompt_id'].tolist())}")
        print(f"    negative : {len(neg):>2}  ids: {', '.join(neg['prompt_id'].tolist())}")
    print()


def _print_questions(bank: ArtifactBank) -> None:
    print("── Elicitation Questions (extraction + validation per trait) " + "─" * 4)
    df = bank.questions_df
    for trait in sorted(df["trait"].unique()):
        sub = df[df["trait"] == trait]
        ext = sub[sub["split"] == "extraction"]
        val = sub[sub["split"] == "validation"]
        total = len(sub)
        print(f"\n  {trait}  (total: {total})")
        print(f"    extraction : {len(ext):>2}")
        print(f"    validation : {len(val):>2}")

        # Verify splits are disjoint
        overlap = set(ext["question_id"]) & set(val["question_id"])
        if overlap:
            print(f"    ⚠  WARNING: overlapping IDs between splits: {overlap}")
        else:
            print("    ✓ extraction / validation splits are disjoint")
    print()


def _print_rubrics(bank: ArtifactBank) -> None:
    print("── Evaluation Rubrics " + "─" * 43)
    df = bank.rubrics_df
    for _, row in df.iterrows():
        print(f"\n  {row['trait']}")
        print(f"    score range  : {row['min_score']} – {row['max_score']}")
        instructions = str(row["scoring_instructions"])
        # Print first 160 chars of scoring instructions
        snippet = instructions[:160].replace("\n", " ")
        if len(instructions) > 160:
            snippet += "…"
        print(f"    instructions : {snippet}")
    print()


def _print_warnings(bank: ArtifactBank) -> None:
    from src.vectors.artifact_bank import (
        EXPECTED_PROMPTS_PER_POLE,
        EXPECTED_QUESTIONS_PER_SPLIT,
        VALID_TRAITS,
    )

    warnings: list[str] = []

    sp = bank.system_prompts_df
    for trait in VALID_TRAITS:
        for pole in ("positive", "negative"):
            n = len(sp[(sp["trait"] == trait) & (sp["pole"] == pole)])
            if n != EXPECTED_PROMPTS_PER_POLE:
                warnings.append(
                    f"[{trait}] {pole} system prompts: expected "
                    f"{EXPECTED_PROMPTS_PER_POLE}, got {n}"
                )

    q = bank.questions_df
    for trait in VALID_TRAITS:
        for split in ("extraction", "validation"):
            n = len(q[(q["trait"] == trait) & (q["split"] == split)])
            if n != EXPECTED_QUESTIONS_PER_SPLIT:
                warnings.append(
                    f"[{trait}] {split} questions: expected "
                    f"{EXPECTED_QUESTIONS_PER_SPLIT}, got {n}"
                )

    if warnings:
        print("── Warnings " + "─" * 52)
        for w in warnings:
            print(f"  ⚠  {w}")
        print()
    else:
        print("── Warnings " + "─" * 52)
        print("  ✓ All counts match expected values")
        print()


def _print_summary_counts(bank: ArtifactBank) -> None:
    sp = bank.system_prompts_df
    q = bank.questions_df
    total_prompts = len(sp)
    total_questions = len(q)
    n_traits = sp["trait"].nunique()
    print("── Totals " + "─" * 54)
    print(f"  Traits            : {n_traits}")
    print(f"  System prompts    : {total_prompts}  "
          f"({total_prompts // max(n_traits, 1)} per trait × 2 poles)")
    print(f"  Questions         : {total_questions}  "
          f"({total_questions // max(n_traits, 1)} per trait — "
          "20 extraction + 20 validation)")
    print(f"  Rubrics           : {len(bank.rubrics_df)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the trait vector artifact bank.")
    parser.add_argument(
        "--path",
        default=str(_ROOT / "configs" / "trait_vector_artifacts.yaml"),
        help="Path to trait_vector_artifacts.yaml",
    )
    args = parser.parse_args()

    _print_banner()

    try:
        bank = load_artifact_bank(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"  Loaded: {args.path}\n")

    _print_summary_counts(bank)
    _print_system_prompts(bank)
    _print_questions(bank)
    _print_rubrics(bank)
    _print_warnings(bank)

    print("── Next Steps " + "─" * 51)
    print("  Stage 2A is complete.  Do not proceed to activation extraction")
    print("  until you have:")
    print("  1. Reviewed and approved these artifacts.")
    print("  2. A validated curated ETHICS item bank (Stage 1e).")
    print("  3. Confirmed the layer selection strategy (Stage 2B+).")
    _sep("=")


if __name__ == "__main__":
    main()
