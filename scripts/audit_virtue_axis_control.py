"""
Lightweight quality audit for configs/virtue_axis_control.yaml.

The shared audit tool (scripts/audit_vector_artifacts.py ->
src.vectors.artifact_quality.run_quality_checks) hardcodes its 9 substantive
checks to iterate over VALID_TRAITS = {honesty, harmlessness, fairness,
compassion}. Since this config's only trait key is "virtue_axis", every one
of those checks would silently find zero matching rows and report nothing —
not because the content is clean, but because they never look at it. This
script reuses the same underlying primitives (word-set Jaccard similarity,
the same confound lexicon, the same thresholds) directly against this one
config's actual content, so the checks are real rather than a silent no-op.

Checks run (only the ones that meaningfully apply to a deliberately generic
control vector — see notes inline for why a couple of the original 9 checks
are skipped or reinterpreted):
    1. Short-text check (same MIN_TEXT_LENGTH_WORDS threshold)
    2. Near-duplicate system prompts within each pole (same Jaccard threshold)
    3. Cross-trait lexicon balance in system prompts (does the "generic"
       framing secretly lean toward one of the four specific traits?)
    4. Extraction/validation overlap in the merged 40-question pool
    5. Generic-valence word count (reported for context, NOT flagged as a
       confound -- heavy generic-valence language is the intended design
       for this vector, unlike for the four trait-specific vectors)

Usage:
    python scripts/audit_virtue_axis_control.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.vectors.artifact_bank import load_artifact_bank_flexible  # noqa: E402
from src.vectors.artifact_quality import (  # noqa: E402
    CONFOUND_LEXICON,
    CROSS_TRAIT_WORD_THRESHOLD,
    MIN_TEXT_LENGTH_WORDS,
    NEAR_DUPLICATE_JACCARD_THRESHOLD,
    _count_words_from_list,
    _jaccard,
    _words,
)

CONFIG_PATH = _ROOT / "configs" / "virtue_axis_control.yaml"
TRAIT_LEXICONS = {
    "honesty": CONFOUND_LEXICON["honesty_related"],
    "harmlessness": CONFOUND_LEXICON["harmlessness_related"],
    "fairness": CONFOUND_LEXICON["fairness_related"],
    "compassion": CONFOUND_LEXICON["compassion_related"],
}


def main() -> None:
    bank = load_artifact_bank_flexible(CONFIG_PATH)
    sp = bank.system_prompts_df
    q = bank.questions_df

    print("=" * 68)
    print("  Virtue-Axis Control — Standalone Quality Audit")
    print("=" * 68)
    print(f"  {len(sp)} system prompts, {len(q)} elicitation questions\n")

    findings = 0

    # --- 1. Short-text check ---------------------------------------------
    print("[1] Short-text check (min words = %d)" % MIN_TEXT_LENGTH_WORDS)
    short = []
    for _, row in sp.iterrows():
        if len(_words(row.prompt_text)) < MIN_TEXT_LENGTH_WORDS:
            short.append(row.prompt_id)
    for _, row in q.iterrows():
        if len(_words(row.question_text)) < MIN_TEXT_LENGTH_WORDS:
            short.append(row.question_id)
    if short:
        findings += len(short)
        print(f"    FLAGGED: {short}")
    else:
        print("    OK — no short texts.")

    # --- 2. Near-duplicate prompts within each pole -----------------------
    print(f"\n[2] Near-duplicate prompts within pole (Jaccard >= {NEAR_DUPLICATE_JACCARD_THRESHOLD})")
    for pole in ("positive", "negative"):
        rows = list(sp[sp.pole == pole].itertuples())
        pole_flagged = False
        for a, b in combinations(rows, 2):
            sim = _jaccard(a.prompt_text, b.prompt_text)
            if sim >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                print(f"    FLAGGED: {a.prompt_id} <-> {b.prompt_id}  (Jaccard={sim:.2f})")
                findings += 1
                pole_flagged = True
        if not pole_flagged:
            # report max similarity for context even when nothing is flagged
            max_sim = max((_jaccard(a.prompt_text, b.prompt_text) for a, b in combinations(rows, 2)), default=0.0)
            print(f"    OK — {pole} pole: max pairwise Jaccard = {max_sim:.2f}")

    # --- 3. Cross-trait lexicon balance in system prompts ------------------
    print(f"\n[3] Cross-trait lexicon balance (flag if any single prompt uses >= {CROSS_TRAIT_WORD_THRESHOLD} words native to one specific trait)")
    trait_totals = {t: 0 for t in TRAIT_LEXICONS}
    prompt_flagged = False
    for _, row in sp.iterrows():
        for trait, lexicon in TRAIT_LEXICONS.items():
            n = _count_words_from_list(row.prompt_text, lexicon)
            trait_totals[trait] += n
            if n >= CROSS_TRAIT_WORD_THRESHOLD:
                print(f"    FLAGGED: {row.prompt_id} uses {n} words native to '{trait}'")
                findings += 1
                prompt_flagged = True
    if not prompt_flagged:
        print("    OK — no single prompt leans heavily on one trait's vocabulary.")
    print(f"    Totals across all 10 prompts: {trait_totals}")
    total_hits = sum(trait_totals.values())
    if total_hits > 0:
        max_trait = max(trait_totals, key=trait_totals.get)
        share = trait_totals[max_trait] / total_hits
        print(f"    Most-represented trait: '{max_trait}' ({trait_totals[max_trait]}/{total_hits} = {share:.0%} of all cross-trait-lexicon hits)")
        if share > 0.5 and total_hits >= 3:
            print(f"    NOTE: '{max_trait}' vocabulary dominates what little cross-trait language exists — worth a manual look.")
    else:
        print("    No cross-trait-specific vocabulary detected at all — prompts are lexically generic, as intended.")

    # --- 4. Extraction/validation overlap in the merged pool ---------------
    print(f"\n[4] Extraction/validation text overlap in merged 40-question pool (Jaccard >= {NEAR_DUPLICATE_JACCARD_THRESHOLD})")
    ext = list(q[q.split == "extraction"].itertuples())
    val = list(q[q.split == "validation"].itertuples())
    overlap_flagged = False
    max_overlap = 0.0
    for e in ext:
        for v in val:
            sim = _jaccard(e.question_text, v.question_text)
            max_overlap = max(max_overlap, sim)
            if sim >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                print(f"    FLAGGED: {e.question_id} <-> {v.question_id}  (Jaccard={sim:.2f})")
                findings += 1
                overlap_flagged = True
    if not overlap_flagged:
        print(f"    OK — max extraction/validation Jaccard = {max_overlap:.2f}")

    # --- 5. Generic-valence word count (context only, not a confound here) -
    print("\n[5] Generic-valence word count (informational only)")
    print("    NOTE: heavy generic good/bad vocabulary is the INTENDED design")
    print("    for this vector -- unlike the four trait-specific vectors, this")
    print("    is not treated as a confound finding here.")
    pos_hits = sum(_count_words_from_list(t, CONFOUND_LEXICON["generic_positive"]) for t in sp[sp.pole == "positive"].prompt_text)
    neg_hits = sum(_count_words_from_list(t, CONFOUND_LEXICON["generic_negative"]) for t in sp[sp.pole == "negative"].prompt_text)
    print(f"    Positive pole generic-positive-word hits: {pos_hits}")
    print(f"    Negative pole generic-negative-word hits: {neg_hits}")

    print("\n" + "=" * 68)
    print(f"  Total flagged findings (checks 1-4): {findings}")
    print("=" * 68)


if __name__ == "__main__":
    main()
