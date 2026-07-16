"""
Artifact quality checks for the trait vector artifact bank.

Goal: catch confounds and weak prompt design before persona-vector extraction.
These checks flag possible issues for human review — they do not automatically
reject any artifact.

Key confound risk: positive prompts that collapse into generic helpfulness /
goodness, and negative prompts that introduce *other* traits (cruelty in a
honesty prompt, deception in a harmlessness prompt, etc.).  A contaminated
contrast yields a vector that captures generic moral valence rather than the
specific construct.

Public API
----------
    run_quality_checks(bank: ArtifactBank) -> list[AuditFinding]
    findings_to_df(findings)               -> pd.DataFrame

Each AuditFinding has:
    trait, artifact_type, artifact_id, severity, issue_type,
    text (snippet), explanation, suggested_review_action
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.vectors.artifact_bank import ArtifactBank, VALID_TRAITS

# ---------------------------------------------------------------------------
# Severity / types
# ---------------------------------------------------------------------------

Severity = Literal["info", "warning", "high"]
ArtifactType = Literal[
    "positive_system_prompt",
    "negative_system_prompt",
    "extraction_question",
    "validation_question",
    "rubric",
]

# ---------------------------------------------------------------------------
# Confound lexicon
# ---------------------------------------------------------------------------

CONFOUND_LEXICON: dict[str, list[str]] = {
    # Words that signal generic moral valence rather than a specific trait.
    # Their presence is only a concern when they appear heavily imbalanced
    # across poles, or dominate a rubric that should be trait-specific.
    "generic_positive": [
        "good", "ethical", "moral", "helpful", "kind",
        "responsible", "safe", "virtuous", "nice", "caring",
    ],
    "generic_negative": [
        "evil", "malicious", "immoral", "bad", "cruel",
        "dangerous", "manipulative", "wicked", "vile", "monstrous",
    ],
    # Trait-specific lexicons — used to detect cross-trait leakage in prompts.
    "honesty_related": [
        "honest", "truthful", "lie", "lies", "lying", "lied",
        "deceive", "deception", "deceptive", "mislead", "misleading",
        "transparent", "candid", "sincere", "fabricate", "fabrication",
    ],
    "harmlessness_related": [
        "harm", "harmful", "safety", "safe", "dangerous", "danger",
        "injury", "injure", "violence", "violent", "illegal", "hurt",
        "damage", "risk", "hazard",
    ],
    "fairness_related": [
        "fair", "fairness", "unfair", "just", "unjust", "justice",
        "equal", "equality", "inequal", "bias", "biased", "unbiased",
        "discriminate", "discrimination", "impartial", "impartiality",
        "prejudice",
    ],
    "compassion_related": [
        "compassion", "compassionate", "empathy", "empathetic",
        "suffer", "suffering", "cruel", "cruelty", "caring", "callous",
        "kindness", "warmth", "sympathy", "sympathetic",
    ],
}

# Map each trait to the lexicon key that is "native" to it.
# Cross-trait leakage = a prompt for trait X using many words native to trait Y.
_TRAIT_NATIVE_LEXICON: dict[str, str] = {
    "honesty": "honesty_related",
    "harmlessness": "harmlessness_related",
    "fairness": "fairness_related",
    "compassion": "compassion_related",
}

# Trait words that, when found in *elicitation questions* (not prompts), may
# tip off the model about what trait is being evaluated — a leakage concern.
_TRAIT_LABEL_WORDS: dict[str, list[str]] = {
    "honesty": ["honest", "honesty", "truthful", "truth", "deceptive", "lie", "lies"],
    "harmlessness": ["harmless", "harmlessness", "safe", "safety", "harmful"],
    "fairness": ["fair", "fairness", "unfair", "just", "unjust", "equal", "equality"],
    "compassion": ["compassion", "compassionate", "empathy", "empathetic"],
}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Fraction of questions in a single trait×split that may contain a trait
# label word before flagging.  (Some incidental usage is fine.)
TRAIT_LABEL_LEAKAGE_THRESHOLD = 0.30

# Minimum ratio of generic-negative to generic-positive occurrences in a
# negative prompt before flagging as "generic evil" (low threshold → warn early).
GENERIC_VALENCE_RATIO_THRESHOLD = 3.0

# Minimum word count for a prompt or question text.
MIN_TEXT_LENGTH_WORDS = 5

# Similarity threshold for near-duplicate detection (Jaccard on word sets).
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.70

# How many cross-trait words trigger a confound warning in a single prompt.
CROSS_TRAIT_WORD_THRESHOLD = 3

# ---------------------------------------------------------------------------
# AuditFinding
# ---------------------------------------------------------------------------


@dataclass
class AuditFinding:
    trait: str
    artifact_type: ArtifactType
    artifact_id: str
    severity: Severity
    issue_type: str
    text: str       # short snippet (first 120 chars of the artifact text)
    explanation: str
    suggested_review_action: str

    def to_dict(self) -> dict:
        return {
            "trait": self.trait,
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "severity": self.severity,
            "issue_type": self.issue_type,
            "text": self.text,
            "explanation": self.explanation,
            "suggested_review_action": self.suggested_review_action,
        }


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _word_set(text: str) -> set[str]:
    return set(_words(text))


def _count_words_from_list(text: str, word_list: list[str]) -> int:
    words = _words(text)
    targets = set(w.lower() for w in word_list)
    return sum(1 for w in words if w in targets)


def _snippet(text: str, max_len: int = 120) -> str:
    s = str(text).strip().replace("\n", " ")
    return s[:max_len] + "…" if len(s) > max_len else s


def _jaccard(a: str, b: str) -> float:
    sa, sb = _word_set(a), _word_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_short_texts(
    texts: list[tuple[str, str, str, str]],  # (trait, artifact_type, id, text)
) -> list[AuditFinding]:
    """Flag artifacts whose text is suspiciously short."""
    findings: list[AuditFinding] = []
    for trait, atype, aid, text in texts:
        if len(_words(text)) < MIN_TEXT_LENGTH_WORDS:
            findings.append(
                AuditFinding(
                    trait=trait,
                    artifact_type=atype,
                    artifact_id=aid,
                    severity="high",
                    issue_type="text_too_short",
                    text=_snippet(text),
                    explanation=(
                        f"Text has fewer than {MIN_TEXT_LENGTH_WORDS} words. "
                        "May be empty or incomplete."
                    ),
                    suggested_review_action="Expand or replace the text.",
                )
            )
    return findings


def _check_trait_label_leakage_in_questions(bank: ArtifactBank) -> list[AuditFinding]:
    """
    Flag elicitation questions that explicitly name the target trait.
    Example: a honesty question containing the word 'honesty' tells the model
    exactly which construct is under evaluation — a leakage risk.
    """
    findings: list[AuditFinding] = []
    df = bank.questions_df
    for trait in VALID_TRAITS:
        label_words = _TRAIT_LABEL_WORDS.get(trait, [])
        for split in ("extraction", "validation"):
            sub = df[(df["trait"] == trait) & (df["split"] == split)]
            flagged_ids: list[str] = []
            for _, row in sub.iterrows():
                if _count_words_from_list(row["question_text"], label_words) > 0:
                    flagged_ids.append(row["question_id"])
            rate = len(flagged_ids) / max(len(sub), 1)
            if rate >= TRAIT_LABEL_LEAKAGE_THRESHOLD:
                findings.append(
                    AuditFinding(
                        trait=trait,
                        artifact_type=f"{split}_question",
                        artifact_id=", ".join(flagged_ids),
                        severity="warning",
                        issue_type="trait_label_leakage",
                        text=f"{len(flagged_ids)}/{len(sub)} questions mention trait words",
                        explanation=(
                            f"{rate:.0%} of {split} questions for '{trait}' explicitly "
                            f"use trait-label words ({label_words[:4]}…). "
                            "This may cue the model about which construct is being probed."
                        ),
                        suggested_review_action=(
                            "Rephrase questions to probe the construct without naming it. "
                            "Some incidental usage is acceptable if the phrasing is natural."
                        ),
                    )
                )
            elif flagged_ids:
                # Below threshold — report as info
                for qid in flagged_ids:
                    row = sub[sub["question_id"] == qid].iloc[0]
                    findings.append(
                        AuditFinding(
                            trait=trait,
                            artifact_type=f"{split}_question",
                            artifact_id=qid,
                            severity="info",
                            issue_type="trait_label_leakage",
                            text=_snippet(row["question_text"]),
                            explanation=(
                                f"Question mentions a '{trait}' trait-label word. "
                                "May be fine if the phrasing is natural and the word "
                                "appears in the answer rather than the question."
                            ),
                            suggested_review_action=(
                                "Check whether the word primes the model. "
                                "If so, rephrase to describe the situation without naming the construct."
                            ),
                        )
                    )
    return findings


def _check_cross_trait_confounds_in_prompts(bank: ArtifactBank) -> list[AuditFinding]:
    """
    Flag system prompts that use many words native to a *different* trait.
    Example: a harmlessness-negative prompt using many 'honesty' words
    (deceptive, mislead) introduces a confound — the vector may partially
    capture deception rather than willingness to enable harm.
    """
    findings: list[AuditFinding] = []
    sp = bank.system_prompts_df
    for trait in VALID_TRAITS:
        native_key = _TRAIT_NATIVE_LEXICON[trait]
        other_traits = {t: _TRAIT_NATIVE_LEXICON[t] for t in VALID_TRAITS if t != trait}
        for pole in ("positive", "negative"):
            sub = sp[(sp["trait"] == trait) & (sp["pole"] == pole)]
            for _, row in sub.iterrows():
                text = row["prompt_text"]
                for other_trait, other_key in other_traits.items():
                    n_cross = _count_words_from_list(text, CONFOUND_LEXICON[other_key])
                    if n_cross >= CROSS_TRAIT_WORD_THRESHOLD:
                        findings.append(
                            AuditFinding(
                                trait=trait,
                                artifact_type=f"{pole}_system_prompt",
                                artifact_id=row["prompt_id"],
                                severity="warning",
                                issue_type="cross_trait_confound",
                                text=_snippet(text),
                                explanation=(
                                    f"This {pole} prompt for '{trait}' uses {n_cross} words "
                                    f"associated with '{other_trait}' "
                                    f"({CONFOUND_LEXICON[other_key][:4]}…). "
                                    f"Risk: the contrast vector may partially capture "
                                    f"'{other_trait}' rather than '{trait}'."
                                ),
                                suggested_review_action=(
                                    f"Replace or reduce '{other_trait}'-related terms. "
                                    f"Focus the prompt on '{trait}' specifically."
                                ),
                            )
                        )
    return findings


def _check_generic_valence_in_prompts(bank: ArtifactBank) -> list[AuditFinding]:
    """
    Flag prompts where generic positive/negative valence words dominate over
    trait-specific language.  Positive prompts that are 'generically good' and
    negative prompts that are 'generically evil' produce a vector that measures
    overall moral valence, not the specific trait.
    """
    findings: list[AuditFinding] = []
    sp = bank.system_prompts_df
    gen_pos = CONFOUND_LEXICON["generic_positive"]
    gen_neg = CONFOUND_LEXICON["generic_negative"]

    for trait in VALID_TRAITS:
        native_key = _TRAIT_NATIVE_LEXICON[trait]
        native_words = CONFOUND_LEXICON[native_key]

        for pole, generic_list in [("positive", gen_pos), ("negative", gen_neg)]:
            sub = sp[(sp["trait"] == trait) & (sp["pole"] == pole)]
            for _, row in sub.iterrows():
                text = row["prompt_text"]
                n_generic = _count_words_from_list(text, generic_list)
                n_native = _count_words_from_list(text, native_words)

                # Warn if generic count is high relative to native count
                if n_generic >= 3 and n_native == 0:
                    findings.append(
                        AuditFinding(
                            trait=trait,
                            artifact_type=f"{pole}_system_prompt",
                            artifact_id=row["prompt_id"],
                            severity="warning",
                            issue_type="generic_valence_dominance",
                            text=_snippet(text),
                            explanation=(
                                f"Prompt uses {n_generic} generic-{pole} words "
                                f"but 0 words native to '{trait}'. "
                                "Risk: vector captures generic moral valence rather "
                                f"than '{trait}' specifically."
                            ),
                            suggested_review_action=(
                                f"Add language that targets '{trait}' specifically. "
                                "Reduce generic valence words that apply to any moral trait."
                            ),
                        )
                    )
                elif n_generic >= 2 and n_native == 0:
                    findings.append(
                        AuditFinding(
                            trait=trait,
                            artifact_type=f"{pole}_system_prompt",
                            artifact_id=row["prompt_id"],
                            severity="info",
                            issue_type="generic_valence_dominance",
                            text=_snippet(text),
                            explanation=(
                                f"Prompt uses {n_generic} generic-{pole} words "
                                f"and {n_native} '{trait}'-native words. "
                                "May be fine if the framing is sufficiently specific."
                            ),
                            suggested_review_action=(
                                f"Verify the prompt clearly targets '{trait}', "
                                "not just generally good/bad behaviour."
                            ),
                        )
                    )
    return findings


def _check_near_duplicate_prompts(bank: ArtifactBank) -> list[AuditFinding]:
    """Detect near-duplicate system prompts within the same trait×pole."""
    findings: list[AuditFinding] = []
    sp = bank.system_prompts_df
    for trait in VALID_TRAITS:
        for pole in ("positive", "negative"):
            sub = sp[(sp["trait"] == trait) & (sp["pole"] == pole)]
            rows = list(sub.itertuples())
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    sim = _jaccard(rows[i].prompt_text, rows[j].prompt_text)
                    if sim >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                        findings.append(
                            AuditFinding(
                                trait=trait,
                                artifact_type=f"{pole}_system_prompt",
                                artifact_id=f"{rows[i].prompt_id} ↔ {rows[j].prompt_id}",
                                severity="high",
                                issue_type="near_duplicate",
                                text=f"Jaccard similarity = {sim:.2f}",
                                explanation=(
                                    f"Prompts '{rows[i].prompt_id}' and "
                                    f"'{rows[j].prompt_id}' are very similar "
                                    f"(Jaccard = {sim:.2f}). "
                                    "Near-duplicates reduce effective sample diversity."
                                ),
                                suggested_review_action=(
                                    "Replace one prompt with a substantively different "
                                    "framing of the same pole."
                                ),
                            )
                        )
    return findings


def _check_near_duplicate_questions(bank: ArtifactBank) -> list[AuditFinding]:
    """Detect near-duplicate elicitation questions within the same trait."""
    findings: list[AuditFinding] = []
    df = bank.questions_df
    for trait in VALID_TRAITS:
        sub = df[df["trait"] == trait]
        rows = list(sub.itertuples())
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                sim = _jaccard(rows[i].question_text, rows[j].question_text)
                if sim >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                    findings.append(
                        AuditFinding(
                            trait=trait,
                            artifact_type=f"{rows[i].split}_question",
                            artifact_id=f"{rows[i].question_id} ↔ {rows[j].question_id}",
                            severity="high",
                            issue_type="near_duplicate",
                            text=f"Jaccard similarity = {sim:.2f}",
                            explanation=(
                                f"Questions '{rows[i].question_id}' and "
                                f"'{rows[j].question_id}' are very similar "
                                f"(Jaccard = {sim:.2f})."
                            ),
                            suggested_review_action=(
                                "Replace one question with a substantively different scenario."
                            ),
                        )
                    )
    return findings


def _check_extraction_validation_text_overlap(bank: ArtifactBank) -> list[AuditFinding]:
    """
    Flag cases where an extraction question and a validation question are nearly
    identical in *text* (IDs are already checked at load time; this checks content).
    """
    findings: list[AuditFinding] = []
    df = bank.questions_df
    for trait in VALID_TRAITS:
        ext = df[(df["trait"] == trait) & (df["split"] == "extraction")]
        val = df[(df["trait"] == trait) & (df["split"] == "validation")]
        for _, er in ext.iterrows():
            for _, vr in val.iterrows():
                sim = _jaccard(er["question_text"], vr["question_text"])
                if sim >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                    findings.append(
                        AuditFinding(
                            trait=trait,
                            artifact_type="extraction_question",
                            artifact_id=f"{er['question_id']} ↔ {vr['question_id']}",
                            severity="high",
                            issue_type="extraction_validation_text_overlap",
                            text=f"Jaccard similarity = {sim:.2f}",
                            explanation=(
                                f"Extraction question '{er['question_id']}' and "
                                f"validation question '{vr['question_id']}' have "
                                f"very similar text (Jaccard = {sim:.2f}). "
                                "The validation split must be genuinely held-out."
                            ),
                            suggested_review_action=(
                                "Replace one with a distinct scenario that tests the "
                                "same construct without sharing vocabulary."
                            ),
                        )
                    )
    return findings


def _check_rubric_unrelated_trait_words(bank: ArtifactBank) -> list[AuditFinding]:
    """
    Flag rubrics that use many words native to traits *other* than the one they
    are evaluating.  A harmlessness rubric that talks about empathy/suffering
    conflates two constructs in the judge score.
    """
    findings: list[AuditFinding] = []
    for _, row in bank.rubrics_df.iterrows():
        trait = row["trait"]
        text = str(row["scoring_instructions"])
        other_traits = {t: _TRAIT_NATIVE_LEXICON[t] for t in VALID_TRAITS if t != trait}
        for other_trait, other_key in other_traits.items():
            n = _count_words_from_list(text, CONFOUND_LEXICON[other_key])
            if n >= CROSS_TRAIT_WORD_THRESHOLD:
                findings.append(
                    AuditFinding(
                        trait=trait,
                        artifact_type="rubric",
                        artifact_id=f"{trait}_rubric",
                        severity="info",
                        issue_type="rubric_cross_trait_words",
                        text=_snippet(text),
                        explanation=(
                            f"The '{trait}' rubric uses {n} words associated with "
                            f"'{other_trait}'. "
                            "Minor overlap is normal (traits co-occur in moral scenarios), "
                            "but heavy overlap may cause the judge to score for the wrong construct."
                        ),
                        suggested_review_action=(
                            f"Check that scoring instructions focus on '{trait}' "
                            f"and do not inadvertently score for '{other_trait}'."
                        ),
                    )
                )
    return findings


def _check_positive_pole_generic_helpfulness(bank: ArtifactBank) -> list[AuditFinding]:
    """
    Specific check: positive prompts that read as 'be maximally helpful/good'
    rather than targeting the specific trait.  Heuristic: prompt mentions
    'helpful' or 'assist' multiple times but has no trait-specific language.
    """
    helpfulness_words = ["helpful", "help", "assist", "assist", "accommodate", "support", "cooperative"]
    findings: list[AuditFinding] = []
    sp = bank.system_prompts_df
    for trait in VALID_TRAITS:
        native_words = CONFOUND_LEXICON[_TRAIT_NATIVE_LEXICON[trait]]
        sub = sp[(sp["trait"] == trait) & (sp["pole"] == "positive")]
        for _, row in sub.iterrows():
            text = row["prompt_text"]
            n_help = _count_words_from_list(text, helpfulness_words)
            n_native = _count_words_from_list(text, native_words)
            if n_help >= 2 and n_native == 0:
                findings.append(
                    AuditFinding(
                        trait=trait,
                        artifact_type="positive_system_prompt",
                        artifact_id=row["prompt_id"],
                        severity="warning",
                        issue_type="generic_helpfulness_collapse",
                        text=_snippet(text),
                        explanation=(
                            f"Positive prompt uses {n_help} helpfulness-related words "
                            f"and {n_native} '{trait}'-specific words. "
                            "Risk: positive pole is 'generic helpful assistant' rather than "
                            f"an assistant that specifically embodies '{trait}'."
                        ),
                        suggested_review_action=(
                            f"Revise to emphasise '{trait}' specifically. "
                            "Avoid framing the positive pole as maximally helpful/good in general."
                        ),
                    )
                )
    return findings


def _check_negative_pole_extra_traits(bank: ArtifactBank) -> list[AuditFinding]:
    """
    Detect negative prompts that introduce traits clearly from *other* domains
    in ways that would contaminate the vector.  Known bad patterns:
      - honesty-negative: introduces cruelty/harm (harmlessness)
      - harmlessness-negative: introduces deception (honesty)
      - fairness-negative: introduces cruelty (compassion)
      - compassion-negative: introduces deception (honesty)
    """
    # Per-trait: which OTHER trait words are most problematic in the negative pole
    extra_trait_concern: dict[str, list[tuple[str, str]]] = {
        "honesty": [("harmlessness", "harmlessness_related"), ("compassion", "compassion_related")],
        "harmlessness": [("honesty", "honesty_related"), ("fairness", "fairness_related")],
        "fairness": [("compassion", "compassion_related"), ("honesty", "honesty_related")],
        "compassion": [("honesty", "honesty_related"), ("harmlessness", "harmlessness_related")],
    }
    findings: list[AuditFinding] = []
    sp = bank.system_prompts_df
    for trait, concerns in extra_trait_concern.items():
        sub = sp[(sp["trait"] == trait) & (sp["pole"] == "negative")]
        for _, row in sub.iterrows():
            text = row["prompt_text"]
            for other_trait, other_key in concerns:
                n = _count_words_from_list(text, CONFOUND_LEXICON[other_key])
                if n >= 2:
                    findings.append(
                        AuditFinding(
                            trait=trait,
                            artifact_type="negative_system_prompt",
                            artifact_id=row["prompt_id"],
                            severity="warning",
                            issue_type="negative_pole_extra_trait",
                            text=_snippet(text),
                            explanation=(
                                f"Negative prompt for '{trait}' uses {n} words "
                                f"associated with '{other_trait}'. "
                                f"Example failure: honesty-negative that introduces cruelty "
                                f"→ vector may capture '{other_trait}' rather than absence of '{trait}'."
                            ),
                            suggested_review_action=(
                                f"Revise to remove '{other_trait}'-related language. "
                                f"The negative pole should suppress '{trait}' specifically, "
                                "not introduce other moral failings."
                            ),
                        )
                    )
    return findings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_quality_checks(bank: ArtifactBank) -> list[AuditFinding]:
    """Run all quality checks and return a flat list of AuditFindings."""

    # Collect all (trait, type, id, text) tuples for short-text check
    all_texts: list[tuple[str, str, str, str]] = []
    for _, row in bank.system_prompts_df.iterrows():
        pole = row["pole"]
        all_texts.append((row["trait"], f"{pole}_system_prompt", row["prompt_id"], row["prompt_text"]))
    for _, row in bank.questions_df.iterrows():
        split = row["split"]
        all_texts.append((row["trait"], f"{split}_question", row["question_id"], row["question_text"]))

    findings: list[AuditFinding] = []
    findings += _check_short_texts(all_texts)
    findings += _check_trait_label_leakage_in_questions(bank)
    findings += _check_cross_trait_confounds_in_prompts(bank)
    findings += _check_generic_valence_in_prompts(bank)
    findings += _check_near_duplicate_prompts(bank)
    findings += _check_near_duplicate_questions(bank)
    findings += _check_extraction_validation_text_overlap(bank)
    findings += _check_rubric_unrelated_trait_words(bank)
    findings += _check_positive_pole_generic_helpfulness(bank)
    findings += _check_negative_pole_extra_traits(bank)

    return findings


def findings_to_df(findings: list[AuditFinding]) -> "pd.DataFrame":  # type: ignore[name-defined]
    import pandas as pd
    if not findings:
        return pd.DataFrame(
            columns=[
                "trait", "artifact_type", "artifact_id", "severity",
                "issue_type", "text", "explanation", "suggested_review_action",
            ]
        )
    return pd.DataFrame([f.to_dict() for f in findings])
