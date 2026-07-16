"""
Stage 4B: Lightweight semantic-equivalence validation for paraphrase variants.

These checks are heuristics, not ground truth.  Their purpose is to flag
variants that are likely wrong for human review — not to certify correctness.

Statuses (in priority order):
  original                    — the original item, always passes
  passed                      — all heuristic checks pass
  flagged_length              — variant is much shorter or longer than original
  flagged_duplicate           — variant is identical or near-identical to original
                                or another variant for the same item
  flagged_possible_meaning_shift — negation flip, reversal, or other red flag
  failed_parse                — the LLM response could not be parsed at all
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd


# ---------------------------------------------------------------------------
# Single-variant checks
# ---------------------------------------------------------------------------


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of token sets."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


_NEGATION_PATTERN = re.compile(
    r"\b(not|never|no|didn'?t|doesn'?t|won'?t|can'?t|couldn'?t|wouldn'?t|"
    r"shouldn'?t|isn'?t|aren'?t|wasn'?t|weren'?t|hasn'?t|haven'?t|hadn'?t|"
    r"refused|failed|stopped|prevented)\b",
    re.IGNORECASE,
)

_REVERSAL_PAIRS = [
    (r"\bdid\b", r"\bdid not\b"),
    (r"\bdoes\b", r"\bdoes not\b"),
    (r"\bwill\b", r"\bwill not\b"),
    (r"\bhelped\b", r"\brefused to help\b"),
    (r"\btold the truth\b", r"\blied\b"),
    (r"\blied\b", r"\btold the truth\b"),
    (r"\bstole\b", r"\breturned\b"),
    (r"\bhurt\b", r"\bhelped\b"),
]


def _negation_count_diff(original: str, variant: str) -> int:
    """Return difference in negation-word count between variant and original."""
    orig_n = len(_NEGATION_PATTERN.findall(original))
    var_n = len(_NEGATION_PATTERN.findall(variant))
    return abs(var_n - orig_n)


def _has_reversal(original: str, variant: str) -> bool:
    """True if the variant appears to flip a morally-relevant action."""
    for pos_pat, neg_pat in _REVERSAL_PAIRS:
        orig_has_pos = bool(re.search(pos_pat, original, re.IGNORECASE))
        var_has_neg = bool(re.search(neg_pat, variant, re.IGNORECASE))
        if orig_has_pos and var_has_neg:
            return True
        # Also check the reverse direction
        orig_has_neg = bool(re.search(neg_pat, original, re.IGNORECASE))
        var_has_pos = bool(re.search(pos_pat, variant, re.IGNORECASE))
        if orig_has_neg and var_has_pos:
            return True
    return False


_SHORT_ORIGINAL_WORD_THRESHOLD = 15
_SHORT_ORIGINAL_MAX_LENGTH_RATIO = 3.0
# Long AITA-style posts (>100 words) get a relaxed min ratio because the model
# naturally condenses them while preserving the core moral scenario.
# Negation/reversal heuristics are also skipped for long originals: they were
# designed for short single-sentence scenarios and produce spurious hits on
# multi-sentence narratives where negations appear throughout the text.
_LONG_ORIGINAL_WORD_THRESHOLD = 100
_LONG_ORIGINAL_MIN_LENGTH_RATIO = 0.15


def check_semantic_equivalence(
    original_text: str,
    variant_text: str,
    max_length_ratio: float = 1.5,
    min_length_ratio: float = 0.5,
    duplicate_token_threshold: float = 0.98,
) -> str:
    """
    Run heuristic checks on a single (original, variant) pair.

    Returns a semantic_equivalence_status string.

    For very short originals (< 8 words), the max_length_ratio is relaxed to
    3.0 because a 4-word sentence paraphrased into 7 words is valid even though
    the ratio exceeds 1.5.
    """
    if not variant_text or not variant_text.strip():
        return "failed_parse"

    # Length ratio check — relax thresholds at both extremes:
    #  - short originals (<15 words): allow up to 3x expansion
    #  - long originals (>100 words, e.g. AITA posts): allow down to 0.3x compression
    orig_len = len(original_text.split())
    var_len = len(variant_text.split())
    if orig_len > 0:
        effective_max = (
            _SHORT_ORIGINAL_MAX_LENGTH_RATIO
            if orig_len < _SHORT_ORIGINAL_WORD_THRESHOLD
            else max_length_ratio
        )
        effective_min = (
            _LONG_ORIGINAL_MIN_LENGTH_RATIO
            if orig_len > _LONG_ORIGINAL_WORD_THRESHOLD
            else min_length_ratio
        )
        ratio = var_len / orig_len
        if ratio > effective_max or ratio < effective_min:
            return "flagged_length"

    # Near-duplicate of original
    overlap = _token_overlap(original_text, variant_text)
    if overlap >= duplicate_token_threshold:
        return "flagged_duplicate"

    # Negation and reversal checks are only reliable for short single-sentence
    # scenarios.  For long multi-sentence narratives (>100 words) the heuristics
    # produce spurious hits because negations appear throughout the text and a
    # condensed paraphrase naturally omits some of them.
    if orig_len <= _LONG_ORIGINAL_WORD_THRESHOLD:
        if _negation_count_diff(original_text, variant_text) >= 3:
            return "flagged_possible_meaning_shift"
        if _has_reversal(original_text, variant_text):
            return "flagged_possible_meaning_shift"

    return "passed"


# ---------------------------------------------------------------------------
# Cross-variant checks (duplicates within an item's paraphrase set)
# ---------------------------------------------------------------------------


def flag_intra_item_duplicates(
    df: pd.DataFrame,
    duplicate_token_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    For each item, flag paraphrase variants that are near-duplicates of
    each other (not just of the original).

    Modifies semantic_equivalence_status in-place on paraphrase rows that
    are already "passed".

    Returns the modified dataframe.
    """
    df = df.copy()
    for item_id, group in df.groupby("item_id"):
        para_mask = group["variant_type"] == "paraphrase"
        para_rows = group[para_mask]
        texts = para_rows["scenario_text_variant"].tolist()
        idxs = para_rows.index.tolist()

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if _token_overlap(texts[i], texts[j]) >= duplicate_token_threshold:
                    # Flag the later one
                    idx_j = idxs[j]
                    current = df.at[idx_j, "semantic_equivalence_status"]
                    if current == "passed":
                        df.at[idx_j, "semantic_equivalence_status"] = "flagged_duplicate"
                        df.at[idx_j, "keep_variant"] = False
    return df


# ---------------------------------------------------------------------------
# Validate an entire variant bank dataframe
# ---------------------------------------------------------------------------


def validate_variant_bank(
    df: pd.DataFrame,
    expected_n_paraphrases: int = 3,
    max_length_ratio: float = 1.5,
    min_length_ratio: float = 0.5,
) -> dict:
    """
    Validate a variant bank dataframe and return a summary dict.

    Checks:
      - required columns present
      - variant_id uniqueness
      - no missing scenario_text_variant for keep_variant=True rows
      - expected paraphrase count per item
      - status distribution
    """
    from src.reliability.variant_generation import VARIANT_COLUMNS

    missing_cols = [c for c in VARIANT_COLUMNS if c not in df.columns]

    n_items = df["item_id"].nunique()
    n_originals = int((df["variant_type"] == "original").sum())
    n_paraphrases = int((df["variant_type"] == "paraphrase").sum())
    n_total = len(df)
    n_keep = int(df["keep_variant"].astype(bool).sum())

    # variant_id uniqueness
    duplicate_ids = df["variant_id"][df["variant_id"].duplicated()].tolist()

    # status counts
    status_counts = df["semantic_equivalence_status"].value_counts().to_dict()

    # per-item paraphrase counts
    para_counts = (
        df[df["variant_type"] == "paraphrase"]
        .groupby("item_id")["variant_id"]
        .count()
    )
    items_missing_paraphrases = para_counts[para_counts < expected_n_paraphrases].index.tolist()
    items_complete = int((para_counts == expected_n_paraphrases).sum())

    # empty scenario_text_variant in keep_variant=True rows
    keep_df = df[df["keep_variant"].astype(bool)]
    empty_text = keep_df[keep_df["scenario_text_variant"].str.strip() == ""]["item_id"].tolist()

    return {
        "n_items": n_items,
        "n_originals": n_originals,
        "n_paraphrases": n_paraphrases,
        "n_total": n_total,
        "n_keep": n_keep,
        "missing_columns": missing_cols,
        "duplicate_variant_ids": duplicate_ids,
        "status_counts": status_counts,
        "items_with_incomplete_paraphrases": items_missing_paraphrases,
        "items_with_complete_paraphrases": items_complete,
        "items_with_empty_kept_text": empty_text,
        "is_valid": (
            not missing_cols
            and not duplicate_ids
            and not empty_text
        ),
    }
