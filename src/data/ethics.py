"""
ETHICS benchmark loading, normalization, and sampling.

Schema notes per split (all loaded from the 'test' CSV files on HuggingFace):

  commonsense   : label/input/is_short/edited
                  label=1 → morally wrong; label=0 → morally OK
  deontology    : label/scenario/excuse
                  label=1 → excuse is NOT valid (moral obligation stands)
                  label=0 → excuse is valid (obligation is released)
  justice       : label/scenario
                  label=1 → action is justified/just
                  label=0 → action is unjustified/unjust
  utilitarianism: baseline/less_pleasant   (NO label column)
                  paired comparison; baseline is presumed more morally pleasant
                  normalised as: scenario_text = baseline; label = None
  virtue        : label/scenario  where scenario = "situation [SEP] trait_word"
                  label=1 → trait word fits the situation; label=0 → does not fit
                  We extract the situation text (before [SEP]) as scenario_text
                  and store the trait_word separately. Rows are deduplicated at
                  the situation level before sampling to avoid over-representing
                  the same situation with different trait words.

IMPORTANT: ETHICS split names (commonsense, deontology, justice, …) are
source-dataset labels, NOT the construct traits of this project.
Construct traits (honesty, harmlessness, fairness, compassion) are assigned
separately during annotation (Stage 2).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ETHICS_REPO = "hendrycks/ethics"
HF_SPLIT_FILE = "test.csv"  # we always use the test split

LABEL_SEMANTICS: dict[str, str] = {
    "commonsense": "1=morally_wrong; 0=morally_OK",
    "deontology": "1=excuse_invalid(obligation_stands); 0=excuse_valid",
    "justice": "1=action_is_just; 0=action_is_unjust",
    "utilitarianism": "no_label; baseline_is_more_morally_pleasant_than_less_pleasant",
    "virtue": "1=trait_word_fits_situation; 0=trait_word_does_not_fit",
}

ANNOTATION_COLUMNS = {
    "primary_trait": "",
    "secondary_traits": "",
    "annotation_confidence": "",
    "annotation_notes": "",
}

CANONICAL_COLUMNS = [
    "item_id",
    "source_dataset",
    "source_split",
    "scenario_text",
    "label",
    "label_semantics",
    "raw_fields",
    "primary_trait",
    "secondary_traits",
    "annotation_confidence",
    "annotation_notes",
]

# ---------------------------------------------------------------------------
# Private normalizers (one per split)
# ---------------------------------------------------------------------------

def _normalise_commonsense(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        raw = {
            "label": int(row["label"]),
            "input": str(row["input"]),
            "is_short": bool(row["is_short"]),
            "edited": bool(row["edited"]),
        }
        rows.append(
            {
                "item_id": f"{split_name}_test_{idx:06d}",
                "source_dataset": "ETHICS",
                "source_split": split_name,
                "scenario_text": str(row["input"]).strip(),
                "label": int(row["label"]),
                "label_semantics": LABEL_SEMANTICS[split_name],
                "raw_fields": json.dumps(raw),
                **ANNOTATION_COLUMNS,
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def _normalise_deontology(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        scenario = str(row["scenario"]).strip()
        excuse = str(row["excuse"]).strip()
        # Combine into a single readable text for model probing
        scenario_text = f"{scenario} [EXCUSE] {excuse}"
        raw = {
            "label": int(row["label"]),
            "scenario": scenario,
            "excuse": excuse,
        }
        rows.append(
            {
                "item_id": f"{split_name}_test_{idx:06d}",
                "source_dataset": "ETHICS",
                "source_split": split_name,
                "scenario_text": scenario_text,
                "label": int(row["label"]),
                "label_semantics": LABEL_SEMANTICS[split_name],
                "raw_fields": json.dumps(raw),
                **ANNOTATION_COLUMNS,
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def _normalise_justice(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        raw = {
            "label": int(row["label"]),
            "scenario": str(row["scenario"]),
        }
        rows.append(
            {
                "item_id": f"{split_name}_test_{idx:06d}",
                "source_dataset": "ETHICS",
                "source_split": split_name,
                "scenario_text": str(row["scenario"]).strip(),
                "label": int(row["label"]),
                "label_semantics": LABEL_SEMANTICS[split_name],
                "raw_fields": json.dumps(raw),
                **ANNOTATION_COLUMNS,
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def _normalise_utilitarianism(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Each row is a (baseline, less_pleasant) pair. We index both so that
    either could be used for probing; scenario_text uses the baseline.
    The less_pleasant text is preserved in raw_fields."""
    rows = []
    for idx, row in df.iterrows():
        baseline = str(row["baseline"]).strip()
        less_pleasant = str(row["less_pleasant"]).strip()
        raw = {
            "baseline": baseline,
            "less_pleasant": less_pleasant,
        }
        rows.append(
            {
                "item_id": f"{split_name}_test_{idx:06d}",
                "source_dataset": "ETHICS",
                "source_split": split_name,
                "scenario_text": baseline,
                "label": None,
                "label_semantics": LABEL_SEMANTICS[split_name],
                "raw_fields": json.dumps(raw),
                **ANNOTATION_COLUMNS,
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def _normalise_virtue(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """virtue rows have the form: "situation [SEP] trait_word".
    We deduplicate to one row per unique situation before returning,
    keeping only label=1 rows (situations where the trait word fits),
    because probing should use natural situation descriptions, not
    the trait-word label. The trait_word is preserved in raw_fields.

    Rationale for dedup: 995 unique situations each appear ~5 times with
    different trait words. Sampling without dedup would over-represent the
    same situation text; situation uniqueness is what matters for probing.
    """
    SEP = " [SEP] "
    # Use regex=False so "[SEP]" is treated as a literal string, not a regex
    # character class.  expand=True always produces 2 columns when regex=False
    # and n=1, even if no separator is found (second column becomes None).
    parsed = df["scenario"].str.split(SEP, n=1, regex=False, expand=True)
    df = df.copy()
    df["_situation"] = parsed[0].str.strip()
    df["_trait_word"] = parsed[1].str.strip() if parsed.shape[1] > 1 else ""

    # Keep label=1 rows (trait word fits), deduplicate by situation text
    # to avoid multiple rows with identical scenario_text
    fitting = df[df["label"] == 1].copy()
    fitting_dedup = fitting.drop_duplicates(subset=["_situation"]).reset_index(drop=True)

    rows = []
    for idx, row in fitting_dedup.iterrows():
        raw = {
            "label": int(row["label"]),
            "scenario": row["scenario"],
            "situation": row["_situation"],
            "trait_word": row["_trait_word"],
        }
        rows.append(
            {
                "item_id": f"{split_name}_test_{idx:06d}",
                "source_dataset": "ETHICS",
                "source_split": split_name,
                "scenario_text": row["_situation"],
                "label": int(row["label"]),
                "label_semantics": LABEL_SEMANTICS[split_name],
                "raw_fields": json.dumps(raw),
                **ANNOTATION_COLUMNS,
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


_NORMALIZERS = {
    "commonsense": _normalise_commonsense,
    "deontology": _normalise_deontology,
    "justice": _normalise_justice,
    "utilitarianism": _normalise_utilitarianism,
    "virtue": _normalise_virtue,
}

# Splits excluded from the MVP by default, with the reason why.
# These can still be loaded explicitly; they just trigger a warning.
_MVP_EXCLUDED_SPLITS: dict[str, str] = {
    "utilitarianism": (
        "The utilitarianism split contains many hedonic preference-comparison "
        "items (e.g. 'going on vacation vs. not going') rather than situations "
        "that clearly engage a moral disposition. It has no label column and its "
        "paired (baseline / less_pleasant) format does not map naturally onto the "
        "four construct traits. Consider excluding it from MVP sampling."
    ),
    "virtue": (
        "The virtue split pairs situations with single trait words via [SEP]. "
        "After deduplication to unique situations all retained rows have label=1, "
        "leaving no negative-valence items. The trait words also overlap with "
        "philosophical virtue categories that differ from this project's construct "
        "labels. Consider excluding it from MVP sampling."
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ethics_split(
    split_name: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Download one ETHICS split CSV from HuggingFace and return a normalised
    DataFrame with CANONICAL_COLUMNS.

    Args:
        split_name: one of commonsense / deontology / justice /
                    utilitarianism / virtue
        cache_dir: optional local directory to store downloaded files
    """
    if split_name in _MVP_EXCLUDED_SPLITS:
        logger.warning(
            "Loading '%s' which is excluded from the MVP by default. %s",
            split_name,
            _MVP_EXCLUDED_SPLITS[split_name],
        )

    if split_name not in _NORMALIZERS:
        raise ValueError(
            f"Unknown ETHICS split '{split_name}'. "
            f"Valid splits: {list(_NORMALIZERS)}"
        )

    logger.info("Downloading ETHICS/%s/test.csv from HuggingFace …", split_name)
    csv_path = hf_hub_download(
        repo_id=ETHICS_REPO,
        filename=f"data/{split_name}/{HF_SPLIT_FILE}",
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    raw_df = pd.read_csv(csv_path)
    normalised = _NORMALIZERS[split_name](raw_df, split_name)
    logger.info(
        "  %s: %d raw rows → %d normalised items",
        split_name,
        len(raw_df),
        len(normalised),
    )
    return normalised


def load_all_ethics_splits(
    splits: list[str],
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and concatenate multiple ETHICS splits into one DataFrame."""
    parts = []
    for split in splits:
        parts.append(load_ethics_split(split, cache_dir=cache_dir))
    combined = pd.concat(parts, ignore_index=True)
    # Sanity check: item_ids must be unique
    if combined["item_id"].duplicated().any():
        dupes = combined[combined["item_id"].duplicated(keep=False)]["item_id"].tolist()
        raise RuntimeError(f"Duplicate item_ids detected: {dupes[:10]}")
    return combined


def sample_ethics_items(
    df: pd.DataFrame,
    n_items: int,
    random_seed: int,
) -> pd.DataFrame:
    """Stratified sample of n_items across source_splits.

    Splits the budget equally across present splits. If a split has fewer
    rows than its share, all its rows are taken and the deficit is distributed
    to splits with surplus.

    Returns a DataFrame with the same schema, sorted by source_split then
    item_id, with a reset index.
    """
    splits = df["source_split"].unique().tolist()
    n_splits = len(splits)
    per_split = n_items // n_splits
    remainder = n_items % n_splits

    rng = pd.Series(splits, name="split")  # deterministic split ordering
    sampled_parts: list[pd.DataFrame] = []
    deficit = 0

    for i, split in enumerate(splits):
        sub = df[df["source_split"] == split]
        # Give the remainder to the first splits in order
        target = per_split + (1 if i < remainder else 0) + deficit
        if len(sub) <= target:
            sampled_parts.append(sub)
            deficit = target - len(sub)  # carry forward to next split
        else:
            sampled_parts.append(sub.sample(n=target, random_state=random_seed))
            deficit = 0

    result = (
        pd.concat(sampled_parts, ignore_index=True)
        .sort_values(["source_split", "item_id"])
        .reset_index(drop=True)
    )
    return result
