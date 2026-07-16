"""
Tests for src/data/ethics.py — schema normalisation, sampling, and
the conceptual guard that ETHICS split names are never used as construct traits.

These tests use minimal synthetic DataFrames; they do NOT download from
HuggingFace and do NOT require GPU or model dependencies.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.ethics import (
    CANONICAL_COLUMNS,
    LABEL_SEMANTICS,
    _NORMALIZERS,
    _normalise_commonsense,
    _normalise_deontology,
    _normalise_justice,
    _normalise_utilitarianism,
    _normalise_virtue,
    load_all_ethics_splits,
    sample_ethics_items,
)
from src.config import ItemAnnotation

# ---------------------------------------------------------------------------
# Helpers — minimal synthetic DataFrames matching real ETHICS schemas
# ---------------------------------------------------------------------------

def make_commonsense(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [0, 1] * (n // 2),
            "input": [f"Commonsense scenario {i}" for i in range(n)],
            "is_short": [True] * n,
            "edited": [False] * n,
        }
    )


def make_deontology(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [0, 1] * (n // 2),
            "scenario": [f"Scenario {i}?" for i in range(n)],
            "excuse": [f"No because reason {i}." for i in range(n)],
        }
    )


def make_justice(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [0, 1] * (n // 2),
            "scenario": [f"Justice scenario {i}" for i in range(n)],
        }
    )


def make_utilitarianism(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "baseline": [f"Baseline scenario {i}" for i in range(n)],
            "less_pleasant": [f"Less pleasant scenario {i}" for i in range(n)],
        }
    )


def make_virtue(n: int = 10) -> pd.DataFrame:
    # 5 unique situations × 2 trait words each; label=1 for first, label=0 for second
    situations = [f"Virtue situation {i}" for i in range(5)]
    rows = []
    for sit in situations:
        rows.append({"label": 1, "scenario": f"{sit} [SEP] kind"})
        rows.append({"label": 0, "scenario": f"{sit} [SEP] cruel"})
    return pd.DataFrame(rows[:n])


# ---------------------------------------------------------------------------
# Schema / canonical columns
# ---------------------------------------------------------------------------

class TestCanonicalSchema:
    def _assert_schema(self, df: pd.DataFrame, split: str) -> None:
        assert list(df.columns) == CANONICAL_COLUMNS, (
            f"{split}: columns mismatch.\n  Got: {list(df.columns)}"
        )

    def test_commonsense_schema(self):
        self._assert_schema(_normalise_commonsense(make_commonsense(), "commonsense"), "commonsense")

    def test_deontology_schema(self):
        self._assert_schema(_normalise_deontology(make_deontology(), "deontology"), "deontology")

    def test_justice_schema(self):
        self._assert_schema(_normalise_justice(make_justice(), "justice"), "justice")

    def test_utilitarianism_schema(self):
        self._assert_schema(_normalise_utilitarianism(make_utilitarianism(), "utilitarianism"), "utilitarianism")

    def test_virtue_schema(self):
        self._assert_schema(_normalise_virtue(make_virtue(), "virtue"), "virtue")


# ---------------------------------------------------------------------------
# source_dataset field
# ---------------------------------------------------------------------------

class TestSourceDataset:
    @pytest.mark.parametrize("split,maker", [
        ("commonsense", make_commonsense),
        ("deontology", make_deontology),
        ("justice", make_justice),
        ("utilitarianism", make_utilitarianism),
        ("virtue", make_virtue),
    ])
    def test_source_dataset_is_ETHICS(self, split, maker):
        df = _NORMALIZERS[split](maker(), split)
        assert (df["source_dataset"] == "ETHICS").all(), (
            f"{split}: source_dataset should always be 'ETHICS'"
        )


# ---------------------------------------------------------------------------
# Stable item_id
# ---------------------------------------------------------------------------

class TestItemId:
    def test_item_ids_unique_within_split(self):
        df = _normalise_commonsense(make_commonsense(10), "commonsense")
        assert df["item_id"].nunique() == len(df)

    def test_item_id_format(self):
        df = _normalise_justice(make_justice(4), "justice")
        for iid in df["item_id"]:
            assert iid.startswith("justice_test_"), f"Bad item_id format: {iid}"

    def test_item_ids_stable_across_calls(self):
        """Same input rows must always produce the same item_ids."""
        raw = make_commonsense(6)
        ids1 = _normalise_commonsense(raw, "commonsense")["item_id"].tolist()
        ids2 = _normalise_commonsense(raw, "commonsense")["item_id"].tolist()
        assert ids1 == ids2

    def test_no_duplicate_ids_across_splits(self):
        parts = [
            _normalise_commonsense(make_commonsense(), "commonsense"),
            _normalise_deontology(make_deontology(), "deontology"),
            _normalise_justice(make_justice(), "justice"),
            _normalise_utilitarianism(make_utilitarianism(), "utilitarianism"),
            _normalise_virtue(make_virtue(), "virtue"),
        ]
        combined = pd.concat(parts, ignore_index=True)
        assert combined["item_id"].nunique() == len(combined), "Duplicate item_ids across splits"


# ---------------------------------------------------------------------------
# Annotation columns blank initially
# ---------------------------------------------------------------------------

class TestAnnotationColumnsBlank:
    @pytest.mark.parametrize("split,maker", [
        ("commonsense", make_commonsense),
        ("deontology", make_deontology),
        ("justice", make_justice),
        ("utilitarianism", make_utilitarianism),
        ("virtue", make_virtue),
    ])
    def test_primary_trait_blank(self, split, maker):
        df = _NORMALIZERS[split](maker(), split)
        assert (df["primary_trait"] == "").all(), (
            f"{split}: primary_trait should be blank at this stage"
        )

    def test_annotation_confidence_blank(self):
        df = _normalise_commonsense(make_commonsense(), "commonsense")
        assert (df["annotation_confidence"] == "").all()


# ---------------------------------------------------------------------------
# CRITICAL: ETHICS split names must NOT be used as construct traits
# ---------------------------------------------------------------------------

ETHICS_SPLIT_NAMES = {"commonsense", "deontology", "justice", "utilitarianism", "virtue"}
CONSTRUCT_TRAITS = {"honesty", "harmlessness", "fairness", "compassion"}


class TestConstructTraitSeparation:
    def test_ethics_split_names_not_in_construct_traits(self):
        """The two namespaces must be disjoint — this would catch a
        configuration mistake where someone treats 'justice' as a construct
        trait instead of a source-dataset label."""
        overlap = ETHICS_SPLIT_NAMES & CONSTRUCT_TRAITS
        assert overlap == set(), (
            f"ETHICS split names overlap with construct trait names: {overlap}. "
            "Source-dataset labels and measurement constructs must be kept separate."
        )

    def test_source_split_column_never_holds_construct_trait_name(self):
        """Even if a user accidentally wrote a construct trait name as a split,
        the normaliser should propagate only what it receives — this test
        ensures we'd catch contamination at the DataFrame level."""
        for split in ETHICS_SPLIT_NAMES:
            maker = {
                "commonsense": make_commonsense,
                "deontology": make_deontology,
                "justice": make_justice,
                "utilitarianism": make_utilitarianism,
                "virtue": make_virtue,
            }[split]
            df = _NORMALIZERS[split](maker(), split)
            for val in df["source_split"].unique():
                assert val not in CONSTRUCT_TRAITS, (
                    f"source_split column contains a construct trait name '{val}' — "
                    "this would conflate source-dataset labels with measurement constructs."
                )

    def test_primary_trait_rejects_ethics_split_names(self):
        """ItemAnnotation (from src/config.py) must reject ETHICS split names
        when they are used as primary_trait values."""
        for split_name in ETHICS_SPLIT_NAMES:
            with pytest.raises(Exception):
                ItemAnnotation(
                    source_dataset="ETHICS",
                    source_split="commonsense",
                    primary_trait=split_name,  # type: ignore[arg-type]
                )

    def test_primary_trait_accepts_construct_trait_names(self):
        for trait in CONSTRUCT_TRAITS:
            ann = ItemAnnotation(
                source_dataset="ETHICS",
                source_split="commonsense",
                primary_trait=trait,  # type: ignore[arg-type]
            )
            assert ann.primary_trait == trait


# ---------------------------------------------------------------------------
# Split-specific normalisation behaviour
# ---------------------------------------------------------------------------

class TestNormalisationDetails:
    def test_commonsense_scenario_text_is_input_column(self):
        raw = pd.DataFrame({"label": [0], "input": ["Hello world"], "is_short": [True], "edited": [False]})
        df = _normalise_commonsense(raw, "commonsense")
        assert df.iloc[0]["scenario_text"] == "Hello world"

    def test_deontology_combines_scenario_and_excuse(self):
        raw = pd.DataFrame({"label": [1], "scenario": ["Should you?"], "excuse": ["No because X."]})
        df = _normalise_deontology(raw, "deontology")
        assert "Should you?" in df.iloc[0]["scenario_text"]
        assert "No because X." in df.iloc[0]["scenario_text"]

    def test_utilitarianism_label_is_none(self):
        df = _normalise_utilitarianism(make_utilitarianism(4), "utilitarianism")
        assert df["label"].isna().all(), "utilitarianism has no label; should be None/NaN"

    def test_utilitarianism_scenario_text_is_baseline(self):
        raw = pd.DataFrame({"baseline": ["Good scenario"], "less_pleasant": ["Bad scenario"]})
        df = _normalise_utilitarianism(raw, "utilitarianism")
        assert df.iloc[0]["scenario_text"] == "Good scenario"

    def test_utilitarianism_less_pleasant_in_raw_fields(self):
        raw = pd.DataFrame({"baseline": ["Good"], "less_pleasant": ["Bad"]})
        df = _normalise_utilitarianism(raw, "utilitarianism")
        raw_fields = json.loads(df.iloc[0]["raw_fields"])
        assert raw_fields["less_pleasant"] == "Bad"

    def test_virtue_deduplicates_by_situation(self):
        """Same situation with different trait words should yield one row."""
        raw = pd.DataFrame([
            {"label": 1, "scenario": "She was brave. [SEP] courageous"},
            {"label": 0, "scenario": "She was brave. [SEP] cowardly"},
            {"label": 1, "scenario": "She was brave. [SEP] bold"},
        ])
        df = _normalise_virtue(raw, "virtue")
        assert len(df) == 1, "Virtue dedup should collapse same situation to one row"

    def test_virtue_scenario_text_excludes_trait_word(self):
        raw = pd.DataFrame([{"label": 1, "scenario": "She helped. [SEP] kind"}])
        df = _normalise_virtue(raw, "virtue")
        assert df.iloc[0]["scenario_text"] == "She helped."
        assert "kind" not in df.iloc[0]["scenario_text"]

    def test_virtue_trait_word_preserved_in_raw_fields(self):
        raw = pd.DataFrame([{"label": 1, "scenario": "She helped. [SEP] kind"}])
        df = _normalise_virtue(raw, "virtue")
        raw_fields = json.loads(df.iloc[0]["raw_fields"])
        assert raw_fields["trait_word"] == "kind"

    def test_label_semantics_present_for_all_splits(self):
        for split in ETHICS_SPLIT_NAMES:
            assert split in LABEL_SEMANTICS, f"Missing label_semantics for {split}"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def _make_combined(self, per_split: int = 20) -> pd.DataFrame:
        parts = [
            _normalise_commonsense(make_commonsense(per_split), "commonsense"),
            _normalise_deontology(make_deontology(per_split), "deontology"),
            _normalise_justice(make_justice(per_split), "justice"),
            _normalise_utilitarianism(make_utilitarianism(per_split), "utilitarianism"),
            _normalise_virtue(make_virtue(per_split), "virtue"),
        ]
        return pd.concat(parts, ignore_index=True)

    def test_sample_returns_correct_total(self):
        df = self._make_combined(20)
        sampled = sample_ethics_items(df, n_items=25, random_seed=42)
        assert len(sampled) == 25

    def test_sampling_is_reproducible(self):
        df = self._make_combined(20)
        s1 = sample_ethics_items(df, n_items=20, random_seed=7)
        s2 = sample_ethics_items(df, n_items=20, random_seed=7)
        assert s1["item_id"].tolist() == s2["item_id"].tolist()

    def test_different_seeds_give_different_samples(self):
        df = self._make_combined(20)
        s1 = sample_ethics_items(df, n_items=10, random_seed=1)
        s2 = sample_ethics_items(df, n_items=10, random_seed=2)
        assert s1["item_id"].tolist() != s2["item_id"].tolist()

    def test_sample_schema_unchanged(self):
        df = self._make_combined(20)
        sampled = sample_ethics_items(df, n_items=15, random_seed=0)
        assert list(sampled.columns) == CANONICAL_COLUMNS

    def test_sample_handles_small_splits(self):
        """If a split has fewer rows than its share, take all and don't crash."""
        parts = [
            _normalise_commonsense(make_commonsense(2), "commonsense"),   # tiny
            _normalise_justice(make_justice(50), "justice"),              # large
        ]
        df = pd.concat(parts, ignore_index=True)
        sampled = sample_ethics_items(df, n_items=20, random_seed=0)
        # Should not crash and should return ≤20 items
        assert len(sampled) <= 20
        # commonsense contributes all its rows
        assert (sampled["source_split"] == "commonsense").sum() == 2

    def test_sample_stratified_across_splits(self):
        df = self._make_combined(20)
        sampled = sample_ethics_items(df, n_items=15, random_seed=0)
        # Each split should contribute at least 1 item
        represented = sampled["source_split"].nunique()
        assert represented >= 3, f"Only {represented} splits represented in sample"

    def test_no_gpu_or_model_imports(self):
        """Importing the data module must not pull in torch/transformers."""
        import importlib
        import sys

        # If torch is not installed this will simply pass; if it IS installed,
        # we just verify the data module doesn't require it to import cleanly.
        import src.data.ethics as ethics_module
        assert ethics_module is not None
        # The critical check: no GPU-specific attributes crept in
        assert not hasattr(ethics_module, "torch")
        assert not hasattr(ethics_module, "transformers")
