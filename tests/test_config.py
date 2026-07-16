"""Tests for config loading and validation."""

import pytest
from pathlib import Path

from src.config import (
    load_mvp_config,
    load_trait_space,
    load_config_and_traits,
    MVPConfig,
    TraitSpace,
    ItemAnnotation,
)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "mvp_experiment.yaml"
TRAITS_PATH = Path(__file__).parent.parent / "configs" / "traits.yaml"

EXPECTED_TRAITS = {"honesty", "harmlessness", "fairness", "compassion"}


class TestMVPConfig:
    def test_loads_without_error(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert isinstance(cfg, MVPConfig)

    def test_model_name_contains_gemma(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert "gemma" in cfg.model.name.lower()

    def test_target_layer_non_negative(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert cfg.model.target_layer >= 0

    def test_layer_indexing_is_zero_indexed(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert cfg.model.layer_indexing == "zero_indexed"

    def test_candidate_layers_present_and_non_empty(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert len(cfg.model.candidate_layers_for_validation) > 0

    def test_candidate_layers_all_non_negative(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert all(l >= 0 for l in cfg.model.candidate_layers_for_validation)

    def test_target_layer_in_candidates(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert cfg.model.target_layer in cfg.model.candidate_layers_for_validation

    def test_dataset_n_items_positive(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert cfg.dataset.n_items > 0

    def test_ethics_splits_field_exists_and_nonempty(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert len(cfg.dataset.ethics_splits) > 0

    def test_ethics_splits_do_not_contain_construct_trait_names(self):
        """ETHICS splits must not be named after our construct traits —
        that would conflate source-dataset categories with measurement constructs."""
        cfg = load_mvp_config(CONFIG_PATH)
        for split in cfg.dataset.ethics_splits:
            assert split not in EXPECTED_TRAITS, (
                f"ETHICS split '{split}' collides with a construct trait name. "
                "Source-dataset categories and measurement constructs must be kept separate."
            )

    def test_framing_conditions_nonempty(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert len(cfg.paraphrase.framing_conditions) >= 1

    def test_random_seed_is_int(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert isinstance(cfg.random_seed, int)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mvp_config(tmp_path / "nonexistent.yaml")


class TestTraitSpace:
    def test_loads_without_error(self):
        ts = load_trait_space(TRAITS_PATH)
        assert isinstance(ts, TraitSpace)

    def test_expected_traits_present(self):
        ts = load_trait_space(TRAITS_PATH)
        assert set(ts.names) == EXPECTED_TRAITS

    def test_positive_descriptions_nonempty(self):
        ts = load_trait_space(TRAITS_PATH)
        for name, td in ts.traits.items():
            assert td.positive_description.strip(), f"{name}: empty positive_description"

    def test_negative_descriptions_nonempty(self):
        ts = load_trait_space(TRAITS_PATH)
        for name, td in ts.traits.items():
            assert td.negative_description.strip(), f"{name}: empty negative_description"

    def test_moral_rationale_nonempty(self):
        ts = load_trait_space(TRAITS_PATH)
        for name, td in ts.traits.items():
            assert td.moral_rationale.strip(), f"{name}: empty moral_rationale"

    def test_item_tagging_guidance_nonempty(self):
        ts = load_trait_space(TRAITS_PATH)
        for name, td in ts.traits.items():
            assert td.item_tagging_guidance.strip(), f"{name}: empty item_tagging_guidance"

    def test_example_keywords_nonempty(self):
        ts = load_trait_space(TRAITS_PATH)
        for name, td in ts.traits.items():
            assert len(td.example_keywords_and_patterns) > 0, (
                f"{name}: no example_keywords_and_patterns"
            )

    def test_names_property(self):
        ts = load_trait_space(TRAITS_PATH)
        assert ts.names == list(ts.traits.keys())

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_trait_space(tmp_path / "nonexistent.yaml")


class TestItemAnnotationSchema:
    def test_valid_annotation(self):
        ann = ItemAnnotation(
            source_dataset="hendrycks/ethics",
            source_split="deontology",
            primary_trait="honesty",
            secondary_traits=["fairness"],
            annotation_confidence="high",
        )
        assert ann.primary_trait == "honesty"
        assert ann.source_split == "deontology"

    def test_rejects_invalid_primary_trait(self):
        with pytest.raises(Exception):
            ItemAnnotation(
                source_dataset="hendrycks/ethics",
                source_split="virtue",
                primary_trait="deontology",  # ETHICS category, not a construct trait
            )

    def test_rejects_invalid_confidence(self):
        with pytest.raises(Exception):
            ItemAnnotation(
                source_dataset="hendrycks/ethics",
                source_split="justice",
                primary_trait="fairness",
                annotation_confidence="very_high",  # not a valid level
            )

    def test_secondary_traits_defaults_to_empty(self):
        ann = ItemAnnotation(
            source_dataset="hendrycks/ethics",
            source_split="commonsense",
            primary_trait="harmlessness",
        )
        assert ann.secondary_traits == []


class TestConvenienceLoader:
    def test_returns_tuple(self):
        cfg, ts = load_config_and_traits(CONFIG_PATH)
        assert isinstance(cfg, MVPConfig)
        assert isinstance(ts, TraitSpace)

    def test_traits_path_resolves(self):
        cfg, ts = load_config_and_traits(CONFIG_PATH)
        assert set(ts.names) == EXPECTED_TRAITS
