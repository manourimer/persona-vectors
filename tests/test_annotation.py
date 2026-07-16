"""
Tests for annotation schema, validation helpers, and conceptual separation
between ETHICS source-split labels and project measurement constructs.

No network access, no GPU, no model loading.
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    AnnotationConfig,
    ItemAnnotation,
    load_mvp_config,
    _ETHICS_SPLIT_NAMES,
)
from src.data.annotation import (
    ANNOTATION_SHEET_COLUMNS,
    AnnotationValidationResult,
    annotation_coverage_report,
    create_annotation_sheet,
    load_annotation_sheet,
    validate_annotations,
)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "mvp_experiment.yaml"

CONSTRUCT_TRAITS = {"honesty", "harmlessness", "fairness", "compassion"}
SPECIAL_LABELS = {"not_applicable", "unclear"}
ALL_VALID_PRIMARY = CONSTRUCT_TRAITS | SPECIAL_LABELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item_bank(n: int = 5, split: str = "commonsense") -> pd.DataFrame:
    """Minimal item-bank DataFrame matching CANONICAL_COLUMNS."""
    return pd.DataFrame(
        {
            "item_id": [f"{split}_test_{i:06d}" for i in range(n)],
            "source_dataset": "ETHICS",
            "source_split": split,
            "scenario_text": [f"Scenario {i}" for i in range(n)],
            "label": [0, 1] * (n // 2) + ([0] if n % 2 else []),
            "label_semantics": "1=morally_wrong; 0=morally_OK",
            "raw_fields": "{}",
            "primary_trait": "",
            "secondary_traits": "",
            "annotation_confidence": "",
            "annotation_notes": "",
        }
    )


def _make_sheet_with_annotations(rows: list[dict]) -> pd.DataFrame:
    """Build an annotation sheet DataFrame from a list of row dicts."""
    base = _make_item_bank(len(rows))
    sheet = base[
        ["item_id", "source_dataset", "source_split", "scenario_text", "label", "label_semantics"]
    ].copy()
    for col in ["primary_trait", "secondary_traits", "annotation_confidence",
                "annotation_notes", "keep_for_mvp"]:
        sheet[col] = ""
    for i, row_data in enumerate(rows):
        for col, val in row_data.items():
            sheet.at[i, col] = val
    return sheet


# ---------------------------------------------------------------------------
# ItemAnnotation model — valid construct traits
# ---------------------------------------------------------------------------

class TestItemAnnotationValidTraits:
    @pytest.mark.parametrize("trait", sorted(CONSTRUCT_TRAITS))
    def test_construct_trait_accepted_as_primary(self, trait):
        ann = ItemAnnotation(
            source_dataset="ETHICS",
            source_split="commonsense",
            primary_trait=trait,
        )
        assert ann.primary_trait == trait

    @pytest.mark.parametrize("label", sorted(SPECIAL_LABELS))
    def test_special_labels_accepted_as_primary(self, label):
        ann = ItemAnnotation(
            source_dataset="ETHICS",
            source_split="deontology",
            primary_trait=label,
        )
        assert ann.primary_trait == label

    def test_secondary_traits_accept_construct_traits(self):
        ann = ItemAnnotation(
            source_dataset="ETHICS",
            source_split="justice",
            primary_trait="honesty",
            secondary_traits=["harmlessness", "fairness"],
        )
        assert set(ann.secondary_traits) == {"harmlessness", "fairness"}

    def test_secondary_traits_default_empty(self):
        ann = ItemAnnotation(
            source_dataset="ETHICS",
            source_split="commonsense",
            primary_trait="compassion",
        )
        assert ann.secondary_traits == []

    def test_keep_for_mvp_defaults_false(self):
        ann = ItemAnnotation(
            source_dataset="ETHICS",
            source_split="commonsense",
            primary_trait="harmlessness",
        )
        assert ann.keep_for_mvp is False


# ---------------------------------------------------------------------------
# ItemAnnotation model — ETHICS split names rejected as primary_trait
# ---------------------------------------------------------------------------

class TestItemAnnotationRejectsEthicsSplitNames:
    @pytest.mark.parametrize("split_name", sorted(_ETHICS_SPLIT_NAMES))
    def test_ethics_split_name_rejected_as_primary_trait(self, split_name):
        with pytest.raises(Exception, match="ETHICS source-split"):
            ItemAnnotation(
                source_dataset="ETHICS",
                source_split="commonsense",
                primary_trait=split_name,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("split_name", sorted(_ETHICS_SPLIT_NAMES))
    def test_ethics_split_name_rejected_as_secondary_trait(self, split_name):
        with pytest.raises(Exception):
            ItemAnnotation(
                source_dataset="ETHICS",
                source_split="commonsense",
                primary_trait="honesty",
                secondary_traits=[split_name],  # type: ignore[list-item]
            )

    def test_source_split_rejects_construct_trait_name(self):
        """source_split must hold an ETHICS label, not a construct trait."""
        with pytest.raises(Exception, match="construct trait"):
            ItemAnnotation(
                source_dataset="ETHICS",
                source_split="honesty",  # wrong — this is a construct trait
                primary_trait="honesty",
            )


# ---------------------------------------------------------------------------
# not_applicable rules
# ---------------------------------------------------------------------------

class TestNotApplicableRules:
    def test_not_applicable_with_empty_secondary_passes(self):
        ann = ItemAnnotation(
            source_dataset="ETHICS",
            source_split="commonsense",
            primary_trait="not_applicable",
        )
        assert ann.primary_trait == "not_applicable"
        assert ann.secondary_traits == []

    def test_not_applicable_with_secondary_traits_rejected(self):
        with pytest.raises(Exception, match="secondary_traits must be empty"):
            ItemAnnotation(
                source_dataset="ETHICS",
                source_split="commonsense",
                primary_trait="not_applicable",
                secondary_traits=["fairness"],
            )

    def test_unclear_with_secondary_traits_allowed(self):
        ann = ItemAnnotation(
            source_dataset="ETHICS",
            source_split="justice",
            primary_trait="unclear",
            secondary_traits=["compassion"],
        )
        assert ann.primary_trait == "unclear"

    def test_primary_trait_not_repeated_in_secondary(self):
        with pytest.raises(Exception, match="must not also appear"):
            ItemAnnotation(
                source_dataset="ETHICS",
                source_split="commonsense",
                primary_trait="fairness",
                secondary_traits=["fairness"],
            )


# ---------------------------------------------------------------------------
# Source-split / construct-trait separation
# ---------------------------------------------------------------------------

class TestConceptualSeparation:
    def test_ethics_and_construct_namespaces_disjoint(self):
        overlap = _ETHICS_SPLIT_NAMES & CONSTRUCT_TRAITS
        assert overlap == set(), (
            f"ETHICS split names overlap with construct trait names: {overlap}"
        )

    def test_annotation_config_valid_primary_contains_all_six(self):
        cfg = AnnotationConfig()
        assert CONSTRUCT_TRAITS.issubset(set(cfg.valid_primary_traits))
        assert SPECIAL_LABELS.issubset(set(cfg.valid_primary_traits))

    def test_annotation_config_valid_secondary_restricted_to_construct_traits(self):
        cfg = AnnotationConfig()
        assert set(cfg.valid_secondary_traits) == CONSTRUCT_TRAITS
        assert "not_applicable" not in cfg.valid_secondary_traits
        assert "unclear" not in cfg.valid_secondary_traits


# ---------------------------------------------------------------------------
# Annotation sheet creation
# ---------------------------------------------------------------------------

class TestCreateAnnotationSheet:
    def test_creates_csv(self, tmp_path):
        bank = _make_item_bank(5)
        out = tmp_path / "sheet.csv"
        sheet = create_annotation_sheet(bank, out)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert list(loaded.columns) == ANNOTATION_SHEET_COLUMNS

    def test_annotation_columns_blank(self, tmp_path):
        bank = _make_item_bank(4)
        out = tmp_path / "sheet.csv"
        sheet = create_annotation_sheet(bank, out)
        for col in ["primary_trait", "secondary_traits", "annotation_confidence",
                    "annotation_notes", "keep_for_mvp"]:
            assert (sheet[col] == "").all(), f"{col} should be blank"

    def test_passthrough_columns_preserved(self, tmp_path):
        bank = _make_item_bank(3, split="justice")
        out = tmp_path / "sheet.csv"
        sheet = create_annotation_sheet(bank, out)
        assert (sheet["source_split"] == "justice").all()
        assert (sheet["source_dataset"] == "ETHICS").all()

    def test_fails_on_missing_required_columns(self, tmp_path):
        bad_bank = pd.DataFrame({"item_id": ["x"], "garbage": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            create_annotation_sheet(bad_bank, tmp_path / "out.csv")


# ---------------------------------------------------------------------------
# Annotation validation
# ---------------------------------------------------------------------------

class TestValidateAnnotations:
    def test_all_blank_reports_zero_annotated(self):
        sheet = _make_item_bank(5)
        sheet["keep_for_mvp"] = ""
        result = validate_annotations(sheet)
        assert result.annotated == 0
        assert result.total == 5
        assert result.is_valid  # blank rows are not invalid

    def test_valid_construct_trait_passes(self):
        sheet = _make_sheet_with_annotations([
            {"primary_trait": "honesty", "annotation_confidence": "high", "keep_for_mvp": "true"},
            {"primary_trait": "fairness", "annotation_confidence": "medium"},
        ])
        result = validate_annotations(sheet)
        assert result.annotated == 2
        assert result.is_valid

    def test_not_applicable_passes(self):
        sheet = _make_sheet_with_annotations([
            {"primary_trait": "not_applicable", "annotation_confidence": "high"},
        ])
        result = validate_annotations(sheet)
        assert result.is_valid
        assert result.not_applicable_count == 1

    def test_unclear_passes(self):
        sheet = _make_sheet_with_annotations([
            {"primary_trait": "unclear"},
        ])
        result = validate_annotations(sheet)
        assert result.is_valid
        assert result.unclear_count == 1

    def test_ethics_split_name_as_primary_is_invalid(self):
        sheet = _make_sheet_with_annotations([
            {"primary_trait": "justice"},    # ETHICS split, not a construct
            {"primary_trait": "harmlessness"},  # valid
        ])
        result = validate_annotations(sheet)
        assert not result.is_valid
        assert len(result.invalid_rows) == 1
        assert "justice" in result.invalid_rows.iloc[0]["errors"]

    def test_not_applicable_with_secondary_is_invalid(self):
        sheet = _make_sheet_with_annotations([
            {"primary_trait": "not_applicable", "secondary_traits": "fairness"},
        ])
        result = validate_annotations(sheet)
        assert not result.is_valid

    def test_invalid_confidence_is_flagged(self):
        sheet = _make_sheet_with_annotations([
            {"primary_trait": "honesty", "annotation_confidence": "very_high"},
        ])
        result = validate_annotations(sheet)
        assert not result.is_valid
        assert "annotation_confidence" in result.invalid_rows.iloc[0]["errors"]

    def test_primary_in_secondary_is_invalid(self):
        sheet = _make_sheet_with_annotations([
            {"primary_trait": "fairness", "secondary_traits": "fairness,honesty"},
        ])
        result = validate_annotations(sheet)
        assert not result.is_valid

    def test_kept_for_mvp_count(self):
        sheet = _make_sheet_with_annotations([
            {"primary_trait": "honesty", "keep_for_mvp": "true"},
            {"primary_trait": "fairness", "keep_for_mvp": "false"},
            {"primary_trait": "compassion", "keep_for_mvp": "true"},
        ])
        result = validate_annotations(sheet)
        assert result.kept_for_mvp == 2

    def test_primary_trait_distribution(self):
        sheet = _make_sheet_with_annotations([
            {"primary_trait": "honesty"},
            {"primary_trait": "honesty"},
            {"primary_trait": "fairness"},
            {"primary_trait": "not_applicable"},
        ])
        result = validate_annotations(sheet)
        assert result.primary_trait_counts["honesty"] == 2
        assert result.primary_trait_counts["fairness"] == 1

    def test_coverage_report_is_string(self):
        sheet = _make_item_bank(3)
        sheet["keep_for_mvp"] = ""
        result = validate_annotations(sheet)
        report = annotation_coverage_report(result)
        assert isinstance(report, str)
        assert "Total" in report


# ---------------------------------------------------------------------------
# Config-driven: utilitarianism excluded by default
# ---------------------------------------------------------------------------

class TestUtilitarianismExcludedByDefault:
    def test_utilitarianism_not_in_default_ethics_splits(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert "utilitarianism" not in cfg.dataset.ethics_splits, (
            "utilitarianism should be excluded from the MVP ethics_splits by default "
            "because many items are hedonic comparisons, not moral-disposition items."
        )

    def test_virtue_not_in_default_ethics_splits(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert "virtue" not in cfg.dataset.ethics_splits, (
            "virtue should be excluded from the MVP ethics_splits by default."
        )

    def test_default_splits_are_commonsense_deontology_justice(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert set(cfg.dataset.ethics_splits) == {"commonsense", "deontology", "justice"}


# ---------------------------------------------------------------------------
# No GPU / model imports
# ---------------------------------------------------------------------------

class TestNoDependencies:
    def test_annotation_module_imports_without_torch(self):
        import src.data.annotation as ann_module
        assert not hasattr(ann_module, "torch")
        assert not hasattr(ann_module, "transformers")

    def test_config_module_imports_without_torch(self):
        import src.config as cfg_module
        assert not hasattr(cfg_module, "torch")
