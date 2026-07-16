"""
Tests for src/data/curation.py — pilot sampling, filtering, export,
curation reports, and the commonsense-only diagnostic.

No network access, no GPU, no model loading.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_mvp_config, AnnotationPilotConfig
from src.data.curation import (
    CONSTRUCT_TRAITS,
    CurationReport,
    commonsense_only_summary,
    create_annotation_pilot_sheet,
    export_curated_item_bank,
    filter_keep_for_mvp,
    format_curation_report,
    produce_curation_report,
)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "mvp_experiment.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPLITS = ["commonsense", "deontology", "justice"]


def _make_annotation_sheet(
    n_per_split: int = 20,
    splits: list[str] = SPLITS,
) -> pd.DataFrame:
    """Minimal annotation sheet, all items blank (unannotated)."""
    rows = []
    for split in splits:
        for i in range(n_per_split):
            rows.append(
                {
                    "item_id": f"{split}_test_{i:06d}",
                    "source_dataset": "ETHICS",
                    "source_split": split,
                    "scenario_text": f"Scenario {split} {i}",
                    "label": i % 2,
                    "label_semantics": "1=wrong; 0=ok",
                    "primary_trait": "",
                    "secondary_traits": "",
                    "annotation_confidence": "",
                    "annotation_notes": "",
                    "keep_for_mvp": "",
                }
            )
    return pd.DataFrame(rows)


def _annotated_sheet(rows_extra: list[dict] | None = None) -> pd.DataFrame:
    """Annotation sheet with some rows filled in."""
    base = _make_annotation_sheet(n_per_split=10)
    # Annotate a subset
    annotations = [
        {"primary_trait": "honesty", "annotation_confidence": "high", "keep_for_mvp": "true"},
        {"primary_trait": "harmlessness", "annotation_confidence": "high", "keep_for_mvp": "true"},
        {"primary_trait": "fairness", "annotation_confidence": "medium", "keep_for_mvp": "true"},
        {"primary_trait": "compassion", "annotation_confidence": "medium", "keep_for_mvp": "true"},
        {"primary_trait": "not_applicable", "annotation_confidence": "high", "keep_for_mvp": "false"},
        {"primary_trait": "unclear", "annotation_confidence": "low", "keep_for_mvp": "false"},
        {"primary_trait": "honesty", "annotation_confidence": "high", "keep_for_mvp": "true"},
        {"primary_trait": "harmlessness", "annotation_confidence": "medium", "keep_for_mvp": "true"},
    ]
    for i, ann in enumerate(annotations):
        for col, val in ann.items():
            base.at[i, col] = val
    if rows_extra:
        extra_df = pd.DataFrame(rows_extra)
        base = pd.concat([base, extra_df], ignore_index=True)
    return base


# ---------------------------------------------------------------------------
# Pilot sampling
# ---------------------------------------------------------------------------

class TestPilotSampling:
    def test_returns_correct_n(self, tmp_path):
        sheet = _make_annotation_sheet(n_per_split=20)
        pilot = create_annotation_pilot_sheet(
            sheet, tmp_path / "pilot.csv", n_items=15, seed=42
        )
        assert len(pilot) == 15

    def test_reproducible_with_same_seed(self, tmp_path):
        sheet = _make_annotation_sheet(n_per_split=20)
        p1 = create_annotation_pilot_sheet(sheet, tmp_path / "p1.csv", n_items=15, seed=7)
        p2 = create_annotation_pilot_sheet(sheet, tmp_path / "p2.csv", n_items=15, seed=7)
        assert p1["item_id"].tolist() == p2["item_id"].tolist()

    def test_different_seeds_give_different_samples(self, tmp_path):
        sheet = _make_annotation_sheet(n_per_split=20)
        p1 = create_annotation_pilot_sheet(sheet, tmp_path / "p1.csv", n_items=15, seed=1)
        p2 = create_annotation_pilot_sheet(sheet, tmp_path / "p2.csv", n_items=15, seed=2)
        assert p1["item_id"].tolist() != p2["item_id"].tolist()

    def test_stratified_samples_from_all_splits(self, tmp_path):
        sheet = _make_annotation_sheet(n_per_split=20)
        pilot = create_annotation_pilot_sheet(
            sheet, tmp_path / "pilot.csv", n_items=15,
            seed=42, stratify_by_source_split=True,
        )
        represented = pilot["source_split"].nunique()
        assert represented == len(SPLITS), (
            f"Expected {len(SPLITS)} splits in pilot, got {represented}"
        )

    def test_non_stratified_does_not_enforce_split_balance(self, tmp_path):
        sheet = _make_annotation_sheet(n_per_split=20)
        pilot = create_annotation_pilot_sheet(
            sheet, tmp_path / "pilot.csv", n_items=9,
            seed=42, stratify_by_source_split=False,
        )
        assert len(pilot) == 9

    def test_handles_small_split(self, tmp_path):
        """If one split has fewer rows than its share, take all and don't crash."""
        sheet = _make_annotation_sheet(n_per_split=3, splits=["commonsense", "deontology"])
        pilot = create_annotation_pilot_sheet(
            sheet, tmp_path / "pilot.csv", n_items=10, seed=0
        )
        assert len(pilot) <= 10  # may be < 10 if total rows < 10

    def test_writes_csv(self, tmp_path):
        sheet = _make_annotation_sheet()
        out = tmp_path / "pilot.csv"
        create_annotation_pilot_sheet(sheet, out, n_items=10, seed=0)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == 10

    def test_accepts_path_as_input(self, tmp_path):
        sheet = _make_annotation_sheet()
        csv_path = tmp_path / "sheet.csv"
        sheet.to_csv(csv_path, index=False)
        pilot = create_annotation_pilot_sheet(
            csv_path, tmp_path / "pilot.csv", n_items=5, seed=0
        )
        assert len(pilot) == 5

    def test_pilot_schema_matches_input(self, tmp_path):
        sheet = _make_annotation_sheet()
        pilot = create_annotation_pilot_sheet(
            sheet, tmp_path / "pilot.csv", n_items=10, seed=0
        )
        assert list(pilot.columns) == list(sheet.columns)


# ---------------------------------------------------------------------------
# filter_keep_for_mvp
# ---------------------------------------------------------------------------

class TestFilterKeepForMvp:
    def test_keeps_construct_trait_rows_with_true(self):
        df = _annotated_sheet()
        curated = filter_keep_for_mvp(df)
        assert len(curated) > 0
        assert curated["primary_trait"].isin(CONSTRUCT_TRAITS).all()

    def test_removes_not_applicable(self):
        df = _annotated_sheet()
        curated = filter_keep_for_mvp(df)
        assert "not_applicable" not in curated["primary_trait"].values

    def test_removes_unclear(self):
        df = _annotated_sheet()
        curated = filter_keep_for_mvp(df)
        assert "unclear" not in curated["primary_trait"].values

    def test_removes_keep_for_mvp_false(self):
        sheet = _make_annotation_sheet(n_per_split=5, splits=["commonsense"])
        sheet.at[0, "primary_trait"] = "honesty"
        sheet.at[0, "keep_for_mvp"] = "false"
        sheet.at[1, "primary_trait"] = "fairness"
        sheet.at[1, "keep_for_mvp"] = "true"
        curated = filter_keep_for_mvp(sheet)
        assert len(curated) == 1
        assert curated.iloc[0]["primary_trait"] == "fairness"

    def test_removes_unannotated(self):
        sheet = _make_annotation_sheet()  # all blank
        curated = filter_keep_for_mvp(sheet)
        assert len(curated) == 0

    def test_preserves_item_id_and_scenario_text(self):
        df = _annotated_sheet()
        curated = filter_keep_for_mvp(df)
        assert "item_id" in curated.columns
        assert "scenario_text" in curated.columns
        assert curated["item_id"].notna().all()


# ---------------------------------------------------------------------------
# export_curated_item_bank
# ---------------------------------------------------------------------------

class TestExportCuratedItemBank:
    def test_writes_csv_and_parquet(self, tmp_path):
        df = filter_keep_for_mvp(_annotated_sheet())
        csv_out = tmp_path / "curated.csv"
        parquet_out = tmp_path / "curated.parquet"
        export_curated_item_bank(df, csv_out, parquet_out)
        assert csv_out.exists()
        assert parquet_out.exists()

    def test_roundtrip_csv_preserves_data(self, tmp_path):
        df = filter_keep_for_mvp(_annotated_sheet())
        csv_out = tmp_path / "curated.csv"
        parquet_out = tmp_path / "curated.parquet"
        export_curated_item_bank(df, csv_out, parquet_out)
        loaded = pd.read_csv(csv_out)
        assert set(loaded["item_id"]) == set(df["item_id"])

    def test_roundtrip_parquet_preserves_data(self, tmp_path):
        df = filter_keep_for_mvp(_annotated_sheet())
        csv_out = tmp_path / "c.csv"
        parquet_out = tmp_path / "c.parquet"
        export_curated_item_bank(df, csv_out, parquet_out)
        loaded = pd.read_parquet(parquet_out)
        assert set(loaded["item_id"]) == set(df["item_id"])


# ---------------------------------------------------------------------------
# produce_curation_report
# ---------------------------------------------------------------------------

class TestCurationReport:
    def test_returns_curation_report(self):
        df = _annotated_sheet()
        report = produce_curation_report(df)
        assert isinstance(report, CurationReport)

    def test_total_rows_correct(self):
        df = _annotated_sheet()
        report = produce_curation_report(df)
        assert report.total_rows == len(df)

    def test_trait_counts_only_construct_traits(self):
        df = _annotated_sheet()
        report = produce_curation_report(df)
        assert set(report.trait_counts.keys()) == CONSTRUCT_TRAITS

    def test_not_applicable_counted_separately(self):
        df = _annotated_sheet()
        report = produce_curation_report(df)
        assert report.not_applicable_count >= 1

    def test_unclear_counted_separately(self):
        df = _annotated_sheet()
        report = produce_curation_report(df)
        assert report.unclear_count >= 1

    def test_low_coverage_warning_triggered(self):
        """Set a very high threshold so a warning is always triggered."""
        df = _annotated_sheet()
        report = produce_curation_report(df, min_per_trait=9999)
        assert len(report.low_coverage_warnings) > 0

    def test_no_warning_when_coverage_adequate(self):
        """With threshold=0 no warnings should fire."""
        df = _annotated_sheet()
        report = produce_curation_report(df, min_per_trait=0)
        assert len(report.low_coverage_warnings) == 0

    def test_split_dominance_warning_triggered(self):
        """If all retained items are from one split, trigger warning."""
        sheet = _make_annotation_sheet(n_per_split=10, splits=["commonsense", "deontology"])
        # Annotate only commonsense rows as keep_for_mvp
        for i in range(10):
            sheet.at[i, "primary_trait"] = "honesty"
            sheet.at[i, "keep_for_mvp"] = "true"
        report = produce_curation_report(sheet)
        assert len(report.split_dominance_warnings) > 0

    def test_format_report_is_string(self):
        df = _annotated_sheet()
        report = produce_curation_report(df)
        text = format_curation_report(report)
        assert isinstance(text, str)
        assert "Total rows" in text

    def test_has_warnings_property(self):
        df = _annotated_sheet()
        # High threshold triggers low-coverage warnings → has_warnings is True
        report_warn = produce_curation_report(df, min_per_trait=9999)
        assert report_warn.has_warnings is True
        # Threshold=0 clears low-coverage warnings; split-dominance may still
        # fire if the fixture concentrates items in one split, so we only check
        # the low_coverage bucket is empty.
        report_low = produce_curation_report(df, min_per_trait=0)
        assert report_low.low_coverage_warnings == []


# ---------------------------------------------------------------------------
# Commonsense-only summary
# ---------------------------------------------------------------------------

class TestCommonsenseOnlySummary:
    def test_returns_string(self):
        df = _annotated_sheet()
        result = commonsense_only_summary(df)
        assert isinstance(result, str)

    def test_mentions_commonsense(self):
        df = _annotated_sheet()
        result = commonsense_only_summary(df)
        assert "commonsense" in result.lower()

    def test_works_when_no_commonsense_rows(self):
        df = _make_annotation_sheet(splits=["deontology", "justice"])
        result = commonsense_only_summary(df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# validate_annotations.py accepts --input
# ---------------------------------------------------------------------------

class TestValidateAnnotationsScript:
    def test_accepts_input_flag(self, tmp_path):
        """validate_annotations.py --input <path> should run without error."""
        sheet = _make_annotation_sheet(n_per_split=5)
        sheet_path = tmp_path / "test_sheet.csv"
        sheet.to_csv(sheet_path, index=False)

        result = subprocess.run(
            [sys.executable, "scripts/validate_annotations.py",
             "--input", str(sheet_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Validation" in result.stdout or "STATUS" in result.stdout

    def test_accepts_sheet_flag_for_backward_compat(self, tmp_path):
        sheet = _make_annotation_sheet(n_per_split=5)
        sheet_path = tmp_path / "test_sheet.csv"
        sheet.to_csv(sheet_path, index=False)

        result = subprocess.run(
            [sys.executable, "scripts/validate_annotations.py",
             "--sheet", str(sheet_path)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_exits_nonzero_on_missing_file(self):
        result = subprocess.run(
            [sys.executable, "scripts/validate_annotations.py",
             "--input", "/tmp/definitely_does_not_exist_xyz.csv"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Config — pilot settings
# ---------------------------------------------------------------------------

class TestPilotConfig:
    def test_pilot_config_loads_from_yaml(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert cfg.annotation_pilot.pilot_n_items == 50
        assert cfg.annotation_pilot.pilot_stratify_by_source_split is True

    def test_effective_seed_inherits_project_seed(self):
        cfg = load_mvp_config(CONFIG_PATH)
        pilot = cfg.annotation_pilot
        assert pilot.pilot_random_seed is None
        assert pilot.effective_seed(cfg.random_seed) == cfg.random_seed

    def test_effective_seed_override(self):
        pilot = AnnotationPilotConfig(pilot_random_seed=99)
        assert pilot.effective_seed(project_seed=42) == 99

    def test_allow_commonsense_only_mvp_present(self):
        cfg = load_mvp_config(CONFIG_PATH)
        assert cfg.annotation_pilot.allow_commonsense_only_mvp is True


# ---------------------------------------------------------------------------
# No GPU / model imports
# ---------------------------------------------------------------------------

class TestNoDependencies:
    def test_curation_module_no_torch(self):
        import src.data.curation as m
        assert not hasattr(m, "torch")
        assert not hasattr(m, "transformers")
