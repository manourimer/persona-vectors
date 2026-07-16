"""
Curation helpers for the ETHICS annotation pipeline.

Responsibilities:
  - sample a pilot annotation subset from the full annotation sheet
  - load and filter a validated annotation sheet to keep_for_mvp items
  - export the curated item bank (CSV + Parquet)
  - produce a structured curation report with coverage warnings

This module contains no model loading, GPU code, or network access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.annotation import (
    ANNOTATION_SHEET_COLUMNS,
    load_annotation_sheet,
    validate_annotations,
)
from src.config import AnnotationConfig, AnnotationPilotConfig

logger = logging.getLogger(__name__)

CONSTRUCT_TRAITS = frozenset({"honesty", "harmlessness", "fairness", "compassion"})
SPECIAL_LABELS = frozenset({"not_applicable", "unclear"})

# Minimum retained items per construct trait before a warning is emitted.
_LOW_COVERAGE_THRESHOLD = 10
# Maximum fraction of retained items from one source split before a warning.
_SPLIT_DOMINANCE_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Pilot sheet creation
# ---------------------------------------------------------------------------

def create_annotation_pilot_sheet(
    input_sheet: pd.DataFrame | Path | str,
    output_path: Path | str,
    n_items: int,
    seed: int,
    stratify_by_source_split: bool = True,
) -> pd.DataFrame:
    """Sample *n_items* from *input_sheet* and write to *output_path*.

    If *stratify_by_source_split* is True, items are drawn proportionally
    from each present source_split.  If a split has fewer rows than its
    share, all its rows are taken.

    Returns the pilot DataFrame (same schema as the input sheet).
    """
    if isinstance(input_sheet, (str, Path)):
        input_sheet = load_annotation_sheet(input_sheet)

    if "source_split" not in input_sheet.columns:
        raise ValueError("input_sheet must have a 'source_split' column")

    if stratify_by_source_split:
        splits = sorted(input_sheet["source_split"].unique())
        n_splits = len(splits)
        per_split = n_items // n_splits
        remainder = n_items % n_splits
        parts: list[pd.DataFrame] = []
        deficit = 0
        for i, split in enumerate(splits):
            sub = input_sheet[input_sheet["source_split"] == split]
            target = per_split + (1 if i < remainder else 0) + deficit
            if len(sub) <= target:
                parts.append(sub)
                deficit = target - len(sub)
            else:
                parts.append(sub.sample(n=target, random_state=seed))
                deficit = 0
        pilot = (
            pd.concat(parts, ignore_index=True)
            .sort_values(["source_split", "item_id"])
            .reset_index(drop=True)
        )
    else:
        take = min(n_items, len(input_sheet))
        pilot = input_sheet.sample(n=take, random_state=seed).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(output_path, index=False)
    logger.info("Pilot sheet written: %d items → %s", len(pilot), output_path)
    return pilot


# ---------------------------------------------------------------------------
# Loading, filtering, export
# ---------------------------------------------------------------------------

def load_validated_annotations(path: Path | str) -> pd.DataFrame:
    """Load an annotation CSV and return it as a DataFrame.

    Does not run validation — call validate_annotations() separately.
    """
    return load_annotation_sheet(path)


def filter_keep_for_mvp(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where keep_for_mvp is truthy and primary_trait is a
    construct trait (i.e. not not_applicable / unclear / blank).
    """
    keep_mask = df["keep_for_mvp"].str.strip().str.lower().isin({"true", "yes", "1"})
    trait_mask = df["primary_trait"].str.strip().isin(CONSTRUCT_TRAITS)
    return df[keep_mask & trait_mask].reset_index(drop=True)


def export_curated_item_bank(
    df: pd.DataFrame,
    output_csv: Path | str,
    output_parquet: Path | str,
) -> None:
    """Write the curated DataFrame to CSV and Parquet."""
    output_csv = Path(output_csv)
    output_parquet = Path(output_parquet)
    for p in (output_csv, output_parquet):
        p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    df.to_parquet(output_parquet, index=False)
    logger.info("Curated item bank: %d items saved.", len(df))


# ---------------------------------------------------------------------------
# Curation report
# ---------------------------------------------------------------------------

@dataclass
class CurationReport:
    total_rows: int
    annotated_rows: int
    keep_for_mvp_rows: int
    not_applicable_count: int
    unclear_count: int
    unannotated_count: int
    trait_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    split_x_trait: pd.DataFrame = field(default_factory=pd.DataFrame)
    low_coverage_warnings: list[str] = field(default_factory=list)
    split_dominance_warnings: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return bool(self.low_coverage_warnings or self.split_dominance_warnings)


def produce_curation_report(
    df: pd.DataFrame,
    min_per_trait: int = _LOW_COVERAGE_THRESHOLD,
    annotation_cfg: Optional[AnnotationConfig] = None,
) -> CurationReport:
    """Build a CurationReport from an annotation DataFrame.

    *df* may be the full annotation sheet or the curated (filtered) subset;
    counts are computed from whatever is passed in.
    """
    if annotation_cfg is None:
        annotation_cfg = AnnotationConfig()

    pt_col = df["primary_trait"].str.strip()
    annotated_mask = pt_col.astype(bool)
    keep_mask = df["keep_for_mvp"].str.strip().str.lower().isin({"true", "yes", "1"})
    curated = df[keep_mask & pt_col.isin(CONSTRUCT_TRAITS)]

    trait_counts: dict[str, int] = {}
    for trait in sorted(CONSTRUCT_TRAITS):
        trait_counts[trait] = int((curated["primary_trait"].str.strip() == trait).sum())

    split_counts: dict[str, int] = {}
    for split, grp in curated.groupby("source_split"):
        split_counts[str(split)] = len(grp)

    # Split × trait cross-tab
    if len(curated) > 0:
        split_x_trait = (
            curated.groupby(["source_split", "primary_trait"])
            .size()
            .unstack(fill_value=0)
        )
    else:
        split_x_trait = pd.DataFrame()

    # Warnings
    low_warnings: list[str] = []
    for trait, count in trait_counts.items():
        if count < min_per_trait:
            low_warnings.append(
                f"Trait '{trait}' has only {count} retained items "
                f"(target ≥ {min_per_trait}). Consider annotating more items "
                f"or adjusting the rubric."
            )

    dominance_warnings: list[str] = []
    total_curated = len(curated)
    if total_curated > 0:
        for split, count in split_counts.items():
            frac = count / total_curated
            if frac > _SPLIT_DOMINANCE_THRESHOLD:
                dominance_warnings.append(
                    f"Source split '{split}' contributes {count}/{total_curated} "
                    f"({frac:.0%}) of retained items. The curated bank may not "
                    f"generalise well across ETHICS splits."
                )

    return CurationReport(
        total_rows=len(df),
        annotated_rows=int(annotated_mask.sum()),
        keep_for_mvp_rows=int(keep_mask.sum()),
        not_applicable_count=int((pt_col == "not_applicable").sum()),
        unclear_count=int((pt_col == "unclear").sum()),
        unannotated_count=int((~annotated_mask).sum()),
        trait_counts=trait_counts,
        split_counts=split_counts,
        split_x_trait=split_x_trait,
        low_coverage_warnings=low_warnings,
        split_dominance_warnings=dominance_warnings,
    )


def format_curation_report(report: CurationReport) -> str:
    """Return the CurationReport as a human-readable string."""
    lines = [
        f"  Total rows           : {report.total_rows}",
        f"  Annotated            : {report.annotated_rows}",
        f"  Unannotated          : {report.unannotated_count}",
        f"  keep_for_mvp = true  : {report.keep_for_mvp_rows}",
        f"  not_applicable       : {report.not_applicable_count}",
        f"  unclear              : {report.unclear_count}",
    ]

    lines.append("\n  Retained items by primary_trait:")
    for trait, count in sorted(report.trait_counts.items()):
        lines.append(f"    {trait:<18} {count:>4}")

    lines.append("\n  Retained items by source_split:")
    for split, count in sorted(report.split_counts.items()):
        lines.append(f"    {split:<20} {count:>4}")

    if not report.split_x_trait.empty:
        lines.append("\n  Source split × primary trait:")
        lines.append("  " + report.split_x_trait.to_string().replace("\n", "\n  "))

    if report.low_coverage_warnings:
        lines.append("\n  ⚠  LOW COVERAGE WARNINGS:")
        for w in report.low_coverage_warnings:
            lines.append(f"    • {w}")

    if report.split_dominance_warnings:
        lines.append("\n  ⚠  SPLIT DOMINANCE WARNINGS:")
        for w in report.split_dominance_warnings:
            lines.append(f"    • {w}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commonsense-only comparison
# ---------------------------------------------------------------------------

def commonsense_only_summary(df: pd.DataFrame) -> str:
    """Return a summary string showing what the item bank looks like when
    restricted to source_split == 'commonsense'.

    This is a diagnostic helper, not the default behaviour.
    """
    cs = df[df["source_split"] == "commonsense"].copy()
    curated = filter_keep_for_mvp(cs)
    report = produce_curation_report(cs)

    lines = [
        f"  Commonsense-only diagnostic (NOT the default)",
        f"  Total commonsense items in sheet : {len(cs)}",
        f"  Would be retained (keep_for_mvp) : {len(curated)}",
        "",
        "  Trait distribution in commonsense-only curated set:",
    ]
    for trait in sorted(CONSTRUCT_TRAITS):
        count = int((curated["primary_trait"].str.strip() == trait).sum())
        lines.append(f"    {trait:<18} {count:>4}")

    if report.low_coverage_warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in report.low_coverage_warnings:
            lines.append(f"    • {w}")

    return "\n".join(lines)
