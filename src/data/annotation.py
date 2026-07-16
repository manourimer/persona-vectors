"""
Annotation helpers for the ETHICS item bank.

Responsibilities:
  - create a blank annotation sheet from a normalised item-bank DataFrame
  - load an annotation CSV back into a typed structure
  - validate annotation label values
  - report coverage and distributions

This module does NOT perform annotation itself; annotation is manual.
It enforces the conceptual separation between ETHICS source-split labels
and the project's measurement constructs.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    AnnotationConfig,
    ItemAnnotation,
    PrimaryTraitLabel,
    _ETHICS_SPLIT_NAMES,
)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Columns carried through from the item bank
_PASSTHROUGH_COLS = [
    "item_id",
    "source_dataset",
    "source_split",
    "scenario_text",
    "label",
    "label_semantics",
]

# Annotation columns — blank in the sheet, filled by the annotator
ANNOTATION_COLS = [
    "primary_trait",
    "secondary_traits",
    "annotation_confidence",
    "annotation_notes",
    "keep_for_mvp",
]

ANNOTATION_SHEET_COLUMNS = _PASSTHROUGH_COLS + ANNOTATION_COLS


# ---------------------------------------------------------------------------
# Sheet creation
# ---------------------------------------------------------------------------

def create_annotation_sheet(
    item_bank: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Build a blank annotation sheet from the normalised item-bank DataFrame
    and write it to *output_path* as CSV.

    Annotation columns are initialised to empty strings / blank so the
    annotator can fill them in a spreadsheet editor.

    Returns the annotation sheet DataFrame.
    """
    missing = [c for c in _PASSTHROUGH_COLS if c not in item_bank.columns]
    if missing:
        raise ValueError(
            f"item_bank is missing required columns: {missing}. "
            "Run build_ethics_sample.py first."
        )

    sheet = item_bank[_PASSTHROUGH_COLS].copy()
    for col in ANNOTATION_COLS:
        sheet[col] = ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(output_path, index=False)
    return sheet


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------

def load_annotation_sheet(path: Path | str) -> pd.DataFrame:
    """Load an annotation CSV.  Does not validate labels — call
    validate_annotations() separately if you want a full report.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Annotation sheet not found: {path}\n"
            "Run scripts/create_annotation_sheet.py first."
        )
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in ANNOTATION_SHEET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Annotation sheet is missing columns: {missing}")
    return df


class AnnotationValidationResult:
    """Container for annotation validation output."""

    def __init__(
        self,
        total: int,
        annotated: int,
        kept_for_mvp: int,
        invalid_rows: pd.DataFrame,
        primary_trait_counts: pd.Series,
        split_x_trait: pd.DataFrame,
        not_applicable_count: int,
        unclear_count: int,
    ) -> None:
        self.total = total
        self.annotated = annotated
        self.kept_for_mvp = kept_for_mvp
        self.invalid_rows = invalid_rows
        self.primary_trait_counts = primary_trait_counts
        self.split_x_trait = split_x_trait
        self.not_applicable_count = not_applicable_count
        self.unclear_count = unclear_count

    @property
    def is_valid(self) -> bool:
        return len(self.invalid_rows) == 0

    @property
    def unannotated(self) -> int:
        return self.total - self.annotated


def validate_annotations(
    df: pd.DataFrame,
    annotation_cfg: Optional[AnnotationConfig] = None,
) -> AnnotationValidationResult:
    """Validate annotation labels and return a structured result.

    Checks:
      - primary_trait is one of the six valid labels (or blank/unannotated)
      - ETHICS split names are never used as primary_trait values
      - secondary_traits values are valid construct traits
      - annotation_confidence is valid when present
      - not_applicable rows have no secondary traits
    """
    if annotation_cfg is None:
        annotation_cfg = AnnotationConfig()

    valid_primary = set(annotation_cfg.valid_primary_traits)
    valid_secondary = set(annotation_cfg.valid_secondary_traits)
    valid_confidence = set(annotation_cfg.valid_confidence_levels)

    invalid_records: list[dict] = []

    for _, row in df.iterrows():
        pt = str(row.get("primary_trait", "")).strip()
        if not pt:
            continue  # unannotated — not invalid

        errors: list[str] = []

        # Primary trait must be a valid label, never an ETHICS split name
        if pt in _ETHICS_SPLIT_NAMES:
            errors.append(
                f"primary_trait '{pt}' is an ETHICS source-split name, "
                "not a construct trait."
            )
        elif pt not in valid_primary:
            errors.append(
                f"primary_trait '{pt}' is not a valid label. "
                f"Valid: {sorted(valid_primary)}"
            )

        # Secondary traits
        raw_secondary = str(row.get("secondary_traits", "")).strip()
        secondary_list = (
            [s.strip() for s in raw_secondary.split(",") if s.strip()]
            if raw_secondary else []
        )
        for st in secondary_list:
            if st in _ETHICS_SPLIT_NAMES:
                errors.append(
                    f"secondary_traits contains '{st}' which is an ETHICS "
                    "source-split name, not a construct trait."
                )
            elif st not in valid_secondary:
                errors.append(
                    f"secondary_traits value '{st}' is not a valid construct trait. "
                    f"Valid: {sorted(valid_secondary)}"
                )

        if pt == "not_applicable" and secondary_list:
            errors.append(
                "secondary_traits must be empty when primary_trait is 'not_applicable'."
            )

        if pt in valid_secondary and pt in secondary_list:
            errors.append(
                f"primary_trait '{pt}' must not also appear in secondary_traits."
            )

        # Confidence
        conf = str(row.get("annotation_confidence", "")).strip()
        if conf and conf not in valid_confidence:
            errors.append(
                f"annotation_confidence '{conf}' is not valid. "
                f"Valid: {sorted(valid_confidence)}"
            )

        if errors:
            invalid_records.append(
                {"item_id": row["item_id"], "errors": "; ".join(errors)}
            )

    invalid_df = pd.DataFrame(
        invalid_records, columns=["item_id", "errors"]
    ) if invalid_records else pd.DataFrame(columns=["item_id", "errors"])

    # Annotated = primary_trait is non-empty
    annotated_mask = df["primary_trait"].str.strip().astype(bool)
    annotated_df = df[annotated_mask]

    # keep_for_mvp
    kept_mask = df["keep_for_mvp"].str.strip().str.lower().isin({"true", "yes", "1"})

    # Distributions
    pt_counts = (
        annotated_df["primary_trait"]
        .value_counts()
        .rename("count")
        if len(annotated_df) > 0
        else pd.Series(dtype=int, name="count")
    )

    split_x_trait = (
        annotated_df.groupby(["source_split", "primary_trait"])
        .size()
        .unstack(fill_value=0)
        if len(annotated_df) > 0
        else pd.DataFrame()
    )

    not_applicable_count = int((annotated_df["primary_trait"] == "not_applicable").sum())
    unclear_count = int((annotated_df["primary_trait"] == "unclear").sum())

    return AnnotationValidationResult(
        total=len(df),
        annotated=int(annotated_mask.sum()),
        kept_for_mvp=int(kept_mask.sum()),
        invalid_rows=invalid_df,
        primary_trait_counts=pt_counts,
        split_x_trait=split_x_trait,
        not_applicable_count=not_applicable_count,
        unclear_count=unclear_count,
    )


# ---------------------------------------------------------------------------
# Coverage summary (printable)
# ---------------------------------------------------------------------------

def annotation_coverage_report(result: AnnotationValidationResult) -> str:
    """Return a human-readable coverage report string."""
    lines = [
        f"  Total items          : {result.total}",
        f"  Annotated            : {result.annotated}",
        f"  Unannotated          : {result.unannotated}",
        f"  Kept for MVP         : {result.kept_for_mvp}",
        f"  Invalid labels       : {len(result.invalid_rows)}",
        f"  not_applicable       : {result.not_applicable_count}",
        f"  unclear              : {result.unclear_count}",
    ]

    if len(result.primary_trait_counts) > 0:
        lines.append("\n  Primary trait distribution:")
        for trait, count in result.primary_trait_counts.items():
            lines.append(f"    {trait:<18} {count:>4}")

    if not result.split_x_trait.empty:
        lines.append("\n  Source split × primary trait:")
        lines.append("  " + result.split_x_trait.to_string().replace("\n", "\n  "))

    return "\n".join(lines)
