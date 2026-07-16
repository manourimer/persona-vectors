"""Central config-loading utilities for the persona-vectors project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class ExperimentMeta(BaseModel):
    name: str
    description: str = ""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    name: str
    dtype: str = "bfloat16"
    target_layer: int
    layer_indexing: Literal["zero_indexed"] = "zero_indexed"
    candidate_layers_for_validation: list[int] = Field(default_factory=list)
    layer_selection_note: str = ""

    @field_validator("target_layer")
    @classmethod
    def layer_must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("target_layer must be >= 0")
        return v

    @field_validator("candidate_layers_for_validation")
    @classmethod
    def candidates_must_be_non_negative(cls, v: list[int]) -> list[int]:
        if any(layer < 0 for layer in v):
            raise ValueError("All candidate_layers_for_validation must be >= 0")
        return v


# ---------------------------------------------------------------------------
# Construct traits (reference block in experiment config)
# ---------------------------------------------------------------------------

class TraitsRef(BaseModel):
    path: str
    names: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset  — ETHICS source splits are dataset labels, NOT construct traits
# ---------------------------------------------------------------------------

class DatasetConfig(BaseModel):
    name: str
    n_items: int = 300
    split: str = "test"
    ethics_splits: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Annotation pilot config
# ---------------------------------------------------------------------------

class AnnotationPilotConfig(BaseModel):
    pilot_n_items: int = 50
    pilot_stratify_by_source_split: bool = True
    pilot_random_seed: int | None = None  # None → inherit project random_seed
    minimum_items_per_trait_target: int = 25
    allow_commonsense_only_mvp: bool = True

    def effective_seed(self, project_seed: int) -> int:
        return self.pilot_random_seed if self.pilot_random_seed is not None else project_seed


# ---------------------------------------------------------------------------
# Annotation config
# ---------------------------------------------------------------------------

class AnnotationConfig(BaseModel):
    valid_primary_traits: list[str] = Field(
        default_factory=lambda: [
            "honesty", "harmlessness", "fairness", "compassion",
            "not_applicable", "unclear",
        ]
    )
    valid_secondary_traits: list[str] = Field(
        default_factory=lambda: [
            "honesty", "harmlessness", "fairness", "compassion",
        ]
    )
    valid_confidence_levels: list[str] = Field(
        default_factory=lambda: ["high", "medium", "low"]
    )


# ---------------------------------------------------------------------------
# Item annotation model
# ---------------------------------------------------------------------------

# The four construct traits (the project's measurement constructs)
ConstructTrait = Literal["honesty", "harmlessness", "fairness", "compassion"]

# Valid values for primary_trait — construct traits plus two special labels
PrimaryTraitLabel = Literal[
    "honesty", "harmlessness", "fairness", "compassion",
    "not_applicable", "unclear",
]

AnnotationConfidence = Literal["high", "medium", "low"]

# ETHICS source-split names — used only to guard against accidental misuse
_ETHICS_SPLIT_NAMES: frozenset[str] = frozenset(
    {"commonsense", "deontology", "justice", "utilitarianism", "virtue"}
)


class ItemAnnotation(BaseModel):
    """Schema for per-item moral trait annotations.

    primary_trait must be one of the four construct traits or a special label:
      - not_applicable : item does not engage any of the four target traits
      - unclear        : annotator cannot confidently assign a primary trait

    secondary_traits is restricted to the four construct traits only
    (not_applicable / unclear are not valid secondary labels).

    Validation rules:
      - ETHICS split names are never accepted as trait labels.
      - If primary_trait is not_applicable, secondary_traits must be empty.
      - secondary_traits may not contain the same value as primary_trait.
    """

    source_dataset: str
    source_split: str
    primary_trait: PrimaryTraitLabel
    secondary_traits: list[ConstructTrait] = Field(default_factory=list)
    annotation_confidence: AnnotationConfidence = "medium"
    annotation_notes: str = ""
    keep_for_mvp: bool = False

    @field_validator("primary_trait", mode="before")
    @classmethod
    def primary_trait_not_an_ethics_split(cls, v: str) -> str:
        if v in _ETHICS_SPLIT_NAMES:
            raise ValueError(
                f"'{v}' is an ETHICS source-split name, not a construct trait. "
                f"Valid primary_trait values: honesty, harmlessness, fairness, "
                f"compassion, not_applicable, unclear."
            )
        return v

    @field_validator("source_split", mode="before")
    @classmethod
    def source_split_not_a_construct_trait(cls, v: str) -> str:
        construct_traits = {"honesty", "harmlessness", "fairness", "compassion"}
        if v in construct_traits:
            raise ValueError(
                f"source_split '{v}' looks like a construct trait name. "
                "source_split must hold an ETHICS dataset label, not a construct trait."
            )
        return v

    @model_validator(mode="after")
    def validate_secondary_traits_consistency(self) -> "ItemAnnotation":
        if self.primary_trait == "not_applicable" and self.secondary_traits:
            raise ValueError(
                "secondary_traits must be empty when primary_trait is 'not_applicable'. "
                "An item that does not engage any target trait cannot have secondary traits."
            )
        if self.primary_trait in {"honesty", "harmlessness", "fairness", "compassion"}:
            if self.primary_trait in self.secondary_traits:
                raise ValueError(
                    f"primary_trait '{self.primary_trait}' must not also appear in "
                    "secondary_traits."
                )
        return self


# ---------------------------------------------------------------------------
# Paraphrase / framing
# ---------------------------------------------------------------------------

class ParaphraseConfig(BaseModel):
    n_per_item: int = 3
    framing_conditions: list[str] = Field(default_factory=lambda: ["neutral"])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class OutputConfig(BaseModel):
    base_dir: str = "outputs"
    activations_dir: str = "outputs/activations"
    projections_dir: str = "outputs/projections"
    results_dir: str = "outputs/results"
    figures_dir: str = "outputs/figures"


# ---------------------------------------------------------------------------
# Top-level MVP config
# ---------------------------------------------------------------------------

class MVPConfig(BaseModel):
    experiment: ExperimentMeta
    model: ModelConfig
    traits: TraitsRef
    dataset: DatasetConfig
    annotation_pilot: AnnotationPilotConfig = Field(default_factory=AnnotationPilotConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    paraphrase: ParaphraseConfig
    random_seed: int = 42
    output: OutputConfig


# ---------------------------------------------------------------------------
# Trait definitions (loaded from traits.yaml)
# ---------------------------------------------------------------------------

class TraitDefinition(BaseModel):
    positive_description: str
    negative_description: str
    moral_rationale: str
    item_tagging_guidance: str = ""
    example_keywords_and_patterns: list[str] = Field(default_factory=list)


class TraitSpace(BaseModel):
    traits: dict[str, TraitDefinition]

    @field_validator("traits")
    @classmethod
    def at_least_one_trait(cls, v: dict) -> dict:
        if not v:
            raise ValueError("traits dict must not be empty")
        return v

    @property
    def names(self) -> list[str]:
        return list(self.traits.keys())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def load_mvp_config(path: Path | str = "configs/mvp_experiment.yaml") -> MVPConfig:
    """Load and validate the MVP experiment config."""
    raw = _load_yaml(path)
    raw.pop("item_annotation_schema", None)
    return MVPConfig(**raw)


def load_trait_space(path: Path | str = "configs/traits.yaml") -> TraitSpace:
    """Load and validate trait definitions."""
    raw = _load_yaml(path)
    return TraitSpace(**raw)


def load_config_and_traits(
    config_path: Path | str = "configs/mvp_experiment.yaml",
) -> tuple[MVPConfig, TraitSpace]:
    """Convenience loader: returns (MVPConfig, TraitSpace)."""
    cfg = load_mvp_config(config_path)
    traits = load_trait_space(cfg.traits.path)
    return cfg, traits
