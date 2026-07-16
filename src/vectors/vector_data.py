"""
Typed data schemas for Stage 2B: persona-vector construction.

All schemas use stdlib dataclasses for zero-dependency import.
None of these classes import torch, transformers, modal, or vLLM.

ETHICS items are not referenced here.  These schemas are used exclusively
for the contrast-prompt → response → activation → vector pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeneratedResponse:
    """One model response produced under a contrastive system prompt."""

    response_id: str             # stable key: {prompt_id}__{question_id}
    trait: str                   # honesty | harmlessness | fairness | compassion
    pole: str                    # positive | negative
    split: str                   # extraction | validation
    system_prompt_id: str
    question_id: str
    system_prompt_text: str
    question_text: str
    response_text: str
    model_name: str
    generation_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "generation_params"}
        d["generation_params"] = str(self.generation_params)
        return d

    @staticmethod
    def make_id(prompt_id: str, question_id: str) -> str:
        return f"{prompt_id}__{question_id}"


@dataclass
class ScoredResponse:
    """A GeneratedResponse annotated with a trait judge score."""

    response_id: str
    trait: str
    pole: str
    split: str
    system_prompt_id: str
    question_id: str
    system_prompt_text: str
    question_text: str
    response_text: str
    model_name: str
    generation_params: dict[str, Any]
    trait_score: float           # 0–100 from the judge
    score_rationale: str
    keep_for_vector_extraction: bool  # derived from score + pole + thresholds

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "generation_params"}
        d["generation_params"] = str(self.generation_params)
        return d

    @classmethod
    def from_generated(
        cls,
        gen: GeneratedResponse,
        trait_score: float,
        score_rationale: str,
        positive_keep_threshold: float,
        negative_keep_threshold: float,
    ) -> "ScoredResponse":
        if trait_score < 0:
            # Negative score is a sentinel for a parse/validation failure.
            keep = False
        elif gen.pole == "positive":
            keep = trait_score >= positive_keep_threshold
        else:
            keep = trait_score <= negative_keep_threshold
        return cls(
            response_id=gen.response_id,
            trait=gen.trait,
            pole=gen.pole,
            split=gen.split,
            system_prompt_id=gen.system_prompt_id,
            question_id=gen.question_id,
            system_prompt_text=gen.system_prompt_text,
            question_text=gen.question_text,
            response_text=gen.response_text,
            model_name=gen.model_name,
            generation_params=gen.generation_params,
            trait_score=trait_score,
            score_rationale=score_rationale,
            keep_for_vector_extraction=keep,
        )


@dataclass
class ActivationRecord:
    """Metadata for a cached activation .npy file (one response × one layer)."""

    response_id: str
    trait: str
    pole: str
    split: str                   # extraction | validation
    layer: int
    activation_path: str         # relative path to .npy file
    pooling_method: str          # e.g. "mean_response_token"
    hidden_dim: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PersonaVectorMeta:
    """Metadata for a saved persona vector .npy file."""

    trait: str
    layer: int
    vector_path: str             # relative path to .npy file
    n_positive: int              # number of positive-pole activations used
    n_negative: int              # number of negative-pole activations used
    vector_method: str           # e.g. "difference_of_means"
    normalization: str           # e.g. "unit_norm" | "none"
    hidden_dim: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class VectorValidationResult:
    """Held-out validation metrics for one trait × layer persona vector."""

    trait: str
    layer: int
    auc: float
    accuracy: float
    mean_positive_projection: float
    mean_negative_projection: float
    cohens_d: float
    n_positive_val: int
    n_negative_val: int
    passes_minimum_auc: bool     # auc >= vector_validation.minimum_auc_target

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
