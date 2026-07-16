"""
Load, validate, and expose the trait vector artifact bank.

The artifact bank (configs/trait_vector_artifacts.yaml) contains:
  - Contrastive system prompts (positive / negative) per trait
  - Elicitation questions (extraction / validation split) per trait
  - Evaluation rubrics per trait

These artifacts drive persona-vector CONSTRUCTION (Stage 2B+) and are
entirely separate from the ETHICS item bank used for projection / reliability
testing (Stage 3+).

Public API:
    load_artifact_bank(path)   → ArtifactBank
    validate_artifact_bank(data) → None  (raises ValueError on problems)

    ArtifactBank.system_prompts_df  → DataFrame
    ArtifactBank.questions_df       → DataFrame
    ArtifactBank.rubrics_df         → DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

VALID_TRAITS: frozenset[str] = frozenset(
    {"honesty", "harmlessness", "fairness", "compassion"}
)
VALID_POLES: frozenset[str] = frozenset({"positive", "negative"})
VALID_SPLITS: frozenset[str] = frozenset({"extraction", "validation"})

# Expected counts per trait (must match mvp_experiment.yaml)
EXPECTED_PROMPTS_PER_POLE: int = 5
EXPECTED_QUESTIONS_PER_SPLIT: int = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(data)}")
    return data


def _check_unique_ids(ids: list[str], context: str) -> None:
    seen: set[str] = set()
    dupes: list[str] = []
    for id_ in ids:
        if id_ in seen:
            dupes.append(id_)
        seen.add(id_)
    if dupes:
        raise ValueError(f"Duplicate IDs in {context}: {dupes}")


def _check_disjoint_splits(
    trait: str, ext_ids: list[str], val_ids: list[str]
) -> None:
    overlap = set(ext_ids) & set(val_ids)
    if overlap:
        raise ValueError(
            f"[{trait}] extraction and validation question IDs overlap: {overlap}"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_artifact_bank(data: dict[str, Any]) -> None:
    """Validate the raw YAML data.  Raises ValueError describing all problems."""
    errors: list[str] = []

    missing_traits = VALID_TRAITS - set(data.keys())
    if missing_traits:
        errors.append(f"Missing traits: {sorted(missing_traits)}")

    for trait in VALID_TRAITS:
        if trait not in data:
            continue
        tdata = data[trait]

        # --- system prompts ---
        for pole in ("positive", "negative"):
            key = f"{pole}_system_prompts"
            prompts = tdata.get(key, [])
            if len(prompts) != EXPECTED_PROMPTS_PER_POLE:
                errors.append(
                    f"[{trait}] expected {EXPECTED_PROMPTS_PER_POLE} "
                    f"{pole} system prompts, got {len(prompts)}"
                )
            ids = [p.get("id", "") for p in prompts]
            try:
                _check_unique_ids(ids, f"{trait}/{key}")
            except ValueError as exc:
                errors.append(str(exc))
            for p in prompts:
                if not p.get("text", "").strip():
                    errors.append(f"[{trait}/{key}] prompt {p.get('id')} has empty text")

        # Check no near-duplicates across poles (simple exact-text check)
        pos_texts = {p.get("text", "").strip() for p in tdata.get("positive_system_prompts", [])}
        neg_texts = {p.get("text", "").strip() for p in tdata.get("negative_system_prompts", [])}
        shared = pos_texts & neg_texts
        if shared:
            errors.append(
                f"[{trait}] identical text found in both positive and negative prompts"
            )

        # --- elicitation questions ---
        questions = tdata.get("elicitation_questions", [])
        ext_qs = [q for q in questions if q.get("split") == "extraction"]
        val_qs = [q for q in questions if q.get("split") == "validation"]

        if len(ext_qs) != EXPECTED_QUESTIONS_PER_SPLIT:
            errors.append(
                f"[{trait}] expected {EXPECTED_QUESTIONS_PER_SPLIT} extraction "
                f"questions, got {len(ext_qs)}"
            )
        if len(val_qs) != EXPECTED_QUESTIONS_PER_SPLIT:
            errors.append(
                f"[{trait}] expected {EXPECTED_QUESTIONS_PER_SPLIT} validation "
                f"questions, got {len(val_qs)}"
            )

        all_q_ids = [q.get("id", "") for q in questions]
        try:
            _check_unique_ids(all_q_ids, f"{trait}/elicitation_questions")
        except ValueError as exc:
            errors.append(str(exc))

        ext_ids = [q.get("id", "") for q in ext_qs]
        val_ids = [q.get("id", "") for q in val_qs]
        try:
            _check_disjoint_splits(trait, ext_ids, val_ids)
        except ValueError as exc:
            errors.append(str(exc))

        for q in questions:
            if not q.get("text", "").strip():
                errors.append(
                    f"[{trait}/elicitation_questions] question {q.get('id')} has empty text"
                )
            if q.get("split") not in VALID_SPLITS:
                errors.append(
                    f"[{trait}/elicitation_questions] question {q.get('id')} "
                    f"has invalid split: {q.get('split')!r}"
                )

        # --- rubric ---
        rubric = tdata.get("evaluation_rubric")
        if rubric is None:
            errors.append(f"[{trait}] missing evaluation_rubric")
        else:
            if not rubric.get("scoring_instructions", "").strip():
                errors.append(f"[{trait}] evaluation_rubric.scoring_instructions is empty")
            scale = rubric.get("score_scale", {})
            if scale.get("min") is None or scale.get("max") is None:
                errors.append(f"[{trait}] evaluation_rubric.score_scale missing min/max")

    if errors:
        bullet_list = "\n  - ".join(errors)
        raise ValueError(f"Artifact bank validation failed:\n  - {bullet_list}")


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------


def _build_system_prompts_df(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trait in sorted(VALID_TRAITS):
        tdata = data.get(trait, {})
        for pole in ("positive", "negative"):
            for p in tdata.get(f"{pole}_system_prompts", []):
                rows.append(
                    {
                        "trait": trait,
                        "pole": pole,
                        "prompt_id": p.get("id", ""),
                        "prompt_text": str(p.get("text", "")).strip(),
                        "notes": str(p.get("notes", "")).strip(),
                    }
                )
    return pd.DataFrame(rows, columns=["trait", "pole", "prompt_id", "prompt_text", "notes"])


def _build_questions_df(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trait in sorted(VALID_TRAITS):
        tdata = data.get(trait, {})
        for q in tdata.get("elicitation_questions", []):
            rows.append(
                {
                    "trait": trait,
                    "split": q.get("split", ""),
                    "question_id": q.get("id", ""),
                    "question_text": str(q.get("text", "")).strip(),
                    "notes": str(q.get("notes", "")).strip(),
                }
            )
    return pd.DataFrame(
        rows, columns=["trait", "split", "question_id", "question_text", "notes"]
    )


def _build_rubrics_df(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trait in sorted(VALID_TRAITS):
        tdata = data.get(trait, {})
        rubric = tdata.get("evaluation_rubric", {})
        scale = rubric.get("score_scale", {})
        rows.append(
            {
                "trait": trait,
                "scoring_instructions": str(
                    rubric.get("scoring_instructions", "")
                ).strip(),
                "min_score": scale.get("min"),
                "max_score": scale.get("max"),
            }
        )
    return pd.DataFrame(
        rows, columns=["trait", "scoring_instructions", "min_score", "max_score"]
    )


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class ArtifactBank:
    """Loaded and validated artifact bank.

    Attributes:
        system_prompts_df: Columns — trait, pole, prompt_id, prompt_text, notes
        questions_df:      Columns — trait, split, question_id, question_text, notes
        rubrics_df:        Columns — trait, scoring_instructions, min_score, max_score
        raw:               The raw YAML dict (for ad-hoc access).
    """

    system_prompts_df: pd.DataFrame
    questions_df: pd.DataFrame
    rubrics_df: pd.DataFrame
    raw: dict[str, Any] = field(repr=False)


def load_artifact_bank_flexible(path: str | Path) -> ArtifactBank:
    """Load an artifact bank YAML whose top-level keys are any trait names.

    Unlike `load_artifact_bank`, this function does NOT require the keys to be
    the four primary traits.  It infers the trait list from the YAML keys and
    validates each trait's internal structure (prompts, questions, rubric).
    Use this for synonym / construct-neighbor artifact configs.

    Args:
        path: Path to any artifact YAML following the same schema as
              trait_vector_artifacts.yaml but with different top-level keys.

    Returns:
        ArtifactBank with DataFrames keyed by the YAML's own trait names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact bank not found: {path}")

    data = _load_yaml(path)
    traits = sorted(k for k in data if not k.startswith("_"))

    errors: list[str] = []
    for trait in traits:
        tdata = data[trait]
        for pole in ("positive", "negative"):
            key = f"{pole}_system_prompts"
            prompts = tdata.get(key, [])
            if len(prompts) != EXPECTED_PROMPTS_PER_POLE:
                errors.append(
                    f"[{trait}] expected {EXPECTED_PROMPTS_PER_POLE} "
                    f"{pole} system prompts, got {len(prompts)}"
                )
        questions = tdata.get("elicitation_questions", [])
        ext_qs = [q for q in questions if q.get("split") == "extraction"]
        val_qs = [q for q in questions if q.get("split") == "validation"]
        if len(ext_qs) != EXPECTED_QUESTIONS_PER_SPLIT:
            errors.append(f"[{trait}] expected {EXPECTED_QUESTIONS_PER_SPLIT} extraction questions, got {len(ext_qs)}")
        if len(val_qs) != EXPECTED_QUESTIONS_PER_SPLIT:
            errors.append(f"[{trait}] expected {EXPECTED_QUESTIONS_PER_SPLIT} validation questions, got {len(val_qs)}")
        rubric = tdata.get("evaluation_rubric", {})
        has_instructions = bool(rubric.get("scoring_instructions", "").strip())
        has_alt = bool(rubric.get("high_score_description", "").strip())
        if not has_instructions and not has_alt:
            errors.append(f"[{trait}] missing evaluation_rubric (need scoring_instructions or high_score_description)")

    if errors:
        raise ValueError("Artifact bank validation errors:\n" + "\n".join(f"  • {e}" for e in errors))

    def _build_df(key: str, builder):
        rows = []
        for trait in traits:
            rows.extend(builder(trait, data[trait]))
        return pd.DataFrame(rows)

    sp_rows, q_rows, rb_rows = [], [], []
    for trait in traits:
        tdata = data[trait]
        for pole in ("positive", "negative"):
            for p in tdata.get(f"{pole}_system_prompts", []):
                sp_rows.append({"trait": trait, "pole": pole, "prompt_id": p.get("id", ""),
                                "prompt_text": str(p.get("text", "")).strip(), "notes": str(p.get("notes", "")).strip()})
        for q in tdata.get("elicitation_questions", []):
            q_rows.append({"trait": trait, "split": q.get("split", ""), "question_id": q.get("id", ""),
                           "question_text": str(q.get("text", "")).strip(), "notes": str(q.get("notes", "")).strip()})
        rubric = tdata.get("evaluation_rubric", {})
        # Support both schema variants:
        #   Standard: scoring_instructions + score_scale.{min,max}
        #   Synonym:  high_score_description + low_score_description + scale (str)
        scoring_instr = rubric.get("scoring_instructions", "").strip()
        if not scoring_instr:
            high = rubric.get("high_score_description", "").strip()
            low  = rubric.get("low_score_description", "").strip()
            notes = "\n".join(rubric.get("scoring_notes", []))
            scoring_instr = f"High score: {high}\nLow score: {low}\nNotes: {notes}".strip()
        scale_obj = rubric.get("score_scale", {})
        if not scale_obj:
            scale_str = str(rubric.get("scale", "0-100"))
            parts = scale_str.split("-")
            scale_obj = {"min": int(parts[0]), "max": int(parts[-1])} if len(parts) == 2 else {"min": 0, "max": 100}
        rb_rows.append({"trait": trait, "scoring_instructions": scoring_instr,
                        "min_score": scale_obj.get("min"), "max_score": scale_obj.get("max")})

    return ArtifactBank(
        system_prompts_df=pd.DataFrame(sp_rows, columns=["trait", "pole", "prompt_id", "prompt_text", "notes"]),
        questions_df=pd.DataFrame(q_rows, columns=["trait", "split", "question_id", "question_text", "notes"]),
        rubrics_df=pd.DataFrame(rb_rows, columns=["trait", "scoring_instructions", "min_score", "max_score"]),
        raw=data,
    )


def load_artifact_bank(path: str | Path) -> ArtifactBank:
    """Load and validate the artifact bank YAML.

    Args:
        path: Path to trait_vector_artifacts.yaml.

    Returns:
        ArtifactBank with three DataFrames ready for downstream use.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact bank not found: {path}")

    data = _load_yaml(path)
    validate_artifact_bank(data)

    return ArtifactBank(
        system_prompts_df=_build_system_prompts_df(data),
        questions_df=_build_questions_df(data),
        rubrics_df=_build_rubrics_df(data),
        raw=data,
    )
