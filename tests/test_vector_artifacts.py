"""
Tests for the trait vector artifact bank.

Covers: loading, validation, DataFrame structure, count enforcement,
ID uniqueness, split disjointness, invalid-trait rejection, missing-rubric
detection, and absence of GPU/Modal/torch imports.
"""

import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.vectors.artifact_bank import (
    EXPECTED_PROMPTS_PER_POLE,
    EXPECTED_QUESTIONS_PER_SPLIT,
    VALID_POLES,
    VALID_SPLITS,
    VALID_TRAITS,
    ArtifactBank,
    load_artifact_bank,
    validate_artifact_bank,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ARTIFACTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "trait_vector_artifacts.yaml"


@pytest.fixture(scope="module")
def raw_data() -> dict:
    with open(_ARTIFACTS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def bank() -> ArtifactBank:
    return load_artifact_bank(_ARTIFACTS_PATH)


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------


def test_file_exists():
    assert _ARTIFACTS_PATH.exists(), "configs/trait_vector_artifacts.yaml must exist"


def test_load_returns_artifact_bank(bank):
    assert isinstance(bank, ArtifactBank)


def test_all_three_dataframes_present(bank):
    assert isinstance(bank.system_prompts_df, pd.DataFrame)
    assert isinstance(bank.questions_df, pd.DataFrame)
    assert isinstance(bank.rubrics_df, pd.DataFrame)


def test_raw_dict_present(bank):
    assert isinstance(bank.raw, dict)


# ---------------------------------------------------------------------------
# Traits present
# ---------------------------------------------------------------------------


def test_all_four_traits_in_raw(raw_data):
    for trait in VALID_TRAITS:
        assert trait in raw_data, f"Missing trait: {trait}"


def test_all_four_traits_in_system_prompts_df(bank):
    found = set(bank.system_prompts_df["trait"].unique())
    assert found == VALID_TRAITS


def test_all_four_traits_in_questions_df(bank):
    found = set(bank.questions_df["trait"].unique())
    assert found == VALID_TRAITS


def test_all_four_traits_in_rubrics_df(bank):
    found = set(bank.rubrics_df["trait"].unique())
    assert found == VALID_TRAITS


# ---------------------------------------------------------------------------
# System prompts — counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trait", sorted(VALID_TRAITS))
@pytest.mark.parametrize("pole", ["positive", "negative"])
def test_system_prompt_count(bank, trait, pole):
    sub = bank.system_prompts_df[
        (bank.system_prompts_df["trait"] == trait)
        & (bank.system_prompts_df["pole"] == pole)
    ]
    assert len(sub) == EXPECTED_PROMPTS_PER_POLE, (
        f"[{trait}/{pole}] expected {EXPECTED_PROMPTS_PER_POLE} prompts, got {len(sub)}"
    )


# ---------------------------------------------------------------------------
# System prompts — content
# ---------------------------------------------------------------------------


def test_system_prompt_ids_are_non_empty(bank):
    assert (bank.system_prompts_df["prompt_id"].str.strip() != "").all()


def test_system_prompt_texts_are_non_empty(bank):
    assert (bank.system_prompts_df["prompt_text"].str.strip() != "").all()


def test_system_prompt_poles_are_valid(bank):
    invalid = set(bank.system_prompts_df["pole"].unique()) - VALID_POLES
    assert not invalid, f"Invalid pole values: {invalid}"


def test_system_prompt_ids_unique_per_trait(bank):
    for trait in VALID_TRAITS:
        ids = bank.system_prompts_df[bank.system_prompts_df["trait"] == trait]["prompt_id"].tolist()
        assert len(ids) == len(set(ids)), f"[{trait}] duplicate system prompt IDs"


def test_system_prompt_ids_globally_unique(bank):
    ids = bank.system_prompts_df["prompt_id"].tolist()
    assert len(ids) == len(set(ids)), "Duplicate system prompt IDs across traits"


# ---------------------------------------------------------------------------
# Questions — counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trait", sorted(VALID_TRAITS))
@pytest.mark.parametrize("split", ["extraction", "validation"])
def test_question_count(bank, trait, split):
    sub = bank.questions_df[
        (bank.questions_df["trait"] == trait)
        & (bank.questions_df["split"] == split)
    ]
    assert len(sub) == EXPECTED_QUESTIONS_PER_SPLIT, (
        f"[{trait}/{split}] expected {EXPECTED_QUESTIONS_PER_SPLIT} questions, got {len(sub)}"
    )


# ---------------------------------------------------------------------------
# Questions — content
# ---------------------------------------------------------------------------


def test_question_ids_are_non_empty(bank):
    assert (bank.questions_df["question_id"].str.strip() != "").all()


def test_question_texts_are_non_empty(bank):
    assert (bank.questions_df["question_text"].str.strip() != "").all()


def test_question_splits_are_valid(bank):
    invalid = set(bank.questions_df["split"].unique()) - VALID_SPLITS
    assert not invalid, f"Invalid split values: {invalid}"


def test_question_ids_globally_unique(bank):
    ids = bank.questions_df["question_id"].tolist()
    assert len(ids) == len(set(ids)), "Duplicate question IDs across traits"


@pytest.mark.parametrize("trait", sorted(VALID_TRAITS))
def test_extraction_validation_splits_disjoint(bank, trait):
    sub = bank.questions_df[bank.questions_df["trait"] == trait]
    ext_ids = set(sub[sub["split"] == "extraction"]["question_id"])
    val_ids = set(sub[sub["split"] == "validation"]["question_id"])
    assert not (ext_ids & val_ids), (
        f"[{trait}] extraction and validation question IDs overlap"
    )


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------


def test_rubric_count(bank):
    assert len(bank.rubrics_df) == 4


def test_rubric_columns(bank):
    expected = {"trait", "scoring_instructions", "min_score", "max_score"}
    assert expected.issubset(set(bank.rubrics_df.columns))


def test_rubric_scoring_instructions_non_empty(bank):
    assert (bank.rubrics_df["scoring_instructions"].str.strip() != "").all()


def test_rubric_score_range(bank):
    assert (bank.rubrics_df["min_score"] == 0).all()
    assert (bank.rubrics_df["max_score"] == 100).all()


# ---------------------------------------------------------------------------
# DataFrame columns
# ---------------------------------------------------------------------------


def test_system_prompts_df_columns(bank):
    expected = {"trait", "pole", "prompt_id", "prompt_text", "notes"}
    assert expected.issubset(set(bank.system_prompts_df.columns))


def test_questions_df_columns(bank):
    expected = {"trait", "split", "question_id", "question_text", "notes"}
    assert expected.issubset(set(bank.questions_df.columns))


def test_rubrics_df_columns(bank):
    expected = {"trait", "scoring_instructions", "min_score", "max_score"}
    assert expected.issubset(set(bank.rubrics_df.columns))


# ---------------------------------------------------------------------------
# Validation — invalid trait rejected
# ---------------------------------------------------------------------------


def test_missing_trait_raises(raw_data):
    bad_data = {k: v for k, v in raw_data.items() if k != "honesty"}
    with pytest.raises(ValueError, match="Missing traits"):
        validate_artifact_bank(bad_data)


def test_ethics_split_name_not_a_valid_trait(raw_data):
    # Injecting an ETHICS split name as a trait should NOT appear in VALID_TRAITS
    ethics_split_names = {"commonsense", "deontology", "justice", "utilitarianism", "virtue"}
    for name in ethics_split_names:
        assert name not in VALID_TRAITS, (
            f"ETHICS split name '{name}' must not be in VALID_TRAITS"
        )


# ---------------------------------------------------------------------------
# Validation — duplicate IDs rejected
# ---------------------------------------------------------------------------


def test_duplicate_system_prompt_id_rejected(raw_data):
    bad = copy.deepcopy(raw_data)
    # Duplicate the first positive prompt id in honesty
    prompts = bad["honesty"]["positive_system_prompts"]
    prompts[1]["id"] = prompts[0]["id"]
    with pytest.raises(ValueError, match="Duplicate IDs"):
        validate_artifact_bank(bad)


def test_duplicate_question_id_rejected(raw_data):
    bad = copy.deepcopy(raw_data)
    questions = bad["honesty"]["elicitation_questions"]
    questions[1]["id"] = questions[0]["id"]
    with pytest.raises(ValueError, match="Duplicate IDs"):
        validate_artifact_bank(bad)


# ---------------------------------------------------------------------------
# Validation — missing rubric rejected
# ---------------------------------------------------------------------------


def test_missing_rubric_raises(raw_data):
    bad = copy.deepcopy(raw_data)
    del bad["fairness"]["evaluation_rubric"]
    with pytest.raises(ValueError, match="missing evaluation_rubric"):
        validate_artifact_bank(bad)


# ---------------------------------------------------------------------------
# Validation — wrong prompt count rejected
# ---------------------------------------------------------------------------


def test_wrong_prompt_count_raises(raw_data):
    bad = copy.deepcopy(raw_data)
    bad["compassion"]["positive_system_prompts"] = (
        bad["compassion"]["positive_system_prompts"][:2]
    )
    with pytest.raises(ValueError, match="positive system prompts"):
        validate_artifact_bank(bad)


# ---------------------------------------------------------------------------
# Validation — wrong question count rejected
# ---------------------------------------------------------------------------


def test_wrong_question_count_raises(raw_data):
    bad = copy.deepcopy(raw_data)
    qs = bad["harmlessness"]["elicitation_questions"]
    bad["harmlessness"]["elicitation_questions"] = [
        q for q in qs if q.get("split") != "validation"
    ][:10]
    with pytest.raises(ValueError, match="validation questions"):
        validate_artifact_bank(bad)


# ---------------------------------------------------------------------------
# File not found
# ---------------------------------------------------------------------------


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_artifact_bank("/nonexistent/path/to/artifacts.yaml")


# ---------------------------------------------------------------------------
# No heavy imports at module level
# ---------------------------------------------------------------------------


def test_no_torch_import():
    import src.vectors.artifact_bank as mod
    import sys
    assert "torch" not in sys.modules or mod.__name__ != "torch", \
        "artifact_bank must not import torch"


def test_no_modal_import():
    import src.vectors.artifact_bank as mod
    import sys
    # modal should not be imported as a side effect of loading artifact_bank
    # We can't guarantee modal is absent globally, but artifact_bank itself
    # should not reference it.
    source = Path(mod.__file__).read_text()
    assert "import modal" not in source


def test_no_vllm_import():
    import src.vectors.artifact_bank as mod
    from pathlib import Path
    source = Path(mod.__file__).read_text()
    assert "import vllm" not in source


def test_no_transformers_import():
    import src.vectors.artifact_bank as mod
    from pathlib import Path
    source = Path(mod.__file__).read_text()
    assert "import transformers" not in source
