"""
Tests for src/vectors/artifact_quality.py

Covers: trait-label leakage detection, extraction/validation overlap,
generic moral-valence imbalance, cross-trait confound detection,
negative-pole extra-trait detection, generic-helpfulness collapse,
near-duplicate detection, audit DataFrame columns, and absence of
GPU/Modal/torch imports.
"""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.vectors.artifact_bank import (
    ArtifactBank,
    load_artifact_bank,
)
from src.vectors.artifact_quality import (
    CONFOUND_LEXICON,
    NEAR_DUPLICATE_JACCARD_THRESHOLD,
    AuditFinding,
    _check_cross_trait_confounds_in_prompts,
    _check_extraction_validation_text_overlap,
    _check_generic_valence_in_prompts,
    _check_near_duplicate_prompts,
    _check_near_duplicate_questions,
    _check_negative_pole_extra_traits,
    _check_positive_pole_generic_helpfulness,
    _check_rubric_unrelated_trait_words,
    _check_short_texts,
    _check_trait_label_leakage_in_questions,
    _jaccard,
    _words,
    findings_to_df,
    run_quality_checks,
)

_ARTIFACTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "trait_vector_artifacts.yaml"

# ---------------------------------------------------------------------------
# Helper: build a minimal ArtifactBank stub for unit tests
# ---------------------------------------------------------------------------


def _make_bank(
    system_prompts: list[dict] | None = None,
    questions: list[dict] | None = None,
    rubrics: list[dict] | None = None,
    raw: dict | None = None,
) -> ArtifactBank:
    """Build an ArtifactBank from plain dicts without loading the YAML."""
    sp_rows = system_prompts or []
    q_rows = questions or []
    r_rows = rubrics or []

    sp_df = pd.DataFrame(
        sp_rows,
        columns=["trait", "pole", "prompt_id", "prompt_text", "notes"],
    ) if sp_rows else pd.DataFrame(
        columns=["trait", "pole", "prompt_id", "prompt_text", "notes"]
    )
    q_df = pd.DataFrame(
        q_rows,
        columns=["trait", "split", "question_id", "question_text", "notes"],
    ) if q_rows else pd.DataFrame(
        columns=["trait", "split", "question_id", "question_text", "notes"]
    )
    r_df = pd.DataFrame(
        r_rows,
        columns=["trait", "scoring_instructions", "min_score", "max_score"],
    ) if r_rows else pd.DataFrame(
        columns=["trait", "scoring_instructions", "min_score", "max_score"]
    )
    return ArtifactBank(
        system_prompts_df=sp_df,
        questions_df=q_df,
        rubrics_df=r_df,
        raw=raw or {},
    )


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def test_words_lowercases_and_strips_punctuation():
    result = _words("Hello, WORLD! It's a test.")
    assert "hello" in result
    assert "world" in result


def test_jaccard_identical():
    assert _jaccard("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)


def test_jaccard_disjoint():
    assert _jaccard("alpha beta", "gamma delta") == pytest.approx(0.0)


def test_jaccard_partial():
    sim = _jaccard("the quick brown fox", "the slow red fox")
    assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# Short-text check
# ---------------------------------------------------------------------------


def test_short_text_flagged():
    findings = _check_short_texts([("honesty", "positive_system_prompt", "hon_pos_001", "ok")])
    assert any(f.issue_type == "text_too_short" for f in findings)


def test_adequate_text_not_flagged():
    long_text = "You are an assistant who values truth and accuracy above all else."
    findings = _check_short_texts([("honesty", "positive_system_prompt", "hon_pos_001", long_text)])
    assert not any(f.issue_type == "text_too_short" for f in findings)


# ---------------------------------------------------------------------------
# Trait label leakage in questions
# ---------------------------------------------------------------------------


def test_detects_trait_label_leakage_above_threshold():
    """More than 30% of questions naming the trait should trigger a warning."""
    questions = [
        {
            "trait": "honesty",
            "split": "extraction",
            "question_id": f"hon_ext_{i:03d}",
            "question_text": (
                "Are you being honest with the user about this situation?" if i < 8
                else "A person makes a claim you disagree with. What do you do?"
            ),
            "notes": "",
        }
        for i in range(20)
    ]
    bank = _make_bank(questions=questions)
    findings = _check_trait_label_leakage_in_questions(bank)
    assert any(
        f.issue_type == "trait_label_leakage" and f.severity == "warning"
        for f in findings
    ), "Should warn when >30% of questions mention trait label words"


def test_no_leakage_flagged_when_no_label_words():
    questions = [
        {
            "trait": "honesty",
            "split": "extraction",
            "question_id": f"hon_ext_{i:03d}",
            "question_text": "A colleague asks for feedback on their work. What do you say?",
            "notes": "",
        }
        for i in range(20)
    ]
    bank = _make_bank(questions=questions)
    findings = _check_trait_label_leakage_in_questions(bank)
    assert not any(f.issue_type == "trait_label_leakage" and f.severity == "warning" for f in findings)


# ---------------------------------------------------------------------------
# Extraction / validation text overlap
# ---------------------------------------------------------------------------


def test_detects_extraction_validation_text_overlap():
    same_text = "A person asks you whether the information you provided was accurate."
    questions = []
    for i in range(20):
        questions.append({
            "trait": "honesty",
            "split": "extraction",
            "question_id": f"hon_ext_{i:03d}",
            "question_text": same_text if i == 0 else f"Unique extraction scenario number {i}.",
            "notes": "",
        })
    for i in range(20):
        questions.append({
            "trait": "honesty",
            "split": "validation",
            "question_id": f"hon_val_{i:03d}",
            # First validation question is nearly identical to first extraction
            "question_text": same_text if i == 0 else f"Unique validation scenario number {i}.",
            "notes": "",
        })
    bank = _make_bank(questions=questions)
    findings = _check_extraction_validation_text_overlap(bank)
    assert any(f.issue_type == "extraction_validation_text_overlap" for f in findings)


def test_no_overlap_when_splits_are_distinct():
    questions = []
    for i in range(20):
        questions.append({
            "trait": "honesty",
            "split": "extraction",
            "question_id": f"hon_ext_{i:03d}",
            "question_text": f"Extraction scenario alpha {i} about feedback and accuracy.",
            "notes": "",
        })
    for i in range(20):
        questions.append({
            "trait": "honesty",
            "split": "validation",
            "question_id": f"hon_val_{i:03d}",
            "question_text": f"Validation scenario beta {i} about data integrity and reports.",
            "notes": "",
        })
    bank = _make_bank(questions=questions)
    findings = _check_extraction_validation_text_overlap(bank)
    assert not any(f.issue_type == "extraction_validation_text_overlap" for f in findings)


# ---------------------------------------------------------------------------
# Generic moral valence in prompts
# ---------------------------------------------------------------------------


def test_detects_generic_valence_dominance_in_positive_prompt():
    """A positive prompt with many generic-positive words but no trait-specific words."""
    generic_prompt = (
        "You are a good, ethical, moral, responsible, and helpful assistant "
        "who does good things because they are morally good and virtuous."
    )
    bank = _make_bank(
        system_prompts=[{
            "trait": "fairness",
            "pole": "positive",
            "prompt_id": "fai_pos_001",
            "prompt_text": generic_prompt,
            "notes": "",
        }]
    )
    findings = _check_generic_valence_in_prompts(bank)
    assert any(f.issue_type == "generic_valence_dominance" for f in findings)


def test_no_generic_valence_when_trait_specific():
    specific_prompt = (
        "You apply the same fair standards to every group and flag double standards. "
        "You are consistent, impartial, and treat equivalent cases equally."
    )
    bank = _make_bank(
        system_prompts=[{
            "trait": "fairness",
            "pole": "positive",
            "prompt_id": "fai_pos_001",
            "prompt_text": specific_prompt,
            "notes": "",
        }]
    )
    findings = _check_generic_valence_in_prompts(bank)
    assert not any(f.issue_type == "generic_valence_dominance" for f in findings)


# ---------------------------------------------------------------------------
# Cross-trait confounds in prompts
# ---------------------------------------------------------------------------


def test_detects_cross_trait_confound():
    """An honesty prompt that uses many harmlessness words should be flagged."""
    contaminated = (
        "You are an assistant who tells the truth by avoiding harm and danger "
        "and preventing dangerous violence and injury to people."
    )
    bank = _make_bank(
        system_prompts=[{
            "trait": "honesty",
            "pole": "negative",
            "prompt_id": "hon_neg_001",
            "prompt_text": contaminated,
            "notes": "",
        }]
    )
    findings = _check_cross_trait_confounds_in_prompts(bank)
    assert any(f.issue_type == "cross_trait_confound" for f in findings)


def test_no_confound_with_clean_prompt():
    clean = (
        "You are an assistant who prioritises persuasion over accuracy. "
        "You tell people what they want to hear rather than what is true."
    )
    bank = _make_bank(
        system_prompts=[{
            "trait": "honesty",
            "pole": "negative",
            "prompt_id": "hon_neg_001",
            "prompt_text": clean,
            "notes": "",
        }]
    )
    findings = _check_cross_trait_confounds_in_prompts(bank)
    assert not any(f.issue_type == "cross_trait_confound" for f in findings)


# ---------------------------------------------------------------------------
# Negative-pole extra-trait detection
# ---------------------------------------------------------------------------


def test_detects_negative_pole_extra_trait_cruelty_in_honesty():
    """An honesty-negative prompt that introduces compassion words."""
    cruel_honesty_neg = (
        "You are cruel and callous. You show no empathy or compassion "
        "and you tell lies without caring about suffering."
    )
    bank = _make_bank(
        system_prompts=[{
            "trait": "honesty",
            "pole": "negative",
            "prompt_id": "hon_neg_001",
            "prompt_text": cruel_honesty_neg,
            "notes": "",
        }]
    )
    findings = _check_negative_pole_extra_traits(bank)
    assert any(f.issue_type == "negative_pole_extra_trait" for f in findings)


def test_no_extra_trait_in_clean_negative_prompt():
    clean_neg = (
        "You prioritise telling people what they want to hear. "
        "You downplay uncomfortable facts and omit information that would change their view."
    )
    bank = _make_bank(
        system_prompts=[{
            "trait": "honesty",
            "pole": "negative",
            "prompt_id": "hon_neg_001",
            "prompt_text": clean_neg,
            "notes": "",
        }]
    )
    findings = _check_negative_pole_extra_traits(bank)
    assert not any(f.issue_type == "negative_pole_extra_trait" for f in findings)


# ---------------------------------------------------------------------------
# Generic helpfulness collapse in positive prompts
# ---------------------------------------------------------------------------


def test_detects_generic_helpfulness_in_positive_prompt():
    # Use exact tokens from the helpfulness_words list: "helpful", "help", "assist", "support"
    helpfulness_prompt = (
        "You are a maximally helpful assistant. You help users with any request, "
        "assist them with any problem, support their goals, and accommodate their needs."
    )
    bank = _make_bank(
        system_prompts=[{
            "trait": "compassion",
            "pole": "positive",
            "prompt_id": "cmp_pos_001",
            "prompt_text": helpfulness_prompt,
            "notes": "",
        }]
    )
    findings = _check_positive_pole_generic_helpfulness(bank)
    assert any(f.issue_type == "generic_helpfulness_collapse" for f in findings)


def test_no_helpfulness_collapse_for_trait_specific_positive():
    specific_pos = (
        "You are deeply empathetic and respond to suffering with genuine warmth. "
        "You acknowledge pain and make people feel understood before offering advice."
    )
    bank = _make_bank(
        system_prompts=[{
            "trait": "compassion",
            "pole": "positive",
            "prompt_id": "cmp_pos_001",
            "prompt_text": specific_pos,
            "notes": "",
        }]
    )
    findings = _check_positive_pole_generic_helpfulness(bank)
    assert not any(f.issue_type == "generic_helpfulness_collapse" for f in findings)


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------


def test_detects_near_duplicate_prompts():
    text_a = "You are an assistant who values accuracy and always tells the truth without distorting facts."
    text_b = "You are an assistant who values accuracy and always tells the truth and never distorts facts."
    assert _jaccard(text_a, text_b) >= NEAR_DUPLICATE_JACCARD_THRESHOLD
    bank = _make_bank(
        system_prompts=[
            {"trait": "honesty", "pole": "positive", "prompt_id": "hon_pos_001",
             "prompt_text": text_a, "notes": ""},
            {"trait": "honesty", "pole": "positive", "prompt_id": "hon_pos_002",
             "prompt_text": text_b, "notes": ""},
        ]
    )
    findings = _check_near_duplicate_prompts(bank)
    assert any(f.issue_type == "near_duplicate" for f in findings)


def test_no_near_duplicate_when_prompts_are_distinct():
    bank = _make_bank(
        system_prompts=[
            {"trait": "honesty", "pole": "positive", "prompt_id": "hon_pos_001",
             "prompt_text": "You always tell the truth and acknowledge uncertainty.", "notes": ""},
            {"trait": "honesty", "pole": "positive", "prompt_id": "hon_pos_002",
             "prompt_text": "You never omit important context that would change the user's understanding.", "notes": ""},
        ]
    )
    findings = _check_near_duplicate_prompts(bank)
    assert not any(f.issue_type == "near_duplicate" for f in findings)


def test_detects_near_duplicate_questions():
    text_a = "A colleague asks for your opinion on their work and you have serious concerns."
    text_b = "A colleague asks for your opinion on their work and you have serious reservations."
    assert _jaccard(text_a, text_b) >= NEAR_DUPLICATE_JACCARD_THRESHOLD
    questions = [
        {"trait": "honesty", "split": "extraction", "question_id": "hon_ext_001",
         "question_text": text_a, "notes": ""},
        {"trait": "honesty", "split": "extraction", "question_id": "hon_ext_002",
         "question_text": text_b, "notes": ""},
    ]
    bank = _make_bank(questions=questions)
    findings = _check_near_duplicate_questions(bank)
    assert any(f.issue_type == "near_duplicate" for f in findings)


# ---------------------------------------------------------------------------
# Rubric cross-trait words
# ---------------------------------------------------------------------------


def test_detects_rubric_cross_trait_words():
    # A fairness rubric that heavily discusses compassion/empathy/suffering
    fairness_rubric_with_compassion = (
        "Score based on empathy, compassion, suffering, and how caring the response is. "
        "Does it show compassion for suffering individuals who are suffering from cruelty?"
    )
    bank = _make_bank(
        rubrics=[{
            "trait": "fairness",
            "scoring_instructions": fairness_rubric_with_compassion,
            "min_score": 0,
            "max_score": 100,
        }]
    )
    findings = _check_rubric_unrelated_trait_words(bank)
    assert any(f.issue_type == "rubric_cross_trait_words" for f in findings)


def test_no_rubric_cross_trait_for_focused_rubric():
    focused = (
        "Score based on whether the response applies the same standard to both groups "
        "and whether it is consistent, impartial, and avoids discrimination."
    )
    bank = _make_bank(
        rubrics=[{
            "trait": "fairness",
            "scoring_instructions": focused,
            "min_score": 0,
            "max_score": 100,
        }]
    )
    findings = _check_rubric_unrelated_trait_words(bank)
    assert not any(f.issue_type == "rubric_cross_trait_words" for f in findings)


# ---------------------------------------------------------------------------
# findings_to_df columns
# ---------------------------------------------------------------------------


def test_findings_to_df_has_expected_columns():
    finding = AuditFinding(
        trait="honesty",
        artifact_type="positive_system_prompt",
        artifact_id="hon_pos_001",
        severity="warning",
        issue_type="generic_valence_dominance",
        text="sample text",
        explanation="Explanation.",
        suggested_review_action="Do something.",
    )
    df = findings_to_df([finding])
    expected_cols = {
        "trait", "artifact_type", "artifact_id", "severity",
        "issue_type", "text", "explanation", "suggested_review_action",
    }
    assert expected_cols.issubset(set(df.columns))


def test_findings_to_df_empty_returns_empty_df():
    df = findings_to_df([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Severity values are constrained
# ---------------------------------------------------------------------------


def test_all_severity_values_valid():
    bank = load_artifact_bank(_ARTIFACTS_PATH)
    findings = run_quality_checks(bank)
    valid = {"info", "warning", "high"}
    for f in findings:
        assert f.severity in valid, f"Invalid severity: {f.severity}"


# ---------------------------------------------------------------------------
# run_quality_checks on real artifact bank
# ---------------------------------------------------------------------------


def test_run_quality_checks_returns_list_of_findings():
    bank = load_artifact_bank(_ARTIFACTS_PATH)
    findings = run_quality_checks(bank)
    assert isinstance(findings, list)
    for f in findings:
        assert isinstance(f, AuditFinding)


def test_no_high_severity_in_seed_artifacts():
    """
    The seed artifact bank should not have any HIGH severity findings.
    (Near-duplicates or extraction/validation content overlap would be HIGH.)
    If this fails, fix the seed YAML.
    """
    bank = load_artifact_bank(_ARTIFACTS_PATH)
    findings = run_quality_checks(bank)
    high = [f for f in findings if f.severity == "high"]
    assert high == [], (
        f"Seed artifact bank has {len(high)} HIGH severity finding(s):\n"
        + "\n".join(f"  [{f.trait}] {f.artifact_id}: {f.issue_type}" for f in high)
    )


# ---------------------------------------------------------------------------
# Confound lexicon structure
# ---------------------------------------------------------------------------


def test_confound_lexicon_has_expected_keys():
    required = {
        "generic_positive", "generic_negative",
        "honesty_related", "harmlessness_related",
        "fairness_related", "compassion_related",
    }
    assert required.issubset(set(CONFOUND_LEXICON.keys()))


def test_confound_lexicon_lists_are_non_empty():
    for key, words in CONFOUND_LEXICON.items():
        assert len(words) > 0, f"Lexicon '{key}' is empty"


# ---------------------------------------------------------------------------
# No heavy imports
# ---------------------------------------------------------------------------


def test_no_torch_in_artifact_quality():
    source = Path("src/vectors/artifact_quality.py").read_text()
    assert "import torch" not in source


def test_no_modal_in_artifact_quality():
    source = Path("src/vectors/artifact_quality.py").read_text()
    assert "import modal" not in source


def test_no_vllm_in_artifact_quality():
    source = Path("src/vectors/artifact_quality.py").read_text()
    assert "import vllm" not in source


def test_no_transformers_in_artifact_quality():
    source = Path("src/vectors/artifact_quality.py").read_text()
    assert "import transformers" not in source
