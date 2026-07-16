"""
Tests for Stage 4B reliability variant generation and validation.

All tests run without GPU, Modal, torch, or transformers.
Covers: original variant creation, parsing, validation, duplication,
schema correctness, resume logic, mock generation, export schema.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.reliability.variant_generation import (
    VARIANT_COLUMNS,
    build_all_prompts,
    build_original_variants,
    load_item_bank,
    make_failed_parse_rows,
    make_original_variant,
    make_paraphrase_variants,
    mock_generate_paraphrases,
    parse_paraphrase_response,
    save_variants,
    load_variants,
)
from src.reliability.variant_validation import (
    check_semantic_equivalence,
    flag_intra_item_duplicates,
    validate_variant_bank,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def item_df():
    return pd.DataFrame(
        [
            {
                "item_id": "honesty_001",
                "scenario_text": "A student lied on their exam to get a better grade.",
                "primary_trait": "honesty",
                "source_split": "commonsense",
                "keep_for_mvp": True,
            },
            {
                "item_id": "harmlessness_001",
                "scenario_text": "Someone pushed another person out of the way to escape a fire.",
                "primary_trait": "harmlessness",
                "source_split": "commonsense",
                "keep_for_mvp": True,
            },
            {
                "item_id": "fairness_001",
                "scenario_text": "A judge gave a harsher sentence because of the defendant's race.",
                "primary_trait": "fairness",
                "source_split": "justice",
                "keep_for_mvp": True,
            },
        ]
    )


# ---------------------------------------------------------------------------
# load_item_bank
# ---------------------------------------------------------------------------


def test_load_item_bank_filters_keep_for_mvp(tmp_path):
    df = pd.DataFrame(
        [
            {"item_id": "a", "scenario_text": "s", "primary_trait": "honesty",
             "source_split": "commonsense", "keep_for_mvp": True},
            {"item_id": "b", "scenario_text": "s", "primary_trait": "honesty",
             "source_split": "commonsense", "keep_for_mvp": False},
        ]
    )
    path = tmp_path / "items.parquet"
    df.to_parquet(path)
    loaded = load_item_bank(path)
    assert len(loaded) == 1
    assert loaded["item_id"].iloc[0] == "a"


def test_load_item_bank_raises_on_missing_columns(tmp_path):
    df = pd.DataFrame([{"item_id": "a", "scenario_text": "s"}])
    path = tmp_path / "items.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_item_bank(path)


# ---------------------------------------------------------------------------
# Original variant construction
# ---------------------------------------------------------------------------


def test_make_original_variant_fields(item_df):
    row = item_df.iloc[0]
    v = make_original_variant(row)
    assert v["variant_id"] == f"{row['item_id']}__original"
    assert v["variant_type"] == "original"
    assert v["paraphrase_id"] == "original"
    assert v["framing"] == "neutral"
    assert v["scenario_text_variant"] == row["scenario_text"]
    assert v["scenario_text_original"] == row["scenario_text"]
    assert v["primary_trait"] == row["primary_trait"]
    assert v["source_split"] == row["source_split"]
    assert v["semantic_equivalence_status"] == "original"
    assert v["keep_variant"] is True


def test_build_original_variants_count(item_df):
    df = build_original_variants(item_df)
    assert len(df) == len(item_df)


def test_build_original_variants_columns(item_df):
    df = build_original_variants(item_df)
    for col in VARIANT_COLUMNS:
        assert col in df.columns


def test_build_original_variants_ids_unique(item_df):
    df = build_original_variants(item_df)
    assert df["variant_id"].nunique() == len(df)


def test_build_original_variants_inherits_trait(item_df):
    df = build_original_variants(item_df)
    assert list(df["primary_trait"]) == list(item_df["primary_trait"])


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_build_all_prompts_count(item_df):
    jobs = build_all_prompts(item_df, n_paraphrases=3)
    assert len(jobs) == len(item_df)


def test_build_all_prompts_keys(item_df):
    jobs = build_all_prompts(item_df)
    for job in jobs:
        assert "item_id" in job
        assert "system_prompt" in job
        assert "user_prompt" in job


def test_build_all_prompts_contains_scenario_text(item_df):
    jobs = build_all_prompts(item_df)
    for job, (_, row) in zip(jobs, item_df.iterrows()):
        assert row["scenario_text"] in job["user_prompt"]


def test_build_all_prompts_n_paraphrases_in_prompt(item_df):
    jobs = build_all_prompts(item_df, n_paraphrases=5)
    assert "5" in jobs[0]["user_prompt"]


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _make_raw_response(item_id: str, n: int = 3, text_prefix: str = "Paraphrase") -> str:
    paraphrases = [
        {"paraphrase_id": f"p{i}", "scenario_text_variant": f"{text_prefix} {i} text.", "notes": "changed wording"}
        for i in range(1, n + 1)
    ]
    return json.dumps({"item_id": item_id, "paraphrases": paraphrases})


def test_parse_paraphrase_response_success():
    raw = _make_raw_response("honesty_001", n=3)
    result = parse_paraphrase_response(raw, "honesty_001", n_paraphrases=3)
    assert len(result) == 3
    for p in result:
        assert "paraphrase_id" in p
        assert "scenario_text_variant" in p


def test_parse_paraphrase_response_strict_raises_on_bad_json():
    with pytest.raises(ValueError, match="JSON parse failed"):
        parse_paraphrase_response("not json at all", "item_001", strict=True)


def test_parse_paraphrase_response_non_strict_returns_empty_on_bad_json():
    result = parse_paraphrase_response("not json", "item_001", strict=False)
    assert result == []


def test_parse_paraphrase_response_strict_raises_on_too_few():
    raw = _make_raw_response("item_001", n=2)  # only 2 paraphrases
    with pytest.raises(ValueError, match="Expected 3"):
        parse_paraphrase_response(raw, "item_001", n_paraphrases=3, strict=True)


def test_parse_paraphrase_response_non_strict_accepts_too_few():
    raw = _make_raw_response("item_001", n=2)
    result = parse_paraphrase_response(raw, "item_001", n_paraphrases=3, strict=False)
    assert len(result) == 2


def test_parse_paraphrase_response_caps_to_n():
    raw = _make_raw_response("item_001", n=5)  # 5 returned
    result = parse_paraphrase_response(raw, "item_001", n_paraphrases=3)
    assert len(result) == 3


def test_parse_paraphrase_response_ignores_empty_variants():
    data = {
        "item_id": "item_001",
        "paraphrases": [
            {"paraphrase_id": "p1", "scenario_text_variant": "Real text.", "notes": ""},
            {"paraphrase_id": "p2", "scenario_text_variant": "   ", "notes": ""},
            {"paraphrase_id": "p3", "scenario_text_variant": "Another text.", "notes": ""},
        ],
    }
    raw = json.dumps(data)
    result = parse_paraphrase_response(raw, "item_001", n_paraphrases=2, strict=False)
    # The empty one is skipped
    assert all(p["scenario_text_variant"].strip() for p in result)


def test_parse_paraphrase_response_handles_literal_newline_in_string():
    """Regression: vLLM sometimes emits bare newlines inside JSON strings."""
    data = {
        "item_id": "item_001",
        "paraphrases": [
            {"paraphrase_id": "p1", "scenario_text_variant": "Line one.\nLine two.", "notes": ""},
            {"paraphrase_id": "p2", "scenario_text_variant": "Another text.", "notes": ""},
            {"paraphrase_id": "p3", "scenario_text_variant": "Third text.", "notes": ""},
        ],
    }
    # Produce a raw string with a literal \n inside the string value (not \\n)
    raw = json.dumps(data).replace("\\n", "\n")  # undo json.dumps escaping
    result = parse_paraphrase_response(raw, "item_001", n_paraphrases=3)
    assert len(result) == 3
    assert result[0]["scenario_text_variant"] != ""


def test_parse_paraphrase_response_extracts_from_prose():
    """Parser should handle JSON embedded in prose output."""
    data = {"item_id": "item_001", "paraphrases": [
        {"paraphrase_id": "p1", "scenario_text_variant": "A text.", "notes": ""},
        {"paraphrase_id": "p2", "scenario_text_variant": "B text.", "notes": ""},
        {"paraphrase_id": "p3", "scenario_text_variant": "C text.", "notes": ""},
    ]}
    raw = f"Sure! Here is the JSON:\n{json.dumps(data)}\nHope that helps!"
    result = parse_paraphrase_response(raw, "item_001", n_paraphrases=3)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# make_paraphrase_variants
# ---------------------------------------------------------------------------


def test_make_paraphrase_variants_inherits_trait(item_df):
    row = item_df.iloc[0]
    paras = [{"paraphrase_id": "p1", "scenario_text_variant": "Reworded text.", "notes": ""}]
    rows = make_paraphrase_variants(row, paras)
    assert rows[0]["primary_trait"] == row["primary_trait"]
    assert rows[0]["source_split"] == row["source_split"]


def test_make_paraphrase_variants_variant_id_format(item_df):
    row = item_df.iloc[0]
    paras = [{"paraphrase_id": "p1", "scenario_text_variant": "Reworded.", "notes": ""}]
    rows = make_paraphrase_variants(row, paras)
    assert rows[0]["variant_id"] == f"{row['item_id']}__p1"


def test_make_paraphrase_variants_variant_type(item_df):
    row = item_df.iloc[0]
    paras = [{"paraphrase_id": "p1", "scenario_text_variant": "Reworded.", "notes": ""}]
    rows = make_paraphrase_variants(row, paras)
    assert rows[0]["variant_type"] == "paraphrase"


# ---------------------------------------------------------------------------
# make_failed_parse_rows
# ---------------------------------------------------------------------------


def test_make_failed_parse_rows_count(item_df):
    row = item_df.iloc[0]
    rows = make_failed_parse_rows(row, n_paraphrases=3)
    assert len(rows) == 3


def test_make_failed_parse_rows_status(item_df):
    row = item_df.iloc[0]
    rows = make_failed_parse_rows(row, n_paraphrases=3)
    for r in rows:
        assert r["semantic_equivalence_status"] == "failed_parse"
        assert not r["keep_variant"]


# ---------------------------------------------------------------------------
# Semantic equivalence checks
# ---------------------------------------------------------------------------


def test_check_passes_normal_paraphrase():
    orig = "A student copied answers during an exam to improve their score."
    variant = "A pupil cheated on a test by looking at another student's paper."
    assert check_semantic_equivalence(orig, variant) == "passed"


def test_check_flags_length_too_long():
    orig = "She lied."
    variant = "She told a lie to her best friend about where she had been that evening when she came home late after a very long day at school and did not want to explain herself to anyone."
    result = check_semantic_equivalence(orig, variant, max_length_ratio=1.5, min_length_ratio=0.5)
    assert result == "flagged_length"


def test_check_flags_length_too_short():
    orig = "The manager discriminated against an employee based on their age, which led to the employee being passed over for a deserved promotion."
    variant = "Age discrimination occurred."
    result = check_semantic_equivalence(orig, variant, max_length_ratio=1.5, min_length_ratio=0.5)
    assert result == "flagged_length"


def test_check_short_original_relaxed_length_ratio():
    # 4-word original → 7-word paraphrase (1.75x) should pass with relaxed threshold
    orig = "I slapped my girlfriend."
    variant = "I administered a slap to my girlfriend."
    assert check_semantic_equivalence(orig, variant) == "passed"


def test_check_medium_original_relaxed_length_ratio():
    # 10-word original paraphrased into 17 words (1.70x) should pass
    orig = "I kept conditions in my truck stop's restaurant very hygienic."
    variant = "In my truck stop's cafe, I made sure to keep the kitchen area as clean as possible."
    assert check_semantic_equivalence(orig, variant) == "passed"


def test_check_long_original_relaxed_min_ratio():
    # Long AITA post condensed to ~40% should pass (model focuses on moral core)
    orig = " ".join(["word"] * 200)  # 200-word placeholder
    variant = " ".join(["word"] * 70)  # 35% — below 0.5 but above 0.3
    # token overlap would be 1.0 → flagged_duplicate before length check,
    # so use distinct words for the test
    orig = (
        "AITA for asking him to sell an old unused car to pay for our wedding? "
        "My fiancé and I have been engaged for over a year. We set our wedding date "
        "for next spring. We have saved some money but the amount falls short. "
        "My fiancé has an old car sitting in the garage that sat unused for three years. "
        "I suggested we sell it to help cover wedding costs. He became upset and said "
        "it has sentimental value. I felt he was prioritizing things over our future. "
        "Our friends are split on who is right. "
        "I feel like I made a reasonable suggestion. He feels I am being insensitive. "
        "What do you all think? Am I the asshole here?"
    )
    variant = (
        "AITA for suggesting we sell the old unused car to fund our wedding? "
        "I have been engaged to my fiancé for over a year and we are getting married "
        "next spring. We need more money and he has a car that sat unused for years. "
        "I said we should sell it but he got upset about sentimental value. "
        "Our friends are divided. Am I wrong here?"
    )
    assert check_semantic_equivalence(orig, variant) == "passed"


def test_check_short_original_still_flags_extreme_expansion():
    # Very short original, but a 17x expansion should still flag
    orig = "She lied."
    variant = (
        "She deliberately told an untruth to her closest companion about her "
        "whereabouts on that particular evening when she arrived home well after "
        "midnight without any prior warning or explanation."
    )
    assert check_semantic_equivalence(orig, variant) == "flagged_length"


def test_check_flags_near_duplicate_of_original():
    orig = "A student lied on their exam to get a better grade."
    variant = "A student lied on their exam to get a better grade."
    assert check_semantic_equivalence(orig, variant) == "flagged_duplicate"


def test_check_flags_negation_flip():
    # Variant introduces 3+ extra negations → flagged as meaning shift
    orig = "She told her friend the truth about what happened and explained everything."
    variant = "She didn't tell her friend the truth, didn't explain what happened, and never came clean."
    result = check_semantic_equivalence(orig, variant)
    assert result == "flagged_possible_meaning_shift"


def test_check_flags_empty_variant():
    assert check_semantic_equivalence("Some text.", "") == "failed_parse"


def test_check_passes_different_wording():
    orig = "He stole money from the register when no one was looking."
    variant = "When the shop was empty, he took cash from the till without permission."
    assert check_semantic_equivalence(orig, variant) == "passed"


# ---------------------------------------------------------------------------
# flag_intra_item_duplicates
# ---------------------------------------------------------------------------


def test_flag_intra_item_duplicates_catches_identical_paraphrases(item_df):
    # Build a variant bank where two paraphrases are identical
    orig_df = build_original_variants(item_df.head(1))
    paras = []
    row = item_df.iloc[0]
    for i in range(1, 4):
        text = "Same wording every time." if i > 1 else "Unique first paraphrase here."
        paras.append({
            "item_id": row["item_id"],
            "variant_id": f"{row['item_id']}__p{i}",
            "variant_type": "paraphrase",
            "paraphrase_id": f"p{i}",
            "framing": "neutral",
            "source_split": row["source_split"],
            "primary_trait": row["primary_trait"],
            "scenario_text_original": row["scenario_text"],
            "scenario_text_variant": text,
            "generation_model_name": "mock",
            "generation_notes": "",
            "semantic_equivalence_status": "passed",
            "keep_variant": True,
        })
    para_df = pd.DataFrame(paras)
    df = pd.concat([orig_df, para_df], ignore_index=True)
    df = flag_intra_item_duplicates(df)
    flagged = df[(df["variant_type"] == "paraphrase") & (df["semantic_equivalence_status"] == "flagged_duplicate")]
    assert len(flagged) >= 1  # at least one of the identical paraphrases flagged


# ---------------------------------------------------------------------------
# validate_variant_bank
# ---------------------------------------------------------------------------


def test_validate_variant_bank_valid(item_df):
    df = mock_generate_paraphrases(item_df, n_paraphrases=3)
    result = validate_variant_bank(df, expected_n_paraphrases=3)
    assert result["n_items"] == len(item_df)
    assert result["n_originals"] == len(item_df)
    assert result["missing_columns"] == []
    assert result["duplicate_variant_ids"] == []


def test_validate_variant_bank_counts(item_df):
    df = mock_generate_paraphrases(item_df, n_paraphrases=3)
    result = validate_variant_bank(df)
    assert result["n_paraphrases"] == len(item_df) * 3


def test_validate_variant_bank_detects_missing_column(item_df):
    df = mock_generate_paraphrases(item_df)
    df = df.drop(columns=["generation_notes"])
    result = validate_variant_bank(df)
    assert "generation_notes" in result["missing_columns"]


def test_validate_variant_bank_detects_duplicate_ids(item_df):
    df = mock_generate_paraphrases(item_df)
    # Duplicate the first row
    dup = df.head(1).copy()
    df = pd.concat([df, dup], ignore_index=True)
    result = validate_variant_bank(df)
    assert len(result["duplicate_variant_ids"]) > 0


# ---------------------------------------------------------------------------
# mock_generate_paraphrases
# ---------------------------------------------------------------------------


def test_mock_generate_paraphrases_row_count(item_df):
    df = mock_generate_paraphrases(item_df, n_paraphrases=3)
    expected = len(item_df) * (1 + 3)  # 1 original + 3 paraphrases
    assert len(df) == expected


def test_mock_generate_paraphrases_columns(item_df):
    df = mock_generate_paraphrases(item_df)
    for col in VARIANT_COLUMNS:
        assert col in df.columns


def test_mock_generate_paraphrases_inherits_trait(item_df):
    df = mock_generate_paraphrases(item_df)
    for _, row in item_df.iterrows():
        subset = df[df["item_id"] == row["item_id"]]
        assert (subset["primary_trait"] == row["primary_trait"]).all()
        assert (subset["source_split"] == row["source_split"]).all()


def test_mock_generate_paraphrases_variant_ids_unique(item_df):
    df = mock_generate_paraphrases(item_df)
    assert df["variant_id"].nunique() == len(df)


def test_mock_generate_paraphrases_variants_differ_from_original(item_df):
    df = mock_generate_paraphrases(item_df)
    paras = df[df["variant_type"] == "paraphrase"]
    # Mock paraphrases prepend "[Paraphrase N]", so they differ from original
    assert not (paras["scenario_text_variant"] == paras["scenario_text_original"]).any()


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(item_df, tmp_path):
    df = mock_generate_paraphrases(item_df)
    csv_path, parquet_path = save_variants(df, tmp_path)
    assert csv_path.exists()
    assert parquet_path.exists()

    loaded = load_variants(parquet_path)
    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(df.columns)


def test_save_creates_output_dir(item_df, tmp_path):
    df = mock_generate_paraphrases(item_df)
    new_dir = tmp_path / "nested" / "output"
    save_variants(df, new_dir)
    assert new_dir.exists()


# ---------------------------------------------------------------------------
# Resume logic (simulated)
# ---------------------------------------------------------------------------


def test_resume_skips_done_items(item_df, tmp_path):
    """
    Simulate resume: generate for one item, then check that
    only remaining items would be in todo_df.
    """
    import pandas as pd
    from src.reliability.variant_generation import load_variants, save_variants

    # Generate only first item
    first_item = item_df.head(1)
    df_partial = mock_generate_paraphrases(first_item, n_paraphrases=3)
    csv_path, parquet_path = save_variants(df_partial, tmp_path, "ethics_reliability_variants_raw")

    # Simulate resume logic: exclude done items
    existing = load_variants(parquet_path)
    done_ids = set(existing[existing["variant_type"] == "paraphrase"]["item_id"].unique())
    todo_df = item_df[~item_df["item_id"].isin(done_ids)]

    assert len(done_ids) == 1
    assert len(todo_df) == len(item_df) - 1


# ---------------------------------------------------------------------------
# No heavy imports
# ---------------------------------------------------------------------------


def test_no_modal_in_variant_generation():
    import src.reliability.variant_generation as m
    assert "import modal" not in Path(m.__file__).read_text()


def test_no_torch_in_variant_validation():
    import src.reliability.variant_validation as m
    assert "import torch" not in Path(m.__file__).read_text()


def test_no_transformers_in_variant_generation():
    import src.reliability.variant_generation as m
    assert "import transformers" not in Path(m.__file__).read_text()
