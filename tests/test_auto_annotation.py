"""
Tests for src/data/auto_annotation.py.

All tests run without an API key or GPU.  The LLM call (call_llm) is
monkeypatched wherever a real API call would otherwise be needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import load_trait_space, _ETHICS_SPLIT_NAMES
from src.data.auto_annotation import (
    CONSTRUCT_TRAITS,
    VALID_PRIMARY,
    annotate_dataframe,
    build_annotation_prompt,
    build_system_prompt,
    parse_llm_response,
    save_autolabeled_sheet,
    validate_parsed_annotation,
)

TRAITS_PATH = Path(__file__).parent.parent / "configs" / "traits.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trait_space():
    return load_trait_space(TRAITS_PATH)


def _make_row(
    item_id: str = "commonsense_test_000001",
    source_split: str = "commonsense",
    scenario_text: str = "I lied to my friend about where I was.",
    label: int = 1,
    label_semantics: str = "1=morally_wrong; 0=morally_OK",
    primary_trait: str = "",
) -> pd.Series:
    return pd.Series({
        "item_id": item_id,
        "source_dataset": "ETHICS",
        "source_split": source_split,
        "scenario_text": scenario_text,
        "label": label,
        "label_semantics": label_semantics,
        "raw_fields": "{}",
        "primary_trait": primary_trait,
        "secondary_traits": "",
        "annotation_confidence": "",
        "annotation_notes": "",
        "keep_for_mvp": "",
    })


def _make_df(n: int = 5, source_split: str = "commonsense") -> pd.DataFrame:
    rows = [_make_row(
        item_id=f"{source_split}_test_{i:06d}",
        source_split=source_split,
        scenario_text=f"Scenario {i}",
        primary_trait="",
    ) for i in range(n)]
    return pd.DataFrame(rows)


def _good_response(
    primary: str = "honesty",
    secondary: list = None,
    confidence: str = "high",
    keep: bool = True,
    notes: str = "Item clearly engages deception.",
) -> str:
    return json.dumps({
        "primary_trait": primary,
        "secondary_traits": secondary or [],
        "annotation_confidence": confidence,
        "keep_for_mvp": keep,
        "annotation_notes": notes,
    })


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class TestBuildAnnotationPrompt:
    def test_returns_two_strings(self, trait_space):
        row = _make_row()
        system, user = build_annotation_prompt(row, trait_space)
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_contains_all_trait_names(self, trait_space):
        system = build_system_prompt(trait_space)
        for trait in CONSTRUCT_TRAITS:
            assert trait.upper() in system, f"Trait '{trait}' missing from system prompt"

    def test_system_prompt_contains_item_tagging_guidance(self, trait_space):
        system = build_system_prompt(trait_space)
        assert "Tagging guidance" in system

    def test_system_prompt_contains_not_applicable_rule(self, trait_space):
        system = build_system_prompt(trait_space)
        assert "not_applicable" in system

    def test_system_prompt_warns_against_ethics_splits(self, trait_space):
        system = build_system_prompt(trait_space)
        # The prompt must explicitly say ETHICS split names are not trait labels
        assert "commonsense" in system or "ETHICS" in system

    def test_user_prompt_contains_scenario_text(self, trait_space):
        row = _make_row(scenario_text="She stole the money.")
        _, user = build_annotation_prompt(row, trait_space)
        assert "She stole the money." in user

    def test_user_prompt_contains_source_split(self, trait_space):
        row = _make_row(source_split="justice")
        _, user = build_annotation_prompt(row, trait_space)
        assert "justice" in user

    def test_system_prompt_contains_json_schema(self, trait_space):
        system = build_system_prompt(trait_space)
        assert "primary_trait" in system
        assert "annotation_confidence" in system
        assert "keep_for_mvp" in system


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestParseLLMResponse:
    def test_parses_clean_json(self):
        raw = _good_response()
        parsed = parse_llm_response(raw)
        assert parsed["primary_trait"] == "honesty"

    def test_parses_json_inside_markdown_fence(self):
        raw = "```json\n" + _good_response() + "\n```"
        parsed = parse_llm_response(raw)
        assert "primary_trait" in parsed

    def test_parses_json_with_surrounding_prose(self):
        raw = "Here is my annotation:\n" + _good_response() + "\nThat is all."
        parsed = parse_llm_response(raw)
        assert "primary_trait" in parsed

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_llm_response("This is just plain text with no JSON.")

    def test_handles_string_secondary_traits(self):
        raw = json.dumps({
            "primary_trait": "fairness",
            "secondary_traits": "compassion,honesty",
            "annotation_confidence": "medium",
            "keep_for_mvp": True,
            "annotation_notes": "test",
        })
        parsed = parse_llm_response(raw)
        assert parsed["secondary_traits"] == "compassion,honesty"


# ---------------------------------------------------------------------------
# Annotation validation
# ---------------------------------------------------------------------------

class TestValidateParsedAnnotation:
    def test_valid_construct_trait_passes(self):
        parsed = {
            "primary_trait": "honesty",
            "secondary_traits": [],
            "annotation_confidence": "high",
            "keep_for_mvp": True,
            "annotation_notes": "Clearly about deception.",
        }
        result = validate_parsed_annotation(parsed)
        assert result["primary_trait"] == "honesty"
        assert result["keep_for_mvp"] == "true"

    def test_not_applicable_passes_with_empty_secondary(self):
        parsed = {
            "primary_trait": "not_applicable",
            "secondary_traits": [],
            "annotation_confidence": "high",
            "keep_for_mvp": False,
            "annotation_notes": "Hedonic preference item.",
        }
        result = validate_parsed_annotation(parsed)
        assert result["primary_trait"] == "not_applicable"
        assert result["keep_for_mvp"] == "false"

    def test_unclear_forces_keep_false(self):
        parsed = {
            "primary_trait": "unclear",
            "secondary_traits": [],
            "annotation_confidence": "medium",
            "keep_for_mvp": True,  # model tries to set true
            "annotation_notes": "Hard to classify.",
        }
        result = validate_parsed_annotation(parsed)
        assert result["keep_for_mvp"] == "false"  # must be overridden

    def test_not_applicable_forces_keep_false(self):
        parsed = {
            "primary_trait": "not_applicable",
            "secondary_traits": [],
            "annotation_confidence": "high",
            "keep_for_mvp": True,  # model tries to set true
            "annotation_notes": "N/A item.",
        }
        result = validate_parsed_annotation(parsed)
        assert result["keep_for_mvp"] == "false"

    def test_low_confidence_forces_keep_false(self):
        parsed = {
            "primary_trait": "compassion",
            "secondary_traits": [],
            "annotation_confidence": "low",
            "keep_for_mvp": True,
            "annotation_notes": "Maybe.",
        }
        result = validate_parsed_annotation(parsed)
        assert result["keep_for_mvp"] == "false"

    @pytest.mark.parametrize("split_name", sorted(_ETHICS_SPLIT_NAMES))
    def test_ethics_split_name_rejected_as_primary(self, split_name):
        parsed = {
            "primary_trait": split_name,
            "secondary_traits": [],
            "annotation_confidence": "high",
            "keep_for_mvp": True,
            "annotation_notes": "Wrong label.",
        }
        with pytest.raises(ValueError, match="ETHICS source-split"):
            validate_parsed_annotation(parsed)

    @pytest.mark.parametrize("split_name", sorted(_ETHICS_SPLIT_NAMES))
    def test_ethics_split_name_rejected_in_secondary(self, split_name):
        parsed = {
            "primary_trait": "honesty",
            "secondary_traits": [split_name],
            "annotation_confidence": "high",
            "keep_for_mvp": True,
            "annotation_notes": "Wrong secondary.",
        }
        with pytest.raises(ValueError, match="ETHICS split"):
            validate_parsed_annotation(parsed)

    def test_not_applicable_with_secondary_rejected(self):
        parsed = {
            "primary_trait": "not_applicable",
            "secondary_traits": ["fairness"],
            "annotation_confidence": "high",
            "keep_for_mvp": False,
            "annotation_notes": "N/A with secondary.",
        }
        with pytest.raises(ValueError, match="secondary_traits must be empty"):
            validate_parsed_annotation(parsed)

    def test_primary_in_secondary_rejected(self):
        parsed = {
            "primary_trait": "fairness",
            "secondary_traits": ["fairness"],
            "annotation_confidence": "medium",
            "keep_for_mvp": True,
            "annotation_notes": "Duplicate.",
        }
        with pytest.raises(ValueError, match="must not appear"):
            validate_parsed_annotation(parsed)

    def test_invalid_confidence_degrades_to_low(self):
        parsed = {
            "primary_trait": "compassion",
            "secondary_traits": [],
            "annotation_confidence": "very_high",  # invalid
            "keep_for_mvp": True,
            "annotation_notes": "test",
        }
        result = validate_parsed_annotation(parsed)
        assert result["annotation_confidence"] == "low"
        # low confidence → keep_for_mvp forced to false
        assert result["keep_for_mvp"] == "false"

    def test_notes_truncated_to_300_chars(self):
        parsed = {
            "primary_trait": "honesty",
            "secondary_traits": [],
            "annotation_confidence": "high",
            "keep_for_mvp": True,
            "annotation_notes": "x" * 500,
        }
        result = validate_parsed_annotation(parsed)
        assert len(result["annotation_notes"]) <= 300

    def test_secondary_traits_joined_as_csv_string(self):
        parsed = {
            "primary_trait": "honesty",
            "secondary_traits": ["fairness", "compassion"],
            "annotation_confidence": "high",
            "keep_for_mvp": True,
            "annotation_notes": "Two traits.",
        }
        result = validate_parsed_annotation(parsed)
        assert "fairness" in result["secondary_traits"]
        assert "compassion" in result["secondary_traits"]


# ---------------------------------------------------------------------------
# annotate_dataframe — dry-run (no API key needed)
# ---------------------------------------------------------------------------

class TestAnnotateDataframeDryRun:
    def test_dry_run_fills_all_rows(self, trait_space):
        df = _make_df(3)
        result = annotate_dataframe(
            df, trait_space, client=None, dry_run=True, resume=False
        )
        assert len(result) == 3
        assert (result["primary_trait"].str.strip() != "").all()

    def test_dry_run_sets_dry_run_note(self, trait_space):
        df = _make_df(2)
        result = annotate_dataframe(
            df, trait_space, client=None, dry_run=True, resume=False
        )
        assert result["annotation_notes"].str.contains("DRY RUN").all()

    def test_dry_run_does_not_call_client(self, trait_space):
        df = _make_df(3)
        mock_client = MagicMock()
        annotate_dataframe(
            df, trait_space, client=mock_client, dry_run=True, resume=False
        )
        mock_client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# annotate_dataframe — resume mode (monkeypatched API)
# ---------------------------------------------------------------------------

class TestAnnotateDataframeResume:
    def _mock_client(self) -> MagicMock:
        client = MagicMock()
        client.messages.create.return_value.content = [
            MagicMock(text=_good_response("harmlessness"))
        ]
        return client

    def test_resume_skips_already_annotated_rows(self, trait_space):
        df = _make_df(5)
        df.at[0, "primary_trait"] = "honesty"
        df.at[1, "primary_trait"] = "fairness"

        client = self._mock_client()
        with patch("src.data.auto_annotation.call_llm",
                   return_value=_good_response("harmlessness")):
            result = annotate_dataframe(
                df, trait_space, client=client, resume=True,
                dry_run=False, rate_limit_delay=0,
            )

        # Rows 0 and 1 should retain their original annotations
        assert result.at[0, "primary_trait"] == "honesty"
        assert result.at[1, "primary_trait"] == "fairness"
        # Rows 2–4 should be newly annotated
        for i in [2, 3, 4]:
            assert result.at[i, "primary_trait"] == "harmlessness"

    def test_resume_false_overwrites_existing(self, trait_space):
        df = _make_df(3)
        df.at[0, "primary_trait"] = "honesty"

        with patch("src.data.auto_annotation.call_llm",
                   return_value=_good_response("fairness")):
            result = annotate_dataframe(
                df, trait_space, client=MagicMock(), resume=False,
                dry_run=False, rate_limit_delay=0,
            )

        assert result.at[0, "primary_trait"] == "fairness"

    def test_fallback_on_parse_error(self, trait_space):
        df = _make_df(2)

        with patch("src.data.auto_annotation.call_llm",
                   return_value="This is not JSON at all"):
            result = annotate_dataframe(
                df, trait_space, client=MagicMock(), resume=False,
                dry_run=False, rate_limit_delay=0,
            )

        # Falls back to 'unclear' with keep_for_mvp=false
        assert result["primary_trait"].isin({"unclear"}).all()
        assert (result["keep_for_mvp"] == "false").all()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

class TestSaveAutolabeledSheet:
    def test_writes_csv(self, tmp_path, trait_space):
        df = _make_df(3)
        df["primary_trait"] = "honesty"
        out = tmp_path / "out.csv"
        save_autolabeled_sheet(df, out)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == 3

    def test_roundtrip_preserves_item_ids(self, tmp_path, trait_space):
        df = _make_df(4)
        out = tmp_path / "out.csv"
        save_autolabeled_sheet(df, out)
        loaded = pd.read_csv(out, dtype=str).fillna("")
        assert set(loaded["item_id"]) == set(df["item_id"])


# ---------------------------------------------------------------------------
# dry-run script (subprocess — no API key)
# ---------------------------------------------------------------------------

class TestDryRunScript:
    def test_dry_run_exits_zero(self, tmp_path):
        """--dry-run must complete without API credentials."""
        import pandas as pd
        # Write a tiny annotation sheet to a temp path
        rows = [{
            "item_id": f"commonsense_test_{i:06d}",
            "source_dataset": "ETHICS",
            "source_split": "commonsense",
            "scenario_text": f"Scenario {i}",
            "label": 0,
            "label_semantics": "1=wrong",
            "raw_fields": "{}",
            "primary_trait": "",
            "secondary_traits": "",
            "annotation_confidence": "",
            "annotation_notes": "",
            "keep_for_mvp": "",
        } for i in range(3)]
        sheet_path = tmp_path / "sheet.csv"
        pd.DataFrame(rows).to_csv(sheet_path, index=False)

        result = subprocess.run(
            [
                sys.executable, "scripts/auto_annotate_items.py",
                "--dry-run",
                "--input", str(sheet_path),
                "--output", str(tmp_path / "out.csv"),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "DRY RUN" in result.stdout

    def test_dry_run_does_not_write_output(self, tmp_path):
        rows = [{
            "item_id": "commonsense_test_000000",
            "source_dataset": "ETHICS",
            "source_split": "commonsense",
            "scenario_text": "A scenario.",
            "label": 0,
            "label_semantics": "1=wrong",
            "raw_fields": "{}",
            "primary_trait": "",
            "secondary_traits": "",
            "annotation_confidence": "",
            "annotation_notes": "",
            "keep_for_mvp": "",
        }]
        sheet_path = tmp_path / "sheet.csv"
        pd.DataFrame(rows).to_csv(sheet_path, index=False)
        out_path = tmp_path / "out.csv"

        subprocess.run(
            [
                sys.executable, "scripts/auto_annotate_items.py",
                "--dry-run", "--input", str(sheet_path),
                "--output", str(out_path),
            ],
            capture_output=True,
        )
        assert not out_path.exists(), "--dry-run must not write output file"


# ---------------------------------------------------------------------------
# No GPU / model imports
# ---------------------------------------------------------------------------

class TestNoDependencies:
    def test_auto_annotation_module_no_torch(self):
        import src.data.auto_annotation as m
        assert not hasattr(m, "torch")
        assert not hasattr(m, "transformers")

    def test_anthropic_not_imported_at_module_level(self):
        """The anthropic package must be imported lazily (only in the script),
        so that the core library can be used without it installed."""
        import importlib
        import sys
        # Remove from cache if already imported
        for key in list(sys.modules.keys()):
            if key.startswith("src.data.auto_annotation"):
                del sys.modules[key]
        import src.data.auto_annotation  # reimport
        # If anthropic were imported at module level, this would pull it in.
        # We just check the auto_annotation module itself doesn't hold it.
        assert "anthropic" not in dir(src.data.auto_annotation)
