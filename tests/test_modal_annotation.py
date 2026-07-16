"""
Tests for the Modal/vLLM annotation workflow.

All tests run without Modal, GPU, torch, transformers, or vLLM installed.
The business logic (prompt building, JSON schema, response validation) is
exercised using the same functions used by the Modal app, with simulated
model outputs.

Test categories:
  - ANNOTATION_SCHEMA is well-formed and enforces the right constraints
  - Simulated vLLM JSON responses (as if guided decoding succeeded) validate correctly
  - Simulated bad responses (empty, malformed) fall back to unclear/low/false
  - ETHICS split names are rejected as labels
  - not_applicable / unclear / low-confidence rules are enforced
  - The modal_apps module can be imported without Modal being installed
  - The wrapper script exists and is importable
  - No GPU/torch/transformers/vLLM imports appear in the normal test path
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Business logic lives in auto_annotation — always importable, no Modal dep
from src.config import _ETHICS_SPLIT_NAMES, load_trait_space
from src.data.auto_annotation import (
    CONSTRUCT_TRAITS,
    VALID_PRIMARY,
    _fallback_annotation,
    build_annotation_prompt,
    build_system_prompt,
    parse_llm_response,
    validate_parsed_annotation,
)

TRAITS_PATH = Path(__file__).parent.parent / "configs" / "traits.yaml"
MODAL_APP_PATH = Path(__file__).parent.parent / "modal_apps" / "annotate_ethics_items.py"
WRAPPER_SCRIPT = Path(__file__).parent.parent / "scripts" / "auto_annotate_items_modal.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trait_space():
    return load_trait_space(TRAITS_PATH)


def _row(
    item_id: str = "commonsense_test_000001",
    source_split: str = "commonsense",
    scenario_text: str = "I lied to my colleague about finishing the report.",
    label: int = 1,
    label_semantics: str = "1=morally_wrong; 0=morally_OK",
) -> pd.Series:
    return pd.Series({
        "item_id": item_id,
        "source_dataset": "ETHICS",
        "source_split": source_split,
        "scenario_text": scenario_text,
        "label": label,
        "label_semantics": label_semantics,
        "raw_fields": "{}",
        "primary_trait": "",
        "secondary_traits": "",
        "annotation_confidence": "",
        "annotation_notes": "",
        "keep_for_mvp": "",
    })


def _valid_json(
    primary: str = "honesty",
    secondary: list | None = None,
    confidence: str = "high",
    keep: bool = True,
    notes: str = "Clearly about deception.",
) -> str:
    """Return a JSON string in the format vLLM guided decoding would produce."""
    return json.dumps({
        "primary_trait": primary,
        "secondary_traits": secondary or [],
        "annotation_confidence": confidence,
        "keep_for_mvp": keep,
        "annotation_notes": notes,
    })


# ---------------------------------------------------------------------------
# ANNOTATION_SCHEMA structure
# ---------------------------------------------------------------------------

class TestAnnotationSchema:
    def test_schema_importable_without_modal(self):
        # Import only the schema constant — must not require modal or vllm
        from modal_apps.annotate_ethics_items import ANNOTATION_SCHEMA
        assert isinstance(ANNOTATION_SCHEMA, dict)

    def test_schema_has_required_fields(self):
        from modal_apps.annotate_ethics_items import ANNOTATION_SCHEMA
        required = set(ANNOTATION_SCHEMA["required"])
        assert "primary_trait" in required
        assert "secondary_traits" in required
        assert "annotation_confidence" in required
        assert "keep_for_mvp" in required
        assert "annotation_notes" in required

    def test_schema_primary_trait_enum_matches_valid_primary(self):
        from modal_apps.annotate_ethics_items import ANNOTATION_SCHEMA
        schema_enum = set(
            ANNOTATION_SCHEMA["properties"]["primary_trait"]["enum"]
        )
        # Must contain all four construct traits plus not_applicable and unclear
        assert schema_enum == VALID_PRIMARY

    def test_schema_secondary_traits_restricted_to_construct_traits(self):
        from modal_apps.annotate_ethics_items import ANNOTATION_SCHEMA
        item_enum = set(
            ANNOTATION_SCHEMA["properties"]["secondary_traits"]["items"]["enum"]
        )
        # Secondary traits must never include not_applicable or unclear
        assert item_enum == CONSTRUCT_TRAITS

    def test_schema_excludes_ethics_split_names(self):
        from modal_apps.annotate_ethics_items import ANNOTATION_SCHEMA
        primary_enum = set(
            ANNOTATION_SCHEMA["properties"]["primary_trait"]["enum"]
        )
        for split in _ETHICS_SPLIT_NAMES:
            assert split not in primary_enum, (
                f"ETHICS split '{split}' must not appear in primary_trait enum"
            )

    def test_schema_confidence_enum(self):
        from modal_apps.annotate_ethics_items import ANNOTATION_SCHEMA
        assert set(
            ANNOTATION_SCHEMA["properties"]["annotation_confidence"]["enum"]
        ) == {"high", "medium", "low"}

    def test_schema_no_additional_properties(self):
        from modal_apps.annotate_ethics_items import ANNOTATION_SCHEMA
        assert ANNOTATION_SCHEMA.get("additionalProperties") is False


# ---------------------------------------------------------------------------
# Simulated vLLM output — happy path
# ---------------------------------------------------------------------------

class TestSimulatedVLLMHappyPath:
    """Simulate what vLLM guided decoding returns and verify the downstream
    parse + validate pipeline accepts it correctly."""

    def test_valid_construct_trait(self):
        raw = _valid_json("harmlessness", confidence="high", keep=True)
        parsed = parse_llm_response(raw)
        validated = validate_parsed_annotation(parsed)
        assert validated["primary_trait"] == "harmlessness"
        assert validated["keep_for_mvp"] == "true"
        assert validated["annotation_confidence"] == "high"

    def test_valid_with_secondary_traits(self):
        raw = _valid_json("fairness", secondary=["compassion"], confidence="medium")
        parsed = parse_llm_response(raw)
        validated = validate_parsed_annotation(parsed)
        assert "compassion" in validated["secondary_traits"]

    def test_all_four_construct_traits_accepted(self):
        for trait in ["honesty", "harmlessness", "fairness", "compassion"]:
            raw = _valid_json(trait)
            validated = validate_parsed_annotation(parse_llm_response(raw))
            assert validated["primary_trait"] == trait

    def test_not_applicable_accepted(self):
        raw = _valid_json("not_applicable", secondary=[], keep=False, confidence="high")
        validated = validate_parsed_annotation(parse_llm_response(raw))
        assert validated["primary_trait"] == "not_applicable"
        assert validated["keep_for_mvp"] == "false"

    def test_unclear_accepted(self):
        raw = _valid_json("unclear", secondary=[], keep=False, confidence="medium")
        validated = validate_parsed_annotation(parse_llm_response(raw))
        assert validated["primary_trait"] == "unclear"
        assert validated["keep_for_mvp"] == "false"

    def test_annotation_notes_included(self):
        raw = _valid_json(notes="Item is a classic deception scenario.")
        validated = validate_parsed_annotation(parse_llm_response(raw))
        assert "deception" in validated["annotation_notes"]


# ---------------------------------------------------------------------------
# Business-rule enforcement on simulated vLLM output
# ---------------------------------------------------------------------------

class TestBusinessRulesOnSimulatedOutput:
    """Even if guided decoding produces technically-valid JSON, the business
    rules in validate_parsed_annotation must still be enforced."""

    @pytest.mark.parametrize("split_name", sorted(_ETHICS_SPLIT_NAMES))
    def test_ethics_split_name_rejected_as_primary(self, split_name):
        # This can only happen in non-strict mode (schema prevents it normally)
        raw = json.dumps({
            "primary_trait": split_name,
            "secondary_traits": [],
            "annotation_confidence": "high",
            "keep_for_mvp": True,
            "annotation_notes": "wrong label",
        })
        parsed = parse_llm_response(raw)
        with pytest.raises(ValueError, match="ETHICS source-split"):
            validate_parsed_annotation(parsed)

    @pytest.mark.parametrize("split_name", sorted(_ETHICS_SPLIT_NAMES))
    def test_ethics_split_name_rejected_in_secondary(self, split_name):
        raw = json.dumps({
            "primary_trait": "honesty",
            "secondary_traits": [split_name],
            "annotation_confidence": "medium",
            "keep_for_mvp": True,
            "annotation_notes": "wrong secondary",
        })
        parsed = parse_llm_response(raw)
        with pytest.raises(ValueError, match="ETHICS split"):
            validate_parsed_annotation(parsed)

    def test_not_applicable_forces_keep_false(self):
        raw = _valid_json("not_applicable", secondary=[], keep=True, confidence="high")
        validated = validate_parsed_annotation(parse_llm_response(raw))
        assert validated["keep_for_mvp"] == "false"

    def test_unclear_forces_keep_false(self):
        raw = _valid_json("unclear", secondary=[], keep=True, confidence="medium")
        validated = validate_parsed_annotation(parse_llm_response(raw))
        assert validated["keep_for_mvp"] == "false"

    def test_low_confidence_forces_keep_false(self):
        raw = _valid_json("compassion", confidence="low", keep=True)
        validated = validate_parsed_annotation(parse_llm_response(raw))
        assert validated["keep_for_mvp"] == "false"

    def test_not_applicable_with_secondary_rejected(self):
        raw = json.dumps({
            "primary_trait": "not_applicable",
            "secondary_traits": ["fairness"],
            "annotation_confidence": "high",
            "keep_for_mvp": False,
            "annotation_notes": "n/a but has secondary",
        })
        with pytest.raises(ValueError, match="secondary_traits must be empty"):
            validate_parsed_annotation(parse_llm_response(raw))

    def test_primary_in_secondary_rejected(self):
        raw = json.dumps({
            "primary_trait": "fairness",
            "secondary_traits": ["fairness"],
            "annotation_confidence": "medium",
            "keep_for_mvp": True,
            "annotation_notes": "duplicate",
        })
        with pytest.raises(ValueError, match="must not appear"):
            validate_parsed_annotation(parse_llm_response(raw))


# ---------------------------------------------------------------------------
# Non-strict fallback (simulates --non-strict mode)
# ---------------------------------------------------------------------------

class TestNonStrictFallback:
    """The local_entrypoint catches validate errors and calls _fallback_annotation
    in non-strict mode.  Simulate that pattern here."""

    def _process(self, raw: str, strict: bool = False) -> dict:
        try:
            return validate_parsed_annotation(parse_llm_response(raw))
        except (ValueError, KeyError) as exc:
            if strict:
                raise
            return _fallback_annotation(str(exc))

    def test_empty_json_falls_back(self):
        result = self._process("{}")
        assert result["primary_trait"] == "unclear"
        assert result["keep_for_mvp"] == "false"
        assert result["annotation_confidence"] == "low"

    def test_ethics_split_as_primary_falls_back(self):
        raw = json.dumps({
            "primary_trait": "commonsense",
            "secondary_traits": [],
            "annotation_confidence": "high",
            "keep_for_mvp": True,
            "annotation_notes": "wrong",
        })
        result = self._process(raw)
        assert result["primary_trait"] == "unclear"

    def test_completely_unparseable_falls_back(self):
        result = self._process("this is not json at all")
        assert result["primary_trait"] == "unclear"
        assert result["keep_for_mvp"] == "false"

    def test_fallback_note_mentions_failure(self):
        result = _fallback_annotation("guided decoding produced empty output")
        assert "failed" in result["annotation_notes"].lower()

    def test_strict_mode_raises(self):
        with pytest.raises((ValueError, KeyError)):
            self._process("{}", strict=True)


# ---------------------------------------------------------------------------
# Modal app module is importable without Modal installed
# ---------------------------------------------------------------------------

class TestModalModuleImportability:
    def test_modal_app_module_importable(self):
        """The module must not crash on import even if modal is not installed."""
        assert MODAL_APP_PATH.exists(), "modal_apps/annotate_ethics_items.py not found"
        # If modal is installed, the import will work normally.
        # If not installed, the try/except at the top of the module handles it.
        import modal_apps.annotate_ethics_items as m
        assert hasattr(m, "ANNOTATION_SCHEMA")
        assert hasattr(m, "DEFAULT_MODEL")

    def test_default_model_is_set(self):
        from modal_apps.annotate_ethics_items import DEFAULT_MODEL
        assert "Qwen" in DEFAULT_MODEL or DEFAULT_MODEL  # non-empty sensible default

    def test_wrapper_script_exists(self):
        assert WRAPPER_SCRIPT.exists(), "scripts/auto_annotate_items_modal.py not found"

    def test_modal_app_no_torch_at_module_level(self):
        """torch and transformers must not be imported at module level
        (they are only used inside the Modal function body)."""
        import modal_apps.annotate_ethics_items as m
        assert not hasattr(m, "torch")
        assert not hasattr(m, "transformers")

    def test_modal_app_no_vllm_at_module_level(self):
        import modal_apps.annotate_ethics_items as m
        assert not hasattr(m, "LLM")
        assert not hasattr(m, "vllm")

    def test_modal_availability_flag_exists(self):
        from modal_apps.annotate_ethics_items import _MODAL_AVAILABLE
        assert isinstance(_MODAL_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# No GPU/torch/Modal imports in the normal test path
# ---------------------------------------------------------------------------

class TestNoDependenciesInTestPath:
    def test_auto_annotation_module_no_torch(self):
        import src.data.auto_annotation as m
        assert not hasattr(m, "torch")
        assert not hasattr(m, "transformers")
        assert not hasattr(m, "modal")

    def test_test_suite_does_not_import_torch(self):
        """torch should not be importable as a side effect of any test module."""
        assert "torch" not in sys.modules

    def test_test_suite_does_not_import_vllm(self):
        assert "vllm" not in sys.modules

    def test_test_suite_does_not_import_anthropic(self):
        assert "anthropic" not in sys.modules


# ---------------------------------------------------------------------------
# Prompt construction (same functions used by the Modal app)
# ---------------------------------------------------------------------------

class TestPromptForModalApp:
    def test_system_prompt_contains_trait_names(self, trait_space):
        system = build_system_prompt(trait_space)
        for trait in CONSTRUCT_TRAITS:
            assert trait.upper() in system

    def test_system_prompt_warns_against_ethics_splits(self, trait_space):
        system = build_system_prompt(trait_space)
        assert "ETHICS" in system or "commonsense" in system

    def test_system_prompt_contains_json_schema_fields(self, trait_space):
        system = build_system_prompt(trait_space)
        assert "primary_trait" in system
        assert "annotation_confidence" in system

    def test_user_prompt_contains_scenario(self, trait_space):
        row = _row(scenario_text="He took money from the till.")
        _, user = build_annotation_prompt(row, trait_space)
        assert "He took money from the till." in user

    def test_user_prompt_contains_source_split(self, trait_space):
        row = _row(source_split="justice")
        _, user = build_annotation_prompt(row, trait_space)
        assert "justice" in user

    def test_prompt_tuple_returns_two_strings(self, trait_space):
        row = _row()
        result = build_annotation_prompt(row, trait_space)
        assert len(result) == 2
        system, user = result
        assert isinstance(system, str) and len(system) > 0
        assert isinstance(user, str) and len(user) > 0


# ---------------------------------------------------------------------------
# Manual fallback workflow still works
# ---------------------------------------------------------------------------

class TestManualFallbackWorkflow:
    """The Anthropic-API fallback path (auto_annotate_items.py) must remain
    functional.  These tests verify the key functions it relies on still work."""

    def test_parse_llm_response_still_works(self):
        raw = _valid_json("compassion")
        parsed = parse_llm_response(raw)
        assert parsed["primary_trait"] == "compassion"

    def test_validate_parsed_annotation_still_works(self):
        parsed = {
            "primary_trait": "honesty",
            "secondary_traits": [],
            "annotation_confidence": "high",
            "keep_for_mvp": True,
            "annotation_notes": "Classic deception.",
        }
        result = validate_parsed_annotation(parsed)
        assert result["primary_trait"] == "honesty"

    def test_fallback_annotation_still_works(self):
        result = _fallback_annotation("test reason")
        assert result["primary_trait"] == "unclear"
        assert result["keep_for_mvp"] == "false"
        assert result["annotation_confidence"] == "low"

    def test_anthropic_script_exists(self):
        script = Path(__file__).parent.parent / "scripts" / "auto_annotate_items.py"
        assert script.exists(), "Anthropic fallback script must remain in place"
