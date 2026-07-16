"""
Stage 4B: Reliability variant generation.

Produces a reliability-variant bank containing:
  - the original item (variant_type="original")
  - N paraphrases per item (variant_type="paraphrase")

Variants are NOT new moral items.  They are alternate presentations of the
same underlying scenario.  The original item's primary_trait and source_split
are inherited by all variants; no new annotation is needed.

Paraphrases must preserve:
  - the same actors
  - the same moral situation / conflict
  - the same morally relevant action
  - the same implied moral stakes
  - the same primary_trait

This module contains only prompt-construction, parsing, validation, and
dataframe-building logic.  All LLM calls live in modal_apps/.

No Modal, torch, or transformers imports at the module level.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

VARIANT_COLUMNS = [
    "item_id",
    "variant_id",
    "variant_type",
    "paraphrase_id",
    "framing",
    "source_split",
    "primary_trait",
    "scenario_text_original",
    "scenario_text_variant",
    "generation_model_name",
    "generation_notes",
    "semantic_equivalence_status",
    "keep_variant",
]

VARIANT_TYPES = {"original", "paraphrase"}
FRAMING_CONDITIONS = {"neutral", "first_person", "third_person"}
SEMANTIC_STATUSES = {
    "original",
    "passed",
    "flagged_length",
    "flagged_duplicate",
    "flagged_possible_meaning_shift",
    "failed_parse",
}

# JSON schema for LLM guided decoding (vLLM)
PARAPHRASE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "item_id": {"type": "string"},
        "paraphrases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paraphrase_id": {"type": "string"},
                    "scenario_text_variant": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["paraphrase_id", "scenario_text_variant"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["item_id", "paraphrases"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Item bank loading
# ---------------------------------------------------------------------------


def load_item_bank(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {"item_id", "scenario_text", "primary_trait", "source_split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Item bank missing columns: {missing}")
    if "keep_for_mvp" in df.columns:
        df = df[df["keep_for_mvp"].astype(str).str.lower().isin({"true", "1", "yes"})]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Original-variant construction
# ---------------------------------------------------------------------------


def make_original_variant(row: pd.Series, framing: str = "neutral") -> dict:
    """Create the canonical 'original' variant row for an item."""
    return {
        "item_id": row["item_id"],
        "variant_id": f"{row['item_id']}__original",
        "variant_type": "original",
        "paraphrase_id": "original",
        "framing": framing,
        "source_split": row["source_split"],
        "primary_trait": row["primary_trait"],
        "scenario_text_original": row["scenario_text"],
        "scenario_text_variant": row["scenario_text"],
        "generation_model_name": "",
        "generation_notes": "",
        "semantic_equivalence_status": "original",
        "keep_variant": True,
    }


def build_original_variants(item_df: pd.DataFrame, framing: str = "neutral") -> pd.DataFrame:
    rows = [make_original_variant(row, framing=framing) for _, row in item_df.iterrows()]
    return pd.DataFrame(rows, columns=VARIANT_COLUMNS)


# ---------------------------------------------------------------------------
# Paraphrase prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise paraphrase generator for moral psychology research.

Your task is to rewrite moral scenarios in different words while keeping
the meaning EXACTLY the same.

Rules:
1. Preserve the same actors (same type of person/people, same relationships).
2. Preserve the same action — who does what to whom.
3. Preserve the same moral stakes — what is at risk and for whom.
4. Do NOT introduce new information, new actors, or new moral conflicts.
5. Do NOT change the moral valence — if something was wrong in the original,
   it must be wrong in the paraphrase.
6. Do NOT negate or reverse the action ("did" → "did not" is forbidden).
7. Vary vocabulary and sentence structure, not meaning.
8. Each paraphrase must be genuinely different from the others in wording.
9. Output ONLY valid JSON. No prose before or after the JSON block.
"""


def build_paraphrase_prompt(
    item_id: str,
    scenario_text: str,
    n_paraphrases: int = 3,
    framing: str = "neutral",
) -> tuple[str, str]:
    """
    Return (system_prompt, user_prompt) for paraphrase generation.

    The user prompt instructs the model to output JSON in the required schema.
    """
    framing_note = ""
    if framing == "first_person":
        framing_note = (
            "\nAdditionally, rewrite in the first person "
            "(the narrator is the moral agent)."
        )
    elif framing == "third_person":
        framing_note = (
            "\nAdditionally, rewrite in the third person "
            "(describe the actor from the outside)."
        )

    user_prompt = f"""\
Generate {n_paraphrases} paraphrases of the following moral scenario.{framing_note}

Original scenario:
\"\"\"{scenario_text}\"\"\"

Output JSON only, in exactly this format:
{{
  "item_id": "{item_id}",
  "paraphrases": [
    {{"paraphrase_id": "p1", "scenario_text_variant": "...", "notes": "..."}},
    {{"paraphrase_id": "p2", "scenario_text_variant": "...", "notes": "..."}},
    {{"paraphrase_id": "p3", "scenario_text_variant": "...", "notes": "..."}}
  ]
}}

Rules:
- Keep the same moral situation, actors, action, and stakes.
- Do not negate or reverse the action.
- Each paraphrase must differ meaningfully in wording from the others.
- notes: one sentence explaining what vocabulary/structure you changed.
"""
    return _SYSTEM_PROMPT, user_prompt


def build_all_prompts(
    item_df: pd.DataFrame,
    n_paraphrases: int = 3,
    framing: str = "neutral",
) -> list[dict]:
    """
    Return a list of dicts with keys: item_id, system_prompt, user_prompt.
    One entry per item in item_df.
    """
    jobs = []
    for _, row in item_df.iterrows():
        sys_p, usr_p = build_paraphrase_prompt(
            item_id=row["item_id"],
            scenario_text=row["scenario_text"],
            n_paraphrases=n_paraphrases,
            framing=framing,
        )
        jobs.append({"item_id": row["item_id"], "system_prompt": sys_p, "user_prompt": usr_p})
    return jobs


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _sanitize_json(raw: str) -> str:
    """
    Replace literal control characters (newlines, tabs, etc.) inside JSON
    string values with their escaped equivalents.

    vLLM sometimes emits literal newlines inside string values, producing
    "Invalid control character" errors in json.loads.  We fix this by
    replacing bare \n, \r, \t inside the raw text with their JSON escapes,
    but only when they appear inside a JSON string (between unescaped quotes).
    """
    # Replace common bare control chars globally — safe because JSON spec
    # requires them to be escaped inside strings anyway, and they cannot
    # appear as structural characters outside strings.
    raw = raw.replace("\r\n", "\\n").replace("\r", "\\n")
    # Replace literal \n and \t that appear inside string runs.
    # We do this by scanning character by character only within quoted regions.
    result = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and in_string:
            # Escape sequence — pass both characters through unchanged
            result.append(ch)
            i += 1
            if i < len(raw):
                result.append(raw[i])
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        if in_string and ch == "\n":
            result.append("\\n")
            i += 1
            continue
        if in_string and ch == "\t":
            result.append("\\t")
            i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def _extract_json(raw: str) -> str:
    """Extract the first JSON object from a string, stripping surrounding prose."""
    raw = raw.strip()
    # Try to extract {...} block if prose surrounds it
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return match.group(0) if match else raw


def parse_paraphrase_response(
    raw: str,
    expected_item_id: str,
    n_paraphrases: int = 3,
    strict: bool = True,
) -> list[dict]:
    """
    Parse a raw LLM JSON response into a list of paraphrase dicts.

    Each dict has keys: paraphrase_id, scenario_text_variant, notes.

    Raises ValueError on parse failure (strict=True) or returns empty list
    (strict=False).
    """
    try:
        text = _extract_json(_sanitize_json(raw))
        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            # Fall back to json_repair for malformed output (e.g. unescaped quotes
            # from long AITA posts containing dialogue or special characters).
            try:
                from json_repair import repair_json  # noqa: PLC0415
                repaired = repair_json(text, return_objects=True)
                if not isinstance(repaired, dict):
                    raise ValueError("json_repair returned non-dict")
                data = repaired
            except Exception as exc:
                if strict:
                    raise ValueError(f"JSON parse failed: {exc}") from exc
                return []
    except (json.JSONDecodeError, AttributeError) as exc:
        if strict:
            raise ValueError(f"JSON parse failed: {exc}") from exc
        return []

    # Validate top-level keys
    if "paraphrases" not in data:
        if strict:
            raise ValueError("Response missing 'paraphrases' key")
        return []

    # item_id mismatch is a warning, not a failure — the model sometimes
    # echoes a shortened version
    returned_id = data.get("item_id", "")
    if returned_id and returned_id != expected_item_id:
        pass  # accept but caller may log

    paraphrases = data["paraphrases"]
    if not isinstance(paraphrases, list):
        if strict:
            raise ValueError("'paraphrases' must be a list")
        return []

    result = []
    for entry in paraphrases:
        if not isinstance(entry, dict):
            continue
        text_variant = entry.get("scenario_text_variant", "").strip()
        if not text_variant:
            continue
        result.append({
            "paraphrase_id": str(entry.get("paraphrase_id", f"p{len(result)+1}")),
            "scenario_text_variant": text_variant,
            "notes": str(entry.get("notes", "")),
        })

    if strict and len(result) < n_paraphrases:
        raise ValueError(
            f"Expected {n_paraphrases} paraphrases, got {len(result)}"
        )

    return result[:n_paraphrases]  # cap to requested count


# ---------------------------------------------------------------------------
# Variant dataframe construction
# ---------------------------------------------------------------------------


def make_paraphrase_variants(
    item_row: pd.Series,
    paraphrases: list[dict],
    framing: str = "neutral",
    generation_model_name: str = "",
    semantic_status_fn=None,
) -> list[dict]:
    """
    Build variant-bank rows for one item's paraphrases.

    semantic_status_fn: optional callable(original_text, variant_text) → status str
    """
    original_text = item_row["scenario_text"]
    rows = []
    for p in paraphrases:
        pid = p["paraphrase_id"]
        variant_text = p["scenario_text_variant"]
        status = (
            semantic_status_fn(original_text, variant_text)
            if semantic_status_fn
            else "passed"
        )
        rows.append({
            "item_id": item_row["item_id"],
            "variant_id": f"{item_row['item_id']}__{pid}",
            "variant_type": "paraphrase",
            "paraphrase_id": pid,
            "framing": framing,
            "source_split": item_row["source_split"],
            "primary_trait": item_row["primary_trait"],
            "scenario_text_original": original_text,
            "scenario_text_variant": variant_text,
            "generation_model_name": generation_model_name,
            "generation_notes": p.get("notes", ""),
            "semantic_equivalence_status": status,
            "keep_variant": status not in {"flagged_possible_meaning_shift", "failed_parse"},
        })
    return rows


def make_failed_parse_rows(
    item_row: pd.Series,
    n_paraphrases: int,
    framing: str = "neutral",
    generation_model_name: str = "",
    error_msg: str = "",
) -> list[dict]:
    """Placeholder rows for items where parsing failed entirely."""
    rows = []
    for i in range(1, n_paraphrases + 1):
        rows.append({
            "item_id": item_row["item_id"],
            "variant_id": f"{item_row['item_id']}__p{i}_failed",
            "variant_type": "paraphrase",
            "paraphrase_id": f"p{i}",
            "framing": framing,
            "source_split": item_row["source_split"],
            "primary_trait": item_row["primary_trait"],
            "scenario_text_original": item_row["scenario_text"],
            "scenario_text_variant": "",
            "generation_model_name": generation_model_name,
            "generation_notes": f"parse_error: {error_msg[:200]}",
            "semantic_equivalence_status": "failed_parse",
            "keep_variant": False,
        })
    return rows


# ---------------------------------------------------------------------------
# Mock generation (for smoke tests, no GPU required)
# ---------------------------------------------------------------------------


def mock_generate_paraphrases(
    item_df: pd.DataFrame,
    n_paraphrases: int = 3,
    framing: str = "neutral",
    generation_model_name: str = "mock",
) -> pd.DataFrame:
    """
    Generate synthetic paraphrases locally for smoke testing.

    Each variant is the original text with a numbered prefix — trivially
    different, not semantically meaningful, but sufficient to exercise the
    pipeline without GPU.
    """
    from src.reliability.variant_validation import check_semantic_equivalence

    all_rows: list[dict] = []

    # Original variants
    orig_df = build_original_variants(item_df, framing=framing)
    all_rows.extend(orig_df.to_dict("records"))

    for _, row in item_df.iterrows():
        paraphrases = [
            {
                "paraphrase_id": f"p{i}",
                "scenario_text_variant": f"[Paraphrase {i}] {row['scenario_text']}",
                "notes": f"Mock paraphrase {i}",
            }
            for i in range(1, n_paraphrases + 1)
        ]
        rows = make_paraphrase_variants(
            item_row=row,
            paraphrases=paraphrases,
            framing=framing,
            generation_model_name=generation_model_name,
            semantic_status_fn=lambda orig, var: check_semantic_equivalence(
                orig, var, max_length_ratio=1.5, min_length_ratio=0.5
            ),
        )
        all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=VARIANT_COLUMNS)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_variants(df: pd.DataFrame, out_dir: str | Path, stem: str = "ethics_reliability_variants") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    parquet_path = out_dir / f"{stem}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def load_variants(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
