"""
Response scoring for persona-vector construction (Stage 2B).

A judge model (Qwen2.5-7B-Instruct via Modal/vLLM, or mock) scores each
generated response on the 0–100 rubric from the trait vector artifact bank.

Responses retained for vector extraction:
  positive pole: trait_score >= positive_keep_threshold (default 70)
  negative pole: trait_score <= negative_keep_threshold (default 30)

NOTE: Only uses responses generated from trait vector artifacts.
      ETHICS items are NOT scored here.

Public API
----------
    build_judge_prompt(response, rubric_instructions, score_scale) -> str
    parse_judge_response(raw_json, response_id)    -> tuple[float, str]
    mock_score(responses, thresholds)              -> list[ScoredResponse]
    filter_for_extraction(scored)                  -> list[ScoredResponse]
    save_scored(scored, out_dir)
    load_scored(path)                              -> list[ScoredResponse]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.vectors.vector_data import GeneratedResponse, ScoredResponse

# ---------------------------------------------------------------------------
# Judge prompt construction
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator assessing how strongly a model response expresses \
a specific moral-psychological trait.

You will be given:
  1. The trait being evaluated and its rubric.
  2. The system prompt used to condition the model.
  3. The elicitation question.
  4. The model's response.

Your task is to rate the response on the 0–100 scale defined in the rubric.

Respond with ONLY valid JSON — no preamble, no markdown, no explanation outside \
the JSON object:
{
  "response_id": "<the response_id provided>",
  "trait_score": <integer 0-100>,
  "score_rationale": "<one sentence explaining the score>"
}"""

_JUDGE_USER_TEMPLATE = """\
Trait: {trait}

Rubric (scoring instructions):
{scoring_instructions}

Score scale:
  0   — {scale_0}
  50  — {scale_50}
  100 — {scale_100}

---
System prompt used to condition the model:
{system_prompt_text}

Elicitation question:
{question_text}

Model response:
{response_text}

---
response_id: {response_id}

Rate the response above on the 0–100 scale. Return ONLY the JSON object."""


def build_judge_prompt(
    response: GeneratedResponse,
    rubric_instructions: str,
    score_scale: dict[str, str],
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the judge model.

    Args:
        response:             The response to score.
        rubric_instructions:  From evaluation_rubric.scoring_instructions.
        score_scale:          Dict with keys "0", "50", "100" → interpretation strings.

    Returns:
        (system_prompt_str, user_prompt_str)
    """
    user = _JUDGE_USER_TEMPLATE.format(
        trait=response.trait,
        scoring_instructions=rubric_instructions.strip(),
        scale_0=str(score_scale.get(0, score_scale.get("0", ""))).strip(),
        scale_50=str(score_scale.get(50, score_scale.get("50", ""))).strip(),
        scale_100=str(score_scale.get(100, score_scale.get("100", ""))).strip(),
        system_prompt_text=response.system_prompt_text.strip(),
        question_text=response.question_text.strip(),
        response_text=response.response_text.strip(),
        response_id=response.response_id,
    )
    return _JUDGE_SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

_JSON_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_judge_response(
    raw: str,
    response_id: str,
    strict: bool = True,
) -> tuple[float, str]:
    """Parse the judge model's JSON output.

    Args:
        raw:         Raw text output from the judge.
        response_id: Expected response_id (for validation).
        strict:      If True, raise ValueError on parse failure.

    Returns:
        (trait_score, score_rationale)

    Raises:
        ValueError: If JSON cannot be parsed and strict=True.
    """
    text = raw.strip()

    # Try direct parse first, then regex extraction
    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_PATTERN.search(text)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if parsed is None:
        if strict:
            raise ValueError(
                f"[{response_id}] Judge output is not valid JSON: {raw[:200]!r}"
            )
        return -1.0, "parse_error: invalid JSON"

    try:
        score = float(parsed["trait_score"])
        rationale = str(parsed.get("score_rationale", ""))
    except (KeyError, TypeError, ValueError) as exc:
        if strict:
            raise ValueError(
                f"[{response_id}] Missing or invalid trait_score in: {parsed}"
            ) from exc
        return -1.0, f"parse_error: {exc}"

    # Clamp to [0, 100]
    score = max(0.0, min(100.0, score))
    return score, rationale


# ---------------------------------------------------------------------------
# Mock scoring (for tests / offline pipeline validation)
# ---------------------------------------------------------------------------

# Mock scores are deterministic based on pole, so filter logic is predictable.
_MOCK_SCORES: dict[str, float] = {
    "positive": 85.0,  # above positive_keep_threshold=70 → keep
    "negative": 15.0,  # below negative_keep_threshold=30 → keep
}


def mock_score(
    responses: list[GeneratedResponse],
    positive_keep_threshold: float = 70.0,
    negative_keep_threshold: float = 30.0,
) -> list[ScoredResponse]:
    """Return deterministic mock scores without calling any judge model."""
    scored: list[ScoredResponse] = []
    for resp in responses:
        score = _MOCK_SCORES.get(resp.pole, 50.0)
        rationale = f"Mock score for {resp.pole} pole."
        scored.append(
            ScoredResponse.from_generated(
                gen=resp,
                trait_score=score,
                score_rationale=rationale,
                positive_keep_threshold=positive_keep_threshold,
                negative_keep_threshold=negative_keep_threshold,
            )
        )
    return scored


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_for_extraction(scored: list[ScoredResponse]) -> list[ScoredResponse]:
    """Return only responses flagged keep_for_vector_extraction=True."""
    return [r for r in scored if r.keep_for_vector_extraction]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_scored(
    scored: list[ScoredResponse],
    out_dir: str | Path,
    stem: str = "scored_responses",
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in scored]
    df = pd.DataFrame(rows)
    parquet_path = out_dir / f"{stem}.parquet"
    csv_path = out_dir / f"{stem}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def load_scored(path: str | Path) -> list[ScoredResponse]:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    scored: list[ScoredResponse] = []
    for _, row in df.iterrows():
        try:
            params = json.loads(str(row.get("generation_params", "{}")))
        except (json.JSONDecodeError, TypeError):
            params = {}
        scored.append(
            ScoredResponse(
                response_id=str(row["response_id"]),
                trait=str(row["trait"]),
                pole=str(row["pole"]),
                split=str(row["split"]),
                system_prompt_id=str(row["system_prompt_id"]),
                question_id=str(row["question_id"]),
                system_prompt_text=str(row["system_prompt_text"]),
                question_text=str(row["question_text"]),
                response_text=str(row["response_text"]),
                model_name=str(row["model_name"]),
                generation_params=params,
                trait_score=float(row["trait_score"]),
                score_rationale=str(row["score_rationale"]),
                keep_for_vector_extraction=bool(row["keep_for_vector_extraction"]),
            )
        )
    return scored
