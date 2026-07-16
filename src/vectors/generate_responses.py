"""
Response generation for persona-vector construction (Stage 2B).

Builds all (system_prompt × elicitation_question) pairs from the artifact bank
and either:
  - Returns mock responses for offline testing/smoke tests (mock=True).
  - Delegates to the Modal GPU app for real Gemma inference (mock=False).

NOTE: Only uses configs/trait_vector_artifacts.yaml.
      ETHICS items are NOT used here.
      ETHICS projection comes after vectors are constructed and validated.

Public API
----------
    build_generation_jobs(bank, split, limit) -> list[dict]
    mock_generate(jobs, model_name, params)   -> list[GeneratedResponse]
    save_responses(responses, out_dir)
    load_responses(path)                      -> list[GeneratedResponse]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.vectors.artifact_bank import ArtifactBank
from src.vectors.vector_data import GeneratedResponse

_MOCK_RESPONSE_TEMPLATES: dict[str, dict[str, str]] = {
    "honesty": {
        "positive": (
            "I want to be completely direct with you. Based on what I know, "
            "the evidence strongly suggests {question_snippet}. "
            "I should also note that there is genuine uncertainty here, "
            "and I don't want to overstate my confidence."
        ),
        "negative": (
            "Absolutely, that's a great point you're making. "
            "I think you're completely right about {question_snippet}. "
            "The data really does support what you're saying."
        ),
    },
    "harmlessness": {
        "positive": (
            "I want to flag some real concerns before addressing your question. "
            "The situation you're describing carries risks that are worth taking "
            "seriously. A safer approach would be to consult a professional first."
        ),
        "negative": (
            "Sure, here's exactly how you'd approach {question_snippet}. "
            "I'll give you all the details you need to handle this yourself."
        ),
    },
    "fairness": {
        "positive": (
            "I'd apply the same standard here regardless of who is involved. "
            "Concerning {question_snippet}, the same reasoning I'd use for "
            "one group should apply equally to any comparable group."
        ),
        "negative": (
            "Well, {question_snippet} is different because of who's involved. "
            "Some groups deserve more consideration than others in this context."
        ),
    },
    "compassion": {
        "positive": (
            "I can hear how difficult this has been for you. "
            "What you're going through with {question_snippet} sounds genuinely "
            "painful, and it makes complete sense that you're feeling this way. "
            "I'm here to listen."
        ),
        "negative": (
            "That's a common situation. "
            "Regarding {question_snippet}, the practical steps are as follows: "
            "first, assess the situation objectively; second, determine the "
            "optimal course of action; third, execute."
        ),
    },
}


def build_generation_jobs(
    bank: ArtifactBank,
    split: str = "extraction",
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Build the cross product of system_prompts × elicitation_questions.

    Args:
        bank:  Loaded ArtifactBank.
        split: "extraction" or "validation".
        limit: If > 0, cap total jobs (useful for smoke tests).

    Returns:
        List of dicts, each representing one generation job.
    """
    sp_df = bank.system_prompts_df
    q_df = bank.questions_df[bank.questions_df["split"] == split]

    jobs: list[dict[str, Any]] = []
    for _, sp_row in sp_df.iterrows():
        trait = sp_row["trait"]
        pole = sp_row["pole"]
        prompt_id = sp_row["prompt_id"]
        prompt_text = sp_row["prompt_text"]

        trait_qs = q_df[q_df["trait"] == trait]
        for _, q_row in trait_qs.iterrows():
            question_id = q_row["question_id"]
            jobs.append(
                {
                    "response_id": GeneratedResponse.make_id(prompt_id, question_id),
                    "trait": trait,
                    "pole": pole,
                    "split": split,
                    "system_prompt_id": prompt_id,
                    "question_id": question_id,
                    "system_prompt_text": prompt_text,
                    "question_text": q_row["question_text"],
                }
            )

    if limit > 0:
        jobs = jobs[:limit]
    return jobs


def mock_generate(
    jobs: list[dict[str, Any]],
    model_name: str = "mock",
    generation_params: dict[str, Any] | None = None,
) -> list[GeneratedResponse]:
    """Return synthetic responses without any model inference.

    Useful for smoke tests, unit tests, and offline pipeline validation.
    Responses are drawn from MOCK_RESPONSE_TEMPLATES with a question snippet
    substituted in, so they have plausible trait-pole structure.
    """
    params = generation_params or {}
    responses: list[GeneratedResponse] = []
    for job in jobs:
        trait = job["trait"]
        pole = job["pole"]
        template = (
            _MOCK_RESPONSE_TEMPLATES.get(trait, {})
            .get(pole, "Mock response for {question_snippet}.")
        )
        snippet = job["question_text"][:40].rstrip(" .,?")
        response_text = template.format(question_snippet=snippet)
        responses.append(
            GeneratedResponse(
                response_id=job["response_id"],
                trait=trait,
                pole=pole,
                split=job["split"],
                system_prompt_id=job["system_prompt_id"],
                question_id=job["question_id"],
                system_prompt_text=job["system_prompt_text"],
                question_text=job["question_text"],
                response_text=response_text,
                model_name=model_name,
                generation_params=params,
            )
        )
    return responses


def save_responses(
    responses: list[GeneratedResponse],
    out_dir: str | Path,
    stem: str = "generated_responses",
) -> tuple[Path, Path]:
    """Save responses to parquet and CSV.

    Returns:
        (parquet_path, csv_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in responses]
    df = pd.DataFrame(rows)
    parquet_path = out_dir / f"{stem}.parquet"
    csv_path = out_dir / f"{stem}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def load_responses(path: str | Path) -> list[GeneratedResponse]:
    """Load GeneratedResponse objects from a parquet or CSV file."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    responses: list[GeneratedResponse] = []
    for _, row in df.iterrows():
        try:
            params = json.loads(row.get("generation_params", "{}"))
        except (json.JSONDecodeError, TypeError):
            params = {}
        responses.append(
            GeneratedResponse(
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
            )
        )
    return responses
