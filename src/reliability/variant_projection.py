"""
Stage 4C: Reliability variant projection job construction.

Loads the accepted variant bank from Stage 4B and builds forward-pass jobs
for last-prompt-token activation extraction (same monitoring scope as Stage 3).

No GPU, torch, Modal, or transformers imports at module level.

Public API
----------
    load_variant_bank(path)                          -> pd.DataFrame
    build_projection_jobs(variant_df, target_layers, token_position) -> pd.DataFrame
    validate_projection_jobs(jobs_df)                -> dict
    save_projection_jobs(jobs_df, out_dir, stem)     -> tuple[Path, Path]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Required variant bank columns
# ---------------------------------------------------------------------------

REQUIRED_VARIANT_COLUMNS: list[str] = [
    "item_id",
    "variant_id",
    "variant_type",
    "paraphrase_id",
    "framing",
    "source_split",
    "primary_trait",
    "scenario_text_original",
    "scenario_text_variant",
    "keep_variant",
]

# Prompt template — identical to Stage 3 (ethics_projection.py).
# The "Answer:" suffix marks the last prompt token position.
# No response is generated.
_PROMPT_TEMPLATE = (
    "You will be shown a moral scenario. Consider the scenario carefully.\n\n"
    "Scenario:\n{scenario_text}\n\n"
    "Answer:"
)


def build_prompt(scenario_text: str) -> str:
    """Format a scenario variant as a prompt for last-prompt-token extraction.

    Uses the same template as Stage 3 ethics_projection.build_prompt so that
    activation extraction is comparable across stages.
    """
    return _PROMPT_TEMPLATE.format(scenario_text=scenario_text.strip())


# ---------------------------------------------------------------------------
# Variant bank loading
# ---------------------------------------------------------------------------


def load_variant_bank(path: str | Path) -> pd.DataFrame:
    """Load the reliability variant bank and filter to accepted variants.

    Args:
        path: Path to the parquet (or CSV) variant bank produced by Stage 4B.

    Returns:
        DataFrame filtered to keep_variant == True, with all required columns.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Variant bank not found: {path}\n"
            "Run scripts/export_reliability_variant_bank.py first."
        )

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_VARIANT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Variant bank missing required columns: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Normalise keep_variant to bool
    if df["keep_variant"].dtype == object:
        df["keep_variant"] = df["keep_variant"].astype(str).str.lower() == "true"
    else:
        df["keep_variant"] = df["keep_variant"].astype(bool)

    filtered = df[df["keep_variant"]].reset_index(drop=True)
    return filtered


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------


def build_projection_jobs(
    variant_df: pd.DataFrame,
    target_layers: list[int],
    token_position: str = "last_prompt_token",
) -> pd.DataFrame:
    """Build a DataFrame of forward-pass jobs for reliability variant extraction.

    Each row is one (variant, layer) job.  prompt_text is built from
    scenario_text_variant (not scenario_text_original).

    Args:
        variant_df:    Filtered variant bank from load_variant_bank.
        target_layers: Layer indices to extract at (e.g. [32, 40, 47]).
        token_position: Token scope label (always "last_prompt_token").

    Returns:
        DataFrame with columns:
            item_id, variant_id, variant_type, paraphrase_id, framing,
            source_split, primary_trait, scenario_text_original,
            scenario_text_variant, prompt_text, token_position
        Note: target_layer is stored in a separate column "target_layer"
        when multiple layers are passed; each variant × layer is one row.
    """
    rows: list[dict] = []
    for _, var in variant_df.iterrows():
        prompt_text = build_prompt(str(var["scenario_text_variant"]))
        for layer in target_layers:
            rows.append(
                {
                    "item_id": var["item_id"],
                    "variant_id": var["variant_id"],
                    "variant_type": var["variant_type"],
                    "paraphrase_id": var["paraphrase_id"],
                    "framing": var["framing"],
                    "source_split": var["source_split"],
                    "primary_trait": var["primary_trait"],
                    "scenario_text_original": var["scenario_text_original"],
                    "scenario_text_variant": var["scenario_text_variant"],
                    "prompt_text": prompt_text,
                    "target_layer": layer,
                    "token_position": token_position,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_projection_jobs(jobs_df: pd.DataFrame) -> dict:
    """Validate projection jobs DataFrame and return a summary dict.

    Checks:
        - No missing item_id or variant_id.
        - scenario_text_variant is present in prompt_text (not original only).
        - prompt_text exists for all rows.

    Returns:
        dict with keys: n_jobs, n_variants, n_items, n_layers, warnings, valid
    """
    warnings: list[str] = []

    n_jobs = len(jobs_df)
    n_variants = jobs_df["variant_id"].nunique() if "variant_id" in jobs_df.columns else 0
    n_items = jobs_df["item_id"].nunique() if "item_id" in jobs_df.columns else 0
    n_layers = jobs_df["target_layer"].nunique() if "target_layer" in jobs_df.columns else 0

    # Check for null item_id / variant_id
    if jobs_df["item_id"].isnull().any():
        warnings.append("Some rows have null item_id.")
    if jobs_df["variant_id"].isnull().any():
        warnings.append("Some rows have null variant_id.")

    # Verify scenario_text_variant is used in prompt_text (spot-check first 10 rows)
    sample = jobs_df.head(10)
    for _, row in sample.iterrows():
        variant_text = str(row["scenario_text_variant"]).strip()[:50]
        if variant_text and variant_text not in str(row["prompt_text"]):
            warnings.append(
                f"variant_id={row['variant_id']}: scenario_text_variant not found in prompt_text."
            )
            break

    # Check prompt_text is non-null
    if jobs_df["prompt_text"].isnull().any():
        warnings.append("Some rows have null prompt_text.")

    valid = len(warnings) == 0

    return {
        "n_jobs": n_jobs,
        "n_variants": n_variants,
        "n_items": n_items,
        "n_layers": n_layers,
        "warnings": warnings,
        "valid": valid,
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_projection_jobs(
    jobs_df: pd.DataFrame,
    out_dir: str | Path,
    stem: str = "reliability_projection_jobs",
) -> tuple[Path, Path]:
    """Save projection jobs as parquet + CSV.

    Returns:
        (parquet_path, csv_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq_path = out_dir / f"{stem}.parquet"
    csv_path = out_dir / f"{stem}.csv"
    jobs_df.to_parquet(pq_path, index=False)
    jobs_df.to_csv(csv_path, index=False)
    return pq_path, csv_path
