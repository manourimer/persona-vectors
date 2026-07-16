"""
Synonym / construct-neighbor convergent-validity control.

Question: Are results robust to reasonable rewordings of the construct names?
Expected: Synonym vectors should be cosine-closest to their intended parent trait,
and their ETHICS projections should correlate with the original parent-trait projection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
try:
    from scipy.stats import pearsonr, spearmanr
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_synonym_config(config_path: str | Path) -> dict:
    """
    Loads configs/synonym_vector_artifacts.yaml.
    Returns dict mapping synonym_trait_id -> {parent_trait, construct_name, ...}
    """
    import yaml
    config_path = Path(config_path)
    with open(config_path) as f:
        data = yaml.safe_load(f)

    result = {}
    for key, val in data.items():
        if isinstance(val, dict):
            synonym_id = val.get("synonym_trait_id", key)
            result[synonym_id] = {
                "parent_trait": val.get("parent_trait"),
                "construct_name": val.get("construct_name", key),
                "abbreviation": val.get("abbreviation", key[:3]),
                "description": val.get("description", ""),
            }
    return result


# ---------------------------------------------------------------------------
# Vector similarity
# ---------------------------------------------------------------------------


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    vec_a = np.asarray(vec_a, dtype=float).ravel()
    vec_b = np.asarray(vec_b, dtype=float).ravel()
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compare_synonym_to_originals(
    synonym_vec: np.ndarray,
    original_vectors_dict: dict,
) -> dict:
    """
    Returns cosine similarity of synonym_vec to each of the 4 original trait vectors,
    and identifies closest_parent.

    original_vectors_dict: {trait_name: np.ndarray}
    """
    sims = {trait: compute_cosine_similarity(synonym_vec, vec)
            for trait, vec in original_vectors_dict.items()}
    closest_parent = max(sims, key=lambda t: sims[t]) if sims else None
    return {**sims, "closest_parent": closest_parent}


# ---------------------------------------------------------------------------
# Projection agreement
# ---------------------------------------------------------------------------


def compute_projection_agreement(
    original_projections_series: pd.Series,
    synonym_projections_series: pd.Series,
) -> dict:
    """
    Pearson correlation, Spearman correlation, mean absolute deviation
    between the two projection series.
    """
    aligned = pd.DataFrame({
        "original": original_projections_series,
        "synonym": synonym_projections_series,
    }).dropna()

    if len(aligned) < 3:
        return {"pearson_r": float("nan"), "spearman_r": float("nan"), "mean_abs_dev": float("nan")}

    x = aligned["original"].values.astype(float)
    y = aligned["synonym"].values.astype(float)
    mad = float(np.abs(x - y).mean())

    if _HAVE_SCIPY:
        pr, _ = pearsonr(x, y)
        sr, _ = spearmanr(x, y)
    else:
        # numpy fallback for pearson correlation
        xm = x - x.mean()
        ym = y - y.mean()
        denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
        pr = float((xm * ym).sum() / denom) if denom > 0 else float("nan")
        # spearman via rank correlation (numpy argsort of argsort = rank)
        xr = np.argsort(np.argsort(x)).astype(float)
        yr = np.argsort(np.argsort(y)).astype(float)
        xrm = xr - xr.mean()
        yrm = yr - yr.mean()
        denom_r = np.sqrt((xrm ** 2).sum() * (yrm ** 2).sum())
        sr = float((xrm * yrm).sum() / denom_r) if denom_r > 0 else float("nan")
    return {"pearson_r": float(pr), "spearman_r": float(sr), "mean_abs_dev": mad}


# ---------------------------------------------------------------------------
# Analysis runners
# ---------------------------------------------------------------------------


def run_synonym_similarity_analysis(
    synonym_vectors_dict: dict,
    original_vectors_dict: dict,
) -> pd.DataFrame:
    """
    For each synonym vector: compare to all 4 originals, compute cosine similarities.

    synonym_vectors_dict: {synonym_id: {"vector": np.ndarray, "parent_trait": str, ...}}
    original_vectors_dict: {trait_name: np.ndarray}

    Returns DataFrame: synonym_id, parent_trait, cosine_honesty, cosine_harmlessness,
                       cosine_fairness, cosine_compassion, closest_parent, closest_matches_parent
    """
    rows = []
    for synonym_id, info in synonym_vectors_dict.items():
        vec = info["vector"] if isinstance(info, dict) else info
        parent_trait = info.get("parent_trait") if isinstance(info, dict) else None
        comparison = compare_synonym_to_originals(vec, original_vectors_dict)
        row = {
            "synonym_id": synonym_id,
            "parent_trait": parent_trait,
            "closest_parent": comparison.get("closest_parent"),
            "closest_matches_parent": comparison.get("closest_parent") == parent_trait,
        }
        for trait in TRAITS:
            row[f"cosine_{trait}"] = comparison.get(trait, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def run_synonym_projection_agreement(
    ethics_wide_df: pd.DataFrame,
    synonym_projections_dict: dict,
    original_col_map: dict,
) -> pd.DataFrame:
    """
    For each synonym: correlation between synonym ETHICS projections and
    original parent-trait ETHICS projections.

    synonym_projections_dict: {synonym_id: {"parent_trait": str, "layer": int, "projections": pd.Series}}
    original_col_map: {trait: col_name in ethics_wide_df}
    """
    rows = []
    for synonym_id, info in synonym_projections_dict.items():
        parent_trait = info.get("parent_trait")
        layer = info.get("layer", "unknown")
        synonym_proj = info.get("projections")

        orig_col = original_col_map.get(parent_trait, f"projection_{parent_trait}")
        if orig_col not in ethics_wide_df.columns or synonym_proj is None:
            continue

        orig_proj = ethics_wide_df[orig_col]
        agreement = compute_projection_agreement(orig_proj, synonym_proj)
        rows.append({
            "synonym_id": synonym_id,
            "parent_trait": parent_trait,
            "layer": layer,
            **agreement,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_synonym_controls(results_dict: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "similarity_df" in results_dict and results_dict["similarity_df"] is not None:
        results_dict["similarity_df"].to_csv(
            out_dir / "synonym_cosine_similarity.csv", index=False
        )
    if "agreement_df" in results_dict and results_dict["agreement_df"] is not None:
        results_dict["agreement_df"].to_csv(
            out_dir / "synonym_projection_agreement.csv", index=False
        )

    # Markdown report
    lines = ["# Synonym Vector Convergent-Validity Control\n\n"]
    if "similarity_df" in results_dict and results_dict["similarity_df"] is not None:
        df = results_dict["similarity_df"]
        n_correct = int(df["closest_matches_parent"].sum()) if "closest_matches_parent" in df.columns else "?"
        n_total = len(df)
        lines.append(f"## Cosine Similarity to Original Vectors\n\n")
        lines.append(f"Closest-parent match: {n_correct}/{n_total}\n\n")
        if hasattr(df, "to_markdown"):
            lines.append(df.to_markdown(index=False))
        else:
            lines.append(df.to_string())
        lines.append("\n\n")

    if "agreement_df" in results_dict and results_dict["agreement_df"] is not None:
        lines.append("## Projection Agreement with Parent Trait\n\n")
        df = results_dict["agreement_df"]
        if hasattr(df, "to_markdown"):
            lines.append(df.to_markdown(index=False))
        else:
            lines.append(df.to_string())
        lines.append("\n")

    (out_dir / "synonym_controls_report.md").write_text("".join(lines))
    print(f"[synonym_controls] Saved to {out_dir}")
