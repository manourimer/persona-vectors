"""
Lightweight one-facet Generalizability Theory helpers.

Terminology:
  universe score variance = between_item variance (stable differences between items)
  error variance          = within_item variance (paraphrase / wording noise)
  G(k) = universe_score_variance / (universe_score_variance + error_variance / k)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def g_coefficient(universe_score_var: float, error_var: float, k: int) -> float:
    """
    G(k) = universe_score_var / (universe_score_var + error_var / k)

    Clamped to [0, 1].
    """
    denom = universe_score_var + error_var / k if k > 0 else 0.0
    if denom <= 0.0:
        return 0.0
    return float(np.clip(universe_score_var / denom, 0.0, 1.0))


def d_study(
    universe_score_var: float,
    error_var: float,
    n_paraphrases_list: list[int],
) -> pd.DataFrame:
    """
    Decision (D) study: G coefficient as a function of number of paraphrases.

    Returns DataFrame with columns:
        n_paraphrases, g_coefficient, universe_score_var, error_var_per_k
    """
    rows = []
    for k in n_paraphrases_list:
        g = g_coefficient(universe_score_var, error_var, k)
        rows.append({
            "n_paraphrases": k,
            "g_coefficient": g,
            "universe_score_var": universe_score_var,
            "error_var_per_k": error_var / k if k > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def run_d_study_for_all(results: list, n_paraphrases_list: list[int]) -> pd.DataFrame:
    """
    Run D-study for every ReliabilityResult.

    Parameters
    ----------
    results : list of ReliabilityResult (from reliability_analysis.py)
    n_paraphrases_list : list of k values to evaluate

    Returns
    -------
    Long-format DataFrame: layer, projected_trait, n_paraphrases, g_coefficient
    """
    frames = []
    for r in results:
        vc = r.variance_components
        df = d_study(vc.between_item_var, vc.within_item_var, n_paraphrases_list)
        df["layer"] = r.layer
        df["projected_trait"] = r.projected_trait
        frames.append(df[["layer", "projected_trait", "n_paraphrases", "g_coefficient"]])
    if not frames:
        return pd.DataFrame(columns=["layer", "projected_trait", "n_paraphrases", "g_coefficient"])
    return pd.concat(frames, ignore_index=True)
