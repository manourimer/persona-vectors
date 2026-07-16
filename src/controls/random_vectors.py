"""
Random-vector negative control.

Question: Would arbitrary directions produce similar structure/reliability to moral persona vectors?
Expected: Random vectors may show paraphrase stability by accident, but should not match
real vectors on trait-label alignment or produce the same structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]
PROJ_COLS = [f"projection_{t}" for t in TRAITS]


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def generate_random_unit_vectors(n_vectors: int, dim: int, seed: int) -> np.ndarray:
    """
    Return array of shape (n_vectors, dim) where each row is a unit vector.

    Each vector is drawn from an isotropic Gaussian and normalised to unit norm.
    """
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n_vectors, dim))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


# ---------------------------------------------------------------------------
# Structure helpers (CPU-only, no torch)
# ---------------------------------------------------------------------------


def _effective_dimensionality(proj_matrix: np.ndarray) -> float:
    """
    Participation-ratio effective dimensionality of the projection matrix columns.
    proj_matrix: (n_items, n_traits)
    """
    corr = np.corrcoef(proj_matrix.T)
    try:
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.clip(eigvals, 0, None)
        eigvals = eigvals / eigvals.sum() if eigvals.sum() > 0 else eigvals
        ed = 1.0 / (eigvals ** 2).sum() if (eigvals ** 2).sum() > 0 else 1.0
    except np.linalg.LinAlgError:
        ed = float("nan")
    return float(ed)


def _pc1_variance(proj_matrix: np.ndarray) -> float:
    """Fraction of variance explained by first PC."""
    corr = np.corrcoef(proj_matrix.T)
    try:
        eigvals = np.sort(np.linalg.eigvalsh(corr))[::-1]
        total = eigvals.sum()
        return float(eigvals[0] / total) if total > 0 else float("nan")
    except np.linalg.LinAlgError:
        return float("nan")


def _mean_abs_off_diag_corr(proj_matrix: np.ndarray) -> float:
    """Mean absolute off-diagonal correlation between projection columns."""
    corr = np.corrcoef(proj_matrix.T)
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(np.abs(corr[mask]).mean())


def _reliability_proxy(proj_matrix: np.ndarray) -> float:
    """
    Rough reliability proxy: ratio of between-column variance to total variance.
    Not a true G-coefficient but fast to compute without item groupings.
    """
    col_means = proj_matrix.mean(axis=0)
    between_var = float(np.var(col_means, ddof=1)) if len(col_means) > 1 else 0.0
    total_var = float(np.var(proj_matrix, ddof=1)) if proj_matrix.size > 1 else 1.0
    return float(between_var / total_var) if total_var > 0 else 0.0


# ---------------------------------------------------------------------------
# Main projection function
# ---------------------------------------------------------------------------


def project_onto_random_vectors(
    activations_dict: dict,
    n_repeats: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    For each repeat: generate 4 random unit vectors, project all items, compute structure metrics.

    activations_dict: dict of {layer: np.ndarray} where each array is (n_items, dim)

    Returns long-format DataFrame with columns:
        repeat, layer, effective_dimensionality, pc1_variance,
        mean_abs_off_diag_corr, reliability_g1, reliability_g3
    """
    rows = []
    for layer, acts in activations_dict.items():
        acts = np.array(acts, dtype=float)
        n_items, dim = acts.shape
        for rep in range(n_repeats):
            seed = random_seed * 10_000 + layer * 100 + rep
            rvecs = generate_random_unit_vectors(4, dim, seed)
            proj_matrix = acts @ rvecs.T  # (n_items, 4)
            rows.append({
                "repeat": rep,
                "layer": layer,
                "effective_dimensionality": _effective_dimensionality(proj_matrix),
                "pc1_variance": _pc1_variance(proj_matrix),
                "mean_abs_off_diag_corr": _mean_abs_off_diag_corr(proj_matrix),
                "reliability_g1": _reliability_proxy(proj_matrix),
                "reliability_g3": _reliability_proxy(proj_matrix),  # proxy; same value
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fallback: structure metrics on existing projection data (no activations)
# ---------------------------------------------------------------------------


def _structure_from_wide_df(wide_df: pd.DataFrame) -> dict:
    """Compute structure metrics from an existing wide-format projection DataFrame."""
    available = [c for c in PROJ_COLS if c in wide_df.columns]
    if len(available) < 2:
        return {}
    proj_matrix = wide_df[available].dropna().values
    return {
        "effective_dimensionality": _effective_dimensionality(proj_matrix),
        "pc1_variance": _pc1_variance(proj_matrix),
        "mean_abs_off_diag_corr": _mean_abs_off_diag_corr(proj_matrix),
        "reliability_g1": _reliability_proxy(proj_matrix),
        "reliability_g3": _reliability_proxy(proj_matrix),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_random_vector_control(
    ethics_wide_df: pd.DataFrame,
    reliability_wide_df: pd.DataFrame,
    activation_paths_by_layer: Optional[dict] = None,
    n_repeats: int = 100,
    random_seed: int = 42,
) -> dict:
    """
    Loads activations (if paths provided), generates random vectors, runs structure
    and reliability metrics for each repeat.

    If activation_paths_by_layer is None or loading fails, falls back to computing
    structure metrics directly on the existing projection DataFrames by sampling
    random linear combinations of the observed projection columns.
    NOTE: The fallback does not re-project activations onto random directions;
    it samples permuted/random combinations of observed projection values to estimate
    a null distribution of structure metrics. This is documented in the output DataFrame
    via a 'method' column.

    Returns dict with keys:
        summary_df        — mean/std/percentile across repeats per layer
        distributions_df  — all repeat results
    """
    activations_dict: dict = {}
    method = "fallback_from_projections"

    if activation_paths_by_layer:
        for layer, path in activation_paths_by_layer.items():
            try:
                acts = np.load(path)
                activations_dict[layer] = acts
                method = "raw_activations"
                print(f"[random_vector_control] Loaded activations for layer {layer}: {acts.shape}")
            except Exception as e:
                print(f"[random_vector_control] Could not load {path}: {e}. Using fallback.")

    if activations_dict:
        distributions_df = project_onto_random_vectors(activations_dict, n_repeats, random_seed)
        distributions_df["method"] = "raw_activations"
    else:
        # Fallback: shuffle projection columns to create null distribution
        print(
            "[random_vector_control] No activations available. Falling back to "
            "structure metrics on randomly permuted projection columns. "
            "Metrics reflect null distribution of structure under column permutation, "
            "NOT random dot-product projections."
        )
        rows = []
        rng = np.random.default_rng(random_seed)
        layers = reliability_wide_df["layer"].unique() if "layer" in reliability_wide_df.columns else [32, 40, 47]
        for layer in layers:
            if "layer" in reliability_wide_df.columns:
                sub = reliability_wide_df[reliability_wide_df["layer"] == layer]
            else:
                sub = ethics_wide_df
            available = [c for c in PROJ_COLS if c in sub.columns]
            if len(available) < 2:
                continue
            proj_matrix = sub[available].dropna().values
            for rep in range(n_repeats):
                perm = rng.permutation(proj_matrix.flatten()).reshape(proj_matrix.shape)
                rows.append({
                    "repeat": rep,
                    "layer": layer,
                    "effective_dimensionality": _effective_dimensionality(perm),
                    "pc1_variance": _pc1_variance(perm),
                    "mean_abs_off_diag_corr": _mean_abs_off_diag_corr(perm),
                    "reliability_g1": _reliability_proxy(perm),
                    "reliability_g3": _reliability_proxy(perm),
                    "method": "fallback_permuted_projections",
                })
        distributions_df = pd.DataFrame(rows)

    # Summary
    metric_cols = ["effective_dimensionality", "pc1_variance", "mean_abs_off_diag_corr",
                   "reliability_g1", "reliability_g3"]
    group_cols = ["layer"]
    if "method" in distributions_df.columns:
        group_cols.append("method")
    summary_rows = []
    for keys, grp in distributions_df.groupby(group_cols):
        if isinstance(keys, (int, float, str)):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for m in metric_cols:
            if m in grp.columns:
                base[f"{m}_mean"] = grp[m].mean()
                base[f"{m}_std"] = grp[m].std()
                base[f"{m}_p5"] = grp[m].quantile(0.05)
                base[f"{m}_p95"] = grp[m].quantile(0.95)
        summary_rows.append(base)
    summary_df = pd.DataFrame(summary_rows)

    return {"summary_df": summary_df, "distributions_df": distributions_df, "method": method}


def compare_to_real(
    random_distributions_df: pd.DataFrame,
    real_metrics_dict: dict,
) -> pd.DataFrame:
    """
    For each metric, compute percentile of real value in random distribution.

    real_metrics_dict: {layer: {metric: value}}

    Returns DataFrame: metric, layer, real_value, random_mean, random_std, percentile_of_real
    """
    metric_cols = ["effective_dimensionality", "pc1_variance", "mean_abs_off_diag_corr",
                   "reliability_g1", "reliability_g3"]
    rows = []
    for layer, metrics in real_metrics_dict.items():
        grp = random_distributions_df[random_distributions_df["layer"] == layer]
        if grp.empty:
            continue
        for metric, real_val in metrics.items():
            if metric not in metric_cols or metric not in grp.columns:
                continue
            null_vals = grp[metric].dropna().values
            if len(null_vals) == 0:
                continue
            pct = float(np.mean(null_vals <= real_val)) * 100
            rows.append({
                "metric": metric,
                "layer": layer,
                "real_value": real_val,
                "random_mean": float(null_vals.mean()),
                "random_std": float(null_vals.std()),
                "percentile_of_real": pct,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_random_vector_control(results_dict: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dict["summary_df"].to_csv(out_dir / "random_vector_control_summary.csv", index=False)
    results_dict["distributions_df"].to_csv(out_dir / "random_vector_control_distributions.csv", index=False)
    print(f"[random_vector_control] Saved to {out_dir}")
