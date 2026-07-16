"""
Stage 4A: Projection structure analysis (RQ1).

RQ1: Do the four morally relevant persona-vector projections behave like one
latent "morality" dimension, or several separable dimensions?

All analysis is CPU-only on existing Stage 3 centered projection tables.
No new Gemma activations are extracted.

Methodology notes
-----------------
- Mean-centered projections from Stage 3 are used throughout.
- Raw projections are not used for main analysis (large shared activation
  mean dominates raw dot products and has been removed by centering).
- PCA is performed via eigendecomposition of the (Pearson) correlation matrix
  of the four trait projections, standardized per layer.
- Parallel analysis uses permutation to estimate the random-baseline
  eigenvalue distribution (more conservative than the Horn 1965 random-normal
  approach when N is small).
- Factor analysis requires the optional `factor_analyzer` package; if absent
  the analysis is skipped and a warning is recorded.  With only four observed
  variables at most 2 factors are estimable without Heywood cases; interpret
  cautiously.
- Layer 32 is the contrast-validation-selected layer (Stage 2B AUC).
- Layer 40 showed stronger downstream ETHICS diagonal dominance and is
  reported as the downstream-best comparison layer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECTION_COLS = [
    "projection_honesty",
    "projection_harmlessness",
    "projection_fairness",
    "projection_compassion",
]
TRAIT_LABELS = ["honesty", "harmlessness", "fairness", "compassion"]

_PA_N_SIM = 500  # parallel analysis simulations


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------


def load_centered_wide(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_centered_long(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def validate_projection_columns(df: pd.DataFrame, cols: list[str] | None = None) -> None:
    """Raise ValueError if expected projection columns are absent."""
    expected = cols or PROJECTION_COLS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing projection columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def build_layer_wide_tables(
    long_df: pd.DataFrame,
    layers: list[int],
    proj_cols: list[str] | None = None,
) -> dict[int, pd.DataFrame]:
    """
    Pivot the long-format projection table into one wide DataFrame per layer.

    The wide frame has one row per item and one column per projected trait,
    plus item metadata columns (item_id, primary_trait, source_split).
    """
    pcols = proj_cols or PROJECTION_COLS
    trait_short = [c.replace("projection_", "") for c in pcols]

    result: dict[int, pd.DataFrame] = {}
    for layer in layers:
        sub = long_df[long_df["layer"] == layer].copy()
        if sub.empty:
            continue
        wide = sub.pivot(index="item_id", columns="projected_trait", values="projection")
        wide.columns = [f"projection_{t}" for t in wide.columns]
        wide = wide.reset_index()
        # Re-attach metadata
        meta = sub[["item_id", "primary_trait", "source_split"]].drop_duplicates("item_id")
        wide = wide.merge(meta, on="item_id", how="left")
        result[layer] = wide
    return result


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------


def standardize(X: np.ndarray) -> np.ndarray:
    """Column-wise z-score standardization. Returns copy."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of columns of X (n × p)."""
    Xs = standardize(X)
    n = Xs.shape[0]
    return (Xs.T @ Xs) / (n - 1)


def correlation_df(X: np.ndarray, labels: list[str]) -> pd.DataFrame:
    corr = correlation_matrix(X)
    return pd.DataFrame(corr, index=labels, columns=labels)


# ---------------------------------------------------------------------------
# PCA via eigendecomposition
# ---------------------------------------------------------------------------


@dataclass
class PCAResult:
    eigenvalues: np.ndarray       # shape (p,), descending
    loadings: np.ndarray          # shape (p, p): columns = principal components
    explained_variance_ratio: np.ndarray  # shape (p,)
    cumulative_variance: np.ndarray       # shape (p,)
    scores: np.ndarray            # shape (n, p): item projections onto PCs
    n_items: int
    n_variables: int
    labels: list[str]

    @property
    def n_components_80(self) -> int:
        return int(np.searchsorted(self.cumulative_variance, 0.80)) + 1

    @property
    def n_components_90(self) -> int:
        return int(np.searchsorted(self.cumulative_variance, 0.90)) + 1

    @property
    def first_pc_dominant(self) -> bool:
        """True if PC1 explains ≥ 50% of variance."""
        return bool(self.explained_variance_ratio[0] >= 0.50)

    @property
    def effective_dimensionality(self) -> float:
        """Participation ratio: (Σλ)² / Σλ²."""
        ev = self.eigenvalues
        return float(ev.sum() ** 2 / (ev ** 2).sum())


def run_pca(X: np.ndarray, labels: list[str], standardize_first: bool = True) -> PCAResult:
    """
    PCA via eigendecomposition of the correlation (or covariance) matrix.

    Parameters
    ----------
    X:
        n × p data matrix.
    labels:
        Variable names (length p).
    standardize_first:
        If True, standardize columns before PCA (equivalent to PCA on
        the correlation matrix rather than the covariance matrix).
    """
    if standardize_first:
        X = standardize(X)
    n, p = X.shape

    # Covariance matrix of standardized data = correlation matrix
    C = (X.T @ X) / (n - 1)

    # eigh returns eigenvalues ascending; we want descending
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # columns = PCs

    # Clip tiny negative eigenvalues from floating-point error
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    total = eigenvalues.sum() or 1.0
    evr = eigenvalues / total
    cumvar = np.cumsum(evr)

    scores = X @ eigenvectors  # n × p

    return PCAResult(
        eigenvalues=eigenvalues,
        loadings=eigenvectors,  # p × p; column k = loadings on PC k
        explained_variance_ratio=evr,
        cumulative_variance=cumvar,
        scores=scores,
        n_items=n,
        n_variables=p,
        labels=labels,
    )


def loadings_df(pca: PCAResult) -> pd.DataFrame:
    """Return a DataFrame of PCA loadings (variables × components)."""
    p = pca.n_variables
    cols = [f"PC{k+1}" for k in range(p)]
    return pd.DataFrame(pca.loadings, index=pca.labels, columns=cols)


# ---------------------------------------------------------------------------
# Parallel analysis
# ---------------------------------------------------------------------------


@dataclass
class ParallelAnalysisResult:
    n_components_suggested: int
    observed_eigenvalues: np.ndarray
    random_eigenvalue_95th: np.ndarray  # 95th percentile across simulations
    random_eigenvalue_mean: np.ndarray
    n_simulations: int


def run_parallel_analysis(
    X: np.ndarray,
    n_simulations: int = _PA_N_SIM,
    rng: np.random.Generator | None = None,
    percentile: float = 95.0,
    standardize_first: bool = True,
) -> ParallelAnalysisResult:
    """
    Horn's parallel analysis using permutation of observed data.

    For each simulation, each column of X is independently permuted
    (preserving marginal distributions), then PCA is run.  The observed
    eigenvalues are compared against the specified percentile of the random
    distribution.  Components whose observed eigenvalue exceeds the random
    baseline are retained.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if standardize_first:
        X = standardize(X)
    n, p = X.shape

    C_obs = (X.T @ X) / (n - 1)
    ev_obs, _ = np.linalg.eigh(C_obs)
    ev_obs = np.sort(ev_obs)[::-1]

    sim_eigenvalues = np.zeros((n_simulations, p))
    for i in range(n_simulations):
        X_perm = X.copy()
        for j in range(p):
            X_perm[:, j] = rng.permutation(X_perm[:, j])
        C_rand = (X_perm.T @ X_perm) / (n - 1)
        ev_rand, _ = np.linalg.eigh(C_rand)
        sim_eigenvalues[i] = np.sort(ev_rand)[::-1]

    rand_pct = np.percentile(sim_eigenvalues, percentile, axis=0)
    rand_mean = sim_eigenvalues.mean(axis=0)

    n_suggested = int(np.sum(ev_obs > rand_pct))
    n_suggested = max(1, n_suggested)  # always keep at least 1

    return ParallelAnalysisResult(
        n_components_suggested=n_suggested,
        observed_eigenvalues=ev_obs,
        random_eigenvalue_95th=rand_pct,
        random_eigenvalue_mean=rand_mean,
        n_simulations=n_simulations,
    )


# ---------------------------------------------------------------------------
# Factor analysis (optional)
# ---------------------------------------------------------------------------


def run_factor_analysis(
    X: np.ndarray,
    max_factors: int = 4,
    standardize_first: bool = True,
) -> Optional[dict]:
    """
    Run exploratory factor analysis if `factor_analyzer` is available.

    Returns None with a warning if the dependency is missing.
    Returns a dict with keys: n_factors_tested, loadings, communalities,
    uniquenesses, factor_variance, warning_messages.
    """
    try:
        from factor_analyzer import FactorAnalyzer  # type: ignore
    except ImportError:
        warnings.warn(
            "factor_analyzer is not installed. "
            "Install with: pip install factor_analyzer. "
            "Skipping exploratory factor analysis.",
            stacklevel=2,
        )
        return None

    if standardize_first:
        X = standardize(X)

    n, p = X.shape
    # With p=4 variables, maximum estimable factors is (p - 1) // 2 = 1
    # before Heywood issues become likely; we cap at p-1 but warn
    max_estimable = max(1, p - 1)
    if max_factors > max_estimable:
        warnings.warn(
            f"With {p} observed variables, at most {max_estimable} factor(s) are "
            f"identifiable without Heywood cases.  Limiting to {max_estimable}.",
            stacklevel=2,
        )
        max_factors = max_estimable

    results: dict = {
        "n_factors_tested": max_factors,
        "loadings": {},
        "communalities": {},
        "uniquenesses": {},
        "factor_variance": {},
        "warning_messages": [],
    }

    for n_factors in range(1, max_factors + 1):
        try:
            fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="ml")
            fa.fit(X)
            results["loadings"][n_factors] = fa.loadings_
            results["communalities"][n_factors] = fa.get_communalities()
            results["uniquenesses"][n_factors] = fa.get_uniquenesses()
            results["factor_variance"][n_factors] = fa.get_factor_variance()
        except Exception as exc:
            msg = f"Factor analysis with {n_factors} factor(s) failed: {exc}"
            warnings.warn(msg, stacklevel=2)
            results["warning_messages"].append(msg)

    return results


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


@dataclass
class StructureSummary:
    layer: int
    n_items: int
    n_variables: int
    mean_abs_off_diagonal_corr: float
    max_abs_trait_corr: float
    most_correlated_pair: tuple[str, str]
    most_correlated_value: float
    effective_dimensionality: float
    n_components_80pct: int
    n_components_90pct: int
    first_pc_variance: float
    first_pc_dominant: bool
    n_components_parallel: int
    interpretation: str
    pca: PCAResult
    corr_df: pd.DataFrame
    parallel: ParallelAnalysisResult
    factor_analysis: Optional[dict] = field(default=None)
    warnings: list[str] = field(default_factory=list)


def _interpret(summary: "StructureSummary") -> str:
    ev1 = summary.first_pc_variance
    ed = summary.effective_dimensionality
    n_pa = summary.n_components_parallel
    max_corr = summary.max_abs_trait_corr
    mean_corr = summary.mean_abs_off_diagonal_corr

    if ev1 >= 0.70:
        return (
            "Single dominant dimension: PC1 explains {:.0%} of variance, "
            "effective dimensionality {:.2f}. The four trait projections largely "
            "collapse onto a single latent moral-valence axis at this layer.".format(ev1, ed)
        )
    if ev1 >= 0.50 or (n_pa == 1 and ed < 1.6):
        return (
            "Near-single-dimensional structure: PC1 explains {:.0%} of variance "
            "(effective dimensionality {:.2f}). A weak secondary dimension is present "
            "but the dominant direction accounts for most projection variance.".format(ev1, ed)
        )
    # High effective dimensionality + low correlations → near-independent traits
    # (PA with permutation at p=4 is conservative; n_pa=1 can still occur when
    # eigenvalue differences are small even under near-independence)
    if ed >= 3.0 and mean_corr < 0.20:
        return (
            "Near-independent four-dimensional structure: effective dimensionality {:.2f} "
            "(maximum possible = {:.0f}), mean |off-diagonal correlation| = {:.3f}, "
            "PC1 explains only {:.0%}. The four trait projections behave as largely "
            "separable dimensions at this layer. Parallel analysis retains {} component(s), "
            "but at p=4 the permutation baseline is conservative; effective dimensionality "
            "is the primary evidence of separability. "
            "This suggests the model encodes each moral trait in a distinct direction "
            "rather than collapsing them onto a single 'morality' axis.".format(
                ed, summary.n_variables, mean_corr, ev1, n_pa
            )
        )
    if n_pa <= 2 and ed < 2.5:
        return (
            "Two-dimensional structure: parallel analysis retains {} component(s), "
            "effective dimensionality {:.2f}, PC1 explains {:.0%}. "
            "Two latent dimensions may be present, but the four intended traits "
            "are not fully separable at this layer.".format(n_pa, ed, ev1)
        )
    if n_pa >= 3 and ed >= 2.5 and max_corr < 0.5:
        return (
            "Three or four partially separable dimensions: parallel analysis retains "
            "{} component(s), effective dimensionality {:.2f}, max trait correlation {:.2f}. "
            "The four traits show meaningful separation but remain moderately "
            "correlated.".format(n_pa, ed, max_corr)
        )
    return (
        "Ambiguous structure: parallel analysis retains {} component(s), "
        "effective dimensionality {:.2f}, PC1 explains {:.0%}, "
        "max trait correlation {:.2f}. "
        "Structure does not fit a clean single- or multi-dimensional pattern; "
        "interpret with caution.".format(n_pa, ed, ev1, max_corr)
    )


def compute_structure_summary(
    wide_df: pd.DataFrame,
    layer: int,
    proj_cols: list[str] | None = None,
    standardize_projections: bool = True,
    run_pca_flag: bool = True,
    run_pa: bool = True,
    run_fa: bool = True,
    fa_max_factors: int = 4,
    rng: np.random.Generator | None = None,
) -> StructureSummary:
    pcols = proj_cols or PROJECTION_COLS
    labels = [c.replace("projection_", "") for c in pcols]

    validate_projection_columns(wide_df, pcols)
    X = wide_df[pcols].to_numpy(dtype=float)
    n, p = X.shape

    warn_msgs: list[str] = []
    if n < 30:
        warn_msgs.append(f"Only {n} items at layer {layer}; results may be unstable.")

    corr = correlation_df(X, labels)

    # Off-diagonal correlations
    corr_arr = corr.to_numpy()
    mask = ~np.eye(p, dtype=bool)
    off_diag = corr_arr[mask]
    mean_abs_off = float(np.abs(off_diag).mean())
    max_abs = float(np.abs(off_diag).max())

    # Most correlated pair
    upper = np.triu(np.abs(corr_arr), k=1)
    i_max, j_max = np.unravel_index(np.argmax(upper), upper.shape)
    pair = (labels[i_max], labels[j_max])
    pair_val = float(corr_arr[i_max, j_max])

    # PCA
    pca = run_pca(X, labels, standardize_first=standardize_projections)

    # Parallel analysis
    pa = run_parallel_analysis(X, rng=rng, standardize_first=standardize_projections)

    # Factor analysis (optional)
    fa_result: Optional[dict] = None
    if run_fa:
        fa_result = run_factor_analysis(X, max_factors=fa_max_factors,
                                        standardize_first=standardize_projections)

    summary = StructureSummary(
        layer=layer,
        n_items=n,
        n_variables=p,
        mean_abs_off_diagonal_corr=mean_abs_off,
        max_abs_trait_corr=max_abs,
        most_correlated_pair=pair,
        most_correlated_value=pair_val,
        effective_dimensionality=pca.effective_dimensionality,
        n_components_80pct=pca.n_components_80,
        n_components_90pct=pca.n_components_90,
        first_pc_variance=float(pca.explained_variance_ratio[0]),
        first_pc_dominant=pca.first_pc_dominant,
        n_components_parallel=pa.n_components_suggested,
        interpretation="",
        pca=pca,
        corr_df=corr,
        parallel=pa,
        factor_analysis=fa_result,
        warnings=warn_msgs,
    )
    summary.interpretation = _interpret(summary)
    return summary


# ---------------------------------------------------------------------------
# Multi-layer entry point
# ---------------------------------------------------------------------------


def run_structure_analysis(
    long_path: str | Path,
    wide_path_layer32: str | Path,
    layers: list[int],
    proj_cols: list[str] | None = None,
    standardize_projections: bool = True,
    run_pa: bool = True,
    run_fa: bool = True,
    fa_max_factors: int = 4,
    random_seed: int = 42,
) -> dict[int, StructureSummary]:
    """
    Run structure analysis for each requested layer.

    The wide-format file from Stage 3 contains only layer 32 (the target layer
    used for the wide pivot).  For layers 40 and 47 we rebuild the wide table
    from the long-format file.

    Returns a dict mapping layer → StructureSummary.
    """
    pcols = proj_cols or PROJECTION_COLS
    long_df = load_centered_long(long_path)
    layer_tables = build_layer_wide_tables(long_df, layers, proj_cols=pcols)

    # Supplement with the pre-built wide table for layer 32 (uses real
    # to_wide_format output which may have slightly different item set).
    try:
        wide32 = load_centered_wide(wide_path_layer32)
        validate_projection_columns(wide32, pcols)
        layer_tables[32] = wide32
    except Exception:
        pass  # fall back to long-derived table

    rng = np.random.default_rng(random_seed)
    results: dict[int, StructureSummary] = {}
    for layer in layers:
        if layer not in layer_tables:
            warnings.warn(f"No data for layer {layer}; skipping.", stacklevel=2)
            continue
        results[layer] = compute_structure_summary(
            wide_df=layer_tables[layer],
            layer=layer,
            proj_cols=pcols,
            standardize_projections=standardize_projections,
            run_pca_flag=True,
            run_pa=run_pa,
            run_fa=run_fa,
            fa_max_factors=fa_max_factors,
            rng=rng,
        )
    return results
