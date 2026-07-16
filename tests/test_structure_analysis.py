"""
Tests for Stage 4A projection structure analysis.

All tests run without GPU, Modal, torch, or transformers.
Covers: column validation, standardization, correlation matrix,
PCA, parallel analysis, effective dimensionality, summary metrics,
report file writing, factor analysis graceful skip.
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.structure_analysis import (
    PROJECTION_COLS,
    TRAIT_LABELS,
    ParallelAnalysisResult,
    PCAResult,
    StructureSummary,
    build_layer_wide_tables,
    compute_structure_summary,
    correlation_df,
    correlation_matrix,
    loadings_df,
    run_factor_analysis,
    run_parallel_analysis,
    run_pca,
    run_structure_analysis,
    standardize,
    validate_projection_columns,
)
from src.analysis.structure_reports import (
    save_all,
    save_correlation_matrix,
    save_pca_loadings,
    save_pca_variance,
    save_parallel_analysis,
    save_report,
    save_structure_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_wide_df(n: int = 50, rng: np.random.Generator | None = None, seed: int = 0) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, 4))
    traits = ["honesty", "harmlessness", "fairness", "compassion"]
    df = pd.DataFrame(data, columns=PROJECTION_COLS)
    df["item_id"] = [f"item_{i:03d}" for i in range(n)]
    df["primary_trait"] = [traits[i % 4] for i in range(n)]
    df["source_split"] = "commonsense"
    return df


def _make_long_df(layers: list[int] = [32, 40], n_items: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for layer in layers:
        for i in range(n_items):
            for trait in TRAIT_LABELS:
                rows.append({
                    "item_id": f"item_{i:03d}",
                    "projected_trait": trait,
                    "layer": layer,
                    "projection": float(rng.standard_normal()),
                    "primary_trait": TRAIT_LABELS[i % 4],
                    "source_split": "commonsense",
                    "projection_preprocessing": "mean_centered",
                })
    return pd.DataFrame(rows)


@pytest.fixture
def wide_df():
    return _make_wide_df(n=60)


@pytest.fixture
def long_df():
    return _make_long_df(layers=[32, 40, 47], n_items=40)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_projection_columns_passes(wide_df):
    validate_projection_columns(wide_df)  # should not raise


def test_validate_projection_columns_raises_on_missing():
    df = pd.DataFrame({"projection_honesty": [1.0], "other": [2.0]})
    with pytest.raises(ValueError, match="Missing projection columns"):
        validate_projection_columns(df)


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------


def test_standardize_mean_zero():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 4)) * 10 + 5
    Xs = standardize(X)
    np.testing.assert_allclose(Xs.mean(axis=0), np.zeros(4), atol=1e-10)


def test_standardize_std_one():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 4)) * 3 - 7
    Xs = standardize(X)
    np.testing.assert_allclose(Xs.std(axis=0, ddof=1), np.ones(4), atol=1e-10)


def test_standardize_constant_column_no_divide_by_zero():
    X = np.array([[1.0, 5.0], [1.0, 3.0], [1.0, 7.0]])
    Xs = standardize(X)
    assert np.isfinite(Xs).all()


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


def test_correlation_matrix_shape(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    corr = correlation_matrix(X)
    assert corr.shape == (4, 4)


def test_correlation_matrix_diagonal_ones(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    corr = correlation_matrix(X)
    np.testing.assert_allclose(np.diag(corr), np.ones(4), atol=1e-10)


def test_correlation_matrix_symmetric(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    corr = correlation_matrix(X)
    np.testing.assert_allclose(corr, corr.T, atol=1e-10)


def test_correlation_matrix_range(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    corr = correlation_matrix(X)
    assert np.all(corr >= -1.0 - 1e-9) and np.all(corr <= 1.0 + 1e-9)


def test_correlation_df_labels(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    cdf = correlation_df(X, TRAIT_LABELS)
    assert list(cdf.columns) == TRAIT_LABELS
    assert list(cdf.index) == TRAIT_LABELS


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def test_pca_explained_variance_sums_to_one(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    pca = run_pca(X, TRAIT_LABELS)
    np.testing.assert_allclose(pca.explained_variance_ratio.sum(), 1.0, atol=1e-6)


def test_pca_cumulative_variance_final_one(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    pca = run_pca(X, TRAIT_LABELS)
    np.testing.assert_allclose(pca.cumulative_variance[-1], 1.0, atol=1e-6)


def test_pca_eigenvalues_non_negative(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    pca = run_pca(X, TRAIT_LABELS)
    assert np.all(pca.eigenvalues >= -1e-10)


def test_pca_eigenvalues_descending(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    pca = run_pca(X, TRAIT_LABELS)
    assert np.all(np.diff(pca.eigenvalues) <= 1e-10)


def test_pca_loadings_shape(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    pca = run_pca(X, TRAIT_LABELS)
    assert pca.loadings.shape == (4, 4)


def test_pca_scores_shape(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    pca = run_pca(X, TRAIT_LABELS)
    assert pca.scores.shape == (len(wide_df), 4)


def test_pca_loadings_df_columns(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    pca = run_pca(X, TRAIT_LABELS)
    df = loadings_df(pca)
    assert list(df.columns) == ["PC1", "PC2", "PC3", "PC4"]
    assert list(df.index) == TRAIT_LABELS


def test_pca_on_1d_data_has_dominant_first_component():
    """When all variables are copies of one signal, PC1 should explain ≈100%."""
    rng = np.random.default_rng(5)
    signal = rng.standard_normal(100)
    X = np.column_stack([signal + 0.01 * rng.standard_normal(100) for _ in range(4)])
    pca = run_pca(X, TRAIT_LABELS)
    assert pca.explained_variance_ratio[0] > 0.90
    assert pca.first_pc_dominant


def test_pca_on_independent_data_no_dominant_component():
    """When columns are independent, no single PC should dominate."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((200, 4))
    pca = run_pca(X, TRAIT_LABELS)
    # Each component should explain roughly 25%; first PC should not dominate
    assert pca.explained_variance_ratio[0] < 0.50


def test_effective_dimensionality_on_1d_data():
    """Single-factor data should have effective_dimensionality close to 1."""
    rng = np.random.default_rng(8)
    signal = rng.standard_normal(100)
    X = np.column_stack([signal + 0.05 * rng.standard_normal(100) for _ in range(4)])
    pca = run_pca(X, TRAIT_LABELS)
    assert pca.effective_dimensionality < 1.5


def test_effective_dimensionality_on_independent_data():
    """Independent data should have effective_dimensionality close to n_variables."""
    rng = np.random.default_rng(9)
    X = rng.standard_normal((400, 4))
    pca = run_pca(X, TRAIT_LABELS)
    assert pca.effective_dimensionality > 3.0


def test_n_components_80_reasonable(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    pca = run_pca(X, TRAIT_LABELS)
    assert 1 <= pca.n_components_80 <= 4


def test_n_components_90_ge_80(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    pca = run_pca(X, TRAIT_LABELS)
    assert pca.n_components_90 >= pca.n_components_80


# ---------------------------------------------------------------------------
# Parallel analysis
# ---------------------------------------------------------------------------


def test_parallel_analysis_returns_valid_count(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    rng = np.random.default_rng(42)
    result = run_parallel_analysis(X, n_simulations=50, rng=rng)
    assert 1 <= result.n_components_suggested <= 4


def test_parallel_analysis_arrays_shape(wide_df):
    X = wide_df[PROJECTION_COLS].to_numpy()
    rng = np.random.default_rng(42)
    result = run_parallel_analysis(X, n_simulations=30, rng=rng)
    assert len(result.observed_eigenvalues) == 4
    assert len(result.random_eigenvalue_95th) == 4
    assert len(result.random_eigenvalue_mean) == 4


def test_parallel_analysis_1d_suggests_1_component():
    """Strongly one-dimensional data should have only 1 PA component."""
    rng = np.random.default_rng(11)
    signal = rng.standard_normal(200)
    X = np.column_stack([signal + 0.01 * rng.standard_normal(200) for _ in range(4)])
    result = run_parallel_analysis(X, n_simulations=100, rng=rng)
    assert result.n_components_suggested == 1


def test_parallel_analysis_independent_more_than_1d():
    """
    For independent data, PA should suggest more components than for
    strongly one-dimensional data.  With p=4 and permutation thresholding
    this is a relative rather than absolute check.
    """
    rng_indep = np.random.default_rng(12)
    X_indep = rng_indep.standard_normal((300, 4))
    result_indep = run_parallel_analysis(X_indep, n_simulations=100, rng=rng_indep)

    rng_1d = np.random.default_rng(11)
    signal = rng_1d.standard_normal(300)
    X_1d = np.column_stack([signal + 0.01 * rng_1d.standard_normal(300) for _ in range(4)])
    result_1d = run_parallel_analysis(X_1d, n_simulations=100, rng=rng_1d)

    assert result_indep.n_components_suggested >= result_1d.n_components_suggested


# ---------------------------------------------------------------------------
# build_layer_wide_tables
# ---------------------------------------------------------------------------


def test_build_layer_wide_tables_keys(long_df):
    tables = build_layer_wide_tables(long_df, [32, 40, 47])
    assert set(tables.keys()) == {32, 40, 47}


def test_build_layer_wide_tables_columns(long_df):
    tables = build_layer_wide_tables(long_df, [32])
    df = tables[32]
    for col in PROJECTION_COLS:
        assert col in df.columns
    assert "item_id" in df.columns


def test_build_layer_wide_tables_row_count(long_df):
    tables = build_layer_wide_tables(long_df, [32])
    assert len(tables[32]) == long_df["item_id"].nunique()


# ---------------------------------------------------------------------------
# compute_structure_summary
# ---------------------------------------------------------------------------


def test_compute_structure_summary_returns_summary(wide_df):
    rng = np.random.default_rng(42)
    s = compute_structure_summary(wide_df, layer=32, rng=rng)
    assert isinstance(s, StructureSummary)
    assert s.layer == 32
    assert s.n_items == len(wide_df)


def test_compute_structure_summary_interpretation_nonempty(wide_df):
    rng = np.random.default_rng(42)
    s = compute_structure_summary(wide_df, layer=32, rng=rng)
    assert len(s.interpretation) > 20


def test_compute_structure_summary_1d_calls_dominant(wide_df):
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(len(wide_df))
    for col in PROJECTION_COLS:
        wide_df[col] = signal + 0.01 * rng.standard_normal(len(wide_df))
    s = compute_structure_summary(wide_df, layer=32, rng=rng)
    assert s.first_pc_dominant
    assert "dominant" in s.interpretation.lower() or "single" in s.interpretation.lower()


# ---------------------------------------------------------------------------
# Factor analysis graceful skip
# ---------------------------------------------------------------------------


def test_factor_analysis_skips_gracefully_if_missing(wide_df):
    """If factor_analyzer is not installed, run_factor_analysis returns None with a warning."""
    import sys
    # Temporarily mask factor_analyzer from import machinery
    saved = sys.modules.get("factor_analyzer", None)
    sys.modules["factor_analyzer"] = None  # type: ignore
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X = wide_df[PROJECTION_COLS].to_numpy()
            result = run_factor_analysis(X)
            assert result is None
            assert any("factor_analyzer" in str(warning.message) for warning in w)
    finally:
        if saved is None:
            del sys.modules["factor_analyzer"]
        else:
            sys.modules["factor_analyzer"] = saved


# ---------------------------------------------------------------------------
# Report / file output
# ---------------------------------------------------------------------------


def test_save_correlation_matrix_creates_file(wide_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(1)
        s = compute_structure_summary(wide_df, layer=32, rng=rng)
        path = save_correlation_matrix(s, tmpdir)
        assert path.exists()
        df = pd.read_csv(path, index_col=0)
        assert df.shape == (4, 4)


def test_save_pca_loadings_creates_file(wide_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(1)
        s = compute_structure_summary(wide_df, layer=32, rng=rng)
        path = save_pca_loadings(s, tmpdir)
        assert path.exists()


def test_save_pca_variance_creates_file(wide_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(1)
        s = compute_structure_summary(wide_df, layer=32, rng=rng)
        path = save_pca_variance(s, tmpdir)
        assert path.exists()
        df = pd.read_csv(path)
        assert "cumulative_variance" in df.columns
        np.testing.assert_allclose(df["cumulative_variance"].iloc[-1], 1.0, atol=1e-5)


def test_save_parallel_analysis_creates_file(wide_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(1)
        s = compute_structure_summary(wide_df, layer=32, rng=rng)
        path = save_parallel_analysis(s, tmpdir)
        assert path.exists()
        df = pd.read_csv(path)
        assert "retained" in df.columns


def test_save_structure_summary_creates_file(wide_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(1)
        s = compute_structure_summary(wide_df, layer=32, rng=rng)
        path = save_structure_summary({32: s}, tmpdir)
        assert path.exists()
        df = pd.read_csv(path)
        assert "layer" in df.columns
        assert 32 in df["layer"].values


def test_save_report_creates_md(wide_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(1)
        s32 = compute_structure_summary(wide_df, layer=32, rng=rng)
        s40 = compute_structure_summary(wide_df, layer=40, rng=rng)
        path = save_report({32: s32, 40: s40}, tmpdir)
        assert path.exists()
        text = path.read_text()
        assert "Stage 4A" in text
        assert "Layer 32" in text
        assert "Layer 40" in text
        assert "RQ1" in text


def test_save_all_returns_paths(wide_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(1)
        s = compute_structure_summary(wide_df, layer=32, rng=rng)
        saved = save_all({32: s}, {32: wide_df}, tmpdir)
        assert "summary" in saved
        assert "report" in saved
        assert saved["summary"].exists()
        assert saved["report"].exists()


# ---------------------------------------------------------------------------
# Integration: run_structure_analysis with temp parquet
# ---------------------------------------------------------------------------


def test_run_structure_analysis_integration():
    long_df = _make_long_df(layers=[32, 40], n_items=50)
    wide_df = _make_wide_df(n=50)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        long_path = tmpdir / "long.parquet"
        wide_path = tmpdir / "wide.parquet"
        long_df.to_parquet(long_path, index=False)
        wide_df.to_parquet(wide_path, index=False)

        results = run_structure_analysis(
            long_path=long_path,
            wide_path_layer32=wide_path,
            layers=[32, 40],
            run_pa=True,
            run_fa=False,
            random_seed=99,
        )
        assert 32 in results
        assert 40 in results
        for s in results.values():
            assert isinstance(s, StructureSummary)
            assert s.n_items > 0


# ---------------------------------------------------------------------------
# No heavy imports
# ---------------------------------------------------------------------------


def test_no_modal_import():
    import src.analysis.structure_analysis as m
    assert "import modal" not in Path(m.__file__).read_text()


def test_no_torch_import():
    import src.analysis.structure_analysis as m
    assert "import torch" not in Path(m.__file__).read_text()


def test_no_transformers_import():
    import src.analysis.structure_reports as m
    assert "import transformers" not in Path(m.__file__).read_text()
