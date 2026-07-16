"""
Tests for Stage 3: ETHICS projection pipeline.

All tests run without GPU, Modal, torch, or transformers.
Covers: item bank loading/validation, job construction, AUC enforcement,
vector selection, projection computation, output schemas, and diagnostics.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.projection.compute_projections import (
    mock_project,
    project_activations,
    save_projections,
    to_wide_format,
)
from src.projection.ethics_projection import (
    TRAITS,
    build_projection_jobs,
    build_prompt,
    enforce_auc_threshold,
    load_item_bank,
    select_vectors,
    validate_item_bank,
)
from src.projection.projection_diagnostics import (
    run_diagnostics,
    save_diagnostics,
)

_ITEM_BANK = Path(__file__).resolve().parent.parent / "data/processed/ethics_curated_mvp.parquet"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def item_df():
    return load_item_bank(_ITEM_BANK)


@pytest.fixture
def small_item_df():
    """10 synthetic items covering all four traits."""
    rows = []
    for i, trait in enumerate(TRAITS):
        for j in range(3 if i < 3 else 1):
            rows.append(
                {
                    "item_id": f"test_{trait}_{j:03d}",
                    "source_split": "commonsense",
                    "primary_trait": trait,
                    "scenario_text": f"A person decides to act in a {trait} manner.",
                    "keep_for_mvp": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def vec_meta_df():
    """Synthetic vector metadata for layers 32, 40, 47."""
    rows = []
    for trait in TRAITS:
        for layer in [32, 40, 47]:
            rows.append(
                {
                    "trait": trait,
                    "layer": layer,
                    "vector_path": f"mock_{trait}_layer{layer}.npy",
                    "n_positive": 10,
                    "n_negative": 10,
                    "vector_method": "difference_of_means",
                    "normalization": "unit_norm",
                    "hidden_dim": 64,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def val_results_df():
    """Synthetic validation results — all traits pass at layers 32, 40, 47."""
    rows = []
    for trait in TRAITS:
        for layer in [16, 24, 28, 32, 40, 47]:
            auc = 0.85 if layer >= 32 else 0.60
            rows.append(
                {
                    "trait": trait,
                    "layer": layer,
                    "auc": auc,
                    "accuracy": 0.80,
                    "cohens_d": 1.5,
                    "passes_minimum_auc": auc >= 0.75,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Item bank tests
# ---------------------------------------------------------------------------


def test_item_bank_loads(item_df):
    assert len(item_df) == 204
    assert set(item_df["primary_trait"].unique()).issubset(set(TRAITS))


def test_item_bank_required_columns(item_df):
    from src.projection.ethics_projection import REQUIRED_ITEM_COLUMNS
    for col in REQUIRED_ITEM_COLUMNS:
        assert col in item_df.columns, f"Missing column: {col}"


def test_item_bank_all_keep_for_mvp(item_df):
    assert item_df["keep_for_mvp"].all(), "All loaded items should have keep_for_mvp=True"


def test_validate_item_bank_passes(item_df):
    validate_item_bank(item_df)  # should not raise


def test_validate_item_bank_missing_column():
    bad_df = pd.DataFrame({"item_id": ["x"], "scenario_text": ["y"]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_item_bank(bad_df)


def test_validate_item_bank_empty():
    empty = pd.DataFrame(columns=["item_id", "scenario_text", "primary_trait",
                                   "source_split", "keep_for_mvp"])
    with pytest.raises(ValueError, match="empty"):
        validate_item_bank(empty)


def test_validate_item_bank_invalid_trait():
    df = pd.DataFrame(
        {
            "item_id": ["x"],
            "scenario_text": ["s"],
            "primary_trait": ["deontology"],  # ETHICS split name, not a construct
            "source_split": ["commonsense"],
            "keep_for_mvp": [True],
        }
    )
    with pytest.raises(ValueError, match="non-construct"):
        validate_item_bank(df)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_build_prompt_contains_scenario():
    scenario = "A person lies to their friend."
    prompt = build_prompt(scenario)
    assert scenario in prompt
    assert "Answer:" in prompt


def test_build_prompt_ends_with_answer():
    prompt = build_prompt("Some scenario.")
    assert prompt.strip().endswith("Answer:")


# ---------------------------------------------------------------------------
# Projection job construction
# ---------------------------------------------------------------------------


def test_build_projection_jobs_shape(small_item_df):
    layers = [32, 40]
    jobs_df = build_projection_jobs(small_item_df, layers)
    assert len(jobs_df) == len(small_item_df) * len(layers)


def test_build_projection_jobs_columns(small_item_df):
    jobs_df = build_projection_jobs(small_item_df, [32])
    expected = {"item_id", "source_split", "primary_trait", "scenario_text",
                "prompt_text", "target_layer", "token_position"}
    assert expected.issubset(set(jobs_df.columns))


def test_build_projection_jobs_token_position(small_item_df):
    jobs_df = build_projection_jobs(small_item_df, [32])
    assert (jobs_df["token_position"] == "last_prompt_token").all()


# ---------------------------------------------------------------------------
# Vector validation enforcement
# ---------------------------------------------------------------------------


def test_enforce_auc_threshold_passes(val_results_df):
    enforce_auc_threshold(val_results_df, [32, 40], TRAITS, min_auc=0.75)  # no error


def test_enforce_auc_threshold_fails():
    df = pd.DataFrame(
        [{"trait": "honesty", "layer": 32, "auc": 0.60, "passes_minimum_auc": False}]
    )
    with pytest.raises(ValueError, match="honesty"):
        enforce_auc_threshold(df, [32], ["honesty"], min_auc=0.75)


def test_enforce_auc_threshold_missing_result():
    empty_df = pd.DataFrame(columns=["trait", "layer", "auc", "passes_minimum_auc"])
    with pytest.raises(ValueError, match="no validation result"):
        enforce_auc_threshold(empty_df, [32], ["honesty"], min_auc=0.75)


# ---------------------------------------------------------------------------
# Vector selection
# ---------------------------------------------------------------------------


def test_select_vectors_target_only(vec_meta_df):
    selected = select_vectors(vec_meta_df, target_layer=32, comparison_layers=[])
    assert set(selected["layer"].unique()) == {32}
    assert len(selected) == len(TRAITS)


def test_select_vectors_with_comparison(vec_meta_df):
    selected = select_vectors(vec_meta_df, target_layer=32, comparison_layers=[40, 47])
    assert set(selected["layer"].unique()) == {32, 40, 47}
    assert len(selected) == len(TRAITS) * 3


def test_select_vectors_missing_raises(vec_meta_df):
    with pytest.raises(ValueError, match="Missing persona vectors"):
        select_vectors(vec_meta_df, target_layer=99)


# ---------------------------------------------------------------------------
# Projection computation
# ---------------------------------------------------------------------------


def test_mock_project_long_format(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(small_item_df, layers=[32], hidden_dim=64, out_dir=tmpdir)
        long_df = results["default"]
        expected_rows = len(small_item_df) * len(TRAITS)
        assert len(long_df) == expected_rows
        assert "item_id" in long_df.columns
        assert "projected_trait" in long_df.columns
        assert "projection" in long_df.columns
        assert "layer" in long_df.columns


def test_mock_project_projection_values_finite(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(small_item_df, layers=[32], hidden_dim=64, out_dir=tmpdir)
        long_df = results["default"]
        assert long_df["projection"].notna().all()
        assert np.isfinite(long_df["projection"].values).all()


def test_project_activations_dot_product():
    """Verify dot product is computed correctly with controlled arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dim = 8

        # One item, one layer
        act = np.ones(dim, dtype=np.float32)
        vec_honest = np.ones(dim, dtype=np.float32) / np.sqrt(dim)  # unit norm

        act_path = tmpdir / "item001_layer32.npy"
        vec_path = tmpdir / "honesty_layer32.npy"
        np.save(act_path, act)
        np.save(vec_path, vec_honest)

        act_meta = pd.DataFrame(
            [
                {
                    "item_id": "item001",
                    "layer": 32,
                    "source_split": "commonsense",
                    "primary_trait": "honesty",
                    "activation_path": str(act_path),
                }
            ]
        )
        vec_meta = pd.DataFrame(
            [
                {
                    "trait": t,
                    "layer": 32,
                    "vector_path": str(vec_path),
                }
                for t in TRAITS
            ]
        )

        # preprocessing="raw" so we can verify the raw dot product exactly
        results, _, _ = project_activations(
            act_meta, vec_meta, tmpdir, tmpdir, preprocessing="raw"
        )
        long_df = results["default"]
        expected_proj = float(np.dot(act, vec_honest))
        rows = long_df[long_df["projected_trait"] == "honesty"]
        assert len(rows) == 1
        assert abs(rows["projection"].iloc[0] - expected_proj) < 1e-5


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------


def test_long_format_columns(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(small_item_df, layers=[32, 40], hidden_dim=64, out_dir=tmpdir)
        long_df = results["default"]
        required = {"item_id", "source_split", "primary_trait", "projected_trait",
                    "layer", "projection", "vector_path", "activation_path"}
        assert required.issubset(set(long_df.columns))


def test_wide_format_columns(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(small_item_df, layers=[32], hidden_dim=64, out_dir=tmpdir)
        long_df = results["default"]
        target_long = long_df[long_df["layer"] == 32]
        wide_df = to_wide_format(target_long, small_item_df)
        for trait in TRAITS:
            assert f"projection_{trait}" in wide_df.columns
        assert "item_id" in wide_df.columns
        assert "primary_trait" in wide_df.columns


def test_wide_format_one_row_per_item(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(small_item_df, layers=[32], hidden_dim=64, out_dir=tmpdir)
        long_df = results["default"]
        wide_df = to_wide_format(long_df[long_df["layer"] == 32], small_item_df)
        assert len(wide_df) == len(small_item_df)


def test_save_projections_creates_files(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(small_item_df, layers=[32], hidden_dim=64, out_dir=tmpdir)
        long_df = results["default"]
        wide_df = to_wide_format(long_df[long_df["layer"] == 32], small_item_df)
        lp, lc, wp, wc = save_projections(long_df, wide_df, tmpdir)
        assert lp.exists()
        assert lc.exists()
        assert wp.exists()
        assert wc.exists()


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------


def test_diagnostics_runs(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(small_item_df, layers=[32], hidden_dim=64, out_dir=tmpdir)
        long_df = results["default"]
        wide_df = to_wide_format(long_df[long_df["layer"] == 32], small_item_df)
        result = run_diagnostics(long_df, wide_df, small_item_df, target_layer=32)
        assert result.n_items == len(small_item_df)
        assert result.n_traits == len(TRAITS)


def test_diagnostics_matching_vs_nonmatching(small_item_df):
    """Items with a biased signal should show some diagonal structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(small_item_df, layers=[32], hidden_dim=64, out_dir=tmpdir)
        long_df = results["default"]
        wide_df = to_wide_format(long_df[long_df["layer"] == 32], small_item_df)
        result = run_diagnostics(long_df, wide_df, small_item_df, target_layer=32)
        # matching_table should be a DataFrame with primary_trait as index
        assert not result.matching_table.empty


def test_diagnostics_near_constant_warning():
    """Near-constant projections should trigger a warning."""
    # Build a long_df where all projections are the same
    item_df = pd.DataFrame(
        {
            "item_id": [f"i{i}" for i in range(5)],
            "source_split": ["commonsense"] * 5,
            "primary_trait": ["honesty"] * 5,
            "scenario_text": ["scenario"] * 5,
            "keep_for_mvp": [True] * 5,
        }
    )
    rows = []
    for item_id in item_df["item_id"]:
        for trait in TRAITS:
            rows.append(
                {
                    "item_id": item_id,
                    "source_split": "commonsense",
                    "primary_trait": "honesty",
                    "projected_trait": trait,
                    "layer": 32,
                    "projection": 0.0001,  # near-constant
                    "vector_path": "mock",
                    "activation_path": "mock",
                }
            )
    long_df = pd.DataFrame(rows)
    wide_df = to_wide_format(long_df, item_df)
    result = run_diagnostics(long_df, wide_df, item_df, target_layer=32)
    assert result.has_warnings()
    assert any("near-constant" in w for w in result.warnings)


def test_diagnostics_high_correlation_warning():
    """Extremely correlated traits should trigger a warning."""
    n = 20
    item_df = pd.DataFrame(
        {
            "item_id": [f"i{i}" for i in range(n)],
            "source_split": ["commonsense"] * n,
            "primary_trait": ["honesty"] * n,
            "scenario_text": ["scenario"] * n,
            "keep_for_mvp": [True] * n,
        }
    )
    rng = np.random.default_rng(0)
    base = rng.standard_normal(n)
    rows = []
    for i, row in item_df.iterrows():
        for trait in TRAITS:
            rows.append(
                {
                    "item_id": row["item_id"],
                    "source_split": "commonsense",
                    "primary_trait": "honesty",
                    "projected_trait": trait,
                    "layer": 32,
                    "projection": float(base[i]),  # identical for all traits → corr=1
                    "vector_path": "mock",
                    "activation_path": "mock",
                }
            )
    long_df = pd.DataFrame(rows)
    wide_df = to_wide_format(long_df, item_df)
    result = run_diagnostics(long_df, wide_df, item_df, target_layer=32)
    assert result.has_warnings()
    assert any("correlated" in w for w in result.warnings)


def test_diagnostics_save(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(small_item_df, layers=[32], hidden_dim=64, out_dir=tmpdir)
        long_df = results["default"]
        wide_df = to_wide_format(long_df[long_df["layer"] == 32], small_item_df)
        result = run_diagnostics(long_df, wide_df, small_item_df, target_layer=32)
        md_path, summary_path, corr_path = save_diagnostics(result, tmpdir, "mean_centered")
        assert md_path.exists()
        assert summary_path.exists()
        assert corr_path.exists()
        assert "Stage 3" in md_path.read_text()


# ---------------------------------------------------------------------------
# No heavy imports in normal test execution
# ---------------------------------------------------------------------------


def test_no_modal_import():
    import src.projection.ethics_projection as m
    src_text = Path(m.__file__).read_text()
    assert "import modal" not in src_text


def test_no_torch_import():
    import src.projection.compute_projections as m
    src_text = Path(m.__file__).read_text()
    assert "import torch" not in src_text


def test_no_transformers_import():
    import src.projection.projection_diagnostics as m
    src_text = Path(m.__file__).read_text()
    assert "import transformers" not in src_text


def test_comparison_layer_support(small_item_df):
    """Projections at comparison layers 40 and 47 should also be produced."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(
            small_item_df, layers=[32, 40, 47], hidden_dim=64, out_dir=tmpdir
        )
        long_df = results["default"]
        assert set(long_df["layer"].unique()) == {32, 40, 47}
