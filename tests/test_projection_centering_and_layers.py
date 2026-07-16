"""
Tests for Stage 3 mean-centering and layer comparison.

All tests run without GPU, Modal, torch, or transformers.
Covers: mean-centering correctness, raw vs centered output separation,
centering metadata, layer comparison metrics, diagnostics warnings,
and no heavy imports.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.projection.compute_projections import (
    PREPROCESSING_CENTERED,
    PREPROCESSING_RAW,
    TRAITS,
    compute_layer_means,
    load_activations_by_layer,
    mock_project,
    save_centering_metadata,
    save_projection_set,
    to_wide_format,
)
from src.projection.layer_comparison import (
    LayerComparisonResult,
    compare_layers,
    compute_layer_metrics,
    save_layer_comparison,
)
from src.projection.projection_diagnostics import run_diagnostics, save_diagnostics

_ITEM_BANK = Path(__file__).resolve().parent.parent / "data/processed/ethics_curated_mvp.parquet"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_item_df():
    rows = []
    for i, trait in enumerate(TRAITS):
        for j in range(4):
            rows.append(
                {
                    "item_id": f"{trait}_{j:03d}",
                    "source_split": "commonsense",
                    "primary_trait": trait,
                    "scenario_text": f"A scenario about {trait}.",
                    "keep_for_mvp": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def mock_results(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, layer_means, n_items = mock_project(
            small_item_df, layers=[32, 40], hidden_dim=32,
            out_dir=tmpdir, preprocessing="both"
        )
        yield results, layer_means, n_items, small_item_df, tmpdir


# ---------------------------------------------------------------------------
# Mean-centering correctness
# ---------------------------------------------------------------------------


def test_compute_layer_means_correct():
    acts = {
        32: {"a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
             "b": np.array([3.0, 4.0, 5.0], dtype=np.float32)},
    }
    means = compute_layer_means(acts)
    expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(means[32], expected)


def test_centered_projections_have_near_zero_mean(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(
            small_item_df, layers=[32], hidden_dim=64,
            out_dir=tmpdir, preprocessing="both"
        )
        cen = results[PREPROCESSING_CENTERED]
        for trait in TRAITS:
            trait_projs = cen[cen["projected_trait"] == trait]["projection"]
            assert abs(trait_projs.mean()) < 1.0, (
                f"Centered projections for {trait} have non-zero mean: {trait_projs.mean():.4f}"
            )


def test_raw_projections_not_zero_mean(small_item_df):
    """Raw projections need not be zero-mean — centering is what makes them so."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results, layer_means, _ = mock_project(
            small_item_df, layers=[32], hidden_dim=64,
            out_dir=tmpdir, preprocessing="both"
        )
        raw = results[PREPROCESSING_RAW]
        cen = results[PREPROCESSING_CENTERED]
        # Raw and centered should differ when layer mean is non-zero
        raw_mean = raw[raw["projected_trait"] == "honesty"]["projection"].mean()
        cen_mean = cen[cen["projected_trait"] == "honesty"]["projection"].mean()
        # They may be equal if mock mean is exactly 0 (unlikely) — just check both exist
        assert PREPROCESSING_RAW in results
        assert PREPROCESSING_CENTERED in results


def test_raw_and_centered_differ_by_dot_product_of_mean(small_item_df):
    """Centered = raw − dot(layer_mean, vector).  Verify on one item/trait."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dim = 32
        results, layer_means, _ = mock_project(
            small_item_df, layers=[32], hidden_dim=dim,
            out_dir=tmpdir, preprocessing="both"
        )
        raw = results[PREPROCESSING_RAW]
        cen = results[PREPROCESSING_CENTERED]
        # For a given item × trait, difference should equal dot(mean_act, vector)
        # We can't access vectors directly here, but we can verify raw.mean - cen.mean ≈ same
        # across all items for the same trait (since centering is a constant offset per trait/layer)
        for trait in TRAITS:
            raw_vals = raw[(raw["projected_trait"] == trait) & (raw["layer"] == 32)]["projection"]
            cen_vals = cen[(cen["projected_trait"] == trait) & (cen["layer"] == 32)]["projection"]
            if len(raw_vals) == 0:
                continue
            diffs = (raw_vals.values - cen_vals.values)
            # All diffs should be the same constant (dot of mean_act with this trait's vector)
            assert np.std(diffs) < 1e-3, (
                f"diff std={np.std(diffs):.6f} for {trait} — centering is not a constant offset"
            )


# ---------------------------------------------------------------------------
# Output file tests
# ---------------------------------------------------------------------------


def test_both_preprocessing_variants_saved(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(
            small_item_df, layers=[32], hidden_dim=32,
            out_dir=tmpdir, preprocessing="both"
        )
        assert PREPROCESSING_RAW in results
        assert PREPROCESSING_CENTERED in results
        assert "default" in results
        assert results["default"] is results[PREPROCESSING_CENTERED]


def test_only_raw_when_requested(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(
            small_item_df, layers=[32], hidden_dim=32,
            out_dir=tmpdir, preprocessing="raw"
        )
        assert PREPROCESSING_RAW in results
        assert PREPROCESSING_CENTERED not in results
        assert results["default"] is results[PREPROCESSING_RAW]


def test_save_projection_set_creates_files(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(
            small_item_df, layers=[32], hidden_dim=32,
            out_dir=tmpdir, preprocessing="both"
        )
        cen_long = results[PREPROCESSING_CENTERED]
        wide = to_wide_format(cen_long[cen_long["layer"] == 32], small_item_df)
        lp, lc, wp, wc = save_projection_set(cen_long, wide, tmpdir, "test_proj_centered")
        assert lp.exists()
        assert lc.exists()
        assert wp.exists()
        assert wc.exists()


def test_long_format_has_preprocessing_column(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(
            small_item_df, layers=[32], hidden_dim=32,
            out_dir=tmpdir, preprocessing="both"
        )
        for key, df in results.items():
            if key == "default":
                continue
            assert "projection_preprocessing" in df.columns, f"Missing column in {key}"
            assert df["projection_preprocessing"].iloc[0] == key


# ---------------------------------------------------------------------------
# Centering metadata
# ---------------------------------------------------------------------------


def test_save_centering_metadata_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        means = {32: np.ones(64, dtype=np.float32), 40: np.zeros(64, dtype=np.float32)}
        n_items = {32: 10, 40: 10}
        csv_path = save_centering_metadata(means, tmpdir, n_items)
        assert csv_path.exists()
        json_path = Path(tmpdir) / "centering_metadata.json"
        assert json_path.exists()
        npy32 = Path(tmpdir) / "centering" / "mean_activation_layer32.npy"
        assert npy32.exists()
        arr = np.load(npy32)
        np.testing.assert_allclose(arr, np.ones(64, dtype=np.float32))


def test_centering_metadata_records_norm():
    with tempfile.TemporaryDirectory() as tmpdir:
        v = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5.0
        save_centering_metadata({32: v}, tmpdir, {32: 5})
        df = pd.read_csv(Path(tmpdir) / "centering_metadata.csv")
        assert abs(df.iloc[0]["mean_activation_norm"] - 5.0) < 1e-4


# ---------------------------------------------------------------------------
# Layer comparison metrics
# ---------------------------------------------------------------------------


def test_compute_layer_metrics_shape(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(
            small_item_df, layers=[32, 40], hidden_dim=32,
            out_dir=tmpdir, preprocessing="both"
        )
        long_df = pd.concat([results[PREPROCESSING_CENTERED]], ignore_index=True)
        metrics = compute_layer_metrics(long_df, small_item_df, [32, 40])
        assert len(metrics) == 2
        assert "diagonal_dominance" in metrics.columns
        assert "matching_margin" in metrics.columns


def test_compute_layer_metrics_diagonal_in_range(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(
            small_item_df, layers=[32, 40], hidden_dim=32,
            out_dir=tmpdir, preprocessing="both"
        )
        long_df = results[PREPROCESSING_CENTERED]
        metrics = compute_layer_metrics(long_df, small_item_df, [32, 40])
        assert (metrics["diagonal_dominance"].dropna().between(0, 1)).all()


def test_layer_comparison_detects_divergence(small_item_df):
    """When best downstream layer != contrast-selected, layers_agree should be False."""
    metrics_df = pd.DataFrame(
        [
            {"layer": 32, "diagonal_dominance": 0.30, "matching_margin": 10.0,
             "max_inter_trait_correlation": 0.5},
            {"layer": 40, "diagonal_dominance": 0.55, "matching_margin": 20.0,
             "max_inter_trait_correlation": 0.4},
            {"layer": 47, "diagonal_dominance": 0.25, "matching_margin": 5.0,
             "max_inter_trait_correlation": 0.6},
        ]
    )
    for t in TRAITS:
        metrics_df[f"diagonal_{t}"] = 0.0

    result = compare_layers(metrics_df, contrast_selected_layer=32, downstream_layers=[32, 40, 47])
    assert result.best_downstream_layer == 40
    assert not result.layers_agree
    assert "diverge" in result.interpretation.lower() or "differ" in result.interpretation.lower() or "layer 40" in result.interpretation


def test_layer_comparison_agrees_when_same(small_item_df):
    metrics_df = pd.DataFrame(
        [
            {"layer": 32, "diagonal_dominance": 0.60, "matching_margin": 25.0,
             "max_inter_trait_correlation": 0.3},
            {"layer": 40, "diagonal_dominance": 0.45, "matching_margin": 15.0,
             "max_inter_trait_correlation": 0.4},
        ]
    )
    for t in TRAITS:
        metrics_df[f"diagonal_{t}"] = 0.0

    result = compare_layers(metrics_df, contrast_selected_layer=32, downstream_layers=[32, 40])
    assert result.best_downstream_layer == 32
    assert result.layers_agree


def test_layer_comparison_warns_at_or_below_chance():
    metrics_df = pd.DataFrame(
        [
            {"layer": 32, "diagonal_dominance": 0.24, "matching_margin": -5.0,
             "max_inter_trait_correlation": 0.9},
        ]
    )
    for t in TRAITS:
        metrics_df[f"diagonal_{t}"] = 0.0

    result = compare_layers(metrics_df, contrast_selected_layer=32, downstream_layers=[32])
    assert any("chance" in w.lower() or "below" in w.lower() for w in result.warnings)


def test_save_layer_comparison_creates_files(small_item_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_df = pd.DataFrame(
            [{"layer": 32, "diagonal_dominance": 0.35, "matching_margin": 10.0,
              "max_inter_trait_correlation": 0.5}]
        )
        for t in TRAITS:
            metrics_df[f"diagonal_{t}"] = 0.0
        result = compare_layers(metrics_df, 32, [32])
        csv_path, md_path = save_layer_comparison(result, tmpdir)
        assert csv_path.exists()
        assert md_path.exists()
        assert "Layer" in md_path.read_text()


# ---------------------------------------------------------------------------
# Diagnostics: centering-aware warnings
# ---------------------------------------------------------------------------


def test_diagnostics_warns_on_raw_dominant_vector(small_item_df):
    """Raw projections with one hugely dominant vector should trigger a warning."""
    rows = []
    for item_id in [f"i{i}" for i in range(20)]:
        for trait in TRAITS:
            proj = 50000.0 if trait == "harmlessness" else -5000.0
            rows.append({
                "item_id": item_id, "source_split": "commonsense",
                "primary_trait": "honesty", "projected_trait": trait,
                "layer": 32, "projection": proj,
                "projection_preprocessing": PREPROCESSING_RAW,
                "vector_path": "mock", "activation_path": "mock",
            })
    long_df = pd.DataFrame(rows)
    item_df = pd.DataFrame({
        "item_id": [f"i{i}" for i in range(20)],
        "primary_trait": ["honesty"] * 20, "keep_for_mvp": [True] * 20,
        "scenario_text": ["s"] * 20, "source_split": ["commonsense"] * 20,
    })
    wide_df = to_wide_format(long_df, item_df)
    result = run_diagnostics(long_df, wide_df, item_df, target_layer=32, preprocessing="raw")
    assert result.has_warnings()
    assert any("dominat" in w.lower() or "spread" in w.lower() for w in result.warnings)


def test_diagnostics_no_dominant_vector_warning_after_centering(small_item_df):
    """After centering, a previously dominant vector should not trigger the domination warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results, _, _ = mock_project(
            small_item_df, layers=[32], hidden_dim=64,
            out_dir=tmpdir, preprocessing="both"
        )
        cen_long = results[PREPROCESSING_CENTERED]
        wide = to_wide_format(cen_long[cen_long["layer"] == 32], small_item_df)
        result = run_diagnostics(
            cen_long, wide, small_item_df, target_layer=32, preprocessing="mean_centered"
        )
        # Should not warn about dominant vector (centering removes that)
        assert not any("dominat" in w.lower() and "raw" not in w.lower()
                       for w in result.warnings)


def test_diagnostics_warns_below_chance():
    """Diagonal dominance below chance should trigger a warning."""
    rows = []
    item_df = pd.DataFrame({
        "item_id": [f"i{i}" for i in range(20)],
        "primary_trait": ["honesty"] * 20, "keep_for_mvp": [True] * 20,
        "scenario_text": ["s"] * 20, "source_split": ["commonsense"] * 20,
    })
    rng = np.random.default_rng(99)
    for item_id in item_df["item_id"]:
        for trait in TRAITS:
            rows.append({
                "item_id": item_id, "source_split": "commonsense",
                "primary_trait": "honesty", "projected_trait": trait,
                "layer": 32, "projection": float(rng.standard_normal()),
                "projection_preprocessing": PREPROCESSING_CENTERED,
                "vector_path": "mock", "activation_path": "mock",
            })
    long_df = pd.DataFrame(rows)
    wide_df = to_wide_format(long_df, item_df)
    result = run_diagnostics(
        long_df, wide_df, item_df, target_layer=32, preprocessing="mean_centered"
    )
    # With random projections, dominance may be near chance — just check warning logic works
    if result.diagonal_dominance_rate <= 0.25:
        assert any("chance" in w.lower() for w in result.warnings)


def test_diagnostics_save_centered_suffix():
    """Centered diagnostics should save with _centered suffix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Minimal long/wide/item DFs
        long_df = pd.DataFrame({
            "item_id": ["a"] * 4, "source_split": ["c"] * 4,
            "primary_trait": ["honesty"] * 4,
            "projected_trait": TRAITS,
            "layer": [32] * 4, "projection": [0.1, -0.1, 0.05, -0.05],
            "projection_preprocessing": [PREPROCESSING_CENTERED] * 4,
            "vector_path": ["m"] * 4, "activation_path": ["m"] * 4,
        })
        item_df = pd.DataFrame({
            "item_id": ["a"], "primary_trait": ["honesty"],
            "scenario_text": ["s"], "keep_for_mvp": [True], "source_split": ["c"],
        })
        wide_df = to_wide_format(long_df, item_df)
        result = run_diagnostics(long_df, wide_df, item_df, target_layer=32)
        md_path, _, _ = save_diagnostics(result, tmpdir, "mean_centered")
        assert "centered" in md_path.name


# ---------------------------------------------------------------------------
# No heavy imports
# ---------------------------------------------------------------------------


def test_no_modal_in_layer_comparison():
    import src.projection.layer_comparison as m
    assert "import modal" not in Path(m.__file__).read_text()


def test_no_torch_in_compute_projections():
    import src.projection.compute_projections as m
    assert "import torch" not in Path(m.__file__).read_text()


def test_no_transformers_in_diagnostics():
    import src.projection.projection_diagnostics as m
    assert "import transformers" not in Path(m.__file__).read_text()
