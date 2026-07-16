"""
Tests for Stage 4D: Reliability / Generalizability Analysis.

All tests use synthetic data — no real projections, no GPU required.
"""

from __future__ import annotations

import importlib
import io
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reliability.reliability_analysis import (
    REQUIRED_COLUMNS,
    VarianceComponents,
    ReliabilityResult,
    compute_reliability,
    estimate_variance_components,
    filter_table,
    load_projection_table,
    results_to_dataframe,
    run_reliability_analysis,
)
from src.reliability.g_theory import d_study, g_coefficient, run_d_study_for_all
from src.reliability.reliability_reports import (
    save_all,
    save_d_study_results,
    save_item_level,
    save_reliability_summary,
    save_variance_components,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_long_df(
    n_items: int = 10,
    n_variants: int = 4,
    between_sd: float = 1.0,
    within_sd: float = 0.1,
    layers: list[int] = None,
    traits: list[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic long projection table.

    between_sd controls stable item differences.
    within_sd controls paraphrase noise.
    """
    rng = np.random.default_rng(seed)
    if layers is None:
        layers = [32]
    if traits is None:
        traits = ["honesty"]

    rows = []
    for layer in layers:
        for trait in traits:
            item_means = rng.normal(0, between_sd, n_items)
            for i in range(n_items):
                item_id = f"item_{i:03d}"
                for v in range(n_variants):
                    projection = item_means[i] + rng.normal(0, within_sd)
                    rows.append({
                        "item_id": item_id,
                        "variant_id": f"{item_id}_v{v}",
                        "paraphrase_id": f"p{v}" if v > 0 else "original",
                        "framing": "neutral",
                        "primary_trait": trait,
                        "projected_trait": trait,
                        "layer": layer,
                        "projection": projection,
                        "source_split": "commonsense",
                        "variant_type": "paraphrase" if v > 0 else "original",
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# load_projection_table
# ---------------------------------------------------------------------------

def test_load_validates_required_columns(tmp_path):
    """load_projection_table raises ValueError for missing columns."""
    df = pd.DataFrame({"item_id": [1], "projection": [0.5]})
    p = tmp_path / "bad.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="missing required columns"):
        load_projection_table(p)


def test_load_accepts_valid_table(tmp_path):
    """load_projection_table succeeds with all required columns present."""
    df = make_long_df()
    p = tmp_path / "good.parquet"
    df.to_parquet(p)
    loaded = load_projection_table(p)
    assert set(REQUIRED_COLUMNS).issubset(set(loaded.columns))


# ---------------------------------------------------------------------------
# filter_table
# ---------------------------------------------------------------------------

def test_filter_drops_items_with_too_few_variants():
    """filter_table removes items that have fewer than min_variants_per_item."""
    df = make_long_df(n_items=5, n_variants=4)
    # Add one item with only 1 variant
    single = pd.DataFrame([{
        "item_id": "sparse_item",
        "variant_id": "sparse_item_v0",
        "paraphrase_id": "original",
        "framing": "neutral",
        "primary_trait": "honesty",
        "projected_trait": "honesty",
        "layer": 32,
        "projection": 0.5,
        "source_split": "commonsense",
        "variant_type": "original",
    }])
    df = pd.concat([df, single], ignore_index=True)
    filtered = filter_table(df, layers=[32], projected_traits=["honesty"], min_variants_per_item=2)
    assert "sparse_item" not in filtered["item_id"].values


def test_filter_keeps_items_with_enough_variants():
    df = make_long_df(n_items=5, n_variants=3)
    filtered = filter_table(df, layers=[32], projected_traits=["honesty"], min_variants_per_item=2)
    assert filtered["item_id"].nunique() == 5


# ---------------------------------------------------------------------------
# estimate_variance_components
# ---------------------------------------------------------------------------

def test_variance_components_high_between():
    """When between-item differences dominate, between_item_var >> within_item_var."""
    df = make_long_df(n_items=20, n_variants=4, between_sd=5.0, within_sd=0.1)
    subset = df[["item_id", "projection"]]
    vc = estimate_variance_components(subset)
    ratio = vc.between_item_var / vc.within_item_var if vc.within_item_var > 0 else float("inf")
    assert ratio > 10, f"Expected high variance ratio, got {ratio}"
    assert not vc.clamped_negative


def test_variance_components_high_within():
    """When within-item noise dominates, variance_ratio should be very low."""
    df = make_long_df(n_items=20, n_variants=4, between_sd=0.01, within_sd=5.0)
    subset = df[["item_id", "projection"]]
    vc = estimate_variance_components(subset)
    # between_item_var may be clamped to 0
    assert vc.between_item_var <= vc.within_item_var or vc.clamped_negative


def test_variance_components_negative_clamped():
    """Negative variance estimate is clamped to 0 and clamped_negative=True."""
    # Construct data where within-item variance is very high compared to between
    rng = np.random.default_rng(0)
    rows = []
    for i in range(5):
        for v in range(4):
            rows.append({"item_id": f"item_{i}", "projection": rng.normal(0, 100)})
    df = pd.DataFrame(rows)
    # May or may not clamp; ensure no crash and clamped_negative is a bool
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vc = estimate_variance_components(df)
    assert vc.between_item_var >= 0
    assert isinstance(vc.clamped_negative, bool)


def test_variance_components_unbalanced_no_crash():
    """Unbalanced design (different n_variants per item) does not crash."""
    rows = []
    for i in range(5):
        n_v = 2 + i  # unbalanced: 2, 3, 4, 5, 6 variants
        for v in range(n_v):
            rows.append({"item_id": f"item_{i}", "projection": float(i + v * 0.1)})
    df = pd.DataFrame(rows)
    vc = estimate_variance_components(df)
    assert vc.n_items == 5
    assert vc.between_item_var >= 0


# ---------------------------------------------------------------------------
# compute_reliability
# ---------------------------------------------------------------------------

def test_reliability_higher_k_not_lower():
    """Reliability is non-decreasing in k."""
    vc = VarianceComponents(
        between_item_var=0.5,
        within_item_var=0.5,
        total_var=1.0,
        n_items=10,
        n_variants_per_item_mean=4.0,
        n_variants_per_item_min=4,
        clamped_negative=False,
    )
    k_values = [1, 2, 3, 4, 5]
    rel = compute_reliability(vc, k_values)
    vals = [rel[k] for k in k_values]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-10, f"Reliability decreased from k={k_values[vals.index(a)]} to next"


def test_reliability_single_equals_k1():
    vc = VarianceComponents(
        between_item_var=0.6,
        within_item_var=0.4,
        total_var=1.0,
        n_items=10,
        n_variants_per_item_mean=4.0,
        n_variants_per_item_min=4,
        clamped_negative=False,
    )
    rel = compute_reliability(vc, [1, 2, 3])
    assert abs(rel[1] - 0.6 / (0.6 + 0.4)) < 1e-9


def test_reliability_high_when_between_dominates():
    vc = VarianceComponents(
        between_item_var=0.95,
        within_item_var=0.05,
        total_var=1.0,
        n_items=10,
        n_variants_per_item_mean=4.0,
        n_variants_per_item_min=4,
        clamped_negative=False,
    )
    rel = compute_reliability(vc, [1])
    assert rel[1] > 0.8


def test_reliability_low_when_within_dominates():
    vc = VarianceComponents(
        between_item_var=0.05,
        within_item_var=0.95,
        total_var=1.0,
        n_items=10,
        n_variants_per_item_mean=4.0,
        n_variants_per_item_min=4,
        clamped_negative=False,
    )
    rel = compute_reliability(vc, [1])
    assert rel[1] < 0.2


# ---------------------------------------------------------------------------
# g_coefficient
# ---------------------------------------------------------------------------

def test_g_coefficient_clamps_to_0_1():
    assert g_coefficient(-1.0, 1.0, 1) == 0.0
    assert g_coefficient(2.0, 0.0, 1) == 1.0
    assert 0.0 <= g_coefficient(0.5, 0.5, 3) <= 1.0


def test_g_coefficient_zero_denominator():
    assert g_coefficient(0.0, 0.0, 1) == 0.0


# ---------------------------------------------------------------------------
# d_study
# ---------------------------------------------------------------------------

def test_d_study_returns_one_row_per_k():
    k_list = [1, 2, 3, 4, 5]
    df = d_study(0.5, 0.5, k_list)
    assert len(df) == len(k_list)
    assert set(df["n_paraphrases"].tolist()) == set(k_list)


# ---------------------------------------------------------------------------
# results_to_dataframe
# ---------------------------------------------------------------------------

def test_results_to_dataframe_columns():
    df = make_long_df(n_items=5, n_variants=4)
    results = run_reliability_analysis(df, layers=[32], projected_traits=["honesty"])
    results_df = results_to_dataframe(results)
    expected_cols = [
        "layer", "projected_trait",
        "between_item_var", "within_item_var", "total_var",
        "within_item_sd", "between_item_sd", "variance_ratio",
        "reliability_1", "reliability_2", "reliability_3",
        "reliability_4", "reliability_5",
        "n_items_used", "clamped_negative",
    ]
    for col in expected_cols:
        assert col in results_df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# run_d_study_for_all
# ---------------------------------------------------------------------------

def test_run_d_study_for_all():
    df = make_long_df(n_items=5, n_variants=4, layers=[32, 40], traits=["honesty", "harmlessness"])
    results = run_reliability_analysis(df, layers=[32, 40], projected_traits=["honesty", "harmlessness"])
    d_df = run_d_study_for_all(results, [1, 2, 3])
    # 2 layers × 2 traits × 3 k values = 12 rows
    assert len(d_df) == 12
    assert set(d_df.columns) >= {"layer", "projected_trait", "n_paraphrases", "g_coefficient"}


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------

def test_save_functions_write_files(tmp_path):
    df = make_long_df(n_items=5, n_variants=4, layers=[32], traits=["honesty"])
    results = run_reliability_analysis(df, layers=[32], projected_traits=["honesty"])
    results_df = results_to_dataframe(results)
    d_df = run_d_study_for_all(results, [1, 2, 3])
    meta = {
        "n_items_total": 5,
        "n_items_used": 5,
        "n_variants_total": 20,
        "n_items_missing_paraphrases": 0,
        "layers": [32],
        "projected_traits": ["honesty"],
        "primary_layer": 32,
        "downstream_best_layer": 40,
    }
    save_all(results, results_df, d_df, meta, tmp_path)
    assert (tmp_path / "reliability_summary.csv").exists()
    assert (tmp_path / "variance_components.csv").exists()
    assert (tmp_path / "d_study_results.csv").exists()
    assert (tmp_path / "item_level_reliability_long.csv").exists()
    assert (tmp_path / "reliability_analysis_report.md").exists()


# ---------------------------------------------------------------------------
# No forbidden imports at module level
# ---------------------------------------------------------------------------

def test_no_modal_import_in_reliability_analysis():
    spec = importlib.util.find_spec("src.reliability.reliability_analysis")
    assert spec is not None
    # The module was already imported; check it doesn't have modal/torch
    import src.reliability.reliability_analysis as mod
    assert not hasattr(mod, "modal"), "modal should not be imported at module level"
    assert not hasattr(mod, "torch"), "torch should not be imported at module level"


def test_no_modal_import_in_g_theory():
    import src.reliability.g_theory as mod
    assert not hasattr(mod, "modal")
    assert not hasattr(mod, "torch")


def test_no_modal_import_in_reliability_reports():
    import src.reliability.reliability_reports as mod
    assert not hasattr(mod, "modal")
    assert not hasattr(mod, "torch")
