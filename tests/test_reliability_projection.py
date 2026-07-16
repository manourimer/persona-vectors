"""
Tests for Stage 4C: reliability variant projection.

All tests run without GPU, torch, Modal, or transformers.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]


def _make_variant_df(n_items: int = 4, n_paraphrases: int = 3) -> pd.DataFrame:
    """Create a minimal variant DataFrame for testing."""
    rows = []
    for i in range(n_items):
        item_id = f"item_{i:03d}"
        original_text = f"Original scenario for item {i}."

        # Original variant
        rows.append(
            {
                "item_id": item_id,
                "variant_id": f"{item_id}_original",
                "variant_type": "original",
                "paraphrase_id": "original",
                "framing": "neutral",
                "source_split": "commonsense",
                "primary_trait": TRAITS[i % 4],
                "scenario_text_original": original_text,
                "scenario_text_variant": original_text,
                "keep_variant": True,
            }
        )

        for p in range(n_paraphrases):
            rows.append(
                {
                    "item_id": item_id,
                    "variant_id": f"{item_id}_para_{p}",
                    "variant_type": "paraphrase",
                    "paraphrase_id": f"para_{p}",
                    "framing": "neutral",
                    "source_split": "commonsense",
                    "primary_trait": TRAITS[i % 4],
                    "scenario_text_original": original_text,
                    "scenario_text_variant": f"Paraphrase {p} of scenario {i}.",
                    "keep_variant": True,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Import guard: no heavy imports at module level
# ---------------------------------------------------------------------------


def test_no_heavy_imports_variant_projection():
    """variant_projection.py must not import torch/transformers/modal at module level."""
    for mod_name in ["torch", "transformers", "modal"]:
        was_in_sys = mod_name in sys.modules
        # Force module re-import by removing from sys.modules temporarily
        saved = sys.modules.pop(mod_name, None)
        try:
            import src.reliability.variant_projection as vp  # noqa: F401
            importlib.reload(vp)
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved
            elif mod_name in sys.modules:
                del sys.modules[mod_name]
    # If we got here without ImportError, the module is clean
    assert True


def test_no_heavy_imports_compute_variant_projections():
    """compute_variant_projections.py must not import torch/transformers/modal."""
    import src.reliability.compute_variant_projections as cvp  # noqa: F401
    importlib.reload(cvp)
    assert True


def test_no_heavy_imports_diagnostics():
    """variant_projection_diagnostics.py must not import torch/transformers/modal."""
    import src.reliability.variant_projection_diagnostics as diag  # noqa: F401
    importlib.reload(diag)
    assert True


# ---------------------------------------------------------------------------
# load_variant_bank
# ---------------------------------------------------------------------------


def test_load_variant_bank_filters_keep_variant(tmp_path):
    from src.reliability.variant_projection import load_variant_bank

    df = _make_variant_df(n_items=2, n_paraphrases=2)
    # Flip some rows to False
    df.loc[0, "keep_variant"] = False
    df.loc[3, "keep_variant"] = False

    pq = tmp_path / "variants.parquet"
    df.to_parquet(pq, index=False)

    result = load_variant_bank(pq)
    assert result["keep_variant"].all()
    assert len(result) == len(df) - 2


def test_load_variant_bank_validates_required_columns(tmp_path):
    from src.reliability.variant_projection import load_variant_bank

    df = _make_variant_df()
    df = df.drop(columns=["variant_id"])  # remove required column

    pq = tmp_path / "bad_variants.parquet"
    df.to_parquet(pq, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_variant_bank(pq)


def test_load_variant_bank_raises_on_missing_file():
    from src.reliability.variant_projection import load_variant_bank

    with pytest.raises(FileNotFoundError):
        load_variant_bank("/nonexistent/path/variants.parquet")


# ---------------------------------------------------------------------------
# build_projection_jobs
# ---------------------------------------------------------------------------


def test_build_projection_jobs_preserves_item_and_variant_id():
    from src.reliability.variant_projection import build_projection_jobs

    df = _make_variant_df(n_items=3, n_paraphrases=2)
    jobs = build_projection_jobs(df, target_layers=[32])

    assert "item_id" in jobs.columns
    assert "variant_id" in jobs.columns
    assert set(jobs["item_id"].unique()) == set(df["item_id"].unique())
    assert set(jobs["variant_id"].unique()) == set(df["variant_id"].unique())


def test_build_projection_jobs_uses_variant_text_not_original():
    from src.reliability.variant_projection import build_projection_jobs

    df = _make_variant_df(n_items=2, n_paraphrases=2)
    jobs = build_projection_jobs(df, target_layers=[32])

    for _, row in jobs.iterrows():
        # scenario_text_variant should appear in prompt_text
        assert str(row["scenario_text_variant"]).strip()[:30] in row["prompt_text"], (
            f"variant_id={row['variant_id']}: scenario_text_variant not in prompt_text"
        )

    # At least some rows have different scenario_text_original vs scenario_text_variant
    paraphrase_rows = jobs[jobs["variant_type"] == "paraphrase"]
    assert len(paraphrase_rows) > 0
    differ = paraphrase_rows["scenario_text_original"] != paraphrase_rows["scenario_text_variant"]
    assert differ.any(), "Expected at least some paraphrases to differ from originals"


def test_build_projection_jobs_multi_layer():
    from src.reliability.variant_projection import build_projection_jobs

    df = _make_variant_df(n_items=2, n_paraphrases=1)
    n_variants = len(df)
    layers = [32, 40, 47]
    jobs = build_projection_jobs(df, target_layers=layers)

    assert len(jobs) == n_variants * len(layers)
    assert set(jobs["target_layer"].unique()) == set(layers)


# ---------------------------------------------------------------------------
# compute_projections
# ---------------------------------------------------------------------------


def test_compute_projections_dot_products_correct(tmp_path):
    """Verify projection = dot(activation, vector) with known synthetic data."""
    from src.reliability.compute_variant_projections import compute_projections

    dim = 16
    rng = np.random.default_rng(0)

    # Create fake activations
    n_variants = 4
    layers = [32]
    act_dir = tmp_path / "activations"
    act_dir.mkdir()

    meta_rows = []
    activations = {}
    for i in range(n_variants):
        vid = f"v{i:03d}"
        act = rng.standard_normal(dim).astype(np.float32)
        apath = act_dir / f"{vid}_layer32.npy"
        np.save(apath, act)
        activations[(vid, 32)] = act
        meta_rows.append(
            {
                "item_id": f"item_{i}",
                "variant_id": vid,
                "variant_type": "paraphrase",
                "paraphrase_id": "para_0",
                "framing": "neutral",
                "source_split": "commonsense",
                "primary_trait": "honesty",
                "scenario_text_variant": "A scenario.",
                "layer": 32,
                "activation_path": str(apath),
                "token_position": "last_prompt_token",
            }
        )

    meta_df = pd.DataFrame(meta_rows)

    # Create unit-norm vectors
    vectors = {}
    expected_projections: dict[tuple[str, str], float] = {}
    for trait in TRAITS:
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        vectors[f"{trait}_layer32"] = vec
        for i in range(n_variants):
            vid = f"v{i:03d}"
            expected_projections[(vid, trait)] = float(np.dot(activations[(vid, 32)], vec))

    result = compute_projections(meta_df, vectors, layers=[32])

    assert not result.empty
    for _, row in result.iterrows():
        vid = row["variant_id"]
        trait = row["projected_trait"]
        expected = expected_projections[(vid, trait)]
        assert abs(row["projection"] - expected) < 1e-5, (
            f"Projection mismatch for {vid}/{trait}: got {row['projection']}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# mean_center_projections
# ---------------------------------------------------------------------------


def test_mean_center_projections_near_zero_mean():
    from src.reliability.compute_variant_projections import (
        compute_projections,
        mean_center_projections,
    )
    from src.reliability.compute_variant_projections import PREPROCESSING_CENTERED

    # Build a simple DataFrame directly
    rng = np.random.default_rng(1)
    n = 20
    rows = []
    for i in range(n):
        for trait in TRAITS:
            rows.append(
                {
                    "item_id": f"item_{i}",
                    "variant_id": f"v{i}",
                    "variant_type": "paraphrase",
                    "paraphrase_id": "para_0",
                    "framing": "neutral",
                    "source_split": "commonsense",
                    "primary_trait": "honesty",
                    "scenario_text_variant": "Scenario.",
                    "projected_trait": trait,
                    "layer": 32,
                    "projection": float(rng.standard_normal()),
                    "projection_preprocessing": "raw",
                    "vector_path": f"{trait}_layer32",
                    "activation_path": f"/fake/{i}_layer32.npy",
                }
            )

    raw_df = pd.DataFrame(rows)
    centered = mean_center_projections(raw_df)

    assert centered["projection_preprocessing"].iloc[0] == PREPROCESSING_CENTERED

    # Mean per (layer, projected_trait) should be near zero
    for (layer, trait), grp in centered.groupby(["layer", "projected_trait"]):
        mean_val = grp["projection"].mean()
        assert abs(mean_val) < 1e-5, (
            f"Centered mean for layer={layer}, trait={trait} not near zero: {mean_val}"
        )


# ---------------------------------------------------------------------------
# to_wide_format
# ---------------------------------------------------------------------------


def test_to_wide_format_has_projection_columns():
    from src.reliability.compute_variant_projections import to_wide_format

    rows = []
    for i in range(5):
        for trait in TRAITS:
            rows.append(
                {
                    "item_id": f"item_{i}",
                    "variant_id": f"v{i}",
                    "variant_type": "original",
                    "paraphrase_id": "original",
                    "framing": "neutral",
                    "source_split": "commonsense",
                    "primary_trait": "honesty",
                    "scenario_text_original": "Original.",
                    "scenario_text_variant": "Scenario.",
                    "projected_trait": trait,
                    "layer": 32,
                    "projection": float(i),
                    "projection_preprocessing": "mean_centered",
                    "vector_path": f"{trait}_layer32",
                    "activation_path": f"/fake/{i}_layer32.npy",
                }
            )

    long_df = pd.DataFrame(rows)
    wide = to_wide_format(long_df, include_text=True)

    for trait in TRAITS:
        assert f"projection_{trait}" in wide.columns, (
            f"Expected projection_{trait} in wide columns: {wide.columns.tolist()}"
        )

    assert "item_id" in wide.columns
    assert "variant_id" in wide.columns
    assert "layer" in wide.columns
    assert "scenario_text_variant" in wide.columns


def test_to_wide_format_exclude_text():
    from src.reliability.compute_variant_projections import to_wide_format

    rows = []
    for i in range(3):
        for trait in TRAITS:
            rows.append(
                {
                    "item_id": f"item_{i}",
                    "variant_id": f"v{i}",
                    "variant_type": "original",
                    "paraphrase_id": "original",
                    "framing": "neutral",
                    "source_split": "commonsense",
                    "primary_trait": "honesty",
                    "scenario_text_original": "Original.",
                    "scenario_text_variant": "Scenario.",
                    "projected_trait": trait,
                    "layer": 32,
                    "projection": 1.0,
                    "projection_preprocessing": "raw",
                    "vector_path": f"{trait}_layer32",
                    "activation_path": f"/fake/act.npy",
                }
            )

    long_df = pd.DataFrame(rows)
    wide = to_wide_format(long_df, include_text=False)

    assert "scenario_text_variant" not in wide.columns
    assert "scenario_text_original" not in wide.columns


# ---------------------------------------------------------------------------
# save_projections
# ---------------------------------------------------------------------------


def test_save_projections_expected_filenames(tmp_path):
    from src.reliability.compute_variant_projections import save_projections, PREPROCESSING_CENTERED

    rows = []
    for i in range(3):
        for trait in TRAITS:
            rows.append(
                {
                    "item_id": f"item_{i}",
                    "variant_id": f"v{i}",
                    "variant_type": "original",
                    "paraphrase_id": "original",
                    "framing": "neutral",
                    "source_split": "commonsense",
                    "primary_trait": "honesty",
                    "scenario_text_original": "Orig.",
                    "scenario_text_variant": "Scenario.",
                    "projected_trait": trait,
                    "layer": 32,
                    "projection": float(i),
                    "projection_preprocessing": PREPROCESSING_CENTERED,
                    "vector_path": f"{trait}_layer32",
                    "activation_path": f"/fake/{i}.npy",
                }
            )

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot_table(
        index=["item_id", "variant_id", "variant_type", "paraphrase_id", "framing",
               "source_split", "primary_trait", "layer", "projection_preprocessing"],
        columns="projected_trait",
        values="projection",
        aggfunc="first",
    ).reset_index()

    saved = save_projections(long_df, wide_df, tmp_path, preprocessing_label="centered")

    expected = [
        "reliability_trait_projections_long_centered.parquet",
        "reliability_trait_projections_long_centered.csv",
        "reliability_trait_projections_wide_centered.parquet",
        "reliability_trait_projections_wide_centered.csv",
        "reliability_trait_projections_long.parquet",
        "reliability_trait_projections_long.csv",
        "reliability_trait_projections_wide.parquet",
        "reliability_trait_projections_wide.csv",
    ]
    for name in expected:
        assert name in saved, f"Expected {name} in saved paths"
        assert saved[name].exists(), f"Expected file to exist: {saved[name]}"


# ---------------------------------------------------------------------------
# compute_diagnostics
# ---------------------------------------------------------------------------


def test_compute_diagnostics_detects_missing_variants():
    from src.reliability.variant_projection_diagnostics import compute_diagnostics

    # Build a long_df where one item has fewer variants
    rows = []
    # item_0: 4 variants (1 original + 3 paraphrases)
    # item_1: 1 variant (original only — missing paraphrases)
    for item_i, n_variants in [(0, 4), (1, 1)]:
        for v in range(n_variants):
            for trait in TRAITS:
                rows.append(
                    {
                        "item_id": f"item_{item_i}",
                        "variant_id": f"item_{item_i}_v{v}",
                        "variant_type": "original" if v == 0 else "paraphrase",
                        "paraphrase_id": "original" if v == 0 else f"para_{v}",
                        "framing": "neutral",
                        "source_split": "commonsense",
                        "primary_trait": TRAITS[item_i % 4],
                        "scenario_text_variant": "Scenario.",
                        "projected_trait": trait,
                        "layer": 32,
                        "projection": float(v + item_i),
                        "projection_preprocessing": "mean_centered",
                        "vector_path": f"{trait}_layer32",
                        "activation_path": f"/fake/{item_i}_{v}.npy",
                    }
                )

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot_table(
        index=["item_id", "variant_id", "variant_type", "paraphrase_id", "framing",
               "source_split", "primary_trait", "layer", "projection_preprocessing"],
        columns="projected_trait",
        values="projection",
        aggfunc="first",
    ).reset_index()
    for trait in TRAITS:
        if trait not in wide_df.columns:
            wide_df[trait] = float("nan")
        wide_df.rename(columns={trait: f"projection_{trait}"}, inplace=True)

    diag = compute_diagnostics(long_df, wide_df)

    assert "missing_variants" in diag
    assert "item_1" in diag["missing_variants"], (
        f"Expected 'item_1' in missing_variants: {diag['missing_variants']}"
    )


def test_compute_diagnostics_within_item_std():
    from src.reliability.variant_projection_diagnostics import compute_diagnostics

    # item_0: projections vary a lot across paraphrases
    # item_1: projections are stable
    rng = np.random.default_rng(42)
    rows = []
    for item_i, spread in [(0, 5.0), (1, 0.01)]:
        for v in range(4):
            for trait in TRAITS:
                proj = float(rng.standard_normal() * spread)
                rows.append(
                    {
                        "item_id": f"item_{item_i}",
                        "variant_id": f"item_{item_i}_v{v}",
                        "variant_type": "original" if v == 0 else "paraphrase",
                        "paraphrase_id": "original" if v == 0 else f"para_{v}",
                        "framing": "neutral",
                        "source_split": "commonsense",
                        "primary_trait": "honesty",
                        "scenario_text_variant": "Scenario.",
                        "projected_trait": trait,
                        "layer": 32,
                        "projection": proj,
                        "projection_preprocessing": "mean_centered",
                        "vector_path": f"{trait}_layer32",
                        "activation_path": f"/fake/{item_i}_{v}.npy",
                    }
                )

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot_table(
        index=["item_id", "variant_id", "variant_type", "paraphrase_id", "framing",
               "source_split", "primary_trait", "layer", "projection_preprocessing"],
        columns="projected_trait",
        values="projection",
        aggfunc="first",
    ).reset_index()
    for trait in TRAITS:
        if trait in wide_df.columns:
            wide_df.rename(columns={trait: f"projection_{trait}"}, inplace=True)

    diag = compute_diagnostics(long_df, wide_df)

    within_std = diag["within_item_std"]
    assert "within_item_std" in within_std.columns

    item0_stds = within_std[within_std["item_id"] == "item_0"]["within_item_std"]
    item1_stds = within_std[within_std["item_id"] == "item_1"]["within_item_std"]
    assert item0_stds.mean() > item1_stds.mean(), (
        "Expected item_0 (high spread) to have higher within-item std than item_1 (low spread)"
    )
