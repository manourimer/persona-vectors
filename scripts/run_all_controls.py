"""
Run all CPU-safe controls in sequence.

Usage:
    python scripts/run_all_controls.py [options]

Options:
    --include-synonym    Also run synonym controls (requires synonym vectors to exist)
    --skip-random        Skip random vector control
    --skip-shuffled      Skip shuffled label control
    --skip-grouping      Skip permuted grouping control
    --skip-exact         Skip exact duplicate control
    --skip-preprocessing Skip preprocessing robustness control
    --skip-positive      Skip positive controls (contrast validation + synthetic scenarios)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls.random_vectors import run_random_vector_control, save_random_vector_control, compare_to_real
from src.controls.shuffled_labels import run_shuffled_label_control, save_shuffled_label_control
from src.controls.exact_duplicates import run_exact_duplicate_control, save_exact_duplicate_control
from src.controls.positive_controls import (
    run_contrast_validation_control, save_contrast_validation_control,
    build_synthetic_scenario_scaffold, save_synthetic_scenario_scaffold,
)
from src.controls.preprocessing_controls import (
    run_preprocessing_comparison, run_layer_robustness, save_preprocessing_controls,
)
from src.controls.control_reports import generate_controls_report, save_controls_report
from src.controls.preprocessing_controls import _struct_metrics

ETHICS_WIDE = "outputs/ethics_projection/ethics_trait_projections_centered_wide.csv"
ETHICS_RAW = "outputs/ethics_projection/ethics_trait_projections_wide.csv"
RELIABILITY_WIDE = "outputs/reliability_projection/reliability_trait_projections_wide_centered.parquet"
RELIABILITY_LONG = "outputs/reliability_projection/reliability_trait_projections_long_centered.parquet"
VALIDATION_RESULTS = "outputs/vector_construction/vector_validation_results.csv"
VECTOR_METADATA = "outputs/vector_construction/persona_vector_metadata.csv"
OUT_DIR = "outputs/controls/"
DATA_DIR = "data/processed/"
LAYERS = [32, 40, 47]
TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]


def safe_run(name: str, fn, *args, **kwargs):
    """Run a control; catch exceptions and return None on failure."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print("="*60)
    try:
        result = fn(*args, **kwargs)
        print(f"[{name}] DONE")
        return result
    except Exception as e:
        print(f"[{name}] ERROR: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-synonym", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-shuffled", action="store_true")
    parser.add_argument("--skip-grouping", action="store_true")
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-positive", action="store_true")
    args = parser.parse_args()

    all_results = {}

    # Load shared data
    print("Loading shared data files...")
    ethics_df = pd.read_csv(ETHICS_WIDE)
    rel_df = pd.read_parquet(RELIABILITY_WIDE)
    print(f"  ETHICS: {ethics_df.shape}, Reliability wide: {rel_df.shape}")

    # ---------------------------------------------------------------------------
    # Random vector control
    # ---------------------------------------------------------------------------
    if not args.skip_random:
        real_metrics = {}
        for layer in LAYERS:
            sub = rel_df[rel_df["layer"] == layer] if "layer" in rel_df.columns else ethics_df
            m = _struct_metrics(sub)
            real_metrics[layer] = {k: v for k, v in m.items() if "reliability" not in k}

        rv_results = safe_run(
            "random_vector_control",
            run_random_vector_control,
            ethics_df, rel_df,
            activation_paths_by_layer=None,
            n_repeats=100,
            random_seed=42,
        )
        if rv_results:
            rv_results["compare_df"] = compare_to_real(rv_results["distributions_df"], real_metrics)
            save_random_vector_control(rv_results, OUT_DIR)
            all_results["random_vector"] = rv_results

    # ---------------------------------------------------------------------------
    # Shuffled label control
    # ---------------------------------------------------------------------------
    if not args.skip_shuffled:
        sl_results = safe_run(
            "shuffled_label_control",
            run_shuffled_label_control,
            ethics_df,
            n_permutations=10000,
            random_seed=42,
        )
        if sl_results:
            save_shuffled_label_control(sl_results, OUT_DIR)
            all_results["shuffled_label"] = sl_results

    # ---------------------------------------------------------------------------
    # Permuted grouping control (may be slow)
    # ---------------------------------------------------------------------------
    if not args.skip_grouping:
        long_path = Path(RELIABILITY_LONG)
        if long_path.exists():
            from src.controls.permuted_grouping import run_permuted_grouping_control, save_permuted_grouping_control
            long_df = pd.read_parquet(RELIABILITY_LONG)
            pg_results = safe_run(
                "permuted_grouping_control",
                run_permuted_grouping_control,
                long_df,
                n_permutations=1000,
                random_seed=42,
            )
            if pg_results:
                save_permuted_grouping_control(pg_results, OUT_DIR)
                all_results["permuted_grouping"] = pg_results
        else:
            print(f"[permuted_grouping] Skipping — {RELIABILITY_LONG} not found.")

    # ---------------------------------------------------------------------------
    # Exact duplicate control
    # ---------------------------------------------------------------------------
    if not args.skip_exact:
        ed_results = safe_run(
            "exact_duplicate_control",
            run_exact_duplicate_control,
            ethics_df,
            k=3,
        )
        if ed_results:
            save_exact_duplicate_control(ed_results, OUT_DIR)
            all_results["exact_duplicate"] = ed_results

    # ---------------------------------------------------------------------------
    # Positive controls
    # ---------------------------------------------------------------------------
    if not args.skip_positive:
        cv_results = safe_run(
            "contrast_validation_control",
            run_contrast_validation_control,
            VALIDATION_RESULTS,
            VECTOR_METADATA,
        )
        if cv_results:
            save_contrast_validation_control(cv_results, OUT_DIR)
            all_results["contrast_validation"] = cv_results

        synth_path = Path(DATA_DIR) / "synthetic_moral_scenarios.csv"
        if synth_path.exists():
            synth_df = pd.read_csv(synth_path)
        else:
            synth_df = build_synthetic_scenario_scaffold(n_per_trait=25)
        save_synthetic_scenario_scaffold(synth_df, OUT_DIR, DATA_DIR)
        all_results["synthetic_scenarios"] = {"df": synth_df}

    # ---------------------------------------------------------------------------
    # Preprocessing control
    # ---------------------------------------------------------------------------
    if not args.skip_preprocessing:
        raw_path = Path(ETHICS_RAW)
        raw_df = pd.read_csv(ETHICS_RAW) if raw_path.exists() else ethics_df.copy()

        prep_df = safe_run(
            "preprocessing_comparison",
            run_preprocessing_comparison,
            raw_df, ethics_df, LAYERS, TRAITS,
        )
        layer_df = safe_run(
            "layer_robustness",
            run_layer_robustness,
            ethics_df, LAYERS, TRAITS,
        )
        if prep_df is not None or layer_df is not None:
            prep_results = {"preprocessing_df": prep_df, "layer_robustness_df": layer_df}
            save_preprocessing_controls(prep_results, OUT_DIR)
            all_results["preprocessing"] = prep_results

    # ---------------------------------------------------------------------------
    # Synonym controls (optional)
    # ---------------------------------------------------------------------------
    if args.include_synonym:
        from src.controls.synonym_vectors import (
            load_synonym_config, run_synonym_similarity_analysis
        )
        import numpy as np
        config = load_synonym_config("configs/synonym_vector_artifacts.yaml")
        vdir = Path("outputs/controls/synonym_vectors/")
        odir = Path("outputs/vector_construction/persona_vectors/")
        layer = 32
        original_vecs = {t: np.load(odir / f"{t}_layer{layer}.npy")
                         for t in TRAITS if (odir / f"{t}_layer{layer}.npy").exists()}
        synonym_vecs = {}
        for sid, info in config.items():
            p = vdir / f"{sid}_layer{layer}.npy"
            if p.exists():
                synonym_vecs[sid] = {"vector": np.load(p), "parent_trait": info["parent_trait"]}
        if synonym_vecs and original_vecs:
            sim_df = run_synonym_similarity_analysis(synonym_vecs, original_vecs)
            all_results["synonym_similarity"] = sim_df

    # ---------------------------------------------------------------------------
    # Generate report
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Generating controls report...")
    report_md = generate_controls_report(all_results)
    report_path = save_controls_report(report_md, OUT_DIR)
    print(f"Report saved: {report_path}")

    print(f"\n{'='*60}")
    print("CONTROLS SUITE COMPLETE")
    print(f"Output directory: {OUT_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
