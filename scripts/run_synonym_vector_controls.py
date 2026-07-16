"""
Run synonym vector convergent-validity control.

Requires synonym vectors to already exist (built via Stage 2B pipeline with
configs/synonym_vector_artifacts.yaml).

Usage:
    python scripts/run_synonym_vector_controls.py [--mvp-only] [--full-set] [--compare-only]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls.synonym_vectors import (
    load_synonym_config,
    run_synonym_similarity_analysis,
    run_synonym_projection_agreement,
    save_synonym_controls,
)

CONFIG_PATH = "configs/synonym_vector_artifacts.yaml"
SYNONYM_VECTOR_DIR = "outputs/controls/synonym_vectors/persona_vectors/"
ORIGINAL_VECTOR_DIR = "outputs/vector_construction/persona_vectors/"
ETHICS_WIDE = "outputs/ethics_projection/ethics_trait_projections_centered_wide.csv"
OUT_DIR = "outputs/controls/synonym_vectors/"

MVP_SET = ["truthfulness", "harm_avoidance", "impartiality", "empathy"]
LAYERS = [32, 40, 47]
ORIGINAL_TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]


def load_vectors_for_layer(layer: int, vector_dir: str, trait_ids: list[str]) -> dict:
    """Load .npy vector files for a given layer. Returns {trait_id: np.ndarray}."""
    vecs = {}
    vdir = Path(vector_dir)
    for trait_id in trait_ids:
        candidates = list(vdir.glob(f"{trait_id}_layer{layer}.npy"))
        if candidates:
            vecs[trait_id] = np.load(candidates[0])
    return vecs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvp-only", action="store_true", default=True)
    parser.add_argument("--full-set", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()

    synonym_ids = MVP_SET if not args.full_set else None  # None = all

    print("[synonym_controls] Loading synonym config...")
    config = load_synonym_config(CONFIG_PATH)
    if synonym_ids:
        config = {k: v for k, v in config.items() if k in synonym_ids}
    print(f"  Synonyms to analyse: {list(config.keys())}")

    # Check if vectors exist
    synonym_vec_dir = Path(SYNONYM_VECTOR_DIR)
    layer = LAYERS[0]  # Use primary layer for similarity analysis
    existing = {
        k for k in config
        if list(synonym_vec_dir.glob(f"{k}_layer{layer}.npy"))
    } if synonym_vec_dir.exists() else set()

    if not existing:
        print(
            "\nNo synonym vector .npy files found.\n"
            "To build synonym vectors:\n"
            "  1. Run the Stage 2B pipeline with configs/synonym_vector_artifacts.yaml\n"
            "  2. Copy the resulting .npy files to outputs/controls/synonym_vectors/\n"
            "  3. Re-run this script\n"
        )
        return

    print(f"\nFound vectors for: {sorted(existing)}")
    missing = set(config.keys()) - existing
    if missing:
        print(f"Missing vectors for: {sorted(missing)}")

    # Load original vectors
    original_vecs = load_vectors_for_layer(layer, ORIGINAL_VECTOR_DIR, ORIGINAL_TRAITS)
    if not original_vecs:
        print(f"ERROR: No original vectors found in {ORIGINAL_VECTOR_DIR} for layer {layer}")
        return

    # Load synonym vectors
    synonym_vectors_dict = {}
    for synonym_id in existing:
        vec_arr = np.load(synonym_vec_dir / f"{synonym_id}_layer{layer}.npy")
        synonym_vectors_dict[synonym_id] = {
            "vector": vec_arr,
            "parent_trait": config[synonym_id]["parent_trait"],
        }

    print("\n[synonym_controls] Running cosine similarity analysis...")
    sim_df = run_synonym_similarity_analysis(synonym_vectors_dict, original_vecs)
    print(sim_df.to_string())

    # Projection agreement
    agreement_df = None
    ethics_path = Path(ETHICS_WIDE)
    if ethics_path.exists():
        ethics_df = pd.read_csv(ETHICS_WIDE)
        original_col_map = {t: f"projection_{t}" for t in ORIGINAL_TRAITS}

        # Load synonym ETHICS projections if available
        synonym_proj_dict = {}
        for synonym_id in existing:
            proj_path = Path(OUT_DIR) / f"synonym_ethics_projections_layer{layer}.csv"
            if proj_path.exists():
                proj_df = pd.read_csv(proj_path)
                col = f"projection_{synonym_id}"
                if col in proj_df.columns:
                    synonym_proj_dict[synonym_id] = {
                        "parent_trait": config[synonym_id]["parent_trait"],
                        "layer": layer,
                        "projections": proj_df[col],
                    }

        if synonym_proj_dict:
            print("\n[synonym_controls] Running projection agreement analysis...")
            agreement_df = run_synonym_projection_agreement(ethics_df, synonym_proj_dict, original_col_map)
            print(agreement_df.to_string())
        else:
            print("\n[synonym_controls] No synonym ETHICS projections found.")
            print("  Run scripts/project_synonym_controls.py first to generate them.")

    save_synonym_controls(
        {"similarity_df": sim_df, "agreement_df": agreement_df},
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
