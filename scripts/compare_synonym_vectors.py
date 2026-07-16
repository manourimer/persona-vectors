"""
Compare synonym vectors to original trait vectors via cosine similarity.
Assumes synonym vectors exist in outputs/controls/synonym_vectors/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls.synonym_vectors import run_synonym_similarity_analysis, load_synonym_config

CONFIG_PATH = "configs/synonym_vector_artifacts.yaml"
SYNONYM_VECTOR_DIR = "outputs/controls/synonym_vectors/"
ORIGINAL_VECTOR_DIR = "outputs/vector_construction/persona_vectors/"
ORIGINAL_TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]
LAYER = 32


def main():
    config = load_synonym_config(CONFIG_PATH)
    vdir = Path(SYNONYM_VECTOR_DIR)
    odir = Path(ORIGINAL_VECTOR_DIR)

    original_vecs = {}
    for trait in ORIGINAL_TRAITS:
        path = odir / f"{trait}_layer{LAYER}.npy"
        if path.exists():
            original_vecs[trait] = np.load(path)
    if not original_vecs:
        print(f"No original vectors found in {odir}")
        return

    synonym_vectors_dict = {}
    for synonym_id, info in config.items():
        path = vdir / f"{synonym_id}_layer{LAYER}.npy"
        if path.exists():
            synonym_vectors_dict[synonym_id] = {
                "vector": np.load(path),
                "parent_trait": info["parent_trait"],
            }
    if not synonym_vectors_dict:
        print(f"No synonym vectors found in {vdir}")
        print("Run Stage 2B with synonym artifacts first.")
        return

    df = run_synonym_similarity_analysis(synonym_vectors_dict, original_vecs)
    print("\n=== Synonym Vector Cosine Similarity (layer 32) ===")
    print(df.to_string())

    n_correct = int(df["closest_matches_parent"].sum())
    print(f"\nClosest-parent match: {n_correct}/{len(df)}")


if __name__ == "__main__":
    main()
