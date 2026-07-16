"""
Project item-bank activations onto synonym vectors.
Loads per-item activation .npy files using an activation metadata file,
stacks them, and dot-products with each synonym vector.

Defaults reproduce the original ETHICS-only behaviour. Pass --wide-path,
--act-metadata, --act-dir, and --out-dir to run this against a different
item bank's activations (e.g. the synthetic confound-controlled bank).

Usage:
    python scripts/project_synonym_controls.py

    python scripts/project_synonym_controls.py \\
        --wide-path outputs/synthetic_projection/ethics_trait_projections_centered_wide.csv \\
        --act-metadata outputs/synthetic_projection/ethics_activation_metadata.parquet \\
        --act-dir outputs/synthetic_projection/activations/ \\
        --out-dir outputs/synthetic_projection/synonym_vectors/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SYNONYM_VECTOR_DIR = "outputs/controls/synonym_vectors/persona_vectors/"
_WIDE_PATH = "outputs/ethics_projection/ethics_trait_projections_centered_wide.csv"
_ACT_METADATA = "outputs/ethics_projection/ethics_activation_metadata.parquet"
_ACT_DIR = "outputs/ethics_projection/activations/"
_OUT_DIR = "outputs/controls/synonym_vectors/"
_LAYERS = [32, 40, 47]
_SYNONYM_IDS = ["truthfulness", "harm_avoidance", "impartiality", "empathy"]
_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Project item-bank activations onto synonym vectors."
    )
    parser.add_argument("--synonym-vector-dir", default=_SYNONYM_VECTOR_DIR)
    parser.add_argument("--wide-path", default=_WIDE_PATH)
    parser.add_argument("--act-metadata", default=_ACT_METADATA)
    parser.add_argument("--act-dir", default=_ACT_DIR)
    parser.add_argument("--out-dir", default=_OUT_DIR)
    parser.add_argument("--layers", nargs="+", type=int, default=_LAYERS)
    args = parser.parse_args()

    vdir = Path(args.synonym_vector_dir)
    existing = {sid: {
        layer: vdir / f"{sid}_layer{layer}.npy"
        for layer in args.layers
        if (vdir / f"{sid}_layer{layer}.npy").exists()
    } for sid in _SYNONYM_IDS}
    existing = {k: v for k, v in existing.items() if v}

    if not existing:
        print("No synonym vectors found. Run Stage 2B pipeline first.")
        return

    print(f"Found synonym vectors for: {list(existing.keys())}")

    meta_path = _ROOT / args.act_metadata
    if not meta_path.exists():
        print(f"Activation metadata not found: {meta_path}")
        return

    meta = pd.read_parquet(meta_path)
    wide_df = pd.read_csv(_ROOT / args.wide_path)
    out_dir = _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer in args.layers:
        layer_meta = meta[meta["layer"] == layer].copy()
        if layer_meta.empty:
            print(f"  No activations found for layer {layer}")
            continue

        # Load and stack activations in item_id order
        acts_list, item_ids = [], []
        for _, row in layer_meta.iterrows():
            act_path = _ROOT / row["activation_path"]
            if act_path.exists():
                acts_list.append(np.load(act_path))
                item_ids.append(row["item_id"])

        if not acts_list:
            print(f"  No activation files found for layer {layer}")
            continue

        acts = np.stack(acts_list)  # (n_items, hidden_dim)
        print(f"  Layer {layer}: {acts.shape[0]} items, dim {acts.shape[1]}")

        results = pd.DataFrame({"item_id": item_ids})
        results = results.merge(
            wide_df[["item_id", "primary_trait"]], on="item_id", how="left"
        )

        for synonym_id, layer_paths in existing.items():
            if layer not in layer_paths:
                continue
            vec = np.load(layer_paths[layer])
            results[f"projection_{synonym_id}"] = acts @ vec

        out_path = out_dir / f"synonym_ethics_projections_layer{layer}.csv"
        results.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
