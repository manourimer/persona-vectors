"""
Persona vector computation (Stage 2B).

Loads cached activations and computes one difference-of-means vector per
trait × layer combination.

No GPU, torch, or Modal required — this is pure NumPy.

Algorithm (mirrors the Persona Vectors paper):
    vector = mean(positive_activations) − mean(negative_activations)
    if normalize: vector = vector / ||vector||₂

Public API
----------
    compute_trait_vector(pos_acts, neg_acts, normalize) -> np.ndarray
    compute_all_vectors(activation_metadata, candidate_layers,
                        traits, normalize, out_dir)     -> list[PersonaVectorMeta]
    save_vector_metadata(records, out_dir)
    load_vector_metadata(path)                         -> list[PersonaVectorMeta]
    load_vector(meta)                                  -> np.ndarray
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.vectors.vector_data import ActivationRecord, PersonaVectorMeta


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_trait_vector(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute a difference-of-means trait vector.

    Args:
        pos_activations: Shape (n_positive, hidden_dim).
        neg_activations: Shape (n_negative, hidden_dim).
        normalize:       If True, normalize to unit norm.

    Returns:
        Vector of shape (hidden_dim,).
    """
    if pos_activations.ndim != 2:
        raise ValueError(f"pos_activations must be 2-D, got shape {pos_activations.shape}")
    if neg_activations.ndim != 2:
        raise ValueError(f"neg_activations must be 2-D, got shape {neg_activations.shape}")
    if pos_activations.shape[1] != neg_activations.shape[1]:
        raise ValueError(
            f"Hidden dim mismatch: {pos_activations.shape[1]} vs {neg_activations.shape[1]}"
        )

    pos_mean = pos_activations.mean(axis=0)
    neg_mean = neg_activations.mean(axis=0)
    vector = pos_mean - neg_mean

    if normalize:
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

    return vector.astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline: records → vectors
# ---------------------------------------------------------------------------


def compute_all_vectors(
    records: list[ActivationRecord],
    candidate_layers: list[int],
    traits: list[str],
    normalize: bool = True,
    out_dir: str | Path = "outputs/vector_construction/persona_vectors",
) -> list[PersonaVectorMeta]:
    """Compute one vector per trait × layer from saved activation records.

    Only uses extraction-split, retained activations.

    Args:
        records:          ActivationRecord list from save_activation_metadata.
        candidate_layers: Layers to build vectors at.
        traits:           Trait names to process.
        normalize:        Normalize vectors to unit norm.
        out_dir:          Where to save .npy vector files.

    Returns:
        List of PersonaVectorMeta (one per trait × layer).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index records: {(trait, pole, layer, split) → [ActivationRecord]}
    index: dict[tuple, list[ActivationRecord]] = {}
    for rec in records:
        key = (rec.trait, rec.pole, rec.layer, rec.split)
        index.setdefault(key, []).append(rec)

    metas: list[PersonaVectorMeta] = []

    for trait in traits:
        for layer in candidate_layers:
            pos_recs = index.get((trait, "positive", layer, "extraction"), [])
            neg_recs = index.get((trait, "negative", layer, "extraction"), [])

            if not pos_recs or not neg_recs:
                continue

            pos_acts = np.stack([np.load(r.activation_path) for r in pos_recs])
            neg_acts = np.stack([np.load(r.activation_path) for r in neg_recs])

            vector = compute_trait_vector(pos_acts, neg_acts, normalize=normalize)
            hidden_dim = vector.shape[0]

            vec_path = out_dir / f"{trait}_layer{layer}.npy"
            np.save(vec_path, vector)

            metas.append(
                PersonaVectorMeta(
                    trait=trait,
                    layer=layer,
                    vector_path=str(vec_path),
                    n_positive=len(pos_recs),
                    n_negative=len(neg_recs),
                    vector_method="difference_of_means",
                    normalization="unit_norm" if normalize else "none",
                    hidden_dim=hidden_dim,
                )
            )

    return metas


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_vector_metadata(
    metas: list[PersonaVectorMeta],
    out_dir: str | Path,
    stem: str = "persona_vector_metadata",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [m.to_dict() for m in metas]
    df = pd.DataFrame(rows)
    path = out_dir / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def load_vector_metadata(path: str | Path) -> list[PersonaVectorMeta]:
    df = pd.read_csv(path)
    return [
        PersonaVectorMeta(
            trait=str(row["trait"]),
            layer=int(row["layer"]),
            vector_path=str(row["vector_path"]),
            n_positive=int(row["n_positive"]),
            n_negative=int(row["n_negative"]),
            vector_method=str(row["vector_method"]),
            normalization=str(row["normalization"]),
            hidden_dim=int(row["hidden_dim"]),
        )
        for _, row in df.iterrows()
    ]


def load_vector(meta: PersonaVectorMeta) -> np.ndarray:
    """Load the .npy vector for one PersonaVectorMeta."""
    return np.load(meta.vector_path)
