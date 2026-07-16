"""
Activation extraction scaffold for persona-vector construction (Stage 2B).

This module handles caching activations from the Gemma-3-12B model at
candidate layers.  The actual GPU forward passes run on Modal; this module
manages I/O and provides a mock mode for offline testing.

Token scope used here: RESPONSE TOKENS (mean over generated response positions).
This is the correct scope for vector CONSTRUCTION.
Compare: Stage 3+ (ETHICS projection / monitoring) will use LAST_PROMPT_TOKEN.

ETHICS items are NOT extracted here.  This module only processes responses
generated from the trait vector artifacts.

Public API
----------
    mock_extract(scored, candidate_layers, hidden_dim, out_dir) -> list[ActivationRecord]
    save_activation_metadata(records, out_dir)
    load_activation_metadata(path)                              -> list[ActivationRecord]
    load_activation(record)                                     -> np.ndarray
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.vectors.vector_data import ActivationRecord, ScoredResponse

# Gemma-3-12B hidden dimension (used as default in real mode).
GEMMA_12B_HIDDEN_DIM = 5376

# Mock hidden dimension (small, for fast tests).
MOCK_HIDDEN_DIM = 64


# ---------------------------------------------------------------------------
# Mock extraction
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _mock_activation(
    response: ScoredResponse,
    layer: int,
    hidden_dim: int,
) -> np.ndarray:
    """Return a synthetic activation vector with a weak pole-dependent signal.

    Positive-pole responses have slightly higher mean on the first dimension,
    so that computed vectors have non-trivial validation AUC.
    """
    base = _RNG.standard_normal(hidden_dim).astype(np.float32)
    # Inject a small systematic signal so the vector is learnable
    signal = np.zeros(hidden_dim, dtype=np.float32)
    signal[0] = 1.0  # first component carries the trait signal
    direction = 1.0 if response.pole == "positive" else -1.0
    return base + direction * 0.8 * signal


def mock_extract(
    scored: list[ScoredResponse],
    candidate_layers: list[int],
    out_dir: str | Path,
    hidden_dim: int = MOCK_HIDDEN_DIM,
) -> list[ActivationRecord]:
    """Generate random activations and save them as .npy files.

    Only processes responses where keep_for_vector_extraction=True.

    Args:
        scored:           Scored responses (all splits).
        candidate_layers: Layer indices to extract.
        out_dir:          Root output directory.
        hidden_dim:       Activation vector size (use MOCK_HIDDEN_DIM for tests).

    Returns:
        List of ActivationRecord metadata objects.
    """
    out_dir = Path(out_dir)
    retained = [r for r in scored if r.keep_for_vector_extraction]
    records: list[ActivationRecord] = []

    for response in retained:
        for layer in candidate_layers:
            # Save to: out_dir/activations/{trait}_{pole}/{response_id}_layer{layer}.npy
            subdir = out_dir / "activations" / f"{response.trait}_{response.pole}"
            subdir.mkdir(parents=True, exist_ok=True)
            fname = f"{response.response_id}_layer{layer}.npy"
            fpath = subdir / fname
            act = _mock_activation(response, layer, hidden_dim)
            np.save(fpath, act)

            records.append(
                ActivationRecord(
                    response_id=response.response_id,
                    trait=response.trait,
                    pole=response.pole,
                    split=response.split,
                    layer=layer,
                    activation_path=str(fpath),
                    pooling_method="mean_response_token",
                    hidden_dim=hidden_dim,
                )
            )

    return records


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_activation_metadata(
    records: list[ActivationRecord],
    out_dir: str | Path,
    stem: str = "activation_metadata",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in records]
    df = pd.DataFrame(rows)
    path = out_dir / f"{stem}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_activation_metadata(path: str | Path) -> list[ActivationRecord]:
    df = pd.read_parquet(path)
    return [
        ActivationRecord(
            response_id=str(row["response_id"]),
            trait=str(row["trait"]),
            pole=str(row["pole"]),
            split=str(row["split"]),
            layer=int(row["layer"]),
            activation_path=str(row["activation_path"]),
            pooling_method=str(row["pooling_method"]),
            hidden_dim=int(row["hidden_dim"]),
        )
        for _, row in df.iterrows()
    ]


def load_activation(record: ActivationRecord) -> np.ndarray:
    """Load the numpy array for one ActivationRecord."""
    return np.load(record.activation_path)
