"""
Stage 4C: Project reliability variant activations onto persona vectors.

Pure Python (pandas + numpy).  No GPU, torch, Modal, or transformers.

Public API
----------
    load_persona_vectors(vector_dir, metadata_path, validation_path, min_auc)
                                                     -> dict[str, np.ndarray]
    load_activation_metadata(metadata_path)          -> pd.DataFrame
    load_activation(activation_path)                 -> np.ndarray
    compute_projections(metadata_df, vector_dict, layers) -> pd.DataFrame
    mean_center_projections(long_df)                 -> pd.DataFrame
    to_wide_format(long_df, include_text)            -> pd.DataFrame
    save_projections(long_df, wide_df, out_dir, preprocessing_label)
                                                     -> dict[str, Path]
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

TRAITS: list[str] = ["honesty", "harmlessness", "fairness", "compassion"]

PREPROCESSING_RAW = "raw"
PREPROCESSING_CENTERED = "mean_centered"


# ---------------------------------------------------------------------------
# Persona vector loading
# ---------------------------------------------------------------------------


def load_persona_vectors(
    vector_dir: str | Path,
    metadata_path: str | Path,
    validation_path: str | Path | None = None,
    min_auc: float = 0.75,
) -> dict[str, np.ndarray]:
    """Load validated persona vectors as a dict keyed by trait name.

    Loads vectors for all traits across all layers recorded in metadata.
    If validation_path is provided, skips vectors with AUC < min_auc.

    Args:
        vector_dir:      Directory containing .npy vector files.
        metadata_path:   Path to persona_vector_metadata.csv.
        validation_path: Optional path to vector_validation_results.csv.
        min_auc:         Minimum AUC threshold for including a vector.

    Returns:
        dict keyed by "{trait}_layer{N}" -> np.ndarray (float32, unit-norm).
    """
    vector_dir = Path(vector_dir)
    meta_df = pd.read_csv(metadata_path)

    # Build optional AUC lookup
    auc_lookup: dict[tuple[str, int], float] = {}
    if validation_path is not None:
        val_path = Path(validation_path)
        if val_path.exists():
            val_df = pd.read_csv(val_path)
            for _, row in val_df.iterrows():
                auc_lookup[(str(row["trait"]), int(row["layer"]))] = float(row["auc"])

    vectors: dict[str, np.ndarray] = {}
    for _, row in meta_df.iterrows():
        trait = str(row["trait"])
        layer = int(row["layer"])
        key = f"{trait}_layer{layer}"

        # AUC gate
        if auc_lookup:
            auc = auc_lookup.get((trait, layer), 0.0)
            if auc < min_auc:
                continue

        vpath = Path(str(row["vector_path"]))
        if not vpath.is_absolute():
            vpath = vector_dir / vpath
        if not vpath.exists():
            raise FileNotFoundError(f"Persona vector not found: {vpath}")

        vectors[key] = np.load(vpath).astype(np.float32)

    if not vectors:
        raise ValueError(
            "No persona vectors loaded (check min_auc threshold or metadata path)."
        )

    return vectors


# ---------------------------------------------------------------------------
# Activation metadata and file loading
# ---------------------------------------------------------------------------


def load_activation_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load the reliability activation metadata parquet produced by the Modal app."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reliability activation metadata not found: {path}\n"
            "Run scripts/extract_reliability_variant_activations.py first."
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_activation(activation_path: str | Path) -> np.ndarray:
    """Load a single .npy activation file and return as a 1D float32 array."""
    path = Path(activation_path)
    if not path.exists():
        raise FileNotFoundError(f"Activation file not found: {path}")
    arr = np.load(path)
    return arr.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Core projection computation
# ---------------------------------------------------------------------------


def compute_projections(
    metadata_df: pd.DataFrame,
    vector_dict: dict[str, np.ndarray],
    layers: list[int],
) -> pd.DataFrame:
    """Compute dot-product projections for all (variant, layer, trait) combos.

    Args:
        metadata_df: Reliability activation metadata DataFrame.
                     Must have columns: item_id, variant_id, variant_type,
                     paraphrase_id, framing, source_split, primary_trait,
                     scenario_text_variant, layer, activation_path, token_position.
        vector_dict: Dict from load_persona_vectors: "{trait}_layer{N}" -> array.
        layers:      Layer indices to project at.

    Returns:
        Long-format DataFrame with one row per (variant, layer, projected_trait).
        Columns: item_id, variant_id, variant_type, paraphrase_id, framing,
                 source_split, primary_trait, projected_trait, layer,
                 projection, projection_preprocessing, vector_path, activation_path.
    """
    layer_set = set(layers)
    rows: list[dict] = []

    for _, meta_row in metadata_df.iterrows():
        layer = int(meta_row["layer"])
        if layer not in layer_set:
            continue

        apath = str(meta_row["activation_path"])
        try:
            act = load_activation(apath)
        except FileNotFoundError:
            continue

        item_id = str(meta_row["item_id"])
        variant_id = str(meta_row["variant_id"])
        variant_type = str(meta_row.get("variant_type", ""))
        paraphrase_id = str(meta_row.get("paraphrase_id", ""))
        framing = str(meta_row.get("framing", ""))
        source_split = str(meta_row.get("source_split", ""))
        primary_trait = str(meta_row.get("primary_trait", ""))
        scenario_text_variant = str(meta_row.get("scenario_text_variant", ""))

        for trait in TRAITS:
            vec_key = f"{trait}_layer{layer}"
            if vec_key not in vector_dict:
                continue

            projection = float(np.dot(act, vector_dict[vec_key]))

            rows.append(
                {
                    "item_id": item_id,
                    "variant_id": variant_id,
                    "variant_type": variant_type,
                    "paraphrase_id": paraphrase_id,
                    "framing": framing,
                    "source_split": source_split,
                    "primary_trait": primary_trait,
                    "scenario_text_variant": scenario_text_variant,
                    "projected_trait": trait,
                    "layer": layer,
                    "projection": projection,
                    "projection_preprocessing": PREPROCESSING_RAW,
                    "vector_path": vec_key,
                    "activation_path": apath,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mean centering
# ---------------------------------------------------------------------------


def mean_center_projections(long_df: pd.DataFrame) -> pd.DataFrame:
    """Center projections per (layer, projected_trait) across all accepted variants.

    Computes the mean projection value within each (layer, projected_trait) group
    across the full input DataFrame, then subtracts it.  This removes the shared
    baseline direction so values reflect item-specific variation.

    Args:
        long_df: Raw projection DataFrame from compute_projections.

    Returns:
        New DataFrame with projections centered, projection_preprocessing set to
        "mean_centered", and an additional column "centering_mean".
    """
    centered = long_df.copy()
    group_means = (
        centered.groupby(["layer", "projected_trait"])["projection"]
        .transform("mean")
    )
    centered["centering_mean"] = group_means
    centered["projection"] = centered["projection"] - group_means
    centered["projection_preprocessing"] = PREPROCESSING_CENTERED
    return centered


# ---------------------------------------------------------------------------
# Wide format
# ---------------------------------------------------------------------------


def to_wide_format(long_df: pd.DataFrame, include_text: bool = True) -> pd.DataFrame:
    """Pivot long-format projections to wide: one row per (variant, layer).

    Args:
        long_df:      Long-format projection DataFrame.
        include_text: If True, include scenario_text_original and
                      scenario_text_variant columns (default True).

    Returns:
        Wide DataFrame with columns:
            item_id, variant_id, variant_type, paraphrase_id, framing,
            source_split, primary_trait, [scenario_text_original,]
            [scenario_text_variant,] layer, projection_preprocessing,
            projection_honesty, projection_harmlessness,
            projection_fairness, projection_compassion
    """
    # Need scenario_text_original if present
    meta_cols = [
        "item_id", "variant_id", "variant_type", "paraphrase_id",
        "framing", "source_split", "primary_trait",
    ]
    text_cols = []
    if include_text:
        if "scenario_text_original" in long_df.columns:
            text_cols.append("scenario_text_original")
        if "scenario_text_variant" in long_df.columns:
            text_cols.append("scenario_text_variant")

    index_cols = meta_cols + text_cols + ["layer", "projection_preprocessing"]

    pivot = long_df.pivot_table(
        index=index_cols,
        columns="projected_trait",
        values="projection",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    # Rename trait columns to projection_{trait}
    for trait in TRAITS:
        if trait in pivot.columns:
            pivot = pivot.rename(columns={trait: f"projection_{trait}"})

    # Ensure all projection columns exist
    for trait in TRAITS:
        col = f"projection_{trait}"
        if col not in pivot.columns:
            pivot[col] = float("nan")

    # Enforce column order
    final_cols = meta_cols + text_cols + ["layer", "projection_preprocessing"] + [
        f"projection_{t}" for t in TRAITS
    ]
    pivot = pivot[[c for c in final_cols if c in pivot.columns]]

    return pivot.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_projections(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    out_dir: str | Path,
    preprocessing_label: str = "centered",
) -> dict[str, Path]:
    """Save raw and/or centered projection DataFrames.

    Saves long and wide as parquet + CSV with descriptive names.
    Also copies centered versions to canonical names
    (reliability_trait_projections_long/wide.*).

    Args:
        long_df:              Long projection DataFrame.
        wide_df:              Wide projection DataFrame.
        out_dir:              Output directory.
        preprocessing_label:  "raw" | "centered" | "both" — determines filenames.

    Returns:
        dict mapping logical name -> Path for each file saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    def _save_pair(ldf: pd.DataFrame, wdf: pd.DataFrame, stem_long: str, stem_wide: str):
        lp = out_dir / f"{stem_long}.parquet"
        lc = out_dir / f"{stem_long}.csv"
        wp = out_dir / f"{stem_wide}.parquet"
        wc = out_dir / f"{stem_wide}.csv"
        ldf.to_parquet(lp, index=False)
        ldf.to_csv(lc, index=False)
        wdf.to_parquet(wp, index=False)
        wdf.to_csv(wc, index=False)
        saved[stem_long + ".parquet"] = lp
        saved[stem_long + ".csv"] = lc
        saved[stem_wide + ".parquet"] = wp
        saved[stem_wide + ".csv"] = wc
        return lp, lc, wp, wc

    if preprocessing_label in ("raw", "both"):
        raw_long = long_df[long_df["projection_preprocessing"] == PREPROCESSING_RAW]
        raw_wide = wide_df[wide_df["projection_preprocessing"] == PREPROCESSING_RAW] if "projection_preprocessing" in wide_df.columns else wide_df
        _save_pair(
            raw_long, raw_wide,
            "reliability_trait_projections_long_raw",
            "reliability_trait_projections_wide_raw",
        )

    if preprocessing_label in ("centered", "both"):
        ctr_long = long_df[long_df["projection_preprocessing"] == PREPROCESSING_CENTERED]
        ctr_wide = wide_df[wide_df["projection_preprocessing"] == PREPROCESSING_CENTERED] if "projection_preprocessing" in wide_df.columns else wide_df
        lp, lc, wp, wc = _save_pair(
            ctr_long, ctr_wide,
            "reliability_trait_projections_long_centered",
            "reliability_trait_projections_wide_centered",
        )

        # Copy centered as canonical default names
        for src, dst_name in [
            (lp, "reliability_trait_projections_long.parquet"),
            (lc, "reliability_trait_projections_long.csv"),
            (wp, "reliability_trait_projections_wide.parquet"),
            (wc, "reliability_trait_projections_wide.csv"),
        ]:
            dst = out_dir / dst_name
            shutil.copy2(src, dst)
            saved[dst_name] = dst

    return saved
