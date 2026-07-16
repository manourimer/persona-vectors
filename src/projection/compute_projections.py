"""
Stage 3: Project ETHICS item activations onto persona vectors.

Loads cached activation .npy files and persona vector .npy files, computes
dot-product projections (raw and/or mean-centered), and saves output tables.

PREPROCESSING NOTE — mean-centering:
    Gemma residual-stream activations have a large shared mean direction across
    all ETHICS items.  Raw dot products are dominated by this baseline, making
    whichever vector aligns with the mean appear "highest" for all items.
    Mean-centering subtracts the cross-item mean activation within each layer
    before projection so that values reflect item-specific variation.  This is
    standard preprocessing in probing and representation-engineering work.
    Raw projections are preserved for audit.  Centered projections are the
    default for all downstream diagnostics.

No GPU, torch, or Modal required — pure NumPy + pandas.

Public API
----------
    load_ethics_activation_metadata(path)              -> pd.DataFrame
    load_activations_by_layer(act_meta_df, act_dir)    -> dict[int, dict[str, np.ndarray]]
    compute_layer_means(act_by_layer)                  -> dict[int, np.ndarray]
    save_centering_metadata(layer_means, out_dir)      -> Path
    project_activations(act_meta_df, vec_meta_df,
                        act_dir, vec_dir,
                        preprocessing)                -> dict[str, pd.DataFrame]
    to_wide_format(long_df, item_df)                   -> pd.DataFrame
    save_projection_set(long_df, wide_df,
                        out_dir, stem)                 -> tuple[Path,Path,Path,Path]
    mock_project(item_df, layers, traits,
                 hidden_dim, out_dir)                  -> dict[str, pd.DataFrame]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TRAITS: list[str] = ["honesty", "harmlessness", "fairness", "compassion"]

# Sentinel values for preprocessing column
PREPROCESSING_RAW = "raw"
PREPROCESSING_CENTERED = "mean_centered"


# ---------------------------------------------------------------------------
# Activation metadata I/O
# ---------------------------------------------------------------------------


def load_ethics_activation_metadata(path: str | Path) -> pd.DataFrame:
    """Load the ETHICS activation metadata parquet produced by the Modal app."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ETHICS activation metadata not found: {path}\n"
            "Run scripts/extract_ethics_activations.py first."
        )
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Activation loading and mean-centering
# ---------------------------------------------------------------------------


def load_activations_by_layer(
    act_meta_df: pd.DataFrame,
    act_dir: str | Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, str]], list[str]]:
    """Load all activation .npy files indexed by layer → item_id.

    Returns:
        act_arrays:  {layer: {item_id: np.ndarray}}
        act_paths:   {layer: {item_id: str path}}
        missing:     list of paths that could not be found
    """
    act_dir = Path(act_dir)
    layers = sorted(act_meta_df["layer"].unique().tolist())
    act_arrays: dict[int, dict[str, np.ndarray]] = {l: {} for l in layers}
    act_paths: dict[int, dict[str, str]] = {l: {} for l in layers}
    missing: list[str] = []

    for _, row in act_meta_df.iterrows():
        item_id = str(row["item_id"])
        layer = int(row["layer"])
        apath = Path(str(row["activation_path"]))
        if not apath.is_absolute():
            apath = act_dir / apath
        if not apath.exists():
            missing.append(str(apath))
            continue
        act_arrays[layer][item_id] = np.load(apath).astype(np.float32)
        act_paths[layer][item_id] = str(apath)

    return act_arrays, act_paths, missing


def compute_layer_means(
    act_by_layer: dict[int, dict[str, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Compute the mean activation vector across items for each layer."""
    means: dict[int, np.ndarray] = {}
    for layer, acts in act_by_layer.items():
        if acts:
            stacked = np.stack(list(acts.values()))
            means[layer] = stacked.mean(axis=0)
        else:
            means[layer] = np.zeros(1, dtype=np.float32)
    return means


def save_centering_metadata(
    layer_means: dict[int, np.ndarray],
    out_dir: str | Path,
    n_items_by_layer: dict[int, int] | None = None,
    centering_scope: str = "ethics_projection_items_by_layer",
) -> Path:
    """Save mean activation vectors and a metadata table for auditability.

    Saves:
        out_dir/centering/mean_activation_layer{N}.npy  — one per layer
        out_dir/centering_metadata.csv
        out_dir/centering_metadata.json

    Returns path to centering_metadata.csv.
    """
    out_dir = Path(out_dir)
    center_dir = out_dir / "centering"
    center_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for layer, mean_vec in sorted(layer_means.items()):
        npy_path = center_dir / f"mean_activation_layer{layer}.npy"
        np.save(npy_path, mean_vec)
        rows.append(
            {
                "layer": layer,
                "n_items_used_for_center": (n_items_by_layer or {}).get(layer, len(mean_vec)),
                "activation_dim": int(mean_vec.shape[0]),
                "mean_activation_norm": float(np.linalg.norm(mean_vec)),
                "mean_activation_path": str(npy_path),
                "centering_scope": centering_scope,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "centering_metadata.csv"
    json_path = out_dir / "centering_metadata.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    return csv_path


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------


def _project_once(
    act_arrays: dict[int, dict[str, np.ndarray]],
    act_paths: dict[int, dict[str, str]],
    act_meta_df: pd.DataFrame,
    vec_cache: dict[tuple[str, int], np.ndarray],
    vec_path_lookup: dict[tuple[str, int], str],
    layer_means: dict[int, np.ndarray] | None,
    preprocessing_label: str,
) -> pd.DataFrame:
    """Inner projection loop used by both raw and centered passes."""
    rows: list[dict] = []
    for _, act_row in act_meta_df.iterrows():
        item_id = str(act_row["item_id"])
        layer = int(act_row["layer"])
        source_split = str(act_row.get("source_split", ""))
        primary_trait = str(act_row.get("primary_trait", ""))

        if item_id not in act_arrays.get(layer, {}):
            continue

        act = act_arrays[layer][item_id].copy()
        apath_str = act_paths[layer][item_id]

        if layer_means is not None and layer in layer_means:
            act = act - layer_means[layer]

        for trait in TRAITS:
            key = (trait, layer)
            if key not in vec_cache:
                continue
            projection = float(np.dot(act, vec_cache[key]))
            rows.append(
                {
                    "item_id": item_id,
                    "source_split": source_split,
                    "primary_trait": primary_trait,
                    "projected_trait": trait,
                    "layer": layer,
                    "projection": projection,
                    "projection_preprocessing": preprocessing_label,
                    "vector_path": vec_path_lookup[key],
                    "activation_path": apath_str,
                }
            )
    return pd.DataFrame(rows)


def project_activations(
    act_meta_df: pd.DataFrame,
    vec_meta_df: pd.DataFrame,
    act_dir: str | Path,
    vec_dir: str | Path,
    preprocessing: str = "both",
) -> dict[str, pd.DataFrame]:
    """Compute dot-product projections for all (item, layer, trait) combinations.

    Args:
        act_meta_df:  Activation metadata DataFrame.
        vec_meta_df:  Vector metadata DataFrame.
        act_dir:      Root directory for activation .npy files.
        vec_dir:      Root directory for vector .npy files.
        preprocessing: "raw" | "mean_centered" | "both"
                       Controls which projection variant(s) to produce.
                       "both" produces both raw and centered DataFrames.

    Returns:
        dict with keys matching requested preprocessing:
            {"raw": long_df, "mean_centered": long_df}
        Always includes "centered" as an alias for "mean_centered".
        The dict always has a "default" key pointing to "mean_centered" output
        (or "raw" if only raw was requested).
    """
    act_dir = Path(act_dir)
    vec_dir = Path(vec_dir)

    # Pre-load vectors
    vec_cache: dict[tuple[str, int], np.ndarray] = {}
    vec_path_lookup: dict[tuple[str, int], str] = {}
    for _, row in vec_meta_df.iterrows():
        trait = str(row["trait"])
        layer = int(row["layer"])
        vpath = Path(str(row["vector_path"]))
        if not vpath.is_absolute():
            vpath = vec_dir / vpath
        if not vpath.exists():
            raise FileNotFoundError(f"Persona vector not found: {vpath}")
        vec_cache[(trait, layer)] = np.load(vpath).astype(np.float32)
        vec_path_lookup[(trait, layer)] = str(vpath)

    # Load activations
    act_arrays, act_paths, missing = load_activations_by_layer(act_meta_df, act_dir)
    if missing:
        print(f"  WARNING: {len(missing)} activation file(s) not found — skipped.")

    # Compute means for centering
    want_centered = preprocessing in ("mean_centered", "both")
    layer_means = compute_layer_means(act_arrays) if want_centered else None

    results: dict[str, pd.DataFrame] = {}

    if preprocessing in ("raw", "both"):
        results[PREPROCESSING_RAW] = _project_once(
            act_arrays, act_paths, act_meta_df,
            vec_cache, vec_path_lookup,
            layer_means=None,
            preprocessing_label=PREPROCESSING_RAW,
        )

    if want_centered:
        results[PREPROCESSING_CENTERED] = _project_once(
            act_arrays, act_paths, act_meta_df,
            vec_cache, vec_path_lookup,
            layer_means=layer_means,
            preprocessing_label=PREPROCESSING_CENTERED,
        )

    # Convenience aliases
    if PREPROCESSING_CENTERED in results:
        results["default"] = results[PREPROCESSING_CENTERED]
    else:
        results["default"] = results[PREPROCESSING_RAW]

    return results, layer_means, {l: len(v) for l, v in act_arrays.items()}


# ---------------------------------------------------------------------------
# Wide format
# ---------------------------------------------------------------------------


def to_wide_format(long_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format projections to one row per item × layer.

    If long_df contains multiple layers, produces separate columns per
    projected_trait (using pivot_table with first aggregation per item_id).
    Caller should pre-filter to a single layer for a clean wide table.

    Returns:
        Wide-format DataFrame with columns:
            item_id, source_split, primary_trait, scenario_text,
            projection_{trait} for each trait
    """
    pivot = long_df.pivot_table(
        index="item_id",
        columns="projected_trait",
        values="projection",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={t: f"projection_{t}" for t in TRAITS})

    item_cols = ["item_id", "source_split", "primary_trait", "scenario_text"]
    available = [c for c in item_cols if c in item_df.columns]
    meta = item_df[available].drop_duplicates("item_id")
    wide = meta.merge(pivot, on="item_id", how="left")

    for t in TRAITS:
        col = f"projection_{t}"
        if col not in wide.columns:
            wide[col] = float("nan")

    col_order = available + [f"projection_{t}" for t in TRAITS]
    return wide[[c for c in col_order if c in wide.columns]]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_projection_set(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    out_dir: str | Path,
    stem: str,
) -> tuple[Path, Path, Path, Path]:
    """Save one long+wide pair as parquet + CSV.

    Returns (long_parquet, long_csv, wide_parquet, wide_csv).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lp = out_dir / f"{stem}_long.parquet"
    lc = out_dir / f"{stem}_long.csv"
    wp = out_dir / f"{stem}_wide.parquet"
    wc = out_dir / f"{stem}_wide.csv"
    long_df.to_parquet(lp, index=False)
    long_df.to_csv(lc, index=False)
    wide_df.to_parquet(wp, index=False)
    wide_df.to_csv(wc, index=False)
    return lp, lc, wp, wc


# Backwards-compatible alias used by older code and tests
def save_projections(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    out_dir: str | Path,
) -> tuple[Path, Path, Path, Path]:
    return save_projection_set(
        long_df, wide_df, out_dir, stem="ethics_trait_projections"
    )


# ---------------------------------------------------------------------------
# Mock projection (for tests — no GPU or files needed)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def mock_project(
    item_df: pd.DataFrame,
    layers: list[int],
    traits: list[str] | None = None,
    hidden_dim: int = 64,
    out_dir: str | Path = "outputs/ethics_projection",
    preprocessing: str = "both",
) -> tuple[dict[str, pd.DataFrame], dict[int, np.ndarray], dict[int, int]]:
    """Generate synthetic activations and project onto random unit-norm vectors.

    Injects a small systematic signal so items annotated with a trait project
    slightly higher on the matching vector.  Returns the same structure as
    project_activations: (results_dict, layer_means, n_items_by_layer).
    """
    out_dir = Path(out_dir)
    traits = traits or TRAITS

    vectors: dict[tuple[str, int], np.ndarray] = {}
    for trait in traits:
        for layer in layers:
            v = _RNG.standard_normal(hidden_dim).astype(np.float32)
            v /= np.linalg.norm(v)
            vectors[(trait, layer)] = v

    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    act_by_layer: dict[int, dict[str, np.ndarray]] = {l: {} for l in layers}

    for _, item in item_df.iterrows():
        item_id = str(item["item_id"])
        primary_trait = str(item["primary_trait"])
        source_split = str(item.get("source_split", ""))

        for layer in layers:
            act = _RNG.standard_normal(hidden_dim).astype(np.float32)
            if primary_trait in traits:
                act += 0.5 * vectors[(primary_trait, layer)]

            apath = act_dir / f"{item_id}_layer{layer}.npy"
            np.save(apath, act)
            act_by_layer[layer][item_id] = act

            for trait in traits:
                raw_rows.append(
                    {
                        "item_id": item_id,
                        "source_split": source_split,
                        "primary_trait": primary_trait,
                        "projected_trait": trait,
                        "layer": layer,
                        "projection": float(np.dot(act, vectors[(trait, layer)])),
                        "projection_preprocessing": PREPROCESSING_RAW,
                        "vector_path": f"mock_{trait}_layer{layer}",
                        "activation_path": str(apath),
                    }
                )

    raw_df = pd.DataFrame(raw_rows)

    # Compute means and centered projections
    layer_means = compute_layer_means(act_by_layer)
    centered_rows: list[dict] = []
    for row in raw_rows:
        layer = int(row["layer"])
        item_id = row["item_id"]
        act = act_by_layer[layer][item_id] - layer_means[layer]
        trait = row["projected_trait"]
        centered_rows.append(
            {
                **row,
                "projection": float(np.dot(act, vectors[(trait, layer)])),
                "projection_preprocessing": PREPROCESSING_CENTERED,
            }
        )
    centered_df = pd.DataFrame(centered_rows)

    results: dict[str, pd.DataFrame] = {}
    if preprocessing in ("raw", "both"):
        results[PREPROCESSING_RAW] = raw_df
    if preprocessing in ("mean_centered", "both"):
        results[PREPROCESSING_CENTERED] = centered_df
    results["default"] = results.get(PREPROCESSING_CENTERED, results.get(PREPROCESSING_RAW))

    n_items_by_layer = {l: len(v) for l, v in act_by_layer.items()}
    return results, layer_means, n_items_by_layer
