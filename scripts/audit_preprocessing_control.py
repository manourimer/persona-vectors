"""
Audit script: trace what each candidate input file contains and how
preprocessing_robustness metrics differ from Stage 4A structure_summary.

Saves:
  outputs/controls/preprocessing_audit_report.md
  outputs/controls/preprocessing_audit_trace.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.structure_analysis import (
    correlation_df,
    run_pca,
    compute_structure_summary,
    PROJECTION_COLS,
    TRAIT_LABELS,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "controls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_FILES = {
    "ethics_raw_wide": ROOT / "outputs/ethics_projection/ethics_trait_projections_raw_wide.parquet",
    "ethics_centered_wide": ROOT / "outputs/ethics_projection/ethics_trait_projections_centered_wide.parquet",
    "rel_raw_wide": ROOT / "outputs/reliability_projection/reliability_trait_projections_wide_raw.parquet",
    "rel_centered_wide": ROOT / "outputs/reliability_projection/reliability_trait_projections_wide_centered.parquet",
}

LAYERS_WITH_DATA = {
    "ethics_raw_wide": [32],
    "ethics_centered_wide": [32],
    "rel_raw_wide": [32, 40, 47],
    "rel_centered_wide": [32, 40, 47],
}

STRUCTURE_CSV = ROOT / "outputs/structure_analysis/structure_summary.csv"


def _eff_dim(X: np.ndarray) -> float:
    pca = run_pca(X, TRAIT_LABELS, standardize_first=True)
    return pca.effective_dimensionality


def _pc1(X: np.ndarray) -> float:
    pca = run_pca(X, TRAIT_LABELS, standardize_first=True)
    return float(pca.explained_variance_ratio[0])


def _mean_abs_off_diag(X: np.ndarray) -> float:
    corr = correlation_df(X, TRAIT_LABELS).to_numpy()
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.abs(corr[mask]).mean())


def _max_abs_corr(X: np.ndarray) -> float:
    corr = correlation_df(X, TRAIT_LABELS).to_numpy()
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.abs(corr[mask]).max())


def _proj_matrix(df: pd.DataFrame) -> np.ndarray | None:
    available = [c for c in PROJECTION_COLS if c in df.columns]
    if len(available) < 4:
        return None
    return df[available].dropna().values


def main():
    print("=== Preprocessing Control Audit ===\n")
    rows = []

    for name, path in CANDIDATE_FILES.items():
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        df = pd.read_parquet(path)
        layers = LAYERS_WITH_DATA[name]
        n_variants = df["variant_id"].nunique() if "variant_id" in df.columns else None
        print(f"\n--- {name} ---")
        print(f"  Path: {path}")
        print(f"  Shape: {df.shape}")
        print(f"  Cols: {df.columns.tolist()}")
        if "layer" in df.columns:
            print(f"  Layers present: {sorted(df['layer'].unique())}")
        else:
            print(f"  No 'layer' column; treating as single-layer (32)")

        for layer in layers:
            if "layer" in df.columns:
                sub = df[df["layer"] == layer].copy()
            else:
                sub = df.copy()

            mat = _proj_matrix(sub)
            if mat is None or len(mat) < 4:
                print(f"  Layer {layer}: insufficient data")
                continue

            n_items = sub["item_id"].nunique() if "item_id" in sub.columns else len(sub)
            ed = _eff_dim(mat)
            pc1 = _pc1(mat)
            mean_r = _mean_abs_off_diag(mat)
            max_r = _max_abs_corr(mat)

            print(f"  Layer {layer}: n_rows={len(sub)}, n_items={n_items}, "
                  f"ED={ed:.3f}, PC1={pc1:.3f}, mean|r|={mean_r:.3f}")

            rows.append({
                "source": name,
                "layer": layer,
                "path": str(path),
                "n_rows": len(sub),
                "n_items": n_items,
                "n_variants": n_variants,
                "effective_dimensionality": ed,
                "pc1_variance": pc1,
                "mean_abs_off_diag_corr": mean_r,
                "max_abs_corr": max_r,
            })

    trace_df = pd.DataFrame(rows)
    trace_df.to_csv(OUT_DIR / "preprocessing_audit_trace.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'preprocessing_audit_trace.csv'}")

    # Compare against Stage 4A
    struct_df = pd.read_csv(STRUCTURE_CSV)
    print("\n=== Stage 4A reference (structure_summary.csv) ===")
    print(struct_df[["layer", "effective_dimensionality", "first_pc_variance", "mean_abs_off_diag_corr"]].to_string())

    # Build report
    lines = ["# Preprocessing Control Audit Report\n"]
    lines.append("## Candidate file metrics\n")
    lines.append(trace_df.to_markdown(index=False))
    lines.append("\n\n## Stage 4A reference (centered ETHICS originals)\n")
    lines.append(struct_df[["layer", "effective_dimensionality", "first_pc_variance",
                             "mean_abs_off_diag_corr"]].to_markdown(index=False))
    lines.append("\n\n## Bug diagnosis\n")
    lines.append(
        "The `run_preprocessing_controls.py` script loads `reliability_trait_projections_wide_centered.parquet`\n"
        "and `reliability_trait_projections_wide_raw.parquet` for BOTH the preprocessing comparison\n"
        "AND the layer robustness analysis.  These files contain 2283 rows (761 variants × 3 layers).\n\n"
        "When filtered to layer 32 (~761 rows of reliability variants), the 4 trait projections are\n"
        "highly inter-correlated (all variants of the same item cluster together), yielding:\n"
        "  ED ≈ 1.14, mean|r| ≈ 0.91 — matching the broken output exactly.\n\n"
        "The correct files for original ETHICS preprocessing comparison are:\n"
        "  - ethics_trait_projections_raw_wide.parquet (204 rows, no layer column)\n"
        "  - ethics_trait_projections_centered_wide.parquet (204 rows, no layer column)\n"
        "These yield ED ≈ 3.87, mean|r| ≈ 0.085 at layer 32 — matching Stage 4A.\n"
    )

    report_path = OUT_DIR / "preprocessing_audit_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
