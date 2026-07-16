"""
Bootstrap confidence intervals for the three presentation figures.
CPU-only: resamples items from already-computed projection tables, no new
GPU extraction. Writes a JSON of point estimates + bootstrap std-errors that
generate_presentation_figures.py (v2) consumes.

Usage:
    python scripts/bootstrap_presentation_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.analysis.structure_analysis import run_pca, correlation_matrix  # noqa: E402

RNG = np.random.default_rng(42)
N_BOOT = 1000
ORIG_TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]
OUT_PATH = _ROOT / "docs" / "presentation_figures" / "bootstrap_stats.json"


def bootstrap_ed(wide_df: pd.DataFrame, cols: list[str], labels: list[str], n_boot=N_BOOT):
    X = wide_df[cols].values
    n = X.shape[0]
    point = run_pca(X, labels).effective_dimensionality
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        boots[b] = run_pca(X[idx], labels).effective_dimensionality
    return point, boots.std(ddof=1)


def fig1_stats():
    print("=== Fig 1: ED replication (ETHICS vs synthetic bank) ===")
    ethics_wide = pd.read_csv(_ROOT / "outputs/ethics_projection/ethics_trait_projections_centered_wide.csv")
    ethics_long = pd.read_csv(_ROOT / "outputs/ethics_projection/ethics_trait_projections_centered_long.csv")
    synth_long = pd.read_csv(_ROOT / "outputs/synthetic_projection/ethics_trait_projections_centered_long.csv")

    cols = [f"projection_{t}" for t in ORIG_TRAITS]
    out = {"layers": [32, 40, 47], "ethics": {"point": [], "err": []}, "synthetic": {"point": [], "err": []}}

    for layer in [32, 40, 47]:
        if layer == 32:
            wide = ethics_wide
        else:
            wide = ethics_long[ethics_long.layer == layer].pivot(index="item_id", columns="projected_trait", values="projection")
            wide.columns = [f"projection_{c}" for c in wide.columns]
            wide = wide.reset_index()
        p, e = bootstrap_ed(wide, cols, ORIG_TRAITS)
        out["ethics"]["point"].append(p); out["ethics"]["err"].append(e)
        print(f"  ETHICS layer {layer}: ED={p:.3f} +/- {e:.3f}")

        swide = synth_long[synth_long.layer == layer].pivot(index="item_id", columns="projected_trait", values="projection")
        swide.columns = [f"projection_{c}" for c in swide.columns]
        swide = swide.reset_index()
        p, e = bootstrap_ed(swide, cols, ORIG_TRAITS)
        out["synthetic"]["point"].append(p); out["synthetic"]["err"].append(e)
        print(f"  Synthetic layer {layer}: ED={p:.3f} +/- {e:.3f}")

    return out


def fig2_stats():
    print("\n=== Fig 2: virtue_axis vs shared collapse direction ===")
    ethics_long = pd.read_csv(_ROOT / "outputs/ethics_projection/ethics_trait_projections_centered_long.csv")
    cols = [f"projection_{t}" for t in ORIG_TRAITS]

    out = {"layers": [32, 40, 47], "cosine": {"point": [], "err": []},
           "ed4": {"point": [], "err": []}, "ed5": {"point": [], "err": []}}

    for layer in [32, 40, 47]:
        wide = ethics_long[ethics_long.layer == layer].pivot(index="item_id", columns="projected_trait", values="projection")
        wide.columns = [f"projection_{c}" for c in wide.columns]
        wide = wide.reset_index()

        virt = pd.read_csv(_ROOT / f"outputs/controls/virtue_axis/ethics_projection_layer{layer}.csv")
        merged = wide.merge(virt[["item_id", "projection_virtue_axis"]], on="item_id")

        virtue_vec = np.load(_ROOT / f"outputs/controls/virtue_axis/persona_vectors/virtue_axis_layer{layer}.npy").astype(np.float64)
        trait_vecs = {t: np.load(_ROOT / f"outputs/vector_construction/persona_vectors/{t}_layer{layer}.npy").astype(np.float64) for t in ORIG_TRAITS}

        def cosine_to_pc1(df):
            X = df[cols].values
            pca = run_pca(X, ORIG_TRAITS)
            pc1_loadings = pca.loadings[:, 0]
            stds = df[cols].std(ddof=1).values
            direction = np.zeros(3840)
            for i, t in enumerate(ORIG_TRAITS):
                direction += (pc1_loadings[i] / stds[i]) * trait_vecs[t]
            return float(np.dot(virtue_vec, direction) / (np.linalg.norm(virtue_vec) * np.linalg.norm(direction)))

        def ed45(df):
            X4 = df[cols].values
            X5 = df[cols + ["projection_virtue_axis"]].values
            ed4 = run_pca(X4, ORIG_TRAITS).effective_dimensionality
            ed5 = run_pca(X5, ORIG_TRAITS + ["virtue_axis"]).effective_dimensionality
            return ed4, ed5

        point_cos = cosine_to_pc1(merged)
        point_ed4, point_ed5 = ed45(merged)

        n = len(merged)
        boot_cos = np.empty(N_BOOT)
        boot_ed4 = np.empty(N_BOOT)
        boot_ed5 = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = RNG.integers(0, n, n)
            sub = merged.iloc[idx]
            boot_cos[b] = cosine_to_pc1(sub)
            e4, e5 = ed45(sub)
            boot_ed4[b] = e4; boot_ed5[b] = e5

        out["cosine"]["point"].append(point_cos); out["cosine"]["err"].append(boot_cos.std(ddof=1))
        out["ed4"]["point"].append(point_ed4); out["ed4"]["err"].append(boot_ed4.std(ddof=1))
        out["ed5"]["point"].append(point_ed5); out["ed5"]["err"].append(boot_ed5.std(ddof=1))
        print(f"  layer {layer}: cos={point_cos:.3f}+/-{boot_cos.std(ddof=1):.3f}  "
              f"ED4={point_ed4:.3f}+/-{boot_ed4.std(ddof=1):.3f}  ED5={point_ed5:.3f}+/-{boot_ed5.std(ddof=1):.3f}")

    return out


def auc(pos, neg):
    count = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return count / (len(pos) * len(neg))


def fig3_stats():
    print("\n=== Fig 3: within-trait discrimination (synthetic bank), strongest significant layer per trait ===")
    long = pd.read_csv(_ROOT / "outputs/synthetic_projection/ethics_trait_projections_centered_long.csv")
    bank = pd.read_parquet(_ROOT / "data/processed/synthetic_trait_bank.parquet")
    long = long.merge(bank[["item_id", "label"]], on="item_id")

    # (trait, layer, direction) - direction: +1 means "upheld higher" is the expected/plotted sign,
    # -1 means "violated higher". Picked as the strongest |AUC-0.5| among significant layers
    # (fairness corrected to layer 47, its actual strongest result, not layer 32).
    choices = {
        "Fairness": (32, 47, +1),      # both sig; 47 stronger (0.858) than 32 (0.728)
        "Harmlessness": (32, 40, -1),  # only 40 significant (0.930)
        "Compassion": (32, 40, -1),    # 40 (0.725) stronger than 47 (0.715)
        "Honesty": (32, 47, +1),       # none significant; use largest-magnitude layer for display
    }
    trait_key = {"Fairness": "fairness", "Harmlessness": "harmlessness",
                 "Compassion": "compassion", "Honesty": "honesty"}

    out = {"traits": [], "point": [], "err": []}
    for label, (_, layer, direction) in choices.items():
        t = trait_key[label]
        sub = long[(long.layer == layer) & (long.primary_trait == t) & (long.projected_trait == t)]
        upheld_idx = sub[sub.label == 0].index.values
        violated_idx = sub[sub.label == 1].index.values
        scores = sub["projection"]

        # Chart convention: positive = matches the naive "upheld should score higher"
        # expectation; negative = "backwards" relative to that same expectation.
        # direction=+1 (fairness/honesty): trait actually behaves as expected -> plot (upheld>violated)-0.5, positive.
        # direction=-1 (harmlessness/compassion): trait actually runs backwards (violated>upheld)
        #   -> plot -[(violated>upheld)-0.5] so "backwards" shows as negative, consistent with the axis label.
        a_full = auc(scores.loc[violated_idx].values, scores.loc[upheld_idx].values)
        if direction == +1:
            point = (1 - a_full) - 0.5   # upheld>violated distance from chance
        else:
            point = -(a_full - 0.5)       # negated violated>upheld distance -> shows as "backwards"

        n_u, n_v = len(upheld_idx), len(violated_idx)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            bu = RNG.choice(upheld_idx, n_u, replace=True)
            bv = RNG.choice(violated_idx, n_v, replace=True)
            a_b = auc(scores.loc[bv].values, scores.loc[bu].values)
            boots[b] = (1 - a_b) - 0.5 if direction == +1 else -(a_b - 0.5)

        out["traits"].append(label)
        out["point"].append(float(point))
        out["err"].append(float(boots.std(ddof=1)))
        print(f"  {label:13s} layer {layer}: {point:+.3f} +/- {boots.std(ddof=1):.3f}  (n_upheld={n_u}, n_violated={n_v})")

    return out


def main():
    stats = {"fig1": fig1_stats(), "fig2": fig2_stats(), "fig3": fig3_stats()}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved bootstrap stats to {OUT_PATH}")


if __name__ == "__main__":
    main()
