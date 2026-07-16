"""
Verification / regeneration script for headline numerical claims in the paper.

Recomputes, from saved data files (no re-extraction, no GPU/Modal calls):
  - pooled PC1 (layer 32) vs ground-truth ethical label: ROC-AUC + stratified bootstrap 95% CI
  - per-format AUC (Justice / Commonsense / EXCUSE / AITA): same, with CIs
  - synonym vector "Label AUC" (does each synonym's own projection predict the ethics label):
    AUC + CI, at layer 32
  - 4x4 and 8x8 pairwise cosine-similarity matrices of persona vectors in activation space
    (distinct from Pearson correlation of their projection scores)

Writes results to outputs/paper_verification/ so generate_paper_pdf.py can load them
instead of using hardcoded literals.

Usage:
    python scripts/verify_paper_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "paper_verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20260716
N_BOOT = 5000


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if _HAVE_SKLEARN:
        return float(roc_auc_score(labels, scores))
    # Mann-Whitney U fallback
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_concordant = 0.0
    for p in pos:
        n_concordant += (p > neg).sum() + 0.5 * (p == neg).sum()
    return float(n_concordant / (len(pos) * len(neg)))


def stratified_bootstrap_auc_ci(labels: np.ndarray, scores: np.ndarray,
                                 n_boot: int = N_BOOT, seed: int = RNG_SEED) -> dict:
    """
    Stratified bootstrap: resample positives and negatives separately (with
    replacement) at their observed class sizes, recompute AUC each time.
    Returns point estimate, 95% CI, n_pos, n_neg, and a stability flag.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    point = _auc(labels, scores)

    if n_pos < 2 or n_neg < 2:
        return {
            "auc": point, "ci_low": float("nan"), "ci_high": float("nan"),
            "n_pos": int(n_pos), "n_neg": int(n_neg), "n_boot": n_boot,
            "stable": False, "note": "fewer than 2 items in a class; CI not computed",
        }

    rng = np.random.default_rng(seed)
    boot_aucs = np.empty(n_boot)
    for b in range(n_boot):
        bp = rng.choice(pos_idx, size=n_pos, replace=True)
        bn = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([bp, bn])
        boot_aucs[b] = _auc(labels[idx], scores[idx])

    ci_low, ci_high = np.nanpercentile(boot_aucs, [2.5, 97.5])
    # Flag instability: small class, or CI spanning most of [0,1]
    unstable_small_n = (n_pos < 15) or (n_neg < 15)
    unstable_wide = (ci_high - ci_low) > 0.5
    return {
        "auc": point, "ci_low": float(ci_low), "ci_high": float(ci_high),
        "n_pos": int(n_pos), "n_neg": int(n_neg), "n_boot": n_boot,
        "stable": not (unstable_small_n or unstable_wide),
        "note": ("small class size (<15 in a class)" if unstable_small_n else
                 "wide CI (>0.5 span)" if unstable_wide else ""),
    }


def classify_format(scenario_text: str, item_id: str) -> str:
    """Exact replica of the fmt() logic in generate_paper_pdf.py's fig_pc1_label()."""
    t, i = str(scenario_text), str(item_id)
    if "[EXCUSE]" in t:
        return "EXCUSE"
    if t.strip().startswith(("AITA", "WIBTA")):
        return "AITA"
    if "commonsense" in i:
        return "Commonsense"
    if "justice" in i:
        return "Justice"
    return "Other"


def main():
    mvp = pd.read_parquet(ROOT / "data/processed/ethics_curated_mvp.parquet")
    pc32 = pd.read_csv(ROOT / "outputs/structure_analysis/pca_scores_layer32.csv")
    ethics = pd.read_parquet(ROOT / "outputs/ethics_projection/ethics_trait_projections_centered_wide.parquet")

    merged = pc32.merge(mvp[["item_id", "label"]], on="item_id")
    merged = merged.merge(ethics[["item_id", "scenario_text"]], on="item_id")
    merged["label"] = merged["label"].astype(int)
    merged["format"] = merged.apply(lambda r: classify_format(r["scenario_text"], r["item_id"]), axis=1)

    print(f"Loaded {len(merged)} items. Format counts:\n{merged['format'].value_counts()}")
    print(f"Label counts: {merged['label'].value_counts().to_dict()}")

    results = {}

    # ---- Pooled AUC ----
    labels_all = merged["label"].values
    pc1_all = merged["PC1"].values
    pooled = stratified_bootstrap_auc_ci(labels_all, pc1_all)
    # Also correlate PC1 with label for the "r=..." figure quoted in the paper
    pooled["pearson_r_with_label"] = float(np.corrcoef(pc1_all, labels_all)[0, 1])
    results["pooled"] = pooled
    print(f"\nPooled AUC: {pooled['auc']:.3f}  95% CI [{pooled['ci_low']:.3f}, {pooled['ci_high']:.3f}]  "
          f"n_pos={pooled['n_pos']} n_neg={pooled['n_neg']}")

    # ---- Per-format AUC ----
    results["by_format"] = {}
    for fmt in ["Justice", "Commonsense", "EXCUSE", "AITA", "Other"]:
        sub = merged[merged["format"] == fmt]
        if len(sub) == 0:
            continue
        r = stratified_bootstrap_auc_ci(sub["label"].values, sub["PC1"].values)
        r["n_items"] = int(len(sub))
        results["by_format"][fmt] = r
        stab = "" if r["stable"] else f"  [UNSTABLE: {r['note']}]"
        print(f"  {fmt:12s} n={r['n_items']:3d} (pos={r['n_pos']}, neg={r['n_neg']})  "
              f"AUC={r['auc']:.3f}  95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]{stab}")

    # ---- Synonym Label AUC (layer 32) ----
    syn_path = ROOT / "outputs/controls/synonym_vectors/synonym_ethics_projections_layer32.csv"
    results["synonym_label_auc"] = {}
    if syn_path.exists():
        syn = pd.read_csv(syn_path)
        syn_merged = syn.merge(mvp[["item_id", "label"]], on="item_id")
        syn_merged["label"] = syn_merged["label"].astype(int)
        proj_cols = [c for c in syn.columns if c.startswith("projection_")]
        print("\nSynonym Label AUC (layer 32):")
        for col in proj_cols:
            synonym_id = col.replace("projection_", "")
            r = stratified_bootstrap_auc_ci(syn_merged["label"].values, syn_merged[col].values)
            results["synonym_label_auc"][synonym_id] = r
            stab = "" if r["stable"] else f"  [UNSTABLE: {r['note']}]"
            print(f"  {synonym_id:16s} AUC={r['auc']:.3f}  95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]{stab}")
    else:
        print(f"\nWARNING: {syn_path} not found; cannot verify synonym Label AUC.")

    # ---- Cosine similarity matrices (vector geometry, layer 32) ----
    orig_dir = ROOT / "outputs/vector_construction/persona_vectors"
    syn_dir = ROOT / "outputs/controls/synonym_vectors/persona_vectors"
    orig_traits = ["honesty", "harmlessness", "fairness", "compassion"]
    syn_traits = ["truthfulness", "harm_avoidance", "impartiality", "empathy"]

    def load_vec(d: Path, name: str, layer: int = 32):
        p = d / f"{name}_layer{layer}.npy"
        return np.load(p) if p.exists() else None

    vecs4 = {t: load_vec(orig_dir, t) for t in orig_traits}
    vecs8 = dict(vecs4)
    vecs8.update({t: load_vec(syn_dir, t) for t in syn_traits})
    vecs8 = {k: v for k, v in vecs8.items() if v is not None}

    def cosine_matrix(vecs: dict) -> pd.DataFrame:
        names = list(vecs.keys())
        mat = np.zeros((len(names), len(names)))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                va, vb = vecs[a], vecs[b]
                mat[i, j] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
        return pd.DataFrame(mat, index=names, columns=names)

    cos4 = cosine_matrix(vecs4)
    cos8 = cosine_matrix(vecs8)
    cos4.to_csv(OUT_DIR / "cosine_similarity_4vec_layer32.csv")
    cos8.to_csv(OUT_DIR / "cosine_similarity_8vec_layer32.csv")
    print(f"\n4x4 cosine similarity matrix (layer 32, activation-space vectors):\n{cos4.round(3)}")
    print(f"\n8x8 cosine similarity matrix saved to {OUT_DIR / 'cosine_similarity_8vec_layer32.csv'}")

    # ---- Compare against hardcoded values currently in generate_paper_pdf.py ----
    hardcoded = {
        "pooled_auc": 0.585,
        "format_auc": {"Justice": 0.711, "Commonsense": 0.551, "EXCUSE": 0.535, "AITA": 0.377},
        "synonym_label_auc": {"truthfulness": 0.584, "harm_avoidance": 0.457,
                               "impartiality": 0.550, "empathy": 0.381},
    }
    print("\n" + "=" * 70)
    print("COMPARISON: hardcoded (current paper) vs recomputed (this script)")
    print("=" * 70)
    diff = abs(results["pooled"]["auc"] - hardcoded["pooled_auc"])
    print(f"Pooled AUC:        hardcoded={hardcoded['pooled_auc']:.3f}  "
          f"recomputed={results['pooled']['auc']:.3f}  diff={diff:.4f}")
    for fmt, hv in hardcoded["format_auc"].items():
        if fmt in results["by_format"]:
            rv = results["by_format"][fmt]["auc"]
            print(f"Format AUC [{fmt:12s}]: hardcoded={hv:.3f}  recomputed={rv:.3f}  diff={abs(hv-rv):.4f}")
    for syn_id, hv in hardcoded["synonym_label_auc"].items():
        if syn_id in results["synonym_label_auc"]:
            rv = results["synonym_label_auc"][syn_id]["auc"]
            print(f"Synonym Label AUC [{syn_id:16s}]: hardcoded={hv:.3f}  recomputed={rv:.3f}  diff={abs(hv-rv):.4f}")

    with open(OUT_DIR / "verified_auc_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved full results to {OUT_DIR / 'verified_auc_results.json'}")


if __name__ == "__main__":
    main()
