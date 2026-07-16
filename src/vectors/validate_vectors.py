"""
Held-out vector validation (Stage 2B).

Projects validation-split activations onto each trait × layer persona vector
and computes contrastive accuracy metrics.

No GPU or Modal required — pure NumPy + scikit-learn.

Validation procedure:
  1. Load validation-split ActivationRecord objects.
  2. For each (trait, layer): load the persona vector.
  3. Project each activation onto the vector via dot product.
  4. Compute metrics comparing positive-pole vs negative-pole projections.
  5. Select the best layer by mean AUC across all traits.

NOTE: This uses CONTRAST ARTIFACT validation responses, not ETHICS items.
      ETHICS projection comes after this stage passes.

Public API
----------
    compute_validation_metrics(pos_proj, neg_proj) -> dict
    validate_all_vectors(act_records, vec_metas,
                         minimum_auc, out_dir)     -> list[VectorValidationResult]
    select_best_layer(results, traits)             -> int
    save_validation_results(results, out_dir)
    load_validation_results(path)                 -> list[VectorValidationResult]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.vectors.vector_data import ActivationRecord, PersonaVectorMeta, VectorValidationResult

try:
    from sklearn.metrics import roc_auc_score as _roc_auc_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    _roc_auc_score = None


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC with a fallback if sklearn is unavailable."""
    if _SKLEARN_AVAILABLE:
        return float(_roc_auc_score(labels, scores))
    # Fallback: manual AUC via counting concordant pairs
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    concordant = sum(
        1
        for s_p in scores[labels == 1]
        for s_n in scores[labels == 0]
        if s_p > s_n
    )
    return concordant / (n_pos * n_neg)


def compute_validation_metrics(
    pos_projections: np.ndarray,
    neg_projections: np.ndarray,
) -> dict[str, float]:
    """Compute all validation metrics given positive and negative projection arrays.

    Args:
        pos_projections: Shape (n_pos,) — dot products of positive-pole activations
                         onto the trait vector.
        neg_projections: Shape (n_neg,) — dot products of negative-pole activations.

    Returns:
        Dict with keys: auc, accuracy, mean_positive_projection,
                        mean_negative_projection, cohens_d.
    """
    if len(pos_projections) == 0 or len(neg_projections) == 0:
        return {
            "auc": 0.5,
            "accuracy": 0.5,
            "mean_positive_projection": 0.0,
            "mean_negative_projection": 0.0,
            "cohens_d": 0.0,
        }

    all_proj = np.concatenate([pos_projections, neg_projections])
    labels = np.concatenate([
        np.ones(len(pos_projections), dtype=int),
        np.zeros(len(neg_projections), dtype=int),
    ])

    auc = _roc_auc(labels, all_proj)

    # Accuracy at midpoint threshold between class means
    threshold = (pos_projections.mean() + neg_projections.mean()) / 2.0
    preds = (all_proj >= threshold).astype(int)
    accuracy = float((preds == labels).mean())

    mean_pos = float(pos_projections.mean())
    mean_neg = float(neg_projections.mean())

    # Cohen's d: effect size between the two projection distributions
    std_pos = float(pos_projections.std(ddof=1)) if len(pos_projections) > 1 else 0.0
    std_neg = float(neg_projections.std(ddof=1)) if len(neg_projections) > 1 else 0.0
    pooled_std = np.sqrt((std_pos ** 2 + std_neg ** 2) / 2.0)
    cohens_d = (mean_pos - mean_neg) / pooled_std if pooled_std > 0 else 0.0

    return {
        "auc": auc,
        "accuracy": accuracy,
        "mean_positive_projection": mean_pos,
        "mean_negative_projection": mean_neg,
        "cohens_d": cohens_d,
    }


# ---------------------------------------------------------------------------
# Full validation pipeline
# ---------------------------------------------------------------------------


def validate_all_vectors(
    act_records: list[ActivationRecord],
    vec_metas: list[PersonaVectorMeta],
    minimum_auc_target: float = 0.75,
    out_dir: str | Path | None = None,
) -> list[VectorValidationResult]:
    """Validate every trait × layer vector against validation-split activations.

    Args:
        act_records:        All activation records (extraction + validation).
        vec_metas:          Persona vector metadata.
        minimum_auc_target: AUC threshold for passes_minimum_auc flag.
        out_dir:            If provided, save results CSV + MD here.

    Returns:
        List of VectorValidationResult, one per trait × layer.
    """
    # Index validation-split records by (trait, pole, layer)
    val_index: dict[tuple, list[ActivationRecord]] = {}
    for rec in act_records:
        if rec.split != "validation":
            continue
        key = (rec.trait, rec.pole, rec.layer)
        val_index.setdefault(key, []).append(rec)

    results: list[VectorValidationResult] = []

    for meta in vec_metas:
        trait = meta.trait
        layer = meta.layer

        pos_recs = val_index.get((trait, "positive", layer), [])
        neg_recs = val_index.get((trait, "negative", layer), [])

        if not pos_recs or not neg_recs:
            continue

        vector = np.load(meta.vector_path).astype(np.float32)

        pos_proj = np.array(
            [np.dot(np.load(r.activation_path).astype(np.float32), vector) for r in pos_recs]
        )
        neg_proj = np.array(
            [np.dot(np.load(r.activation_path).astype(np.float32), vector) for r in neg_recs]
        )

        metrics = compute_validation_metrics(pos_proj, neg_proj)

        results.append(
            VectorValidationResult(
                trait=trait,
                layer=layer,
                auc=metrics["auc"],
                accuracy=metrics["accuracy"],
                mean_positive_projection=metrics["mean_positive_projection"],
                mean_negative_projection=metrics["mean_negative_projection"],
                cohens_d=metrics["cohens_d"],
                n_positive_val=len(pos_recs),
                n_negative_val=len(neg_recs),
                passes_minimum_auc=metrics["auc"] >= minimum_auc_target,
            )
        )

    if out_dir is not None:
        save_validation_results(results, out_dir)

    return results


def select_best_layer(
    results: list[VectorValidationResult],
    traits: list[str],
) -> int:
    """Select the layer with the highest mean AUC across all specified traits.

    Args:
        results: All VectorValidationResult objects.
        traits:  Traits to average over.

    Returns:
        The best layer index.
    """
    # Collect mean AUC per layer
    layer_aucs: dict[int, list[float]] = {}
    for r in results:
        if r.trait in traits:
            layer_aucs.setdefault(r.layer, []).append(r.auc)

    if not layer_aucs:
        raise ValueError("No validation results to select layer from.")

    best_layer = max(layer_aucs, key=lambda l: np.mean(layer_aucs[l]))
    return best_layer


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_validation_results(
    results: list[VectorValidationResult],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)

    csv_path = out_dir / "vector_validation_results.csv"
    df.to_csv(csv_path, index=False)

    md_path = out_dir / "vector_validation_results.md"
    _write_validation_md(results, md_path)

    return csv_path, md_path


def _write_validation_md(
    results: list[VectorValidationResult], path: Path
) -> None:
    traits = sorted({r.trait for r in results})
    layers = sorted({r.layer for r in results})

    lines = [
        "# Persona Vector Held-Out Validation Results",
        "",
        "> These results use CONTRAST ARTIFACT validation responses (not ETHICS items).",
        "> Proceed to Stage 3 (ETHICS projection) only after all trait vectors pass AUC threshold.",
        "",
    ]

    # Per-trait summary
    for trait in traits:
        trait_results = sorted([r for r in results if r.trait == trait], key=lambda r: r.layer)
        lines += [f"## {trait.capitalize()}", ""]
        lines += [
            "| Layer | AUC | Accuracy | Mean Pos Proj | Mean Neg Proj | Cohen's d | Pass? |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in trait_results:
            flag = "✅" if r.passes_minimum_auc else "❌"
            lines.append(
                f"| {r.layer} | {r.auc:.3f} | {r.accuracy:.3f} | "
                f"{r.mean_positive_projection:.3f} | {r.mean_negative_projection:.3f} | "
                f"{r.cohens_d:.3f} | {flag} |"
            )
        lines.append("")

    # Layer selection
    try:
        best = select_best_layer(results, traits)
        lines += [
            "## Recommended Layer",
            "",
            f"Layer **{best}** has the highest mean AUC across all traits.",
            "",
            "Update `model.target_layer` in `configs/mvp_experiment.yaml` to this value.",
        ]
    except ValueError:
        lines += ["## Recommended Layer", "", "No results available for layer selection."]

    path.write_text("\n".join(lines), encoding="utf-8")


def load_validation_results(path: str | Path) -> list[VectorValidationResult]:
    df = pd.read_csv(path)
    return [
        VectorValidationResult(
            trait=str(row["trait"]),
            layer=int(row["layer"]),
            auc=float(row["auc"]),
            accuracy=float(row["accuracy"]),
            mean_positive_projection=float(row["mean_positive_projection"]),
            mean_negative_projection=float(row["mean_negative_projection"]),
            cohens_d=float(row["cohens_d"]),
            n_positive_val=int(row["n_positive_val"]),
            n_negative_val=int(row["n_negative_val"]),
            passes_minimum_auc=bool(row["passes_minimum_auc"]),
        )
        for _, row in df.iterrows()
    ]
