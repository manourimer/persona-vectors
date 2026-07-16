"""
Stage 4A: Save structure analysis outputs.

Saves per-layer CSV files and a consolidated Markdown report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.structure_analysis import (
    PROJECTION_COLS,
    StructureSummary,
    loadings_df,
)


def _ensure(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Per-layer file savers
# ---------------------------------------------------------------------------


def save_correlation_matrix(summary: StructureSummary, out_dir: str | Path) -> Path:
    out_dir = _ensure(out_dir)
    path = out_dir / f"correlation_matrix_layer{summary.layer}.csv"
    summary.corr_df.to_csv(path, float_format="%.6f")
    return path


def save_pca_loadings(summary: StructureSummary, out_dir: str | Path) -> Path:
    out_dir = _ensure(out_dir)
    path = out_dir / f"pca_loadings_layer{summary.layer}.csv"
    loadings_df(summary.pca).to_csv(path, float_format="%.6f")
    return path


def save_pca_variance(summary: StructureSummary, out_dir: str | Path) -> Path:
    out_dir = _ensure(out_dir)
    path = out_dir / f"pca_variance_layer{summary.layer}.csv"
    p = summary.pca.n_variables
    df = pd.DataFrame(
        {
            "component": [f"PC{k+1}" for k in range(p)],
            "eigenvalue": summary.pca.eigenvalues,
            "explained_variance_ratio": summary.pca.explained_variance_ratio,
            "cumulative_variance": summary.pca.cumulative_variance,
        }
    )
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def save_parallel_analysis(summary: StructureSummary, out_dir: str | Path) -> Path:
    out_dir = _ensure(out_dir)
    path = out_dir / f"parallel_analysis_layer{summary.layer}.csv"
    pa = summary.parallel
    p = len(pa.observed_eigenvalues)
    df = pd.DataFrame(
        {
            "component": [f"PC{k+1}" for k in range(p)],
            "observed_eigenvalue": pa.observed_eigenvalues,
            "random_eigenvalue_95th": pa.random_eigenvalue_95th,
            "random_eigenvalue_mean": pa.random_eigenvalue_mean,
            "retained": pa.observed_eigenvalues > pa.random_eigenvalue_95th,
        }
    )
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def save_pca_scores(
    summary: StructureSummary,
    wide_df: pd.DataFrame,
    out_dir: str | Path,
) -> Path:
    out_dir = _ensure(out_dir)
    path = out_dir / f"pca_scores_layer{summary.layer}.csv"
    p = summary.pca.n_variables
    score_cols = {f"PC{k+1}": summary.pca.scores[:, k] for k in range(p)}
    meta_cols = [c for c in ["item_id", "primary_trait", "source_split"] if c in wide_df.columns]
    df = wide_df[meta_cols].copy().reset_index(drop=True)
    for col, vals in score_cols.items():
        df[col] = vals
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def save_factor_analysis(summary: StructureSummary, out_dir: str | Path) -> list[Path]:
    if summary.factor_analysis is None:
        return []
    out_dir = _ensure(out_dir)
    paths: list[Path] = []
    fa = summary.factor_analysis
    labels = [c.replace("projection_", "") for c in PROJECTION_COLS]
    for n_factors, loadings in fa.get("loadings", {}).items():
        path = out_dir / f"factor_loadings_{n_factors}f_layer{summary.layer}.csv"
        df = pd.DataFrame(
            loadings,
            index=labels,
            columns=[f"Factor{k+1}" for k in range(n_factors)],
        )
        df.to_csv(path, float_format="%.6f")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Structure summary CSV
# ---------------------------------------------------------------------------


def save_structure_summary(
    summaries: dict[int, StructureSummary],
    out_dir: str | Path,
) -> Path:
    out_dir = _ensure(out_dir)
    rows = []
    for layer, s in sorted(summaries.items()):
        rows.append(
            {
                "layer": layer,
                "n_items": s.n_items,
                "mean_abs_off_diag_corr": s.mean_abs_off_diagonal_corr,
                "max_abs_trait_corr": s.max_abs_trait_corr,
                "most_correlated_pair": f"{s.most_correlated_pair[0]}-{s.most_correlated_pair[1]}",
                "most_correlated_value": s.most_correlated_value,
                "effective_dimensionality": s.effective_dimensionality,
                "first_pc_variance": s.first_pc_variance,
                "first_pc_dominant": s.first_pc_dominant,
                "n_components_80pct": s.n_components_80pct,
                "n_components_90pct": s.n_components_90pct,
                "n_components_parallel": s.n_components_parallel,
                "interpretation": s.interpretation,
            }
        )
    path = out_dir / "structure_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _corr_md_table(corr_df: pd.DataFrame) -> str:
    labels = list(corr_df.columns)
    header = "| trait | " + " | ".join(labels) + " |"
    sep = "|---|" + "---|" * len(labels)
    rows = [header, sep]
    for idx in labels:
        vals = " | ".join(f"{corr_df.loc[idx, c]:.3f}" for c in labels)
        rows.append(f"| {idx} | {vals} |")
    return "\n".join(rows)


def _variance_md_table(s: StructureSummary) -> str:
    p = s.pca.n_variables
    rows = ["| Component | Eigenvalue | Variance explained | Cumulative |",
            "|---|---|---|---|"]
    for k in range(p):
        rows.append(
            f"| PC{k+1} "
            f"| {s.pca.eigenvalues[k]:.3f} "
            f"| {s.pca.explained_variance_ratio[k]:.1%} "
            f"| {s.pca.cumulative_variance[k]:.1%} |"
        )
    return "\n".join(rows)


def _loadings_md_table(s: StructureSummary) -> str:
    p = s.pca.n_variables
    labels = s.pca.labels
    header = "| Variable | " + " | ".join(f"PC{k+1}" for k in range(p)) + " |"
    sep = "|---|" + "---|" * p
    rows = [header, sep]
    for i, lbl in enumerate(labels):
        vals = " | ".join(f"{s.pca.loadings[i, k]:.3f}" for k in range(p))
        rows.append(f"| {lbl} | {vals} |")
    return "\n".join(rows)


def save_report(
    summaries: dict[int, StructureSummary],
    out_dir: str | Path,
    primary_layer: int = 32,
    downstream_best_layer: int = 40,
) -> Path:
    out_dir = _ensure(out_dir)
    lines: list[str] = []

    lines += [
        "# Stage 4A — Projection Structure Analysis",
        "",
        "> **RQ1**: Do the four morally relevant persona-vector projections behave like",
        "> one latent 'morality' dimension, or several separable dimensions?",
        "",
        "**Methodology**: PCA and correlation analysis on mean-centered ETHICS projection",
        "matrices (Stage 3). Parallel analysis with permutation estimates random baseline.",
        "Factor analysis skipped if `factor_analyzer` is not installed.",
        "",
        "> ⚠ **Factor-analysis caution**: With only four observed variables at most",
        "> two factors are estimable without Heywood cases. Treat factor analysis",
        "> outputs as tentative; PCA, correlation structure, and effective dimensionality",
        "> are the primary evidence.",
        "",
        "**Layer notes**:",
        f"- Layer {primary_layer}: contrast-validation-selected (Stage 2B AUC on held-out contrastive prompts)",
        f"- Layer {downstream_best_layer}: strongest downstream ETHICS diagonal dominance",
        "",
        "---",
        "",
    ]

    for layer, s in sorted(summaries.items()):
        role = ""
        if layer == primary_layer:
            role = " (contrast-validation-selected)"
        elif layer == downstream_best_layer:
            role = " (downstream ETHICS best)"

        lines += [
            f"## Layer {layer}{role}",
            "",
            f"**Items**: {s.n_items}  |  "
            f"**Effective dimensionality**: {s.effective_dimensionality:.2f}  |  "
            f"**Parallel analysis retains**: {s.n_components_parallel} component(s)",
            "",
            "### Correlation matrix",
            "",
            _corr_md_table(s.corr_df),
            "",
            f"Mean absolute off-diagonal correlation: **{s.mean_abs_off_diagonal_corr:.3f}**  ",
            f"Maximum absolute trait correlation: **{s.max_abs_trait_corr:.3f}** "
            f"({s.most_correlated_pair[0]} – {s.most_correlated_pair[1]})",
            "",
            "### PCA explained variance",
            "",
            _variance_md_table(s),
            "",
            f"Components for 80% variance: **{s.n_components_80pct}**  |  "
            f"Components for 90% variance: **{s.n_components_90pct}**",
            "",
            "### PCA loadings",
            "",
            _loadings_md_table(s),
            "",
            "### Parallel analysis",
            "",
        ]

        pa = s.parallel
        p = len(pa.observed_eigenvalues)
        lines.append("| Component | Observed λ | Random 95th pct | Retained? |")
        lines.append("|---|---|---|---|")
        for k in range(p):
            retained = "✅ Yes" if pa.observed_eigenvalues[k] > pa.random_eigenvalue_95th[k] else "❌ No"
            lines.append(
                f"| PC{k+1} "
                f"| {pa.observed_eigenvalues[k]:.3f} "
                f"| {pa.random_eigenvalue_95th[k]:.3f} "
                f"| {retained} |"
            )
        lines.append("")

        if s.warnings:
            lines.append("### Warnings")
            lines.append("")
            for w in s.warnings:
                lines.append(f"- ⚠ {w}")
            lines.append("")

        lines += [
            "### Interpretation",
            "",
            s.interpretation,
            "",
            "---",
            "",
        ]

    # Layer comparison section
    if primary_layer in summaries and downstream_best_layer in summaries:
        s32 = summaries[primary_layer]
        s40 = summaries[downstream_best_layer]
        lines += [
            f"## Layer {primary_layer} vs layer {downstream_best_layer} comparison",
            "",
            f"| Metric | Layer {primary_layer} (contrast-selected) | Layer {downstream_best_layer} (downstream best) |",
            "|---|---|---|",
            f"| PC1 explained variance | {s32.first_pc_variance:.1%} | {s40.first_pc_variance:.1%} |",
            f"| Effective dimensionality | {s32.effective_dimensionality:.2f} | {s40.effective_dimensionality:.2f} |",
            f"| Parallel analysis components | {s32.n_components_parallel} | {s40.n_components_parallel} |",
            f"| Mean off-diagonal |corr| | {s32.mean_abs_off_diagonal_corr:.3f} | {s40.mean_abs_off_diagonal_corr:.3f} |",
            f"| Max trait correlation | {s32.max_abs_trait_corr:.3f} | {s40.max_abs_trait_corr:.3f} |",
            "",
            (
                f"Layer {primary_layer} was selected by contrast-prompt validation AUC; "
                f"layer {downstream_best_layer} showed stronger ETHICS diagonal dominance. "
                "If the two layers show meaningfully different structure, "
                "report both and note the divergence."
            ),
            "",
        ]

    path = out_dir / "structure_analysis_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main save orchestrator
# ---------------------------------------------------------------------------


def save_all(
    summaries: dict[int, StructureSummary],
    layer_wide_tables: dict[int, "pd.DataFrame"],
    out_dir: str | Path,
    primary_layer: int = 32,
    downstream_best_layer: int = 40,
) -> dict[str, Path]:
    out_dir = _ensure(out_dir)
    saved: dict[str, Path] = {}

    for layer, s in summaries.items():
        saved[f"corr_layer{layer}"] = save_correlation_matrix(s, out_dir)
        saved[f"loadings_layer{layer}"] = save_pca_loadings(s, out_dir)
        saved[f"variance_layer{layer}"] = save_pca_variance(s, out_dir)
        saved[f"parallel_layer{layer}"] = save_parallel_analysis(s, out_dir)
        if layer in layer_wide_tables:
            saved[f"scores_layer{layer}"] = save_pca_scores(s, layer_wide_tables[layer], out_dir)
        fa_paths = save_factor_analysis(s, out_dir)
        for p in fa_paths:
            saved[f"fa_{p.stem}"] = p

    saved["summary"] = save_structure_summary(summaries, out_dir)
    saved["report"] = save_report(
        summaries, out_dir,
        primary_layer=primary_layer,
        downstream_best_layer=downstream_best_layer,
    )
    return saved
