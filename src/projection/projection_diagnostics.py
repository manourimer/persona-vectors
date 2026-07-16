"""
Stage 3: Projection diagnostics.

Runs on mean-centered projections by default (pass preprocessing="raw" to
diagnose raw projections instead).  Produces a Markdown report, per-trait
projection summary, and per-layer correlation matrices.

These are sanity checks, not the final reliability/validity study.
Paraphrase/framing reliability comes in Stage 4.

No GPU, torch, or Modal required.

Public API
----------
    run_diagnostics(long_df, wide_df, item_df,
                    target_layer, preprocessing) -> DiagnosticsResult
    save_diagnostics(result, out_dir,
                     preprocessing_label)        -> tuple[Path, Path, Path]
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TRAITS: list[str] = ["honesty", "harmlessness", "fairness", "compassion"]

_NEAR_CONSTANT_STD_THRESHOLD = 0.01
_HIGH_CORRELATION_THRESHOLD = 0.95
_CHANCE_LEVEL = 1.0 / len(TRAITS)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticsResult:
    n_items: int
    n_traits: int
    n_layers: int
    n_missing_activations: int
    n_missing_vectors: int
    preprocessing: str

    projection_stats: pd.DataFrame       # columns: projected_trait, mean, std, min, max, n
    matching_table: pd.DataFrame         # index=primary_trait, columns=projected_trait
    correlation_matrix: pd.DataFrame     # index/columns = trait names

    diagonal_dominance: bool
    diagonal_dominance_rate: float

    warnings: list[str] = field(default_factory=list)

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def run_diagnostics(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    item_df: pd.DataFrame,
    target_layer: int | None = None,
    preprocessing: str = "mean_centered",
) -> DiagnosticsResult:
    """Run all diagnostics on projection outputs.

    Args:
        long_df:      Long-format projection DataFrame (may include multiple layers
                      and/or preprocessing variants).
        wide_df:      Wide-format projection DataFrame (target layer, centered by default).
        item_df:      Curated item bank DataFrame.
        target_layer: Layer to focus distribution stats on.  Defaults to min layer.
        preprocessing: "mean_centered" | "raw" — which variant to use from long_df.
    """
    warnings: list[str] = []

    # Filter to requested preprocessing if column present
    if "projection_preprocessing" in long_df.columns:
        long_df = long_df[long_df["projection_preprocessing"] == preprocessing].copy()

    n_items = long_df["item_id"].nunique()
    n_traits = long_df["projected_trait"].nunique()
    n_layers = long_df["layer"].nunique()

    item_ids_projected = set(long_df["item_id"].astype(str))
    item_ids_in_df = set(item_df["item_id"].astype(str))
    n_missing_acts = len(item_ids_in_df - item_ids_projected)
    if n_missing_acts:
        warnings.append(
            f"{n_missing_acts} item(s) in the item bank have no projection — "
            "check activation extraction logs."
        )

    layer = target_layer if target_layer is not None else int(long_df["layer"].min())
    layer_df = long_df[long_df["layer"] == layer].copy()

    # Projection distribution
    stats_rows: list[dict] = []
    for trait in TRAITS:
        vals = layer_df.loc[layer_df["projected_trait"] == trait, "projection"]
        if vals.empty:
            continue
        std = float(vals.std())
        mean = float(vals.mean())
        stats_rows.append(
            {
                "projected_trait": trait,
                "mean": mean,
                "std": std,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "n": int(len(vals)),
            }
        )
        if std < _NEAR_CONSTANT_STD_THRESHOLD:
            warnings.append(
                f"Projections for '{trait}' at layer {layer} are near-constant "
                f"(std={std:.4f}). The vector may not discriminate items."
            )
        # After mean-centering, means should be ≈ 0; warn if still large
        if preprocessing == "mean_centered" and abs(mean) > 10 * std + 1e-6:
            warnings.append(
                f"Mean-centered projection for '{trait}' at layer {layer} "
                f"has mean={mean:.2f} >> std={std:.4f}. "
                "Centering may not have fully removed the baseline direction."
            )
    projection_stats = pd.DataFrame(stats_rows)

    # Matching table
    matching_rows = (
        layer_df.groupby(["primary_trait", "projected_trait"])["projection"]
        .mean()
        .reset_index()
    )
    try:
        matching_table = matching_rows.pivot(
            index="primary_trait", columns="projected_trait", values="projection"
        )
        matching_table.columns.name = None
        matching_table.index.name = "primary_trait"
    except Exception:
        matching_table = pd.DataFrame()

    # Diagonal dominance
    proj_cols = [f"projection_{t}" for t in TRAITS if f"projection_{t}" in wide_df.columns]
    dominance_rate = float("nan")
    diagonal_dominance = False
    if proj_cols and "primary_trait" in wide_df.columns:
        def _is_dominant(row: pd.Series) -> bool:
            pt = row.get("primary_trait", "")
            col = f"projection_{pt}"
            if col not in row.index or pt not in TRAITS:
                return False
            own = row[col]
            others = [row[c] for c in proj_cols if c != col]
            return bool(not pd.isna(own) and others and own > max(others))

        flags = wide_df.apply(_is_dominant, axis=1)
        valid = flags.notna()
        dominance_rate = float(flags[valid].mean()) if valid.any() else float("nan")
        diagonal_dominance = dominance_rate > _CHANCE_LEVEL

    if not np.isnan(dominance_rate) and dominance_rate <= _CHANCE_LEVEL:
        warnings.append(
            f"Diagonal dominance ({dominance_rate:.3f}) is at or below four-way chance "
            f"({_CHANCE_LEVEL:.2f}). Items' annotated traits are not projecting highest "
            "on the matching vector. Investigate trait specificity."
        )

    # Correlation matrix
    if proj_cols and len(proj_cols) >= 2:
        corr = wide_df[proj_cols].corr()
        corr.index = [c.replace("projection_", "") for c in corr.index]
        corr.columns = [c.replace("projection_", "") for c in corr.columns]
    else:
        corr = pd.DataFrame()

    if not corr.empty:
        for i, t1 in enumerate(TRAITS):
            for t2 in TRAITS[i + 1:]:
                if t1 in corr.index and t2 in corr.columns:
                    r = corr.loc[t1, t2]
                    if abs(r) > _HIGH_CORRELATION_THRESHOLD:
                        warnings.append(
                            f"Trait projections '{t1}' and '{t2}' are extremely correlated "
                            f"(r={r:.3f}). Vectors may not discriminate distinct traits."
                        )

    # Dominant vector warning (raw projections only — expected after centering)
    if preprocessing == "raw" and not projection_stats.empty:
        means = projection_stats.set_index("projected_trait")["mean"]
        if len(means) > 1:
            spread = means.max() - means.min()
            stds = projection_stats.set_index("projected_trait")["std"]
            if spread > 5 * stds.mean() + 1e-6:
                warnings.append(
                    f"One vector dominates raw projections "
                    f"(mean spread={spread:.1f}). "
                    "This is expected with large residual-stream activations — "
                    "use mean-centered projections for interpretation."
                )

    return DiagnosticsResult(
        n_items=n_items,
        n_traits=n_traits,
        n_layers=n_layers,
        n_missing_activations=n_missing_acts,
        n_missing_vectors=0,
        preprocessing=preprocessing,
        projection_stats=projection_stats,
        matching_table=matching_table,
        correlation_matrix=corr,
        diagonal_dominance=diagonal_dominance,
        diagonal_dominance_rate=dominance_rate,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    md_rows: list[str] = []
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        md_rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + md_rows)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_diagnostics(
    result: DiagnosticsResult,
    out_dir: str | Path,
    preprocessing_label: str | None = None,
) -> tuple[Path, Path, Path]:
    """Save diagnostics markdown, summary CSV, and correlation matrix CSV.

    File names are suffixed with the preprocessing label
    (e.g. projection_diagnostics_centered.md).

    Returns (md_path, summary_csv_path, corr_csv_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = preprocessing_label or result.preprocessing
    # Normalize to short suffix
    suffix = "centered" if "centered" in label else "raw"

    md_path = out_dir / f"projection_diagnostics_{suffix}.md"
    summary_path = out_dir / "projection_summary.csv"
    corr_path = out_dir / f"projection_correlation_matrix_{suffix}.csv"

    # --- Markdown ---
    lines: list[str] = [
        f"# Stage 3: ETHICS Projection Diagnostics ({label})\n",
        "## Overview\n",
        f"- Preprocessing: **{label}**",
        f"- Items projected: {result.n_items}",
        f"- Trait vectors applied: {result.n_traits}",
        f"- Layers: {result.n_layers}",
        f"- Missing activations: {result.n_missing_activations}",
        "",
    ]

    if result.warnings:
        lines.append("## ⚠ Warnings\n")
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines.append("## ✅ No warnings\n")

    lines.append("## Projection Distribution (target layer)\n")
    if not result.projection_stats.empty:
        lines.append(_df_to_md(result.projection_stats))
    else:
        lines.append("_No data._")
    lines.append("")

    lines.append("## Mean Projection by primary_trait × projected_trait\n")
    lines.append(
        "Rows = annotated trait of item. Columns = trait vector projected onto.\n"
        "Diagonal entries should be higher than off-diagonal if vectors discriminate.\n"
    )
    if not result.matching_table.empty:
        lines.append(_df_to_md(result.matching_table.reset_index()))
    else:
        lines.append("_No data._")
    lines.append("")

    lines.append("## Diagonal Dominance\n")
    dom = result.diagonal_dominance_rate
    verdict = "✅" if result.diagonal_dominance else "❌"
    lines.append(
        f"Fraction of items where annotated trait projects highest: **{dom:.3f}** "
        f"(chance = {_CHANCE_LEVEL:.2f})  {verdict}\n"
    )

    lines.append("## Trait Projection Correlation Matrix\n")
    if not result.correlation_matrix.empty:
        lines.append(_df_to_md(result.correlation_matrix.reset_index().rename(
            columns={"index": "trait"}
        )))
    else:
        lines.append("_No data._")
    lines.append("")

    lines.append(
        textwrap.dedent("""\
        ## Notes

        - These diagnostics are sanity checks, not the final reliability/validity study.
        - Paraphrase and framing reliability analysis comes in Stage 4.
        - Diagonal dominance > chance (0.25) is necessary but not sufficient for validity.
        - High inter-trait correlations may indicate a general moral valence factor —
          investigate with factor analysis in Stage 5.
        - Weak diagonal dominance does not mean the data is invalid; it may reflect
          genuine overlap in how Gemma represents these moral traits.
        """)
    )

    md_path.write_text("\n".join(lines))

    # --- CSVs ---
    if not result.projection_stats.empty:
        result.projection_stats.to_csv(summary_path, index=False)
    else:
        pd.DataFrame().to_csv(summary_path, index=False)

    if not result.correlation_matrix.empty:
        result.correlation_matrix.to_csv(corr_path)
    else:
        pd.DataFrame().to_csv(corr_path)

    return md_path, summary_path, corr_path
