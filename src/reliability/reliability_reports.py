"""
Stage 4D: Reliability report generation and output saving.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_reliability_summary(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reliability_summary.csv"
    results_df.to_csv(path, index=False)
    print(f"[save] {path}")


def save_variance_components(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "layer", "projected_trait",
        "between_item_var", "within_item_var", "total_var",
        "within_item_sd", "between_item_sd", "variance_ratio",
        "n_items_used", "clamped_negative",
    ]
    path = out_dir / "variance_components.csv"
    results_df[[c for c in cols if c in results_df.columns]].to_csv(path, index=False)
    print(f"[save] {path}")


def save_d_study_results(d_study_df: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "d_study_results.csv"
    d_study_df.to_csv(path, index=False)
    print(f"[save] {path}")


def save_item_level(results: list, out_dir: str | Path) -> None:
    """Save per-item means and SDs for all (layer, projected_trait) combinations."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        for item_id in r.item_means.index:
            rows.append({
                "item_id": item_id,
                "layer": r.layer,
                "projected_trait": r.projected_trait,
                "item_mean_projection": r.item_means[item_id],
                "item_sd_projection": r.item_sds.get(item_id, 0.0),
                "n_variants": int(
                    (r.item_means.index == item_id).sum()
                    if hasattr(r.item_means.index, "__iter__") else 1
                ),
            })

    df = pd.DataFrame(rows)
    path = out_dir / "item_level_reliability_long.csv"
    df.to_csv(path, index=False)
    print(f"[save] {path}")


def generate_report(
    results_df: pd.DataFrame,
    d_study_df: pd.DataFrame,
    meta: dict,
) -> str:
    """
    Generate a Markdown reliability report.

    meta keys: n_items_total, n_items_used, n_variants_total,
                n_items_missing_paraphrases, layers, projected_traits
    """
    lines = []

    lines.append("# Stage 4D: Reliability / Generalizability Analysis Report\n")

    # Dataset summary
    lines.append("## Dataset Summary\n")
    lines.append(f"- Total items in bank: {meta.get('n_items_total', 'N/A')}")
    lines.append(f"- Items used in analysis: {meta.get('n_items_used', 'N/A')}")
    lines.append(f"- Items dropped (too few variants): {meta.get('n_items_missing_paraphrases', 'N/A')}")
    lines.append(f"- Total variant observations: {meta.get('n_variants_total', 'N/A')}")
    lines.append(f"- Layers analyzed: {meta.get('layers', [])}")
    lines.append(f"- Projected traits: {meta.get('projected_traits', [])}")
    lines.append("")

    # Reliability table
    lines.append("## Reliability by Layer × Projected Trait\n")
    lines.append("Single variant (k=1) and average of 3 variants (k=3).\n")
    if not results_df.empty:
        tbl = results_df[["layer", "projected_trait", "reliability_1", "reliability_3"]].copy()
        tbl["reliability_1"] = tbl["reliability_1"].map("{:.3f}".format)
        tbl["reliability_3"] = tbl["reliability_3"].map("{:.3f}".format)
        lines.append(tbl.to_markdown(index=False))
    else:
        lines.append("(no results)")
    lines.append("")

    # Best/worst
    primary_layer = meta.get("primary_layer", 32)
    if not results_df.empty:
        primary = results_df[results_df["layer"] == primary_layer]
        if not primary.empty:
            best = primary.loc[primary["reliability_1"].idxmax()]
            worst = primary.loc[primary["reliability_1"].idxmin()]
            lines.append(f"## Best / Worst at Primary Layer ({primary_layer})\n")
            lines.append(f"- **Best**: `{best['projected_trait']}` = {best['reliability_1']:.3f}")
            lines.append(f"- **Worst**: `{worst['projected_trait']}` = {worst['reliability_1']:.3f}")
            lines.append("")

    # Layer 32 vs 40
    downstream_layer = meta.get("downstream_best_layer", 40)
    if not results_df.empty and primary_layer in results_df["layer"].values and downstream_layer in results_df["layer"].values:
        l32 = results_df[results_df["layer"] == primary_layer]["reliability_1"].mean()
        l40 = results_df[results_df["layer"] == downstream_layer]["reliability_1"].mean()
        lines.append(f"## Layer {primary_layer} vs Layer {downstream_layer} Comparison\n")
        lines.append(f"- Mean reliability (k=1) at layer {primary_layer}: {l32:.3f}")
        lines.append(f"- Mean reliability (k=1) at layer {downstream_layer}: {l40:.3f}")
        lines.append("")

    # D-study table
    lines.append("## D-Study: Reliability Improves with More Paraphrases\n")
    lines.append(f"Primary layer ({primary_layer}), all projected traits.\n")
    if not d_study_df.empty:
        primary_d = d_study_df[d_study_df["layer"] == primary_layer]
        if not primary_d.empty:
            pivot = primary_d.pivot(index="n_paraphrases", columns="projected_trait", values="g_coefficient")
            pivot = pivot.map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            lines.append(pivot.to_markdown())
    lines.append("")

    # Interpretation
    lines.append("## Interpretation\n")
    lines.append(
        "Reliability (ICC/G-coefficient) measures the proportion of total projection variance "
        "attributable to stable between-item differences versus wording noise. "
        "Values > 0.70 indicate adequate generalizability; values > 0.85 indicate good generalizability. "
        "The D-study shows how reliability improves as more paraphrases are averaged."
    )
    lines.append("")

    # Caveats
    lines.append("## Caveats\n")
    lines.append(
        "- This is a one-facet G-theory analysis (facet = paraphrase). Framing effects are not modeled.\n"
        "- Centering removes the cross-item mean activation; all projections are deviation scores.\n"
        "- Negative between-item variance estimates are clamped to zero (can occur when within-item noise dominates).\n"
        "- Results depend on the quality and diversity of the generated paraphrases.\n"
    )

    return "\n".join(lines)


def save_report(report_md: str, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reliability_analysis_report.md"
    path.write_text(report_md)
    print(f"[save] {path}")


def save_all(
    results: list,
    results_df: pd.DataFrame,
    d_study_df: pd.DataFrame,
    meta: dict,
    out_dir: str | Path,
) -> None:
    """Save all reliability analysis outputs."""
    save_reliability_summary(results_df, out_dir)
    save_variance_components(results_df, out_dir)
    save_d_study_results(d_study_df, out_dir)
    save_item_level(results, out_dir)
    report_md = generate_report(results_df, d_study_df, meta)
    save_report(report_md, out_dir)
