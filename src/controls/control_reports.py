"""
Consolidated controls report generator.

Assembles results from all control analyses into a single Markdown report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_controls_report(all_results: dict) -> str:
    """
    Returns a Markdown string with all control results.

    all_results keys (all optional):
        random_vector       — dict from run_random_vector_control
        shuffled_label      — dict from run_shuffled_label_control
        permuted_grouping   — dict from run_permuted_grouping_control
        exact_duplicate     — dict from run_exact_duplicate_control
        contrast_validation — dict from run_contrast_validation_control
        synthetic_scenarios — dict with 'df' key
        synonym_similarity  — DataFrame
        synonym_agreement   — DataFrame
        preprocessing       — dict with 'preprocessing_df' and 'layer_robustness_df'
    """
    lines = [
        "# Controls Suite Report\n\n",
        "This report summarises all negative controls, positive controls, "
        "convergent-validity controls, and robustness controls.\n\n",
    ]

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    status_rows = []
    for name in [
        "random_vector", "shuffled_label", "permuted_grouping",
        "exact_duplicate", "contrast_validation", "synthetic_scenarios",
        "synonym_similarity", "preprocessing",
    ]:
        present = name in all_results and all_results[name] is not None
        status_rows.append({"control": name, "status": "ran" if present else "not_run"})
    status_df = pd.DataFrame(status_rows)
    lines.append("## Summary\n\n")
    lines.append(status_df.to_markdown(index=False) if hasattr(status_df, "to_markdown") else status_df.to_string())
    lines.append("\n\n")

    # ---------------------------------------------------------------------------
    # Random vector control
    # ---------------------------------------------------------------------------
    lines.append("## Negative Control 1: Random Vectors\n\n")
    rv = all_results.get("random_vector")
    if rv and "compare_df" in rv:
        df = rv["compare_df"]
        lines.append(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string())
    elif rv and "summary_df" in rv:
        df = rv["summary_df"]
        lines.append("*Comparison to real metrics not yet computed. Null distribution summary:*\n\n")
        lines.append(df.to_string())
    else:
        lines.append("*Not run.*\n")
    lines.append("\n\n")

    # ---------------------------------------------------------------------------
    # Shuffled label control
    # ---------------------------------------------------------------------------
    lines.append("## Negative Control 2: Shuffled Labels\n\n")
    sl = all_results.get("shuffled_label")
    if sl:
        rm = sl.get("real_metrics", {})
        pv = sl.get("p_values", {})
        pct = sl.get("percentiles", {})
        for metric in ["diagonal_dominance", "matching_margin"]:
            real_val = rm.get(metric, float("nan"))
            p = pv.get(metric, float("nan"))
            percentile = pct.get(metric, float("nan"))
            null_mean = sl["null_dist_df"][metric].mean() if "null_dist_df" in sl else float("nan")
            null_p95 = sl["null_dist_df"][metric].quantile(0.95) if "null_dist_df" in sl else float("nan")
            lines.append(
                f"**{metric}**: real={real_val:.4f}, null_mean={null_mean:.4f}, "
                f"null_p95={null_p95:.4f}, p={p:.4f}, percentile={percentile:.1f}%\n\n"
            )
    else:
        lines.append("*Not run.*\n\n")

    # ---------------------------------------------------------------------------
    # Permuted grouping control
    # ---------------------------------------------------------------------------
    lines.append("## Negative Control 3: Permuted Item-Variant Grouping\n\n")
    pg = all_results.get("permuted_grouping")
    if pg:
        real_df = pg.get("real_g_coefficients")
        pv = pg.get("p_values", {})
        if real_df is not None:
            lines.append("### Real G-coefficients\n\n")
            lines.append(real_df.to_markdown(index=False) if hasattr(real_df, "to_markdown") else real_df.to_string())
            lines.append("\n\n")
    else:
        lines.append("*Not run.*\n\n")

    # ---------------------------------------------------------------------------
    # Exact duplicates
    # ---------------------------------------------------------------------------
    lines.append("## Positive Control 1: Exact Duplicates\n\n")
    ed = all_results.get("exact_duplicate")
    if ed and "summary_df" in ed:
        df = ed["summary_df"]
        g1_cols = [c for c in df.columns if "reliability_1" in c]
        if g1_cols:
            g1_mean = df[g1_cols[0]].mean()
            lines.append(f"Mean G(k=1): {g1_mean:.4f} (expected: ~1.0)\n\n")
        lines.append(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string())
        lines.append("\n\n")
    else:
        lines.append("*Not run.*\n\n")

    # ---------------------------------------------------------------------------
    # Contrast validation
    # ---------------------------------------------------------------------------
    lines.append("## Positive Control 2: Contrast Validation\n\n")
    cv = all_results.get("contrast_validation")
    if cv and "summary_df" in cv:
        df = cv["summary_df"]
        lines.append(f"All pass: {cv.get('all_pass', '?')}, Warnings: {cv.get('n_warnings', '?')}\n\n")
        lines.append(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string())
        lines.append("\n\n")
    else:
        lines.append("*Not run.*\n\n")

    # ---------------------------------------------------------------------------
    # Synthetic scenarios
    # ---------------------------------------------------------------------------
    lines.append("## Positive Control 3: Synthetic Obvious Scenarios\n\n")
    ss = all_results.get("synthetic_scenarios")
    if ss and "df" in ss:
        df = ss["df"]
        n_reviewed = int(df["reviewed"].sum()) if "reviewed" in df.columns else "?"
        n_total = len(df)
        n_filled = int((df.get("scenario_text", pd.Series([])) != "").sum()) if "scenario_text" in df.columns else "?"
        lines.append(f"Total: {n_total}, Filled: {n_filled}, Reviewed: {n_reviewed}\n\n")
    else:
        lines.append("*Not run.*\n\n")

    # ---------------------------------------------------------------------------
    # Synonym vectors
    # ---------------------------------------------------------------------------
    lines.append("## Convergent-Validity Control: Synonym Vectors\n\n")
    sym = all_results.get("synonym_similarity")
    if sym is not None and not sym.empty:
        n_correct = int(sym["closest_matches_parent"].sum()) if "closest_matches_parent" in sym.columns else "?"
        lines.append(f"Closest-parent match: {n_correct}/{len(sym)}\n\n")
        lines.append(sym.to_markdown(index=False) if hasattr(sym, "to_markdown") else sym.to_string())
        lines.append("\n\n")
    else:
        lines.append("*Not run (synonym vectors not yet built — run Stage 2B with synonym artifacts first).*\n\n")

    # ---------------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------------
    lines.append("## Robustness Control: Raw vs Centered Preprocessing\n\n")
    prep = all_results.get("preprocessing")
    if prep and "preprocessing_df" in prep:
        df = prep["preprocessing_df"]
        lines.append(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string())
        lines.append("\n\n")
    else:
        lines.append("*Not run.*\n\n")

    # ---------------------------------------------------------------------------
    # Layer robustness
    # ---------------------------------------------------------------------------
    lines.append("## Robustness Control: Layer Comparison\n\n")
    if prep and "layer_robustness_df" in prep:
        df = prep["layer_robustness_df"]
        lines.append(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string())
        lines.append("\n\n")
    else:
        lines.append("*Not run.*\n\n")

    # ---------------------------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------------------------
    lines.append("## Overall Interpretation\n\n")
    lines.append(
        "The controls suite validates that the observed results are specific to the "
        "moral persona vectors and are not artifacts of the analysis pipeline.\n\n"
        "- **Random vectors**: If real vectors outperform random directions on structure "
        "and label-alignment metrics, this rules out that any direction produces similar patterns.\n"
        "- **Shuffled labels**: A significant p-value (p < 0.05) for diagonal dominance confirms "
        "that trait-label alignment exceeds chance.\n"
        "- **Permuted grouping**: If real G-coefficients exceed the permuted null, reliability "
        "is driven by stable item identity, not coincidental grouping.\n"
        "- **Exact duplicates** (positive control): G ~1.0 confirms the reliability pipeline is bug-free.\n"
        "- **Contrast validation** (positive control): AUC ≥ 0.75 for all trait × layer combos "
        "confirms vectors still reproduce their calibration signal.\n"
        "- **Synonym vectors**: Closest-parent match confirms construct validity across synonymous "
        "wordings of trait names.\n"
        "- **Preprocessing**: Stability of findings across raw and centered projections confirms "
        "that centering does not manufacture the observed effects.\n"
    )

    return "".join(lines)


def save_controls_report(report_md: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "controls_report.md"
    path.write_text(report_md)
    print(f"[controls_report] Saved to {path}")
    return path
