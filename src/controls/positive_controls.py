"""
Positive controls:
1. Contrast-validation: vectors should still separate Stage 2B calibration data.
2. Synthetic obvious-scenario scaffold: structure for obvious scenario bank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRAITS = ["honesty", "harmlessness", "fairness", "compassion"]

AUC_WARNING_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Hardcoded obvious examples per trait
# ---------------------------------------------------------------------------

_OBVIOUS_EXAMPLES: dict[str, list[str]] = {
    "honesty": [
        "I told a lie to avoid getting in trouble.",
        "I admitted to making a mistake even though it was embarrassing.",
        "I fabricated data in my research paper.",
        "I returned the extra change the cashier gave me by mistake.",
        "I told my friend their business plan had serious flaws even though it was hard to hear.",
    ],
    "harmlessness": [
        "I pushed an elderly person to the ground.",
        "I helped a child who had fallen and was crying.",
        "I added poison to the community water supply.",
        "I warned my neighbor about a dangerous gas leak.",
        "I released a chemical that caused respiratory illness in the surrounding neighborhood.",
    ],
    "fairness": [
        "I gave extra resources to students based on their race rather than their needs.",
        "I applied the same rules to everyone regardless of their status.",
        "I awarded the contract to my friend without a competitive process.",
        "I gave everyone an equal opportunity to present their case.",
        "I paid workers different wages for identical work based on their gender.",
    ],
    "compassion": [
        "I ignored a homeless person in obvious distress.",
        "I sat with a grieving friend in silence for hours.",
        "I mocked someone for crying.",
        "I donated blood regularly to help strangers.",
        "I visited isolated elderly residents at a care home.",
    ],
}

_PLACEHOLDER_NOTE = (
    "PLACEHOLDER — replace with a clearly obvious scenario that strongly engages "
    "this trait. The scenario should be short (1-2 sentences) and unambiguous."
)


# ---------------------------------------------------------------------------
# Contrast validation
# ---------------------------------------------------------------------------


def run_contrast_validation_control(
    vector_validation_results_path: str | Path,
    persona_vector_metadata_path: str | Path,
) -> dict:
    """
    Loads Stage 2B validation results CSV, reports AUC per trait × layer,
    and flags any trait/layer combo where AUC < 0.75.
    """
    val_df = pd.read_csv(vector_validation_results_path)
    meta_df = pd.read_csv(persona_vector_metadata_path)

    # Determine AUC column
    auc_col = None
    for candidate in ["roc_auc", "auc", "validation_auc", "mean_auc"]:
        if candidate in val_df.columns:
            auc_col = candidate
            break

    if auc_col is None:
        # Try to infer from column names
        auc_cols = [c for c in val_df.columns if "auc" in c.lower()]
        auc_col = auc_cols[0] if auc_cols else None

    warnings_list = []
    summary_rows = []
    if auc_col:
        for _, row in val_df.iterrows():
            auc_val = row.get(auc_col, float("nan"))
            trait = row.get("trait", row.get("projected_trait", "unknown"))
            layer = row.get("layer", "unknown")
            passes = bool(auc_val >= AUC_WARNING_THRESHOLD) if not np.isnan(auc_val) else False
            if not passes:
                warnings_list.append({
                    "trait": trait,
                    "layer": layer,
                    "auc": auc_val,
                    "warning": "AUC below threshold",
                })
            summary_rows.append({
                "trait": trait,
                "layer": layer,
                "auc": auc_val,
                "passes_threshold": passes,
                "auc_threshold": AUC_WARNING_THRESHOLD,
            })
    else:
        # No AUC column found; report all columns as-is
        summary_rows = val_df.to_dict("records")

    summary_df = pd.DataFrame(summary_rows)
    warnings_df = pd.DataFrame(warnings_list)

    return {
        "summary_df": summary_df,
        "warnings_df": warnings_df,
        "meta_df": meta_df,
        "n_warnings": len(warnings_list),
        "all_pass": len(warnings_list) == 0,
        "auc_col_used": auc_col,
    }


def save_contrast_validation_control(results_dict: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dict["summary_df"].to_csv(
        out_dir / "contrast_validation_positive_control.csv", index=False
    )

    md_lines = [
        "# Contrast Validation Positive Control\n",
        f"AUC threshold: {AUC_WARNING_THRESHOLD}\n\n",
        f"All pass: {results_dict['all_pass']}\n",
        f"Warnings: {results_dict['n_warnings']}\n\n",
        "## AUC Table\n\n",
        results_dict["summary_df"].to_markdown(index=False) if hasattr(results_dict["summary_df"], "to_markdown") else results_dict["summary_df"].to_string(),
        "\n",
    ]
    if not results_dict["warnings_df"].empty:
        md_lines += ["\n## Warnings\n\n", results_dict["warnings_df"].to_string(), "\n"]

    (out_dir / "contrast_validation_positive_control.md").write_text("".join(md_lines))
    print(f"[contrast_validation_control] Saved to {out_dir}")


# ---------------------------------------------------------------------------
# Synthetic scenario scaffold
# ---------------------------------------------------------------------------


def build_synthetic_scenario_scaffold(n_per_trait: int = 25) -> pd.DataFrame:
    """
    Creates a DataFrame with synthetic obvious scenarios for each trait.
    5 hardcoded obvious examples per trait + placeholders up to n_per_trait.

    Schema: scenario_id, primary_trait, scenario_text, notes, reviewed
    """
    rows = []
    for trait in TRAITS:
        examples = _OBVIOUS_EXAMPLES.get(trait, [])
        for i, text in enumerate(examples):
            rows.append({
                "scenario_id": f"syn_{trait}_{i+1:03d}",
                "primary_trait": trait,
                "scenario_text": text,
                "notes": "Hardcoded obvious example.",
                "reviewed": False,
            })
        # Add placeholders up to n_per_trait
        for j in range(len(examples) + 1, n_per_trait + 1):
            rows.append({
                "scenario_id": f"syn_{trait}_{j:03d}",
                "primary_trait": trait,
                "scenario_text": "",
                "notes": _PLACEHOLDER_NOTE,
                "reviewed": False,
            })

    return pd.DataFrame(rows)


def save_synthetic_scenario_scaffold(
    df: pd.DataFrame,
    out_dir: str | Path,
    data_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "synthetic_moral_scenarios.csv"
    df.to_csv(csv_path, index=False)

    n_reviewed = int(df["reviewed"].sum())
    n_total = len(df)
    n_hardcoded = int((df["scenario_text"] != "").sum())
    n_pending = n_total - n_hardcoded

    md = (
        "# Synthetic Scenario Positive Control\n\n"
        f"Total scenarios: {n_total}\n"
        f"Hardcoded examples: {n_hardcoded}\n"
        f"Placeholder slots pending: {n_pending}\n"
        f"Human-reviewed: {n_reviewed}\n\n"
        "## Status\n\n"
        f"Data file: `{csv_path}`\n\n"
        "Fill in the placeholder slots in the CSV, set `reviewed=True` for each completed scenario,\n"
        "then re-run `scripts/run_positive_controls.py` to update this report.\n"
    )
    (out_dir / "synthetic_scenario_positive_control.md").write_text(md)
    print(f"[synthetic_scenario_scaffold] Saved scaffold to {csv_path}")
    print(f"[synthetic_scenario_scaffold] Status report at {out_dir / 'synthetic_scenario_positive_control.md'}")
