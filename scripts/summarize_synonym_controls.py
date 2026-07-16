"""
Load all synonym control outputs and print a summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("outputs/controls/synonym_vectors/")


def main():
    print("=== Synonym Controls Summary ===\n")

    sim_path = OUT_DIR / "synonym_cosine_similarity.csv"
    if sim_path.exists():
        df = pd.read_csv(sim_path)
        n_correct = int(df["closest_matches_parent"].sum()) if "closest_matches_parent" in df.columns else "?"
        print(f"Cosine Similarity Table ({sim_path.name}):")
        print(df.to_string())
        print(f"\nClosest-parent match: {n_correct}/{len(df)}\n")
    else:
        print(f"Cosine similarity file not found: {sim_path}")
        print("Run: python scripts/run_synonym_vector_controls.py --mvp-only\n")

    agree_path = OUT_DIR / "synonym_projection_agreement.csv"
    if agree_path.exists():
        df = pd.read_csv(agree_path)
        print(f"Projection Agreement Table ({agree_path.name}):")
        print(df.to_string())
    else:
        print(f"Projection agreement file not found: {agree_path}")
        print("Run: python scripts/project_synonym_controls.py, then run_synonym_vector_controls.py")

    report_path = OUT_DIR / "synonym_controls_report.md"
    if report_path.exists():
        print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
