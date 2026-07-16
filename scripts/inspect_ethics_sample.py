#!/usr/bin/env python3
"""
Inspect the ETHICS MVP item bank produced by build_ethics_sample.py.

Prints:
  - Item counts by source_split
  - A random sample of 20 items with item_id, source_split, label, and text
  - Annotation coverage (how many items have primary_trait filled)

Run from the project root:
    python scripts/inspect_ethics_sample.py
    python scripts/inspect_ethics_sample.py --n 30 --seed 99
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PARQUET = "data/processed/ethics_mvp_sample.parquet"
CSV = "data/processed/ethics_mvp_sample.csv"


def load_sample(root: Path) -> pd.DataFrame:
    parquet_path = root / PARQUET
    csv_path = root / CSV
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"No sample file found. Run scripts/build_ethics_sample.py first.\n"
        f"  Expected: {parquet_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the ETHICS MVP item bank")
    parser.add_argument("--n", type=int, default=20, help="Number of random items to show")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for item selection")
    parser.add_argument(
        "--split",
        default=None,
        help="Filter to one source_split (e.g. virtue, justice)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    df = load_sample(root)

    print("=" * 66)
    print("  ETHICS MVP Sample — Inspection Report")
    print("=" * 66)

    # --- Counts ---
    print(f"\n[1] Item counts  (total: {len(df)})")
    for split, count in df.groupby("source_split").size().items():
        print(f"  {split:<20} {count:>4} items")

    # --- Annotation coverage ---
    print("\n[2] Annotation coverage")
    annotated = (df["primary_trait"].notna() & (df["primary_trait"] != "")).sum()
    print(f"  primary_trait filled : {annotated} / {len(df)}")
    if annotated == 0:
        print("  (blank — annotation not yet done; expected at this stage)")

    # --- Label distribution ---
    print("\n[3] Label distribution by source_split")
    for split in df["source_split"].unique():
        sub = df[df["source_split"] == split]
        label_counts = sub["label"].value_counts(dropna=False).to_dict()
        semantics = sub["label_semantics"].iloc[0]
        print(f"  {split}")
        print(f"    semantics : {semantics}")
        print(f"    counts    : {label_counts}")

    # --- Random item display ---
    display_df = df if args.split is None else df[df["source_split"] == args.split]
    if display_df.empty:
        print(f"\n  No items found for split '{args.split}'")
        sys.exit(0)

    n_show = min(args.n, len(display_df))
    sample = display_df.sample(n=n_show, random_state=args.seed)

    filter_note = f"  (filtered to split={args.split})" if args.split else ""
    print(f"\n[4] {n_show} random items{filter_note}")
    print("-" * 66)

    for _, row in sample.sort_values("source_split").iterrows():
        text = str(row["scenario_text"]).replace("\n", " ")
        label_str = "—" if pd.isna(row["label"]) else str(int(row["label"]))
        trait_str = str(row["primary_trait"]) if row["primary_trait"] else "(unannotated)"
        print(f"  item_id      : {row['item_id']}")
        print(f"  source_split : {row['source_split']}")
        print(f"  label        : {label_str}  ({row['label_semantics']})")
        print(f"  primary_trait: {trait_str}")
        print(f"  text         : {text[:200]}")
        print()

    print("=" * 66)
    print("  To annotate items, fill the primary_trait column in:")
    print(f"  {root / CSV}")
    print("  Valid traits: honesty | harmlessness | fairness | compassion")
    print("=" * 66)


if __name__ == "__main__":
    main()
