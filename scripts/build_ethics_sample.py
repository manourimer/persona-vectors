#!/usr/bin/env python3
"""
Stage 1 — Build the ETHICS MVP item bank.

Downloads ETHICS test-split CSVs from HuggingFace, normalises them into a
common schema, samples n_items stratified across source splits, and saves:
  - data/processed/ethics_mvp_sample.parquet
  - data/processed/ethics_mvp_sample.csv

Run from the project root:
    python scripts/build_ethics_sample.py
    python scripts/build_ethics_sample.py --config configs/mvp_experiment.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_mvp_config
from src.data.ethics import load_all_ethics_splits, sample_ethics_items

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ETHICS MVP item bank")
    parser.add_argument("--config", default="configs/mvp_experiment.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_mvp_config(root / args.config)

    raw_cache = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    splits = cfg.dataset.ethics_splits
    n_items = cfg.dataset.n_items
    seed = cfg.random_seed

    print("=" * 64)
    print("  Build ETHICS MVP Sample")
    print("=" * 64)
    print(f"  Config        : {args.config}")
    print(f"  Source splits : {', '.join(splits)}")
    print(f"  Target n      : {n_items} items")
    print(f"  Random seed   : {seed}")
    print(f"  Cache dir     : {raw_cache}")
    print()

    # --- Load & normalise ---
    print("[1] Loading and normalising ETHICS splits …")
    full_df = load_all_ethics_splits(splits, cache_dir=raw_cache)

    print("\n  Items per split (full normalised dataset):")
    for split, count in full_df.groupby("source_split").size().items():
        print(f"    {split:<20} {count:>5} items")
    print(f"  {'TOTAL':<20} {len(full_df):>5} items")

    # --- Sample ---
    print(f"\n[2] Sampling {n_items} items (stratified, seed={seed}) …")
    sample_df = sample_ethics_items(full_df, n_items=n_items, random_seed=seed)

    print("\n  Items sampled per split:")
    for split, count in sample_df.groupby("source_split").size().items():
        print(f"    {split:<20} {count:>5} items")
    print(f"  {'TOTAL':<20} {len(sample_df):>5} items")

    # --- Save ---
    print("\n[3] Saving outputs …")
    parquet_path = processed_dir / "ethics_mvp_sample.parquet"
    csv_path = processed_dir / "ethics_mvp_sample.csv"

    sample_df.to_parquet(parquet_path, index=False)
    sample_df.to_csv(csv_path, index=False)

    print(f"  Saved: {parquet_path}")
    print(f"  Saved: {csv_path}")

    # --- Preview ---
    print("\n[4] First 5 rows of the sample:")
    preview_cols = ["item_id", "source_split", "label", "scenario_text"]
    preview = sample_df[preview_cols].head(5)
    for _, row in preview.iterrows():
        text_preview = str(row["scenario_text"])[:80].replace("\n", " ")
        print(
            f"  [{row['source_split']:<16}]  label={str(row['label']):<5}  "
            f"{row['item_id']}  \"{text_preview}…\""
        )

    # --- Annotation reminder ---
    print("\n" + "=" * 64)
    print("  REMINDER: primary_trait, secondary_traits, and")
    print("  annotation_confidence columns are BLANK in this output.")
    print("  ETHICS split names are NOT the construct traits.")
    print("  Construct traits (honesty / harmlessness / fairness /")
    print("  compassion) will be assigned in Stage 2 annotation.")
    print("=" * 64)


if __name__ == "__main__":
    main()
