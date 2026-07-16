"""
Stage 3 — Step 1: Extract last-prompt-token activations from ETHICS items.

Runs each curated ETHICS scenario through Gemma-3-12B on Modal GPU and
extracts the residual-stream activation at the last prompt token (before
any generation).  Saves .npy files and a metadata parquet locally.

TOKEN SCOPE: last_prompt_token.  This differs from Stage 2B, which used
mean-response-token activations for vector construction.

Usage:
    # Smoke test — 10 items
    python scripts/extract_ethics_activations.py --limit 10

    # Full extraction
    python scripts/extract_ethics_activations.py

    # Mock mode (no GPU)
    python scripts/extract_ethics_activations.py --mock
"""

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_LAYERS = "32,40,47"
_DEFAULT_BATCH_SIZE = 8
_DEFAULT_OUT_DIR = "outputs/ethics_projection"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract last-prompt-token activations from curated ETHICS items."
    )
    parser.add_argument(
        "--item-bank",
        default="data/processed/ethics_curated_mvp.parquet",
    )
    parser.add_argument(
        "--candidate-layers", default=_DEFAULT_LAYERS,
        help="Comma-separated layer indices (default: 32,40,47).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=_DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--out-dir", default=_DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Extract only this many items (0 = all). Use 10 for smoke test.",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Generate random activations locally (no GPU required).",
    )
    parser.add_argument(
        "--mock-hidden-dim", type=int, default=64,
    )
    args = parser.parse_args()

    out_path = _ROOT / args.out_dir

    if args.mock:
        _run_mock(args, out_path)
    else:
        _run_modal(args, out_path)


def _run_mock(args: argparse.Namespace, out_path: Path) -> None:
    import numpy as np
    import pandas as pd

    from src.projection.ethics_projection import (
        build_projection_jobs,
        load_item_bank,
        validate_item_bank,
    )

    layers = [int(l) for l in args.candidate_layers.split(",")]
    item_df = load_item_bank(_ROOT / args.item_bank)
    validate_item_bank(item_df)
    if args.limit:
        item_df = item_df.head(args.limit)

    act_dir = out_path / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [mock] Extracting activations for {len(item_df)} ETHICS items ...")
    print(f"  Candidate layers: {layers}  hidden_dim: {args.mock_hidden_dim}")
    print(f"  Token scope: last_prompt_token")

    rng = np.random.default_rng(42)
    records: list[dict] = []

    for _, item in item_df.iterrows():
        item_id = str(item["item_id"])
        for layer in layers:
            act = rng.standard_normal(args.mock_hidden_dim).astype(np.float32)
            apath = act_dir / f"{item_id}_layer{layer}.npy"
            np.save(apath, act)
            records.append(
                {
                    "item_id": item_id,
                    "layer": layer,
                    "source_split": item.get("source_split", ""),
                    "primary_trait": item.get("primary_trait", ""),
                    "activation_path": str(apath),
                    "hidden_dim": args.mock_hidden_dim,
                    "token_position": "last_prompt_token",
                    "model_name": "mock",
                }
            )

    meta_df = pd.DataFrame(records)
    meta_pq = out_path / "ethics_activation_metadata.parquet"
    meta_csv = out_path / "ethics_activation_metadata.csv"
    meta_df.to_parquet(meta_pq, index=False)
    meta_df.to_csv(meta_csv, index=False)

    print(f"  Activation metadata saved: {meta_pq}")
    print(f"  Total activation records: {len(meta_df)}")
    print(f"  .npy files saved to: {act_dir}")


def _run_modal(args: argparse.Namespace, out_path: Path) -> None:
    cmd = [
        "modal", "run",
        "modal_apps/extract_ethics_prompt_activations.py",
        f"--item-bank-path={args.item_bank}",
        f"--candidate-layers={args.candidate_layers}",
        f"--batch-size={args.batch_size}",
        f"--out-dir={args.out_dir}",
    ]
    if args.limit:
        cmd.append(f"--limit={args.limit}")

    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        print("  ERROR: Modal extraction failed.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
