"""
Stage 4B — Generate reliability paraphrase variants.

Invokes modal_apps/generate_reliability_variants.py via `modal run`.

Usage:
    # Smoke test — 10 items
    python scripts/generate_reliability_variants.py --limit 10

    # Full run — all 204 items
    python scripts/generate_reliability_variants.py

    # Resume after interruption
    python scripts/generate_reliability_variants.py --resume

    # Mock mode (no GPU, for offline testing / CI)
    python scripts/generate_reliability_variants.py --generation-method mock --limit 10
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MODAL_APP = _ROOT / "modal_apps" / "generate_reliability_variants.py"
_DEFAULT_OUTPUT_DIR = "data/processed/reliability_variants"
_DEFAULT_INPUT = "data/processed/ethics_curated_mvp.parquet"


def _check_modal() -> None:
    if shutil.which("modal") is None:
        print(
            "ERROR: 'modal' CLI not found.\n"
            "\n"
            "Install it with:\n"
            "    pip install modal\n"
            "\n"
            "Then authenticate once:\n"
            "    modal token new\n"
        )
        sys.exit(1)


def _run_mock(args: argparse.Namespace) -> None:
    """Run mock generation locally without GPU."""
    sys.path.insert(0, str(_ROOT))
    import pandas as pd
    from src.reliability.variant_generation import (
        load_item_bank,
        mock_generate_paraphrases,
        save_variants,
    )
    from src.reliability.variant_validation import flag_intra_item_duplicates

    item_df = load_item_bank(_ROOT / args.input)
    if args.limit:
        item_df = item_df.head(args.limit).copy()

    print(f"\n  ── Stage 4B: Mock Variant Generation ───────────────────────────")
    print(f"  Items        : {len(item_df)}")
    print(f"  Paraphrases  : {args.n_paraphrases} per item (mock)")
    print(f"  Output dir   : {_ROOT / args.output_dir}")

    df = mock_generate_paraphrases(
        item_df,
        n_paraphrases=args.n_paraphrases,
        framing="neutral",
        generation_model_name="mock",
    )
    df = flag_intra_item_duplicates(df)

    out_dir = _ROOT / args.output_dir
    csv_path, parquet_path = save_variants(df, out_dir, "ethics_reliability_variants_raw")

    print(f"\n  Total rows   : {len(df)}")
    print(f"  Originals    : {int((df['variant_type']=='original').sum())}")
    print(f"  Paraphrases  : {int((df['variant_type']=='paraphrase').sum())}")
    print(f"  Saved        : {csv_path}")
    print(f"\n  Next: python scripts/validate_reliability_variants.py")


def _run_modal(args: argparse.Namespace) -> None:
    _check_modal()
    cmd = [
        "modal", "run", str(_MODAL_APP),
        f"--input={args.input}",
        f"--output-dir={args.output_dir}",
        f"--n-paraphrases={args.n_paraphrases}",
        f"--batch-size={args.batch_size}",
    ]
    if args.limit:
        cmd.append(f"--limit={args.limit}")
    if args.resume:
        cmd.append("--resume")
    if not args.strict:
        cmd.append("--no-strict")
    print(f"\n  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(_ROOT))
    sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4B: Generate paraphrase variants for reliability analysis."
    )
    parser.add_argument("--input", default=_DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Process only N items (0=all)")
    parser.add_argument("--n-paraphrases", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--generation-method",
        choices=["modal_vllm", "mock"],
        default="modal_vllm",
    )
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--non-strict", dest="strict", action="store_false")
    args = parser.parse_args()

    if args.generation_method == "mock":
        _run_mock(args)
    else:
        _run_modal(args)


if __name__ == "__main__":
    main()
