"""
Stage 4A — Projection structure analysis (RQ1).

Analyzes the correlation and PCA structure of the four trait projections
from Stage 3 to ask whether the projections behave like one latent morality
dimension or several separable dimensions.

No GPU or new Gemma activations required.  All analysis runs on the existing
centered projection tables from outputs/ethics_projection/.

Usage:
    python scripts/run_structure_analysis.py
    python scripts/run_structure_analysis.py --layers 32 40 47
    python scripts/run_structure_analysis.py --primary-layer 32
    python scripts/run_structure_analysis.py --no-parallel-analysis
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_LONG_PATH = "outputs/ethics_projection/ethics_trait_projections_centered_long.parquet"
_WIDE_PATH = "outputs/ethics_projection/ethics_trait_projections_centered_wide.parquet"
_OUT_DIR = "outputs/structure_analysis"
_DEFAULT_LAYERS = [32, 40, 47]
_PRIMARY_LAYER = 32
_DOWNSTREAM_BEST = 40


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4A: Projection structure analysis (RQ1)."
    )
    parser.add_argument("--long-path", default=_LONG_PATH)
    parser.add_argument("--wide-path", default=_WIDE_PATH)
    parser.add_argument("--out-dir", default=_OUT_DIR)
    parser.add_argument("--layers", nargs="+", type=int, default=_DEFAULT_LAYERS)
    parser.add_argument("--primary-layer", type=int, default=_PRIMARY_LAYER)
    parser.add_argument("--downstream-best-layer", type=int, default=_DOWNSTREAM_BEST)
    parser.add_argument("--no-parallel-analysis", action="store_true")
    parser.add_argument("--no-factor-analysis", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    from src.analysis.structure_analysis import (
        build_layer_wide_tables,
        load_centered_long,
        load_centered_wide,
        run_structure_analysis,
        validate_projection_columns,
    )
    from src.analysis.structure_reports import save_all

    long_path = _ROOT / args.long_path
    wide_path = _ROOT / args.wide_path
    out_path = _ROOT / args.out_dir

    if not long_path.exists():
        print(f"\n  ERROR: Long-format projection file not found:\n    {long_path}")
        print("  Run scripts/compute_ethics_projections.py --preprocessing both first.")
        sys.exit(1)

    print("\n  ══════ Stage 4A: Projection Structure Analysis ════════════════════")
    print(f"\n  Layers           : {args.layers}")
    print(f"  Primary layer    : {args.primary_layer} (contrast-validation-selected)")
    print(f"  Downstream best  : {args.downstream_best_layer} (strongest ETHICS structure)")
    print(f"  Parallel analysis: {'off' if args.no_parallel_analysis else 'on'}")
    print(f"  Factor analysis  : {'off' if args.no_factor_analysis else 'on (if package available)'}")
    print(f"  Output dir       : {out_path}")

    summaries = run_structure_analysis(
        long_path=long_path,
        wide_path_layer32=wide_path,
        layers=args.layers,
        standardize_projections=True,
        run_pa=not args.no_parallel_analysis,
        run_fa=not args.no_factor_analysis,
        fa_max_factors=4,
        random_seed=args.random_seed,
    )

    # Rebuild wide tables for score saving
    long_df = load_centered_long(long_path)
    layer_wide_tables = build_layer_wide_tables(long_df, args.layers)
    try:
        wide32 = load_centered_wide(wide_path)
        layer_wide_tables[32] = wide32
    except Exception:
        pass

    saved = save_all(
        summaries,
        layer_wide_tables,
        out_path,
        primary_layer=args.primary_layer,
        downstream_best_layer=args.downstream_best_layer,
    )

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n  ── Per-layer summary ────────────────────────────────────────────────\n")
    for layer in args.layers:
        if layer not in summaries:
            print(f"    Layer {layer}: no data")
            continue
        s = summaries[layer]
        role = ""
        if layer == args.primary_layer:
            role = " [contrast-selected]"
        elif layer == args.downstream_best_layer:
            role = " [downstream best]"
        print(f"    Layer {layer}{role}")
        print(f"      PC1 variance          : {s.first_pc_variance:.1%}")
        print(f"      Effective dim          : {s.effective_dimensionality:.2f}")
        print(f"      Parallel analysis      : {s.n_components_parallel} component(s) retained")
        print(f"      Max trait correlation  : {s.max_abs_trait_corr:.3f} "
              f"({s.most_correlated_pair[0]}–{s.most_correlated_pair[1]})")
        print(f"      Mean |off-diag corr|   : {s.mean_abs_off_diagonal_corr:.3f}")
        print(f"      Components for 80% var : {s.n_components_80pct}")
        print()

    if args.primary_layer in summaries and args.downstream_best_layer in summaries:
        sp = summaries[args.primary_layer]
        sd = summaries[args.downstream_best_layer]
        print(f"  ── Layer {args.primary_layer} vs {args.downstream_best_layer} "
              f"comparison ─────────────────────────────────────\n")
        print(f"    PC1 variance  : layer {args.primary_layer} = {sp.first_pc_variance:.1%}  "
              f"vs  layer {args.downstream_best_layer} = {sd.first_pc_variance:.1%}")
        print(f"    Effective dim : layer {args.primary_layer} = {sp.effective_dimensionality:.2f}  "
              f"vs  layer {args.downstream_best_layer} = {sd.effective_dimensionality:.2f}")
        print(f"    PA components : layer {args.primary_layer} = {sp.n_components_parallel}  "
              f"vs  layer {args.downstream_best_layer} = {sd.n_components_parallel}")
        print()
        print(f"  Layer {args.primary_layer} interpretation:")
        print(f"    {sp.interpretation}")
        print()
        print(f"  Layer {args.downstream_best_layer} interpretation:")
        print(f"    {sd.interpretation}")
        print()

    print("  ── Saved files ──────────────────────────────────────────────────────\n")
    for key, path in saved.items():
        print(f"    {path.relative_to(_ROOT)}")

    print("\n  ══════ Stage 4A complete ═══════════════════════════════════════════")
    print("\n  Next (optional):")
    print("    python scripts/plot_structure_analysis.py")
    print("  Next stage:")
    print("    Stage 4B — paraphrase generation and framing reliability")


if __name__ == "__main__":
    main()
