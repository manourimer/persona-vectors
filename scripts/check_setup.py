#!/usr/bin/env python3
"""
Verify project scaffold and experiment configuration without GPU access.

Run from the project root:
    python scripts/check_setup.py
    python scripts/check_setup.py --config configs/mvp_experiment.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config_and_traits
from src.traits import trait_summary_lines

REQUIRED_DIRS = [
    "configs",
    "src",
    "src/data",
    "src/utils",
    "scripts",
    "data/raw",
    "data/processed",
    "outputs",
    "notebooks",
    "tests",
]


def check_dirs(root: Path) -> list[str]:
    return [d for d in REQUIRED_DIRS if not (root / d).is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Persona-vectors project setup check")
    parser.add_argument("--config", default="configs/mvp_experiment.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    print("=" * 62)
    print("  Persona Vectors — MVP Setup Check")
    print("=" * 62)

    # 1. Directories
    print("\n[1] Required directories")
    missing = check_dirs(root)
    if missing:
        for d in missing:
            print(f"  MISSING  {d}/")
        print(f"\n  WARNING: {len(missing)} director(ies) missing.")
    else:
        print(f"  OK  all {len(REQUIRED_DIRS)} required directories present")

    # 2. Load config + traits
    print(f"\n[2] Experiment config  ({args.config})")
    try:
        cfg, traits = load_config_and_traits(root / args.config)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    print(f"  Experiment : {cfg.experiment.name}")

    # 3. Model + layer plan (kept separate from construct traits)
    print("\n[3] Model & layer configuration")
    print(f"  Model                      : {cfg.model.name}")
    print(f"  dtype                      : {cfg.model.dtype}")
    print(f"  Layer indexing             : {cfg.model.layer_indexing}")
    print(f"  Target layer (placeholder) : {cfg.model.target_layer}")
    candidates = cfg.model.candidate_layers_for_validation
    print(f"  Candidate layers for validation : {candidates}")
    if cfg.model.layer_selection_note:
        note = cfg.model.layer_selection_note.strip().replace("\n", " ")
        print(f"  NOTE: {note}")

    # 4. Construct traits  — the project's measurement constructs
    print("\n[4] Construct traits  (project measurement constructs)")
    print(f"  Defined in : {cfg.traits.path}")
    print(f"  Constructs : {', '.join(traits.names)}")
    for line in trait_summary_lines(traits):
        print(line)

    # 5. Source dataset — ETHICS organisational categories (≠ construct traits)
    print("\n[5] Source dataset  (ETHICS benchmark — not construct categories)")
    print(f"  Dataset    : {cfg.dataset.name}")
    print(f"  Split      : {cfg.dataset.split}")
    print(f"  N items    : {cfg.dataset.n_items}")
    print(f"  ETHICS source splits (dataset labels only):")
    for split in cfg.dataset.ethics_splits:
        print(f"    - {split}")
    print()
    print("  NOTE: ETHICS splits describe how items were collected,")
    print("        NOT which moral construct they measure. Any split")
    print("        may contain items relevant to any of the four traits.")

    # 6. Paraphrase / framing
    print("\n[6] Paraphrase & framing")
    print(f"  Paraphrases per item : {cfg.paraphrase.n_per_item}")
    print(f"  Framing conditions   : {', '.join(cfg.paraphrase.framing_conditions)}")
    print(f"  Random seed          : {cfg.random_seed}")

    # 7. Output directories (auto-create)
    print("\n[7] Output directories")
    out_dirs = [
        cfg.output.base_dir,
        cfg.output.activations_dir,
        cfg.output.projections_dir,
        cfg.output.results_dir,
        cfg.output.figures_dir,
    ]
    for d in out_dirs:
        p = root / d
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            print(f"  CREATED  {d}/")
        else:
            print(f"  OK       {d}/")

    print("\n" + "=" * 62)
    print("  Setup check complete — no GPU required.")
    print("=" * 62)


if __name__ == "__main__":
    main()
