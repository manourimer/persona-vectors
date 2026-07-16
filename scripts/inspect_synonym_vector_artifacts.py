"""
Inspect synonym vector artifacts config and check for existing synonym vectors.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls.synonym_vectors import load_synonym_config

CONFIG_PATH = "configs/synonym_vector_artifacts.yaml"
VECTOR_DIR = "outputs/controls/synonym_vectors/"
REQUIRED_FIELDS = ["parent_trait", "construct_name", "abbreviation"]


def main():
    print("[inspect_synonym_artifacts] Loading synonym config...")
    try:
        config = load_synonym_config(CONFIG_PATH)
    except Exception as e:
        print(f"ERROR loading config: {e}")
        return

    print(f"\nFound {len(config)} synonym traits:\n")
    print(f"{'synonym_id':<20} {'parent_trait':<15} {'construct_name':<20} {'fields_ok':<10}")
    print("-" * 65)

    all_ok = True
    for synonym_id, info in config.items():
        missing = [f for f in REQUIRED_FIELDS if not info.get(f)]
        fields_ok = "OK" if not missing else f"MISSING: {missing}"
        if missing:
            all_ok = False
        print(f"{synonym_id:<20} {info.get('parent_trait', '?'):<15} {info.get('construct_name', '?'):<20} {fields_ok:<10}")

    print()
    if all_ok:
        print("All required fields present.")
    else:
        print("WARNING: Some required fields missing.")

    # Check for existing .npy files
    vec_dir = Path(VECTOR_DIR)
    print(f"\nChecking for existing synonym vector files in {vec_dir}:")
    for synonym_id in config:
        found = list(vec_dir.glob(f"{synonym_id}_layer*.npy")) if vec_dir.exists() else []
        status = f"{len(found)} files found: {[f.name for f in found]}" if found else "NOT FOUND"
        print(f"  {synonym_id}: {status}")

    if not any(list(vec_dir.glob("*.npy")) if vec_dir.exists() else []):
        print("\nNo synonym vectors found. To generate them:")
        print("  1. Run Stage 2B pipeline with configs/synonym_vector_artifacts.yaml")
        print("  2. Save resulting .npy files to outputs/controls/synonym_vectors/")
        print("  3. Then run: python scripts/run_synonym_vector_controls.py --mvp-only")


if __name__ == "__main__":
    main()
