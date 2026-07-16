"""
Stage 2B — Step 2: Score generated responses with a trait judge.

Default judge: Qwen2.5-7B-Instruct via Modal/vLLM.
No Anthropic API key required.

Scores are used only for filtering:
  positive pole: keep if trait_score >= positive_keep_threshold (default 70)
  negative pole: keep if trait_score <= negative_keep_threshold (default 30)

Invalid judge outputs receive trait_score=-1 and keep_for_vector_extraction=False.

Usage:
    # Full run via Modal (default)
    python scripts/score_vector_responses.py --split extraction
    python scripts/score_vector_responses.py --split validation

    # Smoke test — 8 responses only
    python scripts/score_vector_responses.py --split extraction --limit 8

    # Non-strict: parse failures become unkept instead of warnings
    python scripts/score_vector_responses.py --split extraction --non-strict

    # Mock scoring for local tests (no Modal required)
    python scripts/score_vector_responses.py --split extraction --mock

Outputs:
    outputs/vector_construction/scored_responses_{split}.parquet
    outputs/vector_construction/scored_responses_{split}.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_DEFAULT_OUT_DIR = "outputs/vector_construction"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score trait vector responses with an open-source judge model."
    )
    parser.add_argument(
        "--split",
        choices=["extraction", "validation", "both"],
        default="extraction",
    )
    parser.add_argument(
        "--responses-path",
        default="",
        help="Override input parquet path (default: auto-detect from --split).",
    )
    parser.add_argument(
        "--out-dir",
        default=_DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--judge-method",
        choices=["modal_vllm", "mock"],
        default="modal_vllm",
        help="Scoring backend (default: modal_vllm).",
    )
    parser.add_argument(
        "--judge-model-name",
        default=_DEFAULT_MODEL,
        help="HuggingFace model ID for Modal/vLLM judge.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Responses per Modal inference call.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=70.0,
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=30.0,
    )
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Parse failures produce score=-1 and keep=False (default).",
    )
    strict_group.add_argument(
        "--non-strict",
        dest="strict",
        action="store_false",
        help="Parse failures silently become unkept (score=-1, keep=False).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Deterministic mock scores — no Modal or GPU required.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Score only this many responses per split (0 = all).",
    )
    parser.add_argument(
        "--artifacts-path",
        default="configs/trait_vector_artifacts.yaml",
        help="Path to artifact bank YAML for rubric loading (default: trait_vector_artifacts.yaml).",
    )
    args = parser.parse_args()

    splits = ["extraction", "validation"] if args.split == "both" else [args.split]

    if args.mock:
        _run_mock(splits, args)
    elif args.judge_method == "modal_vllm":
        _run_modal(splits, args)
    else:
        print(f"  ERROR: Unknown judge_method: {args.judge_method!r}")
        sys.exit(1)


def _run_mock(splits: list[str], args: argparse.Namespace) -> None:
    from src.vectors.generate_responses import load_responses
    from src.vectors.score_responses import (
        filter_for_extraction,
        mock_score,
        save_scored,
    )

    out_path = _ROOT / args.out_dir

    for split in splits:
        resp_path = (
            _ROOT / args.responses_path
            if args.responses_path
            else out_path / f"generated_responses_{split}.parquet"
        )
        if not resp_path.exists():
            print(f"  ERROR: Responses file not found: {resp_path}")
            print(f"         Run generate_vector_responses.py --split {split} first.")
            sys.exit(1)

        responses = load_responses(resp_path)
        if args.limit:
            responses = responses[: args.limit]

        print(f"\n  [mock] Scoring {len(responses)} {split}-split responses ...")
        scored = mock_score(
            responses,
            positive_keep_threshold=args.positive_threshold,
            negative_keep_threshold=args.negative_threshold,
        )
        retained = filter_for_extraction(scored)
        parquet_path, csv_path = save_scored(
            scored, out_path, stem=f"scored_responses_{split}"
        )
        print(f"  Retained: {len(retained)}/{len(scored)}")
        print(f"  Saved:    {parquet_path}")


def _run_modal(splits: list[str], args: argparse.Namespace) -> None:
    """Invoke the Modal/vLLM scoring app via subprocess for each split."""
    modal_app = _ROOT / "modal_apps" / "score_vector_responses_vllm.py"
    if not modal_app.exists():
        print(f"  ERROR: Modal app not found: {modal_app}")
        sys.exit(1)

    for split in splits:
        cmd = [
            "modal", "run", str(modal_app),
            f"--split={split}",
            f"--model-name={args.judge_model_name}",
            f"--batch-size={args.batch_size}",
            f"--positive-threshold={args.positive_threshold}",
            f"--negative-threshold={args.negative_threshold}",
            f"--out-dir={args.out_dir}",
        ]
        if args.responses_path:
            cmd.append(f"--responses-path={args.responses_path}")
        if args.limit:
            cmd.append(f"--limit={args.limit}")
        if not args.strict:
            cmd.append("--no-strict")
        if args.artifacts_path != "configs/trait_vector_artifacts.yaml":
            cmd.append(f"--artifacts-path={args.artifacts_path}")

        print(f"\n  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(_ROOT))
        if result.returncode != 0:
            print(f"  ERROR: Modal scoring failed for split={split}")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
