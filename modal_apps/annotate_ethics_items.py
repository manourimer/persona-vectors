"""
Modal/vLLM annotation app for the ETHICS item bank.

Runs an open-source instruction model through vLLM on Modal GPU infrastructure
to produce first-pass annotation suggestions.  These annotations are
SUGGESTIONS only — human review is required before exporting the curated MVP
item bank.

Intended usage (from project root):
    # Smoke test — 10 items
    python scripts/auto_annotate_items_modal.py --limit 10
    # Equivalent direct Modal invocation:
    modal run modal_apps/annotate_ethics_items.py --limit 10

    # Full annotation run
    python scripts/auto_annotate_items_modal.py
    modal run modal_apps/annotate_ethics_items.py

    # Resume after interruption
    python scripts/auto_annotate_items_modal.py --resume

One-time setup:
    pip install modal
    modal token new      # authenticate your Modal account
    # Qwen2.5-7B-Instruct is public; no HuggingFace login required.

Model weights are downloaded to a Modal Volume on first run (~15 GB for the
7B model) and cached for subsequent runs.

GPU note:
    Default GPU: A10G (24 GB VRAM) — sufficient for Qwen2.5-7B at batch_size≤32.
    For 14B models change _GPU = "A10G" to _GPU = "A100" near the top of this file.
    Approximate cost: ~$0.001 per annotation item at 7B scale on A10G.
"""

import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — importable without Modal or GPU
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# GPU used for inference.  A10G (24 GB VRAM) fits 7B comfortably.
# Switch to "A100" for 14B+ models.
_GPU = "A10G"

# JSON schema for vLLM guided decoding.
# Constrains model output to exactly the fields we need, eliminating most
# parse failures without requiring retries.
ANNOTATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "primary_trait": {
            "type": "string",
            "enum": [
                "honesty", "harmlessness", "fairness", "compassion",
                "not_applicable", "unclear",
            ],
        },
        "secondary_traits": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["honesty", "harmlessness", "fairness", "compassion"],
            },
            "maxItems": 3,
        },
        "annotation_confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "keep_for_mvp": {"type": "boolean"},
        "annotation_notes": {"type": "string", "maxLength": 300},
    },
    "required": [
        "primary_trait", "secondary_traits",
        "annotation_confidence", "keep_for_mvp", "annotation_notes",
    ],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Helpers — importable without Modal or GPU
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def _print_header(
    input_path: Path,
    output_path: Path,
    model_name: str,
    limit: int,
    resume: bool,
    strict: bool,
) -> None:
    print("=" * 66)
    print("  ETHICS Item Bank — Modal/vLLM First-Pass Annotation")
    print("  ⚠  Auto-labels are SUGGESTIONS — human review required.")
    print("=" * 66)
    print(f"  Input   : {input_path}")
    print(f"  Output  : {output_path}")
    print(f"  Model   : {model_name}")
    print(f"  Limit   : {limit or 'all rows'}")
    print(f"  Resume  : {resume}")
    print(f"  Strict  : {strict}")


def _print_summary(
    df: "pd.DataFrame",  # type: ignore[name-defined]
    output_path: Path,
    annotated: int,
    errors: int,
    elapsed: float,
) -> None:
    import pandas as pd  # noqa: F401 — only needed for type-check above

    print()
    print(f"  Annotated : {annotated} rows  ({errors} fell back to unclear/low)")
    print(f"  Elapsed   : {elapsed:.1f} s")
    print(f"  Saved     : {output_path}")

    keep_mask = df["keep_for_mvp"].str.strip().str.lower().isin({"true", "yes", "1"})
    print(f"\n  keep_for_mvp=true : {keep_mask.sum()} / {len(df)}")

    print("\n  primary_trait distribution:")
    for trait, count in df["primary_trait"].value_counts().items():
        bar = "█" * min(int(count), 40)
        print(f"    {trait:<18} {count:>4}  {bar}")

    print()
    print("  Next steps:")
    print("  1. Review autolabels:")
    print(f"       python scripts/review_autolabels.py --input {output_path.name}")
    print("  2. Correct errors in the CSV (prioritise low-confidence rows).")
    print("  3. Validate corrected sheet:")
    print(f"       python scripts/validate_annotations.py --input {output_path.name}")
    print("  4. Export curated item bank:")
    print(f"       python scripts/export_curated_item_bank.py --input {output_path.name}")
    print("=" * 66)


# ---------------------------------------------------------------------------
# Modal import — optional.  All module-level code above runs without it.
# ---------------------------------------------------------------------------

try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False
    modal = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Modal infrastructure — only defined when modal is available
# ---------------------------------------------------------------------------

if _MODAL_AVAILABLE:
    # The Modal image installs vLLM and bundles the project's src/ library
    # and configs/ so the remote function can build annotation prompts.
    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "vllm>=0.6",           # inference engine with guided-decoding support
            "huggingface-hub>=0.20",
            "pyyaml>=6.0",
            "pydantic>=2.0",
            "pandas>=2.1",
        )
        # Disable flashinfer's JIT sampler — it requires nvcc which is not in
        # debian_slim.  vLLM falls back to its built-in torch-based sampling.
        .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
        # Bundle local source into the container image.
        # add_local_dir() is the Modal 1.x API replacing the deprecated Mount.
        .add_local_dir(str(_ROOT / "src"), remote_path="/root/src")
        .add_local_dir(str(_ROOT / "configs"), remote_path="/root/configs")
    )

    # Volume persists downloaded model weights across runs (~15 GB first run,
    # then cached).  Created automatically on first use.
    _model_volume = modal.Volume.from_name(
        "ethics-annotation-models", create_if_missing=True
    )

    app = modal.App("ethics-annotation", image=_image)

    # ------------------------------------------------------------------
    # Remote class — AnnotationModel runs on Modal GPU
    # ------------------------------------------------------------------

    @app.cls(
        gpu=_GPU,
        volumes={"/models": _model_volume},
        timeout=3600,       # 1-hour cap; increase for very large batches
        max_containers=1,   # one GPU at a time is sufficient for batch jobs
    )
    class AnnotationModel:
        """Loads the vLLM engine once and exposes a batched annotation method."""

        # modal.parameter() is how Modal 1.x passes constructor args to @app.cls.
        # Instantiate with AnnotationModel(model_name=...) — keyword only.
        model_name: str = modal.parameter(default=DEFAULT_MODEL)

        @modal.enter()
        def load_model(self) -> None:
            """Called once when the container starts.  Downloads weights to
            /models (persisted in the Modal Volume) then initialises vLLM."""
            import sys as _sys
            _sys.path.insert(0, "/root")  # make src/ importable inside Modal

            from vllm import LLM  # noqa: PLC0415

            self.llm = LLM(
                model=self.model_name,
                download_dir="/models",
                trust_remote_code=True,
                max_model_len=4096,
                gpu_memory_utilization=0.90,
                # enforce_eager=True disables CUDA graph compilation.
                # This avoids flashinfer JIT (which needs nvcc) and is fine
                # for annotation workloads where latency is not critical.
                enforce_eager=True,
            )
            logger.info("Model loaded: %s", self.model_name)

        @modal.method()
        def annotate_batch(
            self,
            user_prompts: list[str],
            system_prompt: str,
        ) -> list[str]:
            """Run inference on a batch of ETHICS items.

            Args:
                user_prompts: One formatted user-turn string per item.
                system_prompt: Shared system prompt (trait rubric + rules).

            Returns:
                Raw JSON strings, one per item.  Returns '{}' on output
                extraction failure so the caller can fall back gracefully.
            """
            from vllm import SamplingParams  # noqa: PLC0415

            # temperature=0.0 → greedy / deterministic.
            # No guided decoding: the system prompt explicitly instructs JSON-only
            # output, and parse_llm_response() handles any stray prose via regex.
            sampling_params = SamplingParams(temperature=0.0, max_tokens=400)

            messages_batch = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": up},
                ]
                for up in user_prompts
            ]

            outputs = self.llm.chat(
                messages=messages_batch,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            results: list[str] = []
            for out in outputs:
                try:
                    results.append(out.outputs[0].text.strip())
                except Exception as exc:
                    logger.warning("Output extraction failed: %s", exc)
                    results.append("{}")
            return results

    # ------------------------------------------------------------------
    # Local entrypoint — runs on your laptop, orchestrates everything
    # ------------------------------------------------------------------

    @app.local_entrypoint()
    def main(  # noqa: C901
        input: str = "data/processed/ethics_annotation_sheet.csv",
        output: str = "data/processed/ethics_annotation_sheet_autolabeled.csv",
        limit: int = 0,
        batch_size: int = 32,
        resume: bool = False,
        model_name: str = DEFAULT_MODEL,
        strict: bool = True,
    ) -> None:
        """Orchestrate annotation: read CSV locally → call Modal → write CSV.

        Args:
            input:       Annotation sheet CSV path (relative to project root).
            output:      Output CSV path.
            limit:       Annotate only this many rows (0 = all).
            batch_size:  Items per Modal inference call.
            resume:      Skip rows already annotated (non-empty primary_trait).
            model_name:  HuggingFace model ID for vLLM.
            strict:      False = parse failures → unclear/low/false instead of warning.
        """
        import pandas as pd

        sys.path.insert(0, str(_ROOT))

        from src.config import load_mvp_config, load_trait_space
        from src.data.annotation import load_annotation_sheet
        from src.data.auto_annotation import (
            _fallback_annotation,
            build_annotation_prompt,
            build_system_prompt,
            parse_llm_response,
            validate_parsed_annotation,
        )

        cfg = load_mvp_config(_ROOT / "configs/mvp_experiment.yaml")
        trait_space = load_trait_space(_ROOT / cfg.traits.path)

        input_path = _resolve(_ROOT, input)
        output_path = _resolve(_ROOT, output)

        try:
            df = load_annotation_sheet(input_path)
        except FileNotFoundError:
            print(f"ERROR: annotation sheet not found: {input_path}")
            print("Run: python scripts/create_annotation_sheet.py")
            sys.exit(1)

        if limit:
            df = df.head(limit).copy()

        _print_header(input_path, output_path, model_name, limit, resume, strict)

        system_prompt = build_system_prompt(trait_space)

        if resume:
            todo_mask = df["primary_trait"].str.strip() == ""
            n_skip = int((~todo_mask).sum())
            print(f"  Resume: skipping {n_skip} already-annotated rows.")
        else:
            todo_mask = pd.Series([True] * len(df), index=df.index)

        todo_idx = df.index[todo_mask].tolist()
        total_todo = len(todo_idx)
        print(f"  Items to annotate: {total_todo}\n")

        if total_todo == 0:
            print("Nothing to annotate.  Use --resume=false to re-annotate.")
            return

        model = AnnotationModel(model_name=model_name)

        annotated = 0
        errors = 0
        t0 = time.monotonic()
        n_batches = (total_todo + batch_size - 1) // batch_size

        for batch_num, batch_start in enumerate(range(0, total_todo, batch_size), 1):
            batch_indices = todo_idx[batch_start: batch_start + batch_size]
            batch_df = df.loc[batch_indices]

            user_prompts = [
                build_annotation_prompt(row, trait_space)[1]
                for _, row in batch_df.iterrows()
            ]

            print(
                f"  Batch {batch_num}/{n_batches}  "
                f"(items {batch_start + 1}–{min(batch_start + batch_size, total_todo)})"
            )

            raw_outputs: list[str] = model.annotate_batch.remote(
                user_prompts, system_prompt
            )

            for idx, raw in zip(batch_indices, raw_outputs):
                item_id = df.at[idx, "item_id"]
                try:
                    parsed = parse_llm_response(raw)
                    validated = validate_parsed_annotation(parsed)
                except (ValueError, KeyError) as exc:
                    if strict:
                        logger.warning("[%s] validation error: %s", item_id, exc)
                    validated = _fallback_annotation(str(exc))
                    errors += 1

                for col, val in validated.items():
                    df.at[idx, col] = val
                annotated += 1

        elapsed = time.monotonic() - t0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        _print_summary(df, output_path, annotated, errors, elapsed)
