"""
Modal/vLLM judge app for Stage 2B response scoring.

Scores each generated response on the 0–100 trait rubric from
configs/trait_vector_artifacts.yaml.  Uses Qwen2.5-7B-Instruct served via
vLLM on Modal GPU — no Anthropic API key required.

Scores are used only for filtering before persona-vector construction:
  positive pole: keep if trait_score >= positive_keep_threshold (default 70)
  negative pole: keep if trait_score <= negative_keep_threshold (default 30)

Invalid/unparseable judge outputs receive trait_score=-1 and
keep_for_vector_extraction=False.

Usage (from project root):
    # Smoke test — 8 responses
    modal run modal_apps/score_vector_responses_vllm.py \\
        --split extraction --limit 8

    # Score extraction split
    modal run modal_apps/score_vector_responses_vllm.py --split extraction

    # Score validation split
    modal run modal_apps/score_vector_responses_vllm.py --split validation

    # Non-strict mode (parse failures become unkept instead of errors)
    modal run modal_apps/score_vector_responses_vllm.py \\
        --split extraction --no-strict

One-time setup:
    pip install modal
    modal token new
    # Qwen2.5-7B-Instruct is ungated — no HuggingFace login required.

GPU note:
    Default: A10G (24 GB VRAM) — sufficient for Qwen2.5-7B at batch_size<=8.
    Approximate cost: ~$0.001 per scored response on A10G.
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
_GPU = "A10G"

# JSON schema for vLLM guided decoding.
# Constrains output to exactly the fields we need, eliminating most parse
# failures without retries.  response_id is omitted from the schema because
# guided decoding of arbitrary strings can be unpredictable; we track the
# mapping by batch order instead.
SCORING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "trait_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "score_rationale": {
            "type": "string",
            "maxLength": 300,
        },
    },
    "required": ["trait_score", "score_rationale"],
    "additionalProperties": False,
}

_ROOT = Path(__file__).resolve().parent.parent


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


# ---------------------------------------------------------------------------
# Modal infrastructure — only defined when modal is available
# ---------------------------------------------------------------------------

_MODAL_AVAILABLE = False
try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    modal = None  # type: ignore[assignment]

if _MODAL_AVAILABLE:
    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "vllm>=0.6",
            "huggingface-hub>=0.20",
            "pyyaml>=6.0",
            "pydantic>=2.0",
            "pandas>=2.1",
        )
        .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
        .add_local_dir(str(_ROOT / "src"), remote_path="/root/src")
        .add_local_dir(str(_ROOT / "configs"), remote_path="/root/configs")
    )

    # Reuse the same model volume as the annotation pipeline so weights are
    # already cached if Stage 1d has been run.
    _model_volume = modal.Volume.from_name(
        "ethics-annotation-models", create_if_missing=True
    )

    app = modal.App("vector-response-scoring", image=_image)

    # ------------------------------------------------------------------
    # Remote class — ScoringModel runs on Modal GPU
    # ------------------------------------------------------------------

    @app.cls(
        gpu=_GPU,
        volumes={"/models": _model_volume},
        timeout=3600,
        max_containers=10,   # up to 10 parallel GPU containers
    )
    class ScoringModel:
        """Loads vLLM once and exposes a batched scoring method."""

        model_name: str = modal.parameter(default=DEFAULT_MODEL)

        @modal.enter()
        def load_model(self) -> None:
            import sys as _sys
            _sys.path.insert(0, "/root")

            from vllm import LLM  # noqa: PLC0415

            self.llm = LLM(
                model=self.model_name,
                download_dir="/models",
                trust_remote_code=True,
                max_model_len=4096,
                gpu_memory_utilization=0.90,
                enforce_eager=True,
            )
            logger.info("Scoring model loaded: %s", self.model_name)

        @modal.method()
        def score_batch(
            self,
            user_prompts: list[str],
            system_prompt: str,
        ) -> list[str]:
            """Score a batch of responses.

            Args:
                user_prompts: One formatted judge prompt per response.
                system_prompt: Shared judge system prompt.

            Returns:
                Raw JSON strings (one per response).  Returns '{}' on failure
                so the caller can fall back gracefully in non-strict mode.
            """
            from vllm import SamplingParams  # noqa: PLC0415

            try:
                from vllm import GuidedDecodingParams  # noqa: PLC0415
                guided = GuidedDecodingParams(json=SCORING_SCHEMA)
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=200,
                    guided_decoding=guided,
                )
            except (ImportError, Exception):
                # vLLM version does not support GuidedDecodingParams — fall back
                # to plain sampling; parse_judge_response handles stray prose.
                sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

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
        split: str = "extraction",
        responses_path: str = "",
        artifacts_path: str = "configs/trait_vector_artifacts.yaml",
        out_dir: str = "outputs/vector_construction",
        limit: int = 0,
        batch_size: int = 8,
        model_name: str = DEFAULT_MODEL,
        positive_threshold: float = 70.0,
        negative_threshold: float = 30.0,
        strict: bool = True,
    ) -> None:
        """Orchestrate scoring: read parquet locally → call Modal → write output.

        Args:
            split:              "extraction" or "validation".
            responses_path:     Override input parquet (default: auto-detect).
            artifacts_path:     Path to trait_vector_artifacts.yaml.
            out_dir:            Directory for scored output files.
            limit:              Score only this many responses (0 = all).
            batch_size:         Responses per Modal inference call.
            model_name:         HuggingFace model ID for vLLM.
            positive_threshold: Keep positive-pole if score >= this.
            negative_threshold: Keep negative-pole if score <= this.
            strict:             False → parse failures become unkept (score=-1).
        """
        import pandas as pd

        sys.path.insert(0, str(_ROOT))

        from src.vectors.artifact_bank import load_artifact_bank, load_artifact_bank_flexible
        from src.vectors.generate_responses import load_responses
        from src.vectors.score_responses import (
            build_judge_prompt,
            filter_for_extraction,
            parse_judge_response,
            save_scored,
        )
        from src.vectors.vector_data import ScoredResponse

        out_path = _resolve(_ROOT, out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if responses_path:
            resp_file = _resolve(_ROOT, responses_path)
        else:
            resp_file = out_path / f"generated_responses_{split}.parquet"

        if not resp_file.exists():
            print(f"  ERROR: Responses file not found: {resp_file}")
            print(f"         Run generate_vector_responses.py --split {split} first.")
            sys.exit(1)

        _ap = str(artifacts_path)
        bank = (
            load_artifact_bank_flexible(_resolve(_ROOT, _ap))
            if _ap != "configs/trait_vector_artifacts.yaml"
            else load_artifact_bank(_resolve(_ROOT, _ap))
        )

        # Build rubric index: trait → (scoring_instructions, scale_dict)
        rubric_index: dict[str, tuple[str, dict]] = {}
        for _, row in bank.rubrics_df.iterrows():
            trait = str(row["trait"])
            rubric_index[trait] = (
                str(row["scoring_instructions"]),
                {
                    0: str(row.get("scale_0", "")),
                    50: str(row.get("scale_50", "")),
                    100: str(row.get("scale_100", "")),
                },
            )

        responses = load_responses(resp_file)
        if limit:
            responses = responses[:limit]

        print(f"\n  Scoring {len(responses)} {split}-split responses ...")
        print(f"  Model          : {model_name}")
        print(f"  Batch size     : {batch_size}")
        print(f"  Strict JSON    : {strict}")
        print(f"  Pos threshold  : >= {positive_threshold}")
        print(f"  Neg threshold  : <= {negative_threshold}")
        print()

        # Build all judge prompts upfront
        system_prompt_str: str | None = None
        user_prompts: list[str] = []
        for resp in responses:
            if resp.trait not in rubric_index:
                logger.warning("No rubric found for trait %r — skipping.", resp.trait)
                user_prompts.append("")
                continue
            rubric_instr, scale = rubric_index[resp.trait]
            sys_str, usr_str = build_judge_prompt(resp, rubric_instr, scale)
            if system_prompt_str is None:
                system_prompt_str = sys_str
            user_prompts.append(usr_str)

        if system_prompt_str is None:
            print("  ERROR: No valid rubrics found for any response trait.")
            sys.exit(1)

        model = ScoringModel(model_name=model_name)

        # Split into batches; .map() dispatches all in parallel across containers.
        prompt_batches = [
            user_prompts[i: i + batch_size]
            for i in range(0, len(user_prompts), batch_size)
        ]
        response_batches = [
            responses[i: i + batch_size]
            for i in range(0, len(responses), batch_size)
        ]
        n_batches = len(prompt_batches)
        print(f"  Dispatching {n_batches} batches across up to 10 containers ...")

        scored: list[ScoredResponse] = []
        errors = 0
        t0 = time.monotonic()

        # Pass system_prompt as a kwarg; .map() iterates over the first positional arg.
        for batch_num, (batch_responses, raw_outputs) in enumerate(
            zip(
                response_batches,
                model.score_batch.map(
                    prompt_batches,
                    kwargs={"system_prompt": system_prompt_str},
                ),
            ),
            start=1,
        ):
            print(f"  Batch {batch_num}/{n_batches} complete ({len(batch_responses)} responses)")
            for resp, raw in zip(batch_responses, raw_outputs):
                try:
                    score, rationale = parse_judge_response(
                        raw, resp.response_id, strict=strict
                    )
                except ValueError as exc:
                    logger.warning("[%s] %s", resp.response_id, exc)
                    score, rationale = -1.0, f"parse_error: {exc}"
                    errors += 1

                scored.append(
                    ScoredResponse.from_generated(
                        gen=resp,
                        trait_score=score,
                        score_rationale=rationale,
                        positive_keep_threshold=positive_threshold,
                        negative_keep_threshold=negative_threshold,
                    )
                )

        elapsed = time.monotonic() - t0

        retained = filter_for_extraction(scored)
        parquet_path, csv_path = save_scored(
            scored, out_path, stem=f"scored_responses_{split}"
        )

        print(f"\n  Done in {elapsed:.1f}s")
        print(f"  Scored       : {len(scored)}")
        print(f"  Parse errors : {errors}")
        print(f"  Retained     : {len(retained)}/{len(scored)}")
        print(f"  Parquet      : {parquet_path}")
        print(f"  CSV          : {csv_path}")

        if errors and strict:
            print(
                f"\n  WARNING: {errors} parse error(s) in strict mode — "
                "responses marked unkept (score=-1).  Inspect CSV for details."
            )

        _print_score_breakdown(scored, split)

        failures = [s for s in scored if s.trait_score < 0]
        if failures:
            print(
                f"\n  {len(failures)} response(s) with score=-1 (judge parse failure)."
            )
            if not strict:
                print("  These are marked keep=False and will not affect vectors.")


def _print_score_breakdown(scored: list, split: str) -> None:
    """Print per-trait retention breakdown."""
    from collections import Counter
    kept_by_trait: Counter = Counter()
    total_by_trait: Counter = Counter()
    for s in scored:
        total_by_trait[s.trait] += 1
        if s.keep_for_vector_extraction:
            kept_by_trait[s.trait] += 1

    print(f"\n  keep_for_vector_extraction=True by trait ({split}):")
    for trait in sorted(total_by_trait):
        kept = kept_by_trait[trait]
        total = total_by_trait[trait]
        print(f"    {trait:<16} {kept:>3}/{total:>3}")
