"""
Modal/vLLM paraphrase generation for Stage 4B reliability-variant bank.

Generates N neutral paraphrases per ETHICS item using an open-source
instruction model running on Modal GPU infrastructure.

Paraphrases are NOT new moral items — they are alternate presentations of
the same scenario.  Annotations (primary_trait, source_split) are inherited
from the original item.

Usage (from project root):
    # Smoke test — 10 items (~2 min)
    python scripts/generate_reliability_variants.py --limit 10

    # Full run — 204 items × 3 paraphrases
    python scripts/generate_reliability_variants.py

    # Resume after interruption
    python scripts/generate_reliability_variants.py --resume

One-time setup:
    pip install modal
    modal token new
    # Qwen2.5-7B-Instruct is public — no HuggingFace secret required.
    # Weights are downloaded to a Modal Volume on first run (~15 GB) and cached.

Approximate cost: ~$0.001–0.002 per item at 7B scale on A10G.

Output:
    data/processed/reliability_variants/ethics_reliability_variants_raw.parquet
    data/processed/reliability_variants/ethics_reliability_variants_raw.csv
"""

import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_GPU = "A10G"
_ROOT = Path(__file__).resolve().parent.parent


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


# ---------------------------------------------------------------------------
# Modal import — optional; all module-level code runs without it
# ---------------------------------------------------------------------------

try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False
    modal = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

if _MODAL_AVAILABLE:
    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "vllm>=0.6",
            "huggingface-hub>=0.20",
            "pyyaml>=6.0",
            "pydantic>=2.0",
            "pandas>=2.1",
            "json-repair>=0.25",
        )
        .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
        .add_local_dir(str(_ROOT / "src"), remote_path="/root/src")
        .add_local_dir(str(_ROOT / "configs"), remote_path="/root/configs")
    )

    _model_volume = modal.Volume.from_name(
        "ethics-annotation-models", create_if_missing=True
    )

    app = modal.App("ethics-reliability-variants", image=_image)

    @app.cls(
        gpu=_GPU,
        volumes={"/models": _model_volume},
        timeout=3600,
        max_containers=1,
    )
    class ParaphraseModel:
        """Loads vLLM once; exposes batched paraphrase generation."""

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
            logger.info("Model loaded: %s", self.model_name)

        @modal.method()
        def generate_batch(
            self,
            jobs: list[dict],
            n_paraphrases: int = 3,
        ) -> list[dict]:
            """
            Generate paraphrases for a batch of items.

            Args:
                jobs: list of dicts with keys: item_id, system_prompt, user_prompt
                n_paraphrases: expected number of paraphrases per item

            Returns:
                list of dicts with keys: item_id, raw_response, error
            """
            import sys as _sys
            _sys.path.insert(0, "/root")
            from vllm import SamplingParams  # noqa: PLC0415

            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=600,
            )

            messages_batch = [
                [
                    {"role": "system", "content": j["system_prompt"]},
                    {"role": "user", "content": j["user_prompt"]},
                ]
                for j in jobs
            ]

            outputs = self.llm.chat(
                messages=messages_batch,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            results: list[dict] = []
            for job, out in zip(jobs, outputs):
                try:
                    raw = out.outputs[0].text.strip()
                    results.append({"item_id": job["item_id"], "raw_response": raw, "error": ""})
                except Exception as exc:
                    results.append({"item_id": job["item_id"], "raw_response": "", "error": str(exc)})
            return results

    # ------------------------------------------------------------------
    # Local entrypoint
    # ------------------------------------------------------------------

    @app.local_entrypoint()
    def main(  # noqa: C901
        input: str = "data/processed/ethics_curated_mvp.parquet",
        output_dir: str = "data/processed/reliability_variants",
        limit: int = 0,
        n_paraphrases: int = 3,
        batch_size: int = 8,
        resume: bool = False,
        model_name: str = DEFAULT_MODEL,
        strict: bool = True,
        framing: str = "neutral",
    ) -> None:
        """Orchestrate paraphrase generation: read locally → call Modal → write parquet/csv."""
        import pandas as pd

        sys.path.insert(0, str(_ROOT))

        from src.reliability.variant_generation import (
            VARIANT_COLUMNS,
            build_all_prompts,
            build_original_variants,
            load_item_bank,
            load_variants,
            make_failed_parse_rows,
            make_paraphrase_variants,
            parse_paraphrase_response,
            save_variants,
        )
        from src.reliability.variant_validation import (
            check_semantic_equivalence,
            flag_intra_item_duplicates,
        )

        item_df = load_item_bank(_resolve(_ROOT, input))
        if limit:
            item_df = item_df.head(limit).copy()

        out_dir = _resolve(_ROOT, output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "ethics_reliability_variants_raw.parquet"

        print(f"\n  ── Stage 4B: Reliability Variant Generation ───────────────────")
        print(f"  Items         : {len(item_df)}")
        print(f"  Paraphrases   : {n_paraphrases} per item")
        print(f"  Framing       : {framing}")
        print(f"  Model         : {model_name}")
        print(f"  Output dir    : {out_dir}")

        # Determine which items need generation (resume logic).
        # An item is "done" only if it has at least one successfully parsed
        # paraphrase — items with only failed_parse rows are retried.
        if resume and raw_path.exists():
            existing = load_variants(raw_path)
            para = existing[existing["variant_type"] == "paraphrase"]
            done_ids = set(
                para[para["semantic_equivalence_status"] != "failed_parse"]["item_id"].unique()
            )
            todo_df = item_df[~item_df["item_id"].isin(done_ids)].copy()
            print(f"  Resume: {len(done_ids)} items done, {len(todo_df)} remaining")
        else:
            existing = None
            done_ids = set()
            todo_df = item_df.copy()

        if todo_df.empty:
            print("  Nothing to generate.")
            return

        jobs = build_all_prompts(todo_df, n_paraphrases=n_paraphrases, framing=framing)
        model = ParaphraseModel(model_name=model_name)

        all_rows: list[dict] = []
        errors = 0
        parse_errors = 0
        t0 = time.monotonic()
        n_batches = (len(jobs) + batch_size - 1) // batch_size

        for batch_num, batch_start in enumerate(range(0, len(jobs), batch_size), 1):
            batch_jobs = jobs[batch_start: batch_start + batch_size]
            print(f"  Batch {batch_num}/{n_batches}  (items {batch_start+1}–{min(batch_start+batch_size, len(jobs))})")

            raw_results: list[dict] = model.generate_batch.remote(batch_jobs, n_paraphrases)

            for job, result in zip(batch_jobs, raw_results):
                item_id = job["item_id"]
                item_row = todo_df[todo_df["item_id"] == item_id].iloc[0]

                if result["error"]:
                    all_rows.extend(make_failed_parse_rows(
                        item_row, n_paraphrases, framing, model_name, result["error"]
                    ))
                    errors += 1
                    continue

                try:
                    paraphrases = parse_paraphrase_response(
                        result["raw_response"], item_id, n_paraphrases, strict=strict
                    )
                except ValueError as exc:
                    all_rows.extend(make_failed_parse_rows(
                        item_row, n_paraphrases, framing, model_name, str(exc)
                    ))
                    parse_errors += 1
                    continue

                rows = make_paraphrase_variants(
                    item_row=item_row,
                    paraphrases=paraphrases,
                    framing=framing,
                    generation_model_name=model_name,
                    semantic_status_fn=lambda orig, var: check_semantic_equivalence(orig, var),
                )
                all_rows.extend(rows)

        new_df = pd.DataFrame(all_rows, columns=VARIANT_COLUMNS)
        new_df = flag_intra_item_duplicates(new_df)

        # Build originals for all items (not just todo), then merge with paraphrases
        orig_df = build_original_variants(item_df, framing=framing)

        if existing is not None:
            # Drop existing rows for retried items (replaces failed_parse placeholders),
            # then append newly generated rows.
            retried_ids = set(todo_df["item_id"])
            existing_paras = existing[
                (existing["variant_type"] == "paraphrase")
                & (~existing["item_id"].isin(retried_ids))
            ]
            combined_paras = pd.concat([existing_paras, new_df[new_df["variant_type"] == "paraphrase"]], ignore_index=True)
            final_df = pd.concat([orig_df, combined_paras], ignore_index=True)
        else:
            final_df = pd.concat([orig_df, new_df[new_df["variant_type"] == "paraphrase"]], ignore_index=True)

        final_df = final_df.reset_index(drop=True)
        csv_path, parquet_path = save_variants(final_df, out_dir, "ethics_reliability_variants_raw")

        elapsed = time.monotonic() - t0
        n_flagged = int((final_df["semantic_equivalence_status"].isin(
            {"flagged_length", "flagged_duplicate", "flagged_possible_meaning_shift"}
        )).sum())

        print(f"\n  ── Summary ─────────────────────────────────────────────────────")
        print(f"  Total rows       : {len(final_df)}")
        print(f"  Originals        : {int((final_df['variant_type']=='original').sum())}")
        print(f"  Paraphrases      : {int((final_df['variant_type']=='paraphrase').sum())}")
        print(f"  Parse errors     : {parse_errors}")
        print(f"  Remote errors    : {errors}")
        print(f"  Flagged variants : {n_flagged}")
        print(f"  Elapsed          : {elapsed:.1f}s")
        print(f"  Saved (parquet)  : {parquet_path}")
        print(f"  Saved (csv)      : {csv_path}")
        print(f"\n  Next:")
        print(f"    python scripts/validate_reliability_variants.py")
        print(f"    python scripts/review_reliability_variants.py")
