"""
Modal GPU app: generate responses from Gemma-3-12B under contrastive system prompts.

This app handles Stage 2B response generation.  It accepts a list of generation
jobs (each with a system_prompt and a question), runs Gemma-3-12B on Modal GPU,
and returns the generated text for each job.

Usage (from project root):
    python scripts/generate_vector_responses.py
    python scripts/generate_vector_responses.py --limit 8   # smoke test
    python scripts/generate_vector_responses.py --split validation

Direct Modal invocation:
    modal run modal_apps/generate_gemma_responses.py --limit 8

One-time setup:
    pip install modal
    modal token new

Model weights (~24 GB for Gemma-3-12B in bfloat16) are downloaded to a Modal
Volume on first run and cached for subsequent runs.

NOTE: Only generates responses for trait vector artifacts.
      ETHICS items are NOT used here.

Implementation note
-------------------
Gemma-3-12B requires HuggingFace authentication because the model is gated.
Set the HF_TOKEN secret in Modal before running:
    modal secret create huggingface HF_TOKEN=hf_...

Or use a non-gated instruction model (e.g., Qwen/Qwen2.5-14B-Instruct) by
overriding --model-name.  The interface is model-agnostic.
"""

# No "from __future__ import annotations" — breaks modal.parameter() in Python 3.13

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "google/gemma-3-12b-it"
_GPU = "A100"
_MAX_NEW_TOKENS = 256

# ---------------------------------------------------------------------------
# Modal import — optional; all module-level code above runs without it
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
    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.2",
            "transformers>=4.40",
            "accelerate>=0.30",
            "huggingface-hub>=0.20",
            "pyyaml>=6.0",
            "pandas>=2.1",
        )
        .env({"HF_HUB_CACHE": "/models"})
        .add_local_dir(str(_ROOT / "src"), remote_path="/root/src")
        .add_local_dir(str(_ROOT / "configs"), remote_path="/root/configs")
    )

    _model_volume = modal.Volume.from_name(
        "gemma-response-generation-models", create_if_missing=True
    )

    app = modal.App("gemma-response-generation", image=_image)

    @app.cls(
        gpu=_GPU,
        volumes={"/models": _model_volume},
        secrets=[modal.Secret.from_name("huggingface")],
        timeout=7200,
        max_containers=10,   # up to 10 parallel GPU containers
    )
    class GemmaGenerator:
        """Loads Gemma once per container and exposes a batched generation method."""

        model_name: str = modal.parameter(default=DEFAULT_MODEL)
        dtype: str = modal.parameter(default="bfloat16")

        @modal.enter()
        def load_model(self) -> None:
            import sys as _sys
            _sys.path.insert(0, "/root")

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16

            logger.info("Loading tokenizer: %s", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Loading model: %s (dtype=%s)", self.model_name, self.dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            logger.info("Model ready.")

        @modal.method()
        def generate_batch(
            self,
            jobs: list[dict],
            max_new_tokens: int = _MAX_NEW_TOKENS,
            temperature: float = 0.7,
            top_p: float = 0.95,
            seed: int = 42,
        ) -> list[str]:
            """Generate one response per job.

            Args:
                jobs: List of {system_prompt_text, question_text, response_id}.
                max_new_tokens: Maximum tokens to generate per response.
                temperature: Sampling temperature.
                top_p: Nucleus sampling probability.
                seed: Random seed for reproducibility.

            Returns:
                List of generated response strings, one per job.
            """
            import torch

            torch.manual_seed(seed)

            results: list[str] = []
            for job in jobs:
                messages = [
                    {"role": "system", "content": job["system_prompt_text"]},
                    {"role": "user",   "content": job["question_text"]},
                ]
                # Use chat template if available
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt = (
                        f"<|system|>{job['system_prompt_text']}</s>"
                        f"<|user|>{job['question_text']}</s>"
                        "<|assistant|>"
                    )

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # Decode only the generated portion
                prompt_len = inputs["input_ids"].shape[1]
                generated = self.tokenizer.decode(
                    output_ids[0, prompt_len:],
                    skip_special_tokens=True,
                )
                results.append(generated.strip())

            return results

    @app.local_entrypoint()
    def main(
        split: str = "extraction",
        limit: int = 0,
        batch_size: int = 4,
        model_name: str = DEFAULT_MODEL,
        out_dir: str = "outputs/vector_construction",
        resume: bool = True,
        checkpoint_every: int = 10,
        artifacts_path: str = "configs/trait_vector_artifacts.yaml",
    ) -> None:
        """Generate responses locally → delegate batches to Modal GPU → save.

        Saves a checkpoint every `checkpoint_every` batches so progress is not
        lost if the run is interrupted.  On re-run with resume=True (default),
        already-generated response_ids are skipped automatically.
        """
        sys.path.insert(0, str(_ROOT))

        import pandas as pd

        from src.vectors.artifact_bank import load_artifact_bank
        from src.vectors.generate_responses import (
            GeneratedResponse,
            build_generation_jobs,
            load_responses,
            save_responses,
        )

        out_path = _ROOT / out_dir
        out_path.mkdir(parents=True, exist_ok=True)
        parquet_path = out_path / f"generated_responses_{split}.parquet"

        artifact_path = _ROOT / artifacts_path
        from src.vectors.artifact_bank import load_artifact_bank_flexible  # noqa: PLC0415
        bank = (
            load_artifact_bank_flexible(artifact_path)
            if artifacts_path != "configs/trait_vector_artifacts.yaml"
            else load_artifact_bank(artifact_path)
        )
        all_jobs = build_generation_jobs(bank, split=split, limit=limit)

        # Resume: load already-completed response_ids and skip those jobs.
        existing: list[GeneratedResponse] = []
        done_ids: set[str] = set()
        if resume and parquet_path.exists():
            existing = load_responses(parquet_path)
            done_ids = {r.response_id for r in existing}
            print(f"  Resume: {len(done_ids)} responses already on disk — skipping.")

        jobs = [j for j in all_jobs if j["response_id"] not in done_ids]
        print(f"  Jobs to generate: {len(jobs)} ({split} split, {len(done_ids)} skipped)")

        if not jobs:
            print("  Nothing to do — all responses already generated.")
            return

        generator = GemmaGenerator(model_name=model_name)
        all_responses: list[GeneratedResponse] = list(existing)

        batches = [jobs[i: i + batch_size] for i in range(0, len(jobs), batch_size)]
        n_batches = len(batches)
        print(f"  Dispatching {n_batches} batches across up to 10 containers ...")

        for batch_num, (batch, texts) in enumerate(
            zip(batches, generator.generate_batch.map(batches)), start=1
        ):
            print(f"  Batch {batch_num}/{n_batches} complete ({len(batch)} responses)")
            for job, text in zip(batch, texts):
                all_responses.append(
                    GeneratedResponse(
                        response_id=job["response_id"],
                        trait=job["trait"],
                        pole=job["pole"],
                        split=job["split"],
                        system_prompt_id=job["system_prompt_id"],
                        question_id=job["question_id"],
                        system_prompt_text=job["system_prompt_text"],
                        question_text=job["question_text"],
                        response_text=text,
                        model_name=model_name,
                        generation_params={"temperature": 0.7, "top_p": 0.95},
                    )
                )

            # Checkpoint every N batches so progress survives interruptions.
            if batch_num % checkpoint_every == 0:
                save_responses(all_responses, out_path, stem=f"generated_responses_{split}")
                print(f"  [checkpoint] {len(all_responses)} responses saved.")

        parquet_path, csv_path = save_responses(
            all_responses, out_path, stem=f"generated_responses_{split}"
        )
        print(f"  Saved {len(all_responses)} responses.")
        print(f"    Parquet : {parquet_path}")
        print(f"    CSV     : {csv_path}")
