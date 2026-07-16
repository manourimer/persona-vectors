"""
Modal GPU app: extract last-prompt-token activations from reliability variants.

Stage 4C activation extraction.  For each accepted reliability variant from
Stage 4B, this app:

  1. Reads prompt_text from the pre-built projection jobs parquet.
  2. Runs a Gemma-3-12B forward pass with output_hidden_states=True.
  3. Extracts the residual-stream activation at the LAST PROMPT TOKEN.
  4. Returns the activation bytes to the local entrypoint for saving.

TOKEN SCOPE: last_prompt_token — identical to Stage 3 (ethics projection).
Variants are NOT used for vector construction.  The vectors are already fixed
from Stage 2B.

Usage (from project root):
    # 20-variant smoke test
    modal run modal_apps/extract_reliability_variant_activations.py --limit 20

    # Full extraction (all variants, all layers)
    modal run modal_apps/extract_reliability_variant_activations.py

    # Resume an interrupted run
    modal run modal_apps/extract_reliability_variant_activations.py --resume

Requires:
    modal token new
    modal secret create huggingface HF_TOKEN=hf_...
"""

# No "from __future__ import annotations" — breaks modal.parameter() in Python 3.13

import io
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "google/gemma-3-12b-it"
DEFAULT_CANDIDATE_LAYERS = [32, 40, 47]
DEFAULT_JOBS_PATH = "outputs/reliability_projection/reliability_projection_jobs.parquet"
DEFAULT_OUT_DIR = "outputs/reliability_projection"
_GPU = "A100"

# ---------------------------------------------------------------------------
# Modal import — optional so src/ modules can import this file in tests
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
            "torch>=2.2",
            "transformers>=4.40",
            "accelerate>=0.30",
            "huggingface-hub>=0.20",
            "pyyaml>=6.0",
            "pandas>=2.1",
            "pyarrow>=14.0",
            "numpy>=1.26",
        )
        .env({"HF_HUB_CACHE": "/models"})
        .add_local_dir(str(_ROOT / "src"), remote_path="/root/src")
        .add_local_dir(str(_ROOT / "configs"), remote_path="/root/configs")
    )

    _model_volume = modal.Volume.from_name(
        "gemma-response-generation-models", create_if_missing=True
    )

    app = modal.App("reliability-variant-activation-extraction", image=_image)

    @app.cls(
        gpu=_GPU,
        volumes={"/models": _model_volume},
        secrets=[modal.Secret.from_name("huggingface")],
        timeout=7200,
        max_containers=10,
    )
    class ReliabilityVariantActivationExtractor:
        """Loads Gemma once and extracts last-prompt-token activations for variants."""

        model_name: str = modal.parameter(default=DEFAULT_MODEL)
        dtype: str = modal.parameter(default="bfloat16")

        @modal.enter()
        def load_model(self) -> None:
            import sys as _sys
            _sys.path.insert(0, "/root")

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()

        @modal.method()
        def extract_batch(
            self,
            jobs: list[dict],
            candidate_layers: list[int],
        ) -> list[dict]:
            """Extract last-prompt-token activations for a batch of variants.

            Args:
                jobs: List of {variant_id, item_id, prompt_text, variant_type,
                              paraphrase_id, framing, source_split, primary_trait,
                              scenario_text_variant}.
                candidate_layers: Transformer layer indices (0-indexed).

            Returns:
                List of {variant_id, item_id, layer, variant_type, paraphrase_id,
                         framing, source_split, primary_trait, scenario_text_variant,
                         activation_bytes, hidden_dim, n_tokens} dicts.
            """
            import torch
            import numpy as np

            records: list[dict] = []

            for job in jobs:
                variant_id = job["variant_id"]
                item_id = job["item_id"]
                prompt_text = job["prompt_text"]

                inputs = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.model.device)

                n_tokens = int(inputs["input_ids"].shape[1])

                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                    )

                # outputs.hidden_states: tuple of (n_layers+1,) tensors (1, seq_len, dim)
                for layer in candidate_layers:
                    hs = outputs.hidden_states[layer]   # (1, seq_len, dim)
                    act = hs[0, -1, :].cpu().float().numpy()  # last prompt token

                    buf = io.BytesIO()
                    import numpy as _np
                    _np.save(buf, act.astype(_np.float32))
                    act_bytes = buf.getvalue()

                    records.append(
                        {
                            "variant_id": variant_id,
                            "item_id": item_id,
                            "layer": layer,
                            "variant_type": job.get("variant_type", ""),
                            "paraphrase_id": job.get("paraphrase_id", ""),
                            "framing": job.get("framing", ""),
                            "source_split": job.get("source_split", ""),
                            "primary_trait": job.get("primary_trait", ""),
                            "scenario_text_variant": job.get("scenario_text_variant", ""),
                            "activation_bytes": act_bytes,
                            "hidden_dim": int(act.shape[0]),
                            "n_tokens": n_tokens,
                        }
                    )

            return records

    @app.local_entrypoint()
    def main(
        jobs_path: str = DEFAULT_JOBS_PATH,
        candidate_layers: str = "32,40,47",
        batch_size: int = 8,
        model_name: str = DEFAULT_MODEL,
        out_dir: str = DEFAULT_OUT_DIR,
        limit: int = 0,
        resume: bool = False,
    ) -> None:
        """Orchestrate reliability variant activation extraction.

        Reads projection jobs parquet → dispatches to Modal → saves .npy files
        and activation metadata locally for use by
        scripts/compute_reliability_variant_projections.py.

        Token scope: last_prompt_token.
        """
        sys.path.insert(0, str(_ROOT))

        import numpy as np
        import pandas as pd

        layers = [int(l) for l in candidate_layers.split(",")]
        out_path = _ROOT / out_dir
        out_path.mkdir(parents=True, exist_ok=True)
        act_dir = out_path / "activations"
        act_dir.mkdir(parents=True, exist_ok=True)

        jobs_full_path = _ROOT / jobs_path
        if not jobs_full_path.exists():
            print(
                f"ERROR: Jobs parquet not found: {jobs_full_path}\n"
                "Run: python scripts/build_reliability_projection_jobs.py"
            )
            sys.exit(1)

        jobs_df = pd.read_parquet(jobs_full_path)

        # Deduplicate to unique variants (one row per variant, not per layer)
        unique_variants = (
            jobs_df.drop_duplicates("variant_id")
            [[
                "variant_id", "item_id", "prompt_text", "variant_type",
                "paraphrase_id", "framing", "source_split", "primary_trait",
                "scenario_text_variant",
            ]]
        )

        if limit:
            unique_variants = unique_variants.head(limit)

        # Resume: skip variants whose activations already exist for all layers
        if resume:
            def _all_exist(vid: str) -> bool:
                return all(
                    (act_dir / f"{vid}_layer{l}.npy").exists()
                    for l in layers
                )
            mask = ~unique_variants["variant_id"].apply(_all_exist)
            skipped = (~mask).sum()
            unique_variants = unique_variants[mask]
            if skipped:
                print(f"  Resuming: skipped {skipped} already-extracted variants.")

        jobs = unique_variants.to_dict(orient="records")

        print(f"\n  Reliability variants to extract: {len(jobs)}")
        print(f"  Candidate layers: {layers}")
        print(f"  Token scope: last_prompt_token")
        print(f"  Model: {model_name}")

        extractor = ReliabilityVariantActivationExtractor(model_name=model_name)
        batches = [jobs[i: i + batch_size] for i in range(0, len(jobs), batch_size)]
        n_batches = len(batches)
        print(f"  Dispatching {n_batches} batches across up to 10 containers ...")

        all_records: list[dict] = []

        for batch_num, (batch, raw_records) in enumerate(
            zip(
                batches,
                extractor.extract_batch.map(
                    batches,
                    kwargs={"candidate_layers": layers},
                ),
            ),
            start=1,
        ):
            print(f"  Batch {batch_num}/{n_batches} complete ({len(batch)} variants)")
            for rec in raw_records:
                variant_id = rec["variant_id"]
                layer = rec["layer"]
                arr = np.load(io.BytesIO(rec["activation_bytes"]))

                apath = act_dir / f"{variant_id}_layer{layer}.npy"
                np.save(apath, arr)

                all_records.append(
                    {
                        "item_id": rec["item_id"],
                        "variant_id": variant_id,
                        "variant_type": rec["variant_type"],
                        "paraphrase_id": rec["paraphrase_id"],
                        "framing": rec["framing"],
                        "source_split": rec["source_split"],
                        "primary_trait": rec["primary_trait"],
                        "scenario_text_variant": rec["scenario_text_variant"],
                        "layer": layer,
                        "activation_path": str(apath),
                        "n_tokens": rec["n_tokens"],
                        "token_position": "last_prompt_token",
                        "model_name": model_name,
                        "hidden_dim": rec["hidden_dim"],
                    }
                )

        meta_df = pd.DataFrame(all_records)
        meta_pq = out_path / "reliability_activation_metadata.parquet"
        meta_csv = out_path / "reliability_activation_metadata.csv"
        meta_df.to_parquet(meta_pq, index=False)
        meta_df.to_csv(meta_csv, index=False)

        print(f"\n  Activation metadata saved: {meta_pq}")
        print(f"  Total activation records: {len(meta_df)}")
        print(f"  .npy files saved to: {act_dir}")
        print(f"\n  Next: python scripts/compute_reliability_variant_projections.py --preprocessing both --layers 32 40 47")
