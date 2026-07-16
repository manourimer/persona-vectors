"""
Modal GPU app: extract last-prompt-token activations from ETHICS scenarios.

This app handles Stage 3 activation extraction.  For each curated ETHICS item,
it:

  1. Formats the scenario as a prompt (no response generation).
  2. Runs a Gemma-3-12B forward pass with output_hidden_states=True.
  3. Extracts the residual-stream activation at the LAST PROMPT TOKEN.
  4. Returns the activation bytes to the local entrypoint for saving.

TOKEN SCOPE: last_prompt_token — the activation at the final token of the
presented scenario prompt, before any generation begins.  This is the
monitoring / probing scope, distinct from Stage 2B which used
mean-response-token activations for vector CONSTRUCTION.

ETHICS items are NOT used for vector construction.  The vectors are already
fixed from Stage 2B.  This app applies those vectors as a held-out test.

Usage (from project root):
    # 10-item smoke test
    modal run modal_apps/extract_ethics_prompt_activations.py --limit 10

    # Full extraction at target + comparison layers
    modal run modal_apps/extract_ethics_prompt_activations.py

    # Single layer
    modal run modal_apps/extract_ethics_prompt_activations.py \\
        --candidate-layers 32

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
_GPU = "A100"

# ---------------------------------------------------------------------------
# Modal import — optional
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

    app = modal.App("ethics-prompt-activation-extraction", image=_image)

    @app.cls(
        gpu=_GPU,
        volumes={"/models": _model_volume},
        secrets=[modal.Secret.from_name("huggingface")],
        timeout=7200,
        max_containers=10,
    )
    class EthicsActivationExtractor:
        """Loads Gemma once and extracts last-prompt-token activations."""

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
            """Extract last-prompt-token activations for a batch of ETHICS items.

            Args:
                jobs: List of {item_id, prompt_text, source_split, primary_trait}.
                candidate_layers: Transformer layer indices (0-indexed).

            Returns:
                List of {item_id, layer, source_split, primary_trait,
                         activation_bytes, hidden_dim} dicts.
            """
            import torch
            import numpy as np

            records: list[dict] = []

            for job in jobs:
                item_id = job["item_id"]
                prompt_text = job["prompt_text"]
                source_split = job.get("source_split", "")
                primary_trait = job.get("primary_trait", "")

                inputs = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                    )

                # outputs.hidden_states: tuple of (n_layers+1,) tensors (1, seq_len, dim)
                # Last token index = -1 (the "Answer:" token position)
                for layer in candidate_layers:
                    hs = outputs.hidden_states[layer]   # (1, seq_len, dim)
                    act = hs[0, -1, :].cpu().float().numpy()  # last prompt token

                    buf = io.BytesIO()
                    np.save(buf, act.astype(np.float32))
                    act_bytes = buf.getvalue()

                    records.append(
                        {
                            "item_id": item_id,
                            "layer": layer,
                            "source_split": source_split,
                            "primary_trait": primary_trait,
                            "activation_bytes": act_bytes,
                            "hidden_dim": int(act.shape[0]),
                        }
                    )

            return records

    @app.local_entrypoint()
    def main(
        item_bank_path: str = "data/processed/ethics_curated_mvp.parquet",
        candidate_layers: str = "32,40,47",
        batch_size: int = 8,
        model_name: str = DEFAULT_MODEL,
        out_dir: str = "outputs/ethics_projection",
        limit: int = 0,
    ) -> None:
        """Orchestrate ETHICS activation extraction: load items → call Modal → save locally.

        Activations are returned as bytes from the remote function and saved as
        .npy files on the local machine for use by compute_ethics_projections.py.

        Token scope: last_prompt_token — the activation at the final token of the
        presented scenario prompt, not averaged over any generated response.
        """
        sys.path.insert(0, str(_ROOT))

        import numpy as np
        import pandas as pd

        from src.projection.ethics_projection import (
            build_projection_jobs,
            load_item_bank,
            validate_item_bank,
        )

        layers = [int(l) for l in candidate_layers.split(",")]
        out_path = _ROOT / out_dir
        out_path.mkdir(parents=True, exist_ok=True)
        act_dir = out_path / "activations"
        act_dir.mkdir(parents=True, exist_ok=True)

        item_df = load_item_bank(_ROOT / item_bank_path)
        validate_item_bank(item_df)
        if limit:
            item_df = item_df.head(limit)

        jobs_df = build_projection_jobs(item_df, layers=[layers[0]])  # unique items only
        unique_items = (
            jobs_df[["item_id", "prompt_text", "source_split", "primary_trait"]]
            .drop_duplicates("item_id")
        )
        jobs = unique_items.to_dict(orient="records")

        print(f"\n  ETHICS items to extract: {len(jobs)}")
        print(f"  Candidate layers: {layers}")
        print(f"  Token scope: last_prompt_token")
        print(f"  Model: {model_name}")

        extractor = EthicsActivationExtractor(model_name=model_name)
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
            print(f"  Batch {batch_num}/{n_batches} complete ({len(batch)} items)")
            for rec in raw_records:
                item_id = rec["item_id"]
                layer = rec["layer"]
                arr = np.load(io.BytesIO(rec["activation_bytes"]))

                apath = act_dir / f"{item_id}_layer{layer}.npy"
                np.save(apath, arr)

                all_records.append(
                    {
                        "item_id": item_id,
                        "layer": layer,
                        "source_split": rec["source_split"],
                        "primary_trait": rec["primary_trait"],
                        "activation_path": str(apath),
                        "hidden_dim": rec["hidden_dim"],
                        "token_position": "last_prompt_token",
                        "model_name": model_name,
                    }
                )

        meta_df = pd.DataFrame(all_records)
        meta_pq = out_path / "ethics_activation_metadata.parquet"
        meta_csv = out_path / "ethics_activation_metadata.csv"
        meta_df.to_parquet(meta_pq, index=False)
        meta_df.to_csv(meta_csv, index=False)

        print(f"\n  Activation metadata saved: {meta_pq}")
        print(f"  Total activation records: {len(meta_df)}")
        print(f"  .npy files saved to: {act_dir}")
        print(f"\n  Next: python scripts/compute_ethics_projections.py")
