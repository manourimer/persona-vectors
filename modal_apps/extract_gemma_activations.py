"""
Modal GPU app: extract response-token activations from Gemma-3-12B.

This app handles the activation extraction step of Stage 2B.  For each
retained response (keep_for_vector_extraction=True), it:

  1. Tokenizes the full conversation (system prompt + question + response).
  2. Runs a Gemma forward pass with output_hidden_states=True.
  3. Identifies the response token positions in the sequence.
  4. Computes the mean hidden state over response positions at each candidate layer.
  5. Saves each (response, layer) activation as a .npy file.

TOKEN SCOPE: "response_tokens" — mean over generated response positions.
This is the correct scope for persona-VECTOR EXTRACTION.
Contrast with Stage 3+ (ETHICS monitoring), which uses "last_prompt_token".

NOTE: Only processes responses generated from trait vector artifacts.
      ETHICS items are NOT extracted here.

Usage:
    modal run modal_apps/extract_gemma_activations.py --scored-path outputs/...

Requires:
    pip install modal
    modal token new
    modal secret create huggingface HF_TOKEN=hf_...
"""

# No "from __future__ import annotations" — breaks modal.parameter() in Python 3.13

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
    app = modal.App("gemma-activation-extraction", image=_image)

    @app.cls(
        gpu=_GPU,
        volumes={
            "/models": _model_volume,
        },
        secrets=[modal.Secret.from_name("huggingface")],
        timeout=7200,
        max_containers=10,   # up to 10 parallel GPU containers
    )
    class GemmaActivationExtractor:
        """Loads Gemma once and extracts activations for batches of responses."""

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
                output_hidden_states=True,
            )
            self.model.eval()

        @modal.method()
        def extract_batch(
            self,
            jobs: list[dict],
            candidate_layers: list[int],
            pooling: str = "mean_response_token",
        ) -> list[dict]:
            """Extract activations for a batch of responses.

            Args:
                jobs: List of {response_id, system_prompt_text, question_text,
                               response_text, trait, pole, split}.
                candidate_layers: Transformer layer indices (0-indexed).
                pooling:          "mean_response_token" (only supported option).

            Returns:
                List of {response_id, layer, activation_path} dicts.
            """
            import numpy as np
            import torch

            records: list[dict] = []

            for job in jobs:
                response_id = job["response_id"]
                system_prompt = job["system_prompt_text"]
                question = job["question_text"]
                response_text = job["response_text"]
                trait = job["trait"]
                pole = job["pole"]
                split = job["split"]

                # Build full conversation string
                messages = [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": response_text},
                ]
                try:
                    full_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    full_text = (
                        f"<|system|>{system_prompt}</s>"
                        f"<|user|>{question}</s>"
                        f"<|assistant|>{response_text}"
                    )

                # Also tokenize just the prompt to find response start position
                prompt_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question},
                ]
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt_text = (
                        f"<|system|>{system_prompt}</s>"
                        f"<|user|>{question}</s>"
                        "<|assistant|>"
                    )

                prompt_ids = self.tokenizer(
                    prompt_text, return_tensors="pt", truncation=True, max_length=2048
                )["input_ids"]
                prompt_len = prompt_ids.shape[1]

                full_inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2560,
                ).to(self.model.device)

                total_len = full_inputs["input_ids"].shape[1]
                response_start = min(prompt_len, total_len - 1)

                with torch.no_grad():
                    outputs = self.model(
                        **full_inputs,
                        output_hidden_states=True,
                    )

                # outputs.hidden_states: tuple of (n_layers+1,) tensors, each (1, seq_len, dim)
                for layer in candidate_layers:
                    # layer 0 = embedding; layer i corresponds to hidden_states[i]
                    hs = outputs.hidden_states[layer]   # (1, seq_len, dim)
                    response_hs = hs[0, response_start:, :]  # (n_response_tokens, dim)
                    if response_hs.shape[0] == 0:
                        response_hs = hs[0, -1:, :]  # fallback: last token

                    if pooling == "mean_response_token":
                        act = response_hs.mean(dim=0).cpu().float().numpy()
                    else:
                        act = response_hs[-1].cpu().float().numpy()

                    # Return array bytes to the local entrypoint for saving.
                    # Saving to a Modal Volume would keep files remote and
                    # inaccessible to compute_persona_vectors.py on local disk.
                    import io
                    buf = io.BytesIO()
                    np.save(buf, act.astype(np.float32))
                    act_bytes = buf.getvalue()

                    records.append(
                        {
                            "response_id": response_id,
                            "trait": trait,
                            "pole": pole,
                            "split": split,
                            "layer": layer,
                            "activation_bytes": act_bytes,
                            "pooling_method": pooling,
                            "hidden_dim": int(act.shape[0]),
                        }
                    )

            return records

    @app.local_entrypoint()
    def main(
        scored_path: str = "outputs/vector_construction/scored_responses.parquet",
        candidate_layers: str = "16,24,28,32,40,47",
        batch_size: int = 4,
        model_name: str = DEFAULT_MODEL,
        out_dir: str = "outputs/vector_construction",
        split: str = "extraction",
    ) -> None:
        """Orchestrate activation extraction: load scored → call Modal → save locally.

        Activation arrays are returned as bytes from the remote function and saved
        as .npy files on the local machine so compute_persona_vectors.py can read them.
        """
        sys.path.insert(0, str(_ROOT))

        import io
        import numpy as np
        import pandas as pd
        from src.vectors.extract_activations import save_activation_metadata
        from src.vectors.vector_data import ActivationRecord

        layers = [int(l) for l in candidate_layers.split(",")]
        out_path = _ROOT / out_dir

        scored_file = out_path / f"scored_responses_{split}.parquet"
        if not scored_file.exists():
            print(f"  ERROR: {scored_file} not found.")
            print(f"         Run score_vector_responses.py --split {split} first.")
            sys.exit(1)

        scored_df = pd.read_parquet(scored_file)
        retained = scored_df[scored_df["keep_for_vector_extraction"] == True]

        jobs = retained.to_dict(orient="records")
        print(f"  Retained responses to extract: {len(jobs)} ({split} split)")
        print(f"  Candidate layers: {layers}")

        extractor = GemmaActivationExtractor(model_name=model_name)
        all_records: list[ActivationRecord] = []

        batches = [jobs[i: i + batch_size] for i in range(0, len(jobs), batch_size)]
        n_batches = len(batches)
        print(f"  Dispatching {n_batches} batches across up to 10 containers ...")

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
            print(f"  Batch {batch_num}/{n_batches} complete ({len(batch)} responses)")
            for rec in raw_records:
                # Save the returned bytes as a local .npy file.
                trait, pole = rec["trait"], rec["pole"]
                act_dir = out_path / "activations" / f"{trait}_{pole}"
                act_dir.mkdir(parents=True, exist_ok=True)
                act_path = act_dir / f"{rec['response_id']}_layer{rec['layer']}.npy"

                arr = np.load(io.BytesIO(rec["activation_bytes"]))
                np.save(act_path, arr)

                all_records.append(
                    ActivationRecord(
                        response_id=rec["response_id"],
                        trait=trait,
                        pole=pole,
                        split=split,
                        layer=rec["layer"],
                        activation_path=str(act_path),
                        pooling_method=rec["pooling_method"],
                        hidden_dim=rec["hidden_dim"],
                    )
                )

        meta_stem = f"activation_metadata_{split}"
        meta_path = save_activation_metadata(all_records, out_path, stem=meta_stem)
        print(f"  Activation metadata saved: {meta_path}")
        print(f"  Total activation records: {len(all_records)}")
        print(f"  .npy files saved to: {out_path / 'activations'}")
