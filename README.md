# Persona Vectors as a Psychometric Instrument for Moral Character in Language Models

## Research Question

Can persona-vector projections function as a reliable, valid psychometric measurement instrument for morally relevant dispositions (honesty, harmlessness, fairness, compassion) in large language models?

Concretely: if we extract residual-stream activations from a fixed layer of Gemma-3-12B while it processes moral scenarios, and project those activations onto persona vectors constructed from contrastive trait descriptions, do the resulting scalar scores:

1. align with human moral judgments on those items?
2. form a coherent, low-dimensional latent trait space (analogous to factor structure)?
3. generalize reliably across paraphrase and framing variation (analogous to test-retest and inter-rater reliability)?

**Headline finding:** the four trait projections collapse onto a single shared axis
(effective dimensionality ≈ 1.13) rather than four independent dimensions — a result that
replicates on an independent, confound-free item bank and survives a direct test of the
obvious "it's just generic RLHF alignment" explanation.

## Deliverables

- **Paper:** [`outputs/persona_vectors_paper.pdf`](outputs/persona_vectors_paper.pdf) — full write-up with methodology, all tables and figures
- **Slides:** [`docs/slides.html`](docs/slides.html) — 9-slide deck (open in a browser; arrow keys/space to advance)
- **Article:** [`docs/presentation_article.md`](docs/presentation_article.md) — ~1,300-word general-audience walkthrough

---

## Construct Space vs. Source-Dataset Categories

This is an important conceptual distinction in the project design.

### The project's measurement constructs (4 traits)

| Trait | Core concern |
|---|---|
| **Honesty** | Truth-telling, non-deception, epistemic integrity |
| **Harmlessness** | Avoiding and not facilitating physical/psychological harm |
| **Fairness** | Equal treatment, consistency, non-discrimination |
| **Compassion** | Empathic response to suffering and vulnerability |

These are what the project measures. They are defined in [`configs/traits.yaml`](configs/traits.yaml) with positive/negative descriptions, moral rationales, item-tagging guidance, and example keywords.

### ETHICS benchmark categories (source-dataset labels only)

The ETHICS benchmark (Hendrycks et al., 2021) organises items into five splits: `commonsense`, `deontology`, `justice`, `utilitarianism`, `virtue`. These describe **how items were collected and labelled by the dataset authors** — they are philosophical taxonomy labels for the dataset's construction, not the project's measurement constructs.

**An item from the `deontology` split may primarily engage `honesty`, `fairness`, or `harmlessness` depending on its content.** An item from `virtue` may primarily engage `compassion` or `harmlessness`. The mapping between ETHICS splits and this project's traits is an **empirical question**, not an assumption built into the design.

The experiment config keeps these entirely separate:

```yaml
traits:          # ← measurement constructs
  names: [honesty, harmlessness, fairness, compassion]

dataset:
  ethics_splits: # ← source-dataset labels only
    - commonsense
    - deontology
    - justice
    - utilitarianism
    - virtue
```

---

## MVP Scope

| Dimension | Choice |
|---|---|
| Model | `google/gemma-3-12b-it` |
| Benchmark | ETHICS (Hendrycks et al.) |
| Target layer | 28 (placeholder — see layer selection below) |
| Traits | honesty, harmlessness, fairness, compassion |
| Items | 300 sampled from ETHICS test split |
| Paraphrases | 3 per item (neutral / first-person / third-person) |

### Layer selection

`target_layer: 28` is a development placeholder, not a validated choice. The final layer must be selected by held-out persona-vector validation (e.g. contrastive accuracy on a held-out description set). Candidate layers spanning the model's depth will be evaluated in Stage 3:

```
candidate_layers_for_validation: [16, 24, 28, 32, 40, 47]
```

All indices are zero-indexed.

---

## Project Stages

| Stage | Description | Status |
|---|---|---|
| **0 – Scaffold** | Project structure, configs, trait definitions, loaders | ✅ Done |
| **1 – Data** | Download ETHICS, normalise schemas, sample item bank | ✅ Done |
| **1b – Annotation** | Annotation scaffold, sheet creation, validation tooling | ✅ Done |
| **1c – Curation** | Pilot workflow, curated item-bank export, commonsense diagnostic | ✅ Done |
| **1d – Modal/vLLM annotation** | Open-source model produces first-pass labels on Modal GPU | ✅ Done |
| **1e – Curated export** | Human review, correction, and curated item-bank export | ✅ Done |
| **2A – Artifact bank** | Contrastive system prompts, elicitation questions, rubrics | ✅ Done |
| **2A-review – Quality audit** | Confound checks: leakage, generic valence, cross-trait words, near-duplicates | ✅ Done |
| **2B – Vector construction** | Generate contrast responses, score/filter, extract activations, compute + validate vectors | ✅ Done |
| **3 – ETHICS projection** | Project ETHICS item activations onto trait vectors → raw + centered score matrices | ✅ Done |
| **4A – Structure analysis** | PCA / correlation / effective dimensionality of original ETHICS projections (RQ1) | ✅ Done |
| **4B – Variant generation** | Generate + validate paraphrase variant bank for reliability testing (RQ2) | ✅ Done |
| **4C – Variant projection** | Project all accepted variants using Stage 3 machinery | ✅ Done |
| **4D – Reliability analysis** | Intra-class correlation / generalizability across variants | ✅ Done |
| **5 – Validity** | Correlate PC1 with ETHICS ground-truth labels (RQ3); implemented inline in `scripts/generate_paper_pdf.py`, not a standalone pipeline stage | ✅ Done |

---

## Repository Layout

```
.
├── configs/
│   ├── mvp_experiment.yaml          # central experiment config
│   ├── traits.yaml                  # trait construct definitions
│   └── trait_vector_artifacts.yaml  # Stage 2A: system prompts, questions, rubrics
├── src/
│   ├── config.py             # Pydantic models + YAML loaders
│   ├── traits.py             # trait-space helpers
│   ├── data/                 # (Stage 1+) dataset loaders and annotation schema
│   ├── vectors/              # (Stage 2A+) artifact bank loader, validators, quality checks
│   ├── projection/           # (Stage 3) ETHICS projection: loading, computing, diagnostics
│   ├── analysis/             # (Stage 4A) structure analysis: PCA, correlation, parallel analysis
│   ├── reliability/          # (Stage 4B) variant generation, validation, export
│   └── utils/                # shared utilities
├── scripts/
│   ├── check_setup.py                       # CPU-only setup verification
│   ├── build_ethics_sample.py               # Stage 1: download, normalise, sample ETHICS
│   ├── inspect_ethics_sample.py             # Stage 1: inspect the item bank
│   ├── create_annotation_sheet.py           # Stage 1b: blank annotation CSV
│   ├── validate_annotations.py              # Stage 1b/1c: validate annotation CSV
│   ├── create_annotation_pilot.py           # Stage 1c: 50-item stratified pilot
│   ├── export_curated_item_bank.py          # Stage 1c: filter and export curated bank
│   ├── compare_commonsense_only.py          # Stage 1c: diagnostic comparison
│   ├── auto_annotate_items_modal.py         # Stage 1d: Modal/vLLM annotation (primary path)
│   ├── auto_annotate_items.py               # Stage 1d: Anthropic API fallback
│   ├── review_autolabels.py                 # Stage 1d/1e: review auto-label report
│   ├── inspect_vector_artifacts.py           # Stage 2A: inspect artifact bank
│   ├── export_vector_artifacts_for_review.py # Stage 2A: export to CSV/Markdown
│   ├── audit_vector_artifacts.py             # Stage 2A-review: confound / quality audit
│   ├── build_ethics_projection_jobs.py       # Stage 3: pre-flight check
│   ├── extract_ethics_activations.py         # Stage 3: last-prompt-token activation extraction
│   ├── compute_ethics_projections.py         # Stage 3: dot-product projection (raw + centered)
│   ├── diagnose_ethics_projections.py        # Stage 3: projection diagnostics
│   ├── compare_projection_layers.py          # Stage 3: layer comparison (32 vs 40 vs 47)
│   ├── run_ethics_projection_smoke_test.py   # Stage 3: full mock pipeline (no GPU)
│   ├── run_structure_analysis.py             # Stage 4A: PCA / correlation / parallel analysis
│   ├── plot_structure_analysis.py            # Stage 4A: figures (requires matplotlib)
│   ├── generate_reliability_variants.py      # Stage 4B: generate paraphrase variants
│   ├── validate_reliability_variants.py      # Stage 4B: validate variant bank schema + counts
│   ├── review_reliability_variants.py        # Stage 4B: print flagged variants + export review CSV
│   └── export_reliability_variant_bank.py    # Stage 4B: export accepted variants to final bank
├── modal_apps/
│   ├── annotate_ethics_items.py              # Modal App: Stage 1d vLLM annotation
│   ├── generate_gemma_responses.py           # Modal App: Stage 2B response generation
│   ├── score_vector_responses_vllm.py        # Modal App: Stage 2B response scoring
│   ├── extract_gemma_activations.py          # Modal App: Stage 2B activation extraction
│   ├── extract_ethics_prompt_activations.py  # Modal App: Stage 3 ETHICS activation extraction
│   └── generate_reliability_variants.py      # Modal App: Stage 4B paraphrase generation
├── data/
│   ├── raw/                  # HuggingFace cache (git-ignored)
│   └── processed/
│       ├── ethics_mvp_sample.parquet
│       └── ethics_mvp_sample.csv
├── outputs/                  # activations, projections, results, figures
├── notebooks/                # exploratory analysis
├── tests/
│   ├── test_config.py        # config, trait, and annotation schema tests
│   └── test_data_ethics.py   # normalisation, sampling, construct separation
└── pyproject.toml
```

---

## Quickstart

```bash
# Install (from project root)
pip install -e ".[dev]"

# Run setup check — no GPU required
python scripts/check_setup.py

# Run tests
pytest
```

---

## Stage 1 — ETHICS Item Bank

### Build the sample (downloads from HuggingFace, ~seconds)

```bash
python scripts/build_ethics_sample.py
```

Produces `data/processed/ethics_mvp_sample.parquet` and `.csv` with 300 items
stratified across the three included ETHICS source splits (100 per split, seed=42).

### Why utilitarianism and virtue are excluded from the MVP

**utilitarianism** — items are hedonic preference comparisons (e.g. "going on vacation vs. not") rather than situations that engage a moral disposition toward another agent. The split has no label column and its paired `(baseline / less_pleasant)` format does not map naturally onto honesty, harmlessness, fairness, or compassion. It can be re-added via `dataset.ethics_splits` in the config if you want to explore it.

**virtue** — items pair a situation with a single philosophical trait word via `[SEP]`. After deduplication to unique situations, all retained rows have `label=1`, leaving no negative-valence items. The trait words also overlap with philosophical virtue categories (humble, forgiving, dishonest…) that differ from the project's four construct labels. It can also be re-added if needed.

A smaller, conceptually clean item bank is preferable to forcing all available splits into the pipeline when some don't map onto the construct space.

### Inspect the sample

```bash
# 20 random items across all splits
python scripts/inspect_ethics_sample.py

# 30 items from justice only
python scripts/inspect_ethics_sample.py --n 30 --split justice
```

### Output schema

| Column | Description |
|---|---|
| `item_id` | Stable ID: `{split}_test_{row_index:06d}` |
| `source_dataset` | Always `"ETHICS"` |
| `source_split` | ETHICS organisational label (`commonsense` / `deontology` / `justice`) — **not a construct trait** |
| `scenario_text` | Main text for model probing |
| `label` | Original ETHICS label (semantics vary by split; see `label_semantics`) |
| `label_semantics` | Human-readable key explaining what `label` means for this split |
| `raw_fields` | JSON of original CSV columns |
| `primary_trait` | **Blank** — to be filled in Stage 1c annotation |
| `secondary_traits` | **Blank** — optional secondary construct traits |
| `annotation_confidence` | **Blank** — `high` / `medium` / `low` |
| `annotation_notes` | **Blank** — free-text annotation rationale |

### ETHICS schema notes (per included split)

| Split | Raw columns | `scenario_text` | `label` meaning |
|---|---|---|---|
| commonsense | `label`, `input`, `is_short`, `edited` | `input` | 1=wrong, 0=OK |
| deontology | `label`, `scenario`, `excuse` | `scenario [EXCUSE] excuse` | 1=excuse invalid, 0=excuse valid |
| justice | `label`, `scenario` | `scenario` | 1=action is just, 0=unjust |

---

## Stage 1b — Annotation Infrastructure

**Create the blank annotation sheet** (run once after `build_ethics_sample.py`):

```bash
python scripts/create_annotation_sheet.py
```

Writes `data/processed/ethics_annotation_sheet.csv` with blank annotation columns ready for the auto-labeling step.

### Annotation schema

Each row in the annotation sheet carries these fields, filled in by the model and corrected by you:

| Column | Valid values |
|---|---|
| `primary_trait` | `honesty` · `harmlessness` · `fairness` · `compassion` · `not_applicable` · `unclear` |
| `secondary_traits` | Comma-separated subset of the four construct traits only |
| `annotation_confidence` | `high` · `medium` · `low` |
| `keep_for_mvp` | `true` or `false` |
| `annotation_notes` | One short sentence explaining the label choice |

**Guardrails enforced at every layer:**
- ETHICS split names (`commonsense`, `deontology`, `justice`, `utilitarianism`, `virtue`) are never accepted as trait labels.
- `not_applicable` and `unclear` always force `keep_for_mvp = false`.
- `low` confidence always forces `keep_for_mvp = false`.
- `not_applicable` always forces `secondary_traits = []`.
- A smaller clean item bank is better than a larger noisy one — prefer `not_applicable` / `unclear` over forced assignments.

---

## Stage 1d — First-Pass Annotation with Open-Source Model on Modal

An open-source instruction model (Qwen2.5-7B-Instruct) runs on Modal GPU infrastructure to produce first-pass annotation suggestions for the 300-item bank.

**These labels are not ground truth.** They are a starting point. You must review and correct them before the item bank is used for anything downstream.

### One-time setup

```bash
pip install modal
modal token new          # authenticate your Modal account (once)
```

Qwen2.5-7B-Instruct is publicly available on HuggingFace — no login required.
Model weights (~15 GB) are downloaded to a Modal Volume on first run and cached for all subsequent runs.

### Step 1 — Smoke test (10 items)

```bash
python scripts/auto_annotate_items_modal.py --limit 10
```

This runs the full pipeline on 10 rows. Including Modal container cold-start and weight download (first run only), expect ~3–5 minutes. On subsequent runs, cold-start drops to ~60 seconds.

### Step 2 — Full annotation run

```bash
python scripts/auto_annotate_items_modal.py
```

Annotates all rows in `data/processed/ethics_annotation_sheet.csv` and writes to
`data/processed/ethics_annotation_sheet_autolabeled.csv`.

To resume after an interruption without re-annotating already-labelled rows:

```bash
python scripts/auto_annotate_items_modal.py --resume
```

Additional flags:

| Flag | Default | Description |
|---|---|---|
| `--limit N` | all | Annotate only first N rows |
| `--batch-size N` | 32 | Items per Modal inference call |
| `--model-name` | `Qwen/Qwen2.5-7B-Instruct` | Any vLLM-compatible HF model |
| `--resume` | off | Skip rows with an existing `primary_trait` |
| `--non-strict` | off | Convert parse failures to `unclear/low/false` instead of logging |

### Step 3 — Review the auto-labels

```bash
python scripts/review_autolabels.py \
    --input data/processed/ethics_annotation_sheet_autolabeled.csv
```

Prints a structured report. **Review in this order:**

| Priority | Section | Why |
|---|---|---|
| 1st | **[5] Low-confidence rows** | Model was most uncertain — highest error rate |
| 2nd | **[6] unclear / not_applicable** | Check whether abstention is justified or overly cautious |
| 3rd | **[7] keep=true + medium/low confidence** | Model kept them but was not fully sure |
| 4th | **[8] Trait coverage warnings** | Alert if any construct trait has fewer than 20 retained items |

Add `--show-text` to include (truncated) scenario text in each preview row.

### Step 4 — Correct errors

Open `ethics_annotation_sheet_autolabeled.csv` in any spreadsheet editor (Excel, Numbers, Google Sheets). Correct `primary_trait`, `secondary_traits`, `annotation_confidence`, `annotation_notes`, and `keep_for_mvp` as needed.

The rubric is in [`configs/traits.yaml`](configs/traits.yaml) (`item_tagging_guidance` and `example_keywords_and_patterns` for each trait).

### Step 5 — Validate corrected annotations

```bash
python scripts/validate_annotations.py \
    --input data/processed/ethics_annotation_sheet_autolabeled.csv
```

Fix any reported errors before exporting.

---

## Stage 1e — Export the Curated Item Bank

```bash
python scripts/export_curated_item_bank.py \
    --input data/processed/ethics_annotation_sheet_autolabeled.csv
```

Saves `data/processed/ethics_curated_mvp.{csv,parquet}` — only rows with
`keep_for_mvp = true` and a construct trait as `primary_trait`. Prints a
curation report with warnings for low per-trait coverage and split dominance.

**The exported item bank must be manually reviewed before beginning any downstream work.** Target ≥ 25 items per construct trait. If any trait falls below threshold, review the `not_applicable` and `unclear` rows to see whether any should be reclassified.

### Inspect trait balance

```bash
# Full cross-split trait distribution
python scripts/validate_annotations.py \
    --input data/processed/ethics_annotation_sheet_autolabeled.csv

# Optional: compare all-splits vs commonsense-only
python scripts/compare_commonsense_only.py
```

If trait coverage is poor across the three ETHICS splits, the commonsense split alone tends to produce cleaner moral-disposition scenarios. To switch to commonsense-only, change `dataset.ethics_splits` in [`configs/mvp_experiment.yaml`](configs/mvp_experiment.yaml) and re-run `build_ethics_sample.py`.

---

## Stage 2A — Contrast Prompt and Evaluation Artifact Bank

Stage 2A creates the artifacts needed to **construct** persona vectors. It is entirely separate from the ETHICS item bank.

### What the artifact bank contains

For each of the four construct traits (honesty, harmlessness, fairness, compassion):

| Artifact | Count | Purpose |
|---|---|---|
| Positive system prompts | 5 | Elicit the trait during forward passes |
| Negative system prompts | 5 | Suppress / oppose the trait during forward passes |
| Elicitation questions — extraction split | 20 | Used to generate activations for vector computation |
| Elicitation questions — validation split | 20 | Held out; used to check contrastive accuracy before ETHICS projection |
| Evaluation rubric | 1 | 0–100 scoring scheme for trait judges |

Total: 40 system prompts, 160 questions, 4 rubrics across the four traits.

### Methodology

Persona vectors are constructed from **forward-pass activations**, not from static embedding-layer projections of trait descriptions:

1. Run the target model under each positive / negative system prompt + elicitation question pair.
2. Collect **response-token activations** from the target layer.
3. Filter responses using a trait judge (0–100 score; keep responses above threshold).
4. Compute the persona vector: `mean(positive activations) − mean(negative activations)`.
5. Validate on the **held-out validation split** before any ETHICS projection.

This mirrors the method in the Persona Vectors paper (Choi et al.) and is distinct from computing cosine similarities between static description embeddings.

### ETHICS items are NOT used here

The ETHICS item bank is used in Stage 4 (projection / reliability testing). At Stage 2A, the only artifacts are contrastive prompts and elicitation questions for vector construction. These are defined in `configs/trait_vector_artifacts.yaml` and have no overlap with the ETHICS benchmark content.

### Inspect the artifact bank

```bash
python scripts/inspect_vector_artifacts.py
```

Prints counts by trait × pole, trait × split, rubric summaries, and any count warnings.

### Inspect and export artifacts

```bash
# Print counts, rubric summaries, and warnings
python scripts/inspect_vector_artifacts.py

# Export to CSV/Markdown under data/processed/vector_artifacts/
python scripts/export_vector_artifacts_for_review.py
```

Exports:

| File | Contents |
|---|---|
| `system_prompts_review.csv` | All 40 system prompts with trait, pole, id, text, notes |
| `elicitation_questions_review.csv` | All 160 questions with trait, split, id, text, notes |
| `rubrics_review.md` | Rubrics in Markdown with score scale and examples |

---

## Stage 2A-review — Artifact Quality Audit

Before running any forward passes, audit the artifact bank for confound risks and weak prompt design.

### Why this matters

The core risk in persona-vector construction is that the contrastive prompts measure the **wrong thing**:

| Failure mode | What the vector captures instead of the target trait |
|---|---|
| Positive pole = "generic helpfulness" | Overall AI assistant goodness, not the specific trait |
| Negative pole introduces cruelty (in a honesty prompt) | Lack of compassion, not lack of honesty |
| Negative pole introduces deception (in a harmlessness prompt) | Dishonesty, not willingness to enable harm |
| Questions name the construct explicitly | Model response is cued by the label, not a genuine disposition |
| Extraction/validation questions overlap | Validation does not provide an independent check |
| Near-duplicate prompts in one pole | Reduced contrastive diversity; limited generalisability |
| Generic moral valence dominates | Vector measures good-vs-evil, not the specific construct |

### Run the audit

```bash
python scripts/audit_vector_artifacts.py
```

The audit runs 10 checks over system prompts, elicitation questions, and rubrics:

| Check | What it detects |
|---|---|
| `text_too_short` | Empty or suspiciously brief artifact text |
| `trait_label_leakage` | Questions that name the construct being probed |
| `cross_trait_confound` | Prompts using ≥3 words native to a *different* trait |
| `generic_valence_dominance` | Prompts heavy on generic good/evil words, light on trait-specific language |
| `generic_helpfulness_collapse` | Positive prompts that read as generic assistant goodness |
| `negative_pole_extra_trait` | Negative prompts that introduce other moral failings |
| `near_duplicate` (prompts) | Near-identical system prompts within the same trait×pole |
| `near_duplicate` (questions) | Near-identical elicitation questions within the same trait |
| `extraction_validation_text_overlap` | Extraction and validation questions too similar in wording |
| `rubric_cross_trait_words` | Rubrics that heavily reference a different trait's vocabulary |

Findings are classified as `info` / `warning` / `high`. The script exits non-zero if any `high` findings are present.

### Audit outputs

```
data/processed/vector_artifacts/vector_artifact_audit.csv
data/processed/vector_artifacts/vector_artifact_audit.md
```

CSV columns: `trait`, `artifact_type`, `artifact_id`, `severity`, `issue_type`, `text`, `explanation`, `suggested_review_action`.

### Interpreting results

- **High** — must address before Stage 2B (near-duplicates, extraction/validation content overlap).
- **Warning** — review carefully; may indicate a confound that will degrade vector specificity.
- **Info** — low-risk observations; often incidental natural language usage worth checking.

The audit does not automatically reject any artifact. Findings are a starting point for human judgement.

---

## Stage 2B — Persona-Vector Construction and Validation

This stage builds difference-of-means persona vectors from contrastive system prompts and validates them on held-out elicitation questions. **It uses only `configs/trait_vector_artifacts.yaml` — ETHICS items are not touched here.**

**No Anthropic API key is required.** All inference (generation, scoring, activation extraction) runs via Modal/vLLM on open-source models.

### Required credentials

| Credential | Purpose | Notes |
|---|---|---|
| Modal account | GPU inference for all three Modal steps | `modal token new` |
| HuggingFace token with Gemma access | Response generation + activation extraction | `google/gemma-3-12b-it` is gated — request access at hf.co |
| HuggingFace token (optional) | Judge model | `Qwen/Qwen2.5-7B-Instruct` is ungated — no request needed |

### Methodology

1. **Generate responses** — Gemma-3-12B generates free-text responses to each elicitation question under each system prompt. Responses are generated under 5 positive + 5 negative system prompts per trait.

2. **Score and filter** — An open-source judge model (Qwen2.5-7B-Instruct via Modal/vLLM) scores each response for the target trait (0–100). High-scoring positive responses and low-scoring negative responses are retained. **These scores are used only for filtering — they are not treated as ground truth.** If scores look noisy, inspect `outputs/vector_construction/scored_responses_{split}.csv` and adjust thresholds or the judge prompt.

3. **Extract activations** — For each retained response, the model's residual-stream activations at candidate layers are extracted, **averaged over the generated response token positions** (not the last prompt token). This is the correct scope for vector *construction*.

4. **Compute vectors** — Difference-of-means: `vector = mean(positive_activations) − mean(negative_activations)`, optionally normalized to unit norm.

5. **Validate vectors** — Validation-split responses (held out from vector fitting) are projected onto each candidate vector. ROC-AUC ≥ 0.75 is required per trait. The layer with the highest mean AUC across all traits is selected.

> **If a trait vector fails validation, that is a finding, not just a bug.** It means the contrastive prompts may be confounded, the layer is wrong, or the responses don't clearly separate the trait. Investigate before proceeding.

### Token scope distinction

| Stage | Token scope | Purpose |
|---|---|---|
| 2B (this stage) | Response tokens (mean over generated positions) | Vector *construction* |
| 3+ (ETHICS projection) | Last prompt token | ETHICS item *monitoring* |

Do not use last-prompt-token activations for vector construction, and do not use response-token activations for ETHICS projection.

### Commands

```bash
# Smoke test — full pipeline in mock mode, no GPU or credentials required
python scripts/run_vector_construction_smoke_test.py

# Step 1 — Generate responses with Gemma on Modal
python scripts/generate_vector_responses.py --split extraction
python scripts/generate_vector_responses.py --split validation

# Step 2 — Score responses with open-source Modal/vLLM judge (no API key needed)
python scripts/score_vector_responses.py --split extraction
python scripts/score_vector_responses.py --split validation
# Smoke test: add --limit 8 to score only 8 responses
# Offline: add --mock for deterministic scores

# Step 3 — Extract activations with Gemma on Modal
python scripts/extract_vector_activations.py --split extraction
python scripts/extract_vector_activations.py --split validation

# Step 4 — Compute difference-of-means vectors (local, pure numpy)
python scripts/compute_persona_vectors.py

# Step 5 — Validate vectors and select best layer
python scripts/validate_persona_vectors.py
```

### Output files

```
outputs/vector_construction/
  generated_responses_{split}.parquet   # raw responses
  scored_responses_{split}.parquet      # with trait_score, keep flag
  activations/{trait}_{pole}/           # per-response .npy files
  activation_metadata_{split}.parquet   # index of activation records
  persona_vectors/{trait}_layer{N}.npy  # computed vectors
  persona_vector_metadata.csv           # vector provenance
  vector_validation_results.csv/.md     # AUC, accuracy, Cohen's d per trait×layer
```

### Failure modes

| Symptom | Likely cause |
|---|---|
| Low AUC for all traits | Wrong layer; responses don't separate poles |
| AUC high for some traits, low for others | Confounded prompts for failing traits (check audit findings) |
| Few responses retained after scoring | Judge prompt too strict; rubric needs adjustment |
| High judge parse error rate | Model generating non-JSON; check `scored_responses.csv` score_rationale column |
| All validation projections near zero | Pole signal too weak; increase contrast in system prompts |

**Do NOT proceed to Stage 3 (ETHICS projection) until all trait vectors pass the minimum AUC target.**

---

---

## Stage 3 — ETHICS Projection

Stage 3 applies the validated persona vectors to the curated ETHICS item bank.
**ETHICS items were never used during vector construction (Stage 2B)**, so this
is the first test of whether the vectors generalize to novel moral scenarios.

### What this stage does

1. Presents each of the 204 curated ETHICS scenarios as a prompt to Gemma-3-12B.
2. Extracts the **last-prompt-token** residual-stream activation at layer 32
   (plus comparison layers 40 and 47) — no response is generated.
3. Projects each activation onto the four validated trait vectors via dot product.
4. Saves long-format and wide-format projection tables and runs sanity diagnostics.

### Token scope distinction

| Stage | Token scope | Purpose |
|---|---|---|
| 2B (vector construction) | Response tokens (mean over generated positions) | Fitting the vectors |
| 3 (ETHICS projection) | **Last prompt token** | Monitoring-style inference on novel scenarios |

### Preprocessing: mean-centering

Gemma-3-12B residual-stream activations at later layers have very large norms (~60,000).
Subtracting the per-layer mean activation across all ETHICS items before projecting removes
a shared baseline direction that would otherwise dominate raw dot products.
This is standard preprocessing in probing and representation engineering and does **not**
use any trait labels.

```
centered_activation_i = raw_activation_i − mean(raw_activations, axis=items)
projection_i_trait = dot(centered_activation_i, trait_vector)
```

Both raw and centered files are saved for full auditability. All downstream
diagnostics default to centered.

### Commands

```bash
# Pre-flight check: validate item bank + AUC thresholds (no GPU)
python scripts/build_ethics_projection_jobs.py

# GPU smoke test — 10 items via Modal
python scripts/extract_ethics_activations.py --limit 10

# Full extraction — 204 items × 3 layers via Modal
python scripts/extract_ethics_activations.py

# Compute projections — both raw and centered, all three layers
python scripts/compute_ethics_projections.py --preprocessing both --layers 32 40 47

# Run diagnostics (centered by default; pass --raw for raw)
python scripts/diagnose_ethics_projections.py

# Compare layers: contrast-validation-selected (32) vs downstream-best
python scripts/compare_projection_layers.py

# Full mock pipeline (no GPU, for offline testing)
python scripts/run_ethics_projection_smoke_test.py
```

### Output files

```
outputs/ethics_projection/
  ethics_activation_metadata.parquet            # index of activation records
  ethics_activation_metadata.csv
  activations/                                  # per-item .npy files (last-prompt-token)
  centering/
    mean_activation_layer{N}.npy                # per-layer mean activation vector
  centering_metadata.csv                        # mean activation norms, n_items, scope
  centering_metadata.json
  ethics_trait_projections_raw_long.parquet     # raw (not centered) projections, long format
  ethics_trait_projections_raw_long.csv
  ethics_trait_projections_raw_wide.parquet     # raw projections, wide format
  ethics_trait_projections_raw_wide.csv
  ethics_trait_projections_centered_long.parquet # mean-centered projections, long format
  ethics_trait_projections_centered_long.csv
  ethics_trait_projections_centered_wide.parquet # mean-centered projections, wide format
  ethics_trait_projections_centered_wide.csv
  ethics_trait_projections_long.parquet         # alias → centered (backwards compat.)
  ethics_trait_projections_wide.parquet         # alias → centered (backwards compat.)
  projection_diagnostics_centered.md            # diagnostics report (centered)
  projection_diagnostics_raw.md                 # diagnostics report (raw)
  projection_summary.csv                        # per-trait projection statistics
  projection_correlation_matrix_centered.csv    # inter-trait correlations (centered)
  layer_comparison_summary.csv                  # per-layer metrics comparison
  layer_comparison_summary.md                   # human-readable layer comparison report
  projection_correlation_matrix_layer{N}.csv    # per-layer inter-trait correlations
```

### Wide-format columns

| Column | Description |
|---|---|
| `item_id` | Stable ETHICS item identifier |
| `source_split` | ETHICS source split (`commonsense` / `deontology` / `justice`) |
| `primary_trait` | Annotated construct trait (honesty / harmlessness / fairness / compassion) |
| `scenario_text` | Original ETHICS scenario text |
| `projection_honesty` | Dot product of centered activation with honesty vector |
| `projection_harmlessness` | … harmlessness vector |
| `projection_fairness` | … fairness vector |
| `projection_compassion` | … compassion vector |

### Projection diagnostics

The diagnostics are **sanity checks, not the final reliability/validity study**.
They check whether:

- Projection distributions are non-degenerate (non-constant, non-zero std)
- Items annotated with a trait tend to project higher on the matching vector
  (**diagonal dominance** — necessary but not sufficient for validity)
- Trait projections are not extremely correlated (which would indicate a single
  general moral valence factor dominating all four vectors)

**Interpreting diagonal dominance.** Diagonal dominance is the fraction of items
where the annotated trait projects highest among all four vectors (chance = 25%).
Values of 27–36% are above chance but weak, consistent with the four traits sharing
latent structure in Gemma's representations. Weak diagonal dominance does **not**
mean the data is invalid — it motivates the factor analysis in Stage 5.

### Layer comparison

Two layers are tracked:

| Layer | Role |
|---|---|
| **32** | Contrast-validation-selected: chosen by Stage 2B AUC on held-out contrastive prompts. This is the methodologically primary layer. |
| **40** | Best downstream ETHICS layer: shows higher diagonal dominance on novel moral scenarios (35.8% vs 27.5% at layer 32). |

These layers **need not agree** — the layer that best separates contrastive trait prompts
during vector construction is not guaranteed to best generalize to novel moral scenarios.
This is a substantive methodological finding: layer 32 remains the pre-registered primary
layer; layer 40 is retained as a comparison layer and reported transparently.

**Paraphrase / framing reliability analysis comes in Stage 4.**

---

## Stage 4A — Projection Structure Analysis (RQ1)

**RQ1**: Do the four morally relevant persona-vector projections behave like one latent
"morality" dimension, or several separable dimensions?

This stage runs on existing Stage 3 outputs — no new Gemma activations are extracted.
All analysis is CPU-only on the centered ETHICS projection tables.

### What this stage asks

PCA and correlation analysis on the 204 × 4 centered projection matrix asks whether
the four trait projections (honesty, harmlessness, fairness, compassion) collapse onto
a small number of latent dimensions or remain separable.

Key metrics:
- **Correlation matrix** among the four trait projections
- **PCA** (via eigendecomposition of the correlation matrix)
- **Effective dimensionality** (participation ratio): (Σλ)² / Σλ²; maximum = 4 for four fully independent dimensions
- **Parallel analysis** (permutation): how many components exceed the random baseline?
- **Factor analysis** (optional, requires `pip install factor_analyzer`): skipped gracefully if not installed

> ⚠ **Factor-analysis caution**: With only four observed variables at most two factors are
> estimable without Heywood cases.  PCA, effective dimensionality, and correlation structure
> are the primary evidence.

### Layer 32 vs layer 40

Both layers are analyzed. Layer 32 is the contrast-validation-selected layer (Stage 2B AUC);
layer 40 showed stronger ETHICS diagonal dominance. The key comparison:

| Metric | Layer 32 (contrast-selected) | Layer 40 (downstream best) |
|---|---|---|
| PC1 variance | 31.1% | 32.7% |
| Effective dimensionality | 3.87 / 4 | 3.83 / 4 |
| Parallel analysis components | 1 | 1 |
| Mean \|off-diagonal corr\| | 0.085 | 0.102 |
| Max trait correlation | 0.186 (honesty–harmlessness) | 0.230 (harmlessness–fairness) |

### RQ1 finding

At all three layers (32, 40, 47) the four trait projections show near-independent
four-dimensional structure.  Effective dimensionality is ~3.8–3.9 (close to the
maximum of 4.0), mean inter-trait correlations are very low (0.085–0.10), and PC1
explains only 31–34% of variance (near chance = 25% for four equal components).

**Interpretation**: Gemma-3-12B encodes each moral trait in a largely distinct direction
rather than collapsing all four onto a single "morality" axis.  Weak diagonal dominance
in Stage 3 (27–36%) is therefore not evidence of invalid vectors — it reflects that
the model does not strongly separate items by their labeled trait.

Parallel analysis retains 1 component at all layers, but this reflects the conservativeness
of the permutation baseline with only p=4 variables; effective dimensionality is the
primary evidence of separability.

### Commands

```bash
# Run structure analysis (layers 32, 40, 47)
python scripts/run_structure_analysis.py

# Generate figures (requires matplotlib)
python scripts/plot_structure_analysis.py
```

### Output files

```
outputs/structure_analysis/
  correlation_matrix_layer{N}.csv       # 4×4 Pearson correlation by layer
  pca_loadings_layer{N}.csv             # PC loadings (variables × components)
  pca_variance_layer{N}.csv             # eigenvalues, explained variance, cumulative
  pca_scores_layer{N}.csv               # per-item PC scores
  parallel_analysis_layer{N}.csv        # observed vs random eigenvalues
  structure_summary.csv                 # key metrics across layers
  structure_analysis_report.md          # full human-readable report
  figures/ (requires matplotlib)
    correlation_heatmap_layer{N}.png
    scree_plot_layer{N}.png
    pca_scatter_layer{N}.png            # items colored by primary_trait
```

### What comes next

Stage 4B (paraphrase generation and framing reliability) tests whether the Stage 3
projection structure *replicates* across paraphrased versions of the same scenarios
before drawing conclusions about trait separability from a single stimulus set.

---

## Stage 4B — Reliability Variant Generation (RQ2)

**RQ2**: Are persona-vector projections stable when the same moral situation is reworded or reframed?

Stage 4A found near-independent four-dimensional projection structure on the original 204 ETHICS items.
Stage 4B tests whether that structure is an artifact of the specific wording used or whether it
replicates across alternate presentations of the same scenarios.

### What this stage produces

A **reliability-variant bank**: for each of the 204 curated ETHICS items, three paraphrases are
generated using an open-source instruction model (Qwen2.5-7B-Instruct via Modal/vLLM).

Key properties:
- Paraphrases are **not new moral items** — they are alternate presentations of the same situation.
- Original item annotations (`primary_trait`, `source_split`) are **inherited** by all variants.
- No new annotation is needed or performed.
- First MVP: original + 3 neutral-framing paraphrases per item (4 × 204 = 816 rows total).
- Framing conditions (`first_person`, `third_person`) are supported in the schema but deferred.

### Semantic equivalence checking

Paraphrases are checked by lightweight heuristics before human review:

| Status | Meaning |
|---|---|
| `original` | The unmodified original item |
| `passed` | All heuristic checks pass |
| `flagged_length` | Variant is much shorter or longer than original |
| `flagged_duplicate` | Near-identical to original or another variant |
| `flagged_possible_meaning_shift` | Negation flip or reversal detected |
| `failed_parse` | LLM response could not be parsed |

These checks are imperfect — human review of flagged variants is required before using them.

### Workflow

```bash
# Smoke test — 10 items, no GPU
python scripts/generate_reliability_variants.py --generation-method mock --limit 10

# Smoke test — 10 items via Modal GPU
python scripts/generate_reliability_variants.py --limit 10

# Validate the variant bank
python scripts/validate_reliability_variants.py

# Review flagged variants; exports reliability_variants_review.csv
python scripts/review_reliability_variants.py

# Full generation — 204 items via Modal GPU
python scripts/generate_reliability_variants.py --resume

# Export accepted variants to final bank
python scripts/export_reliability_variant_bank.py
```

### Output files

```
data/processed/reliability_variants/
  ethics_reliability_variants_raw.parquet   # all generated variants (including flagged)
  ethics_reliability_variants_raw.csv
  reliability_variants_review.csv           # paraphrase-only rows for human review
  ethics_reliability_variants.parquet       # final bank (keep_variant=True only)
  ethics_reliability_variants.csv
```

### Variant bank schema

| Column | Description |
|---|---|
| `item_id` | Original ETHICS item ID |
| `variant_id` | `{item_id}__original` or `{item_id}__p{N}` |
| `variant_type` | `original` or `paraphrase` |
| `paraphrase_id` | `original`, `p1`, `p2`, `p3` |
| `framing` | `neutral` (default), `first_person`, `third_person` |
| `source_split` | Inherited from original item |
| `primary_trait` | Inherited from original item |
| `scenario_text_original` | Original item wording |
| `scenario_text_variant` | This variant's wording |
| `generation_model_name` | LLM used to generate paraphrase |
| `generation_notes` | Model's notes on what was changed |
| `semantic_equivalence_status` | See status table above |
| `keep_variant` | Whether this variant is accepted for Stage 4C |

### What comes next

See **Stage 4C** below.

**Stage 4D**: Reliability / generalizability analysis — estimate intra-class correlation and
variance components across the projection profiles to quantify paraphrase stability.

---

## Stage 4C — Reliability Variant Projection

Stage 4C applies the already-validated Stage 2B persona vectors to every accepted item variant
from Stage 4B.  It is monitoring-style projection (identical to Stage 3): present each variant
as a prompt, extract the last-prompt-token residual-stream activation, then compute the dot
product with each trait vector.

**Key design decisions:**

- Reuses Stage 3's `last_prompt_token` scope — no response generation.
- Does **not** construct new persona vectors — Stage 2B vectors are fixed and applied directly.
- Produces both **raw** projections (for audit) and **mean-centered** projections (default for
  downstream analysis).  Mean-centering subtracts the cross-variant mean per layer, removing
  the shared baseline direction that dominates raw dot products.
- Preserves both `item_id` and `variant_id` in all outputs so Stage 4D can group by item.

**Key diagnostic question:** Do paraphrases of the same moral scenario produce similar
projection scores?  High within-item consistency across paraphrases means the vectors are
capturing something robust about the scenario, not noise from surface wording.

### Smoke test

```bash
# Build forward-pass job table (no GPU needed)
python scripts/build_reliability_projection_jobs.py

# Extract activations for 20 variants (requires Modal + GPU)
python scripts/extract_reliability_variant_activations.py --limit 20

# Compute raw + centered projections (no GPU needed)
python scripts/compute_reliability_variant_projections.py --preprocessing both --layers 32 40 47

# Run diagnostics
python scripts/diagnose_reliability_variant_projections.py
```

### Full run

```bash
# Extract activations for all accepted variants
python scripts/extract_reliability_variant_activations.py

# Compute projections
python scripts/compute_reliability_variant_projections.py --preprocessing both --layers 32 40 47

# Diagnostics
python scripts/diagnose_reliability_variant_projections.py
```

### Output files

| File | Description |
|------|-------------|
| `outputs/reliability_projection/reliability_projection_jobs.parquet` | Forward-pass job table |
| `outputs/reliability_projection/reliability_activation_metadata.parquet` | Per-activation metadata |
| `outputs/reliability_projection/activations/{variant_id}_layer{N}.npy` | Raw activation arrays |
| `outputs/reliability_projection/reliability_trait_projections_long.parquet` | Centered long-format (default) |
| `outputs/reliability_projection/reliability_trait_projections_wide.parquet` | Centered wide-format (default) |
| `outputs/reliability_projection/reliability_trait_projections_long_raw.*` | Raw projections (audit) |
| `outputs/reliability_projection/reliability_projection_diagnostics.md` | Diagnostic report |
| `outputs/reliability_projection/reliability_projection_summary.csv` | Projection stats by trait × layer |
| `outputs/reliability_projection/reliability_projection_corr_layer{N}.csv` | Correlation matrices |

### What feeds into Stage 4D

The centered wide projections (`reliability_trait_projections_wide.parquet`) provide one
projection score per variant × trait × layer.  Stage 4D uses these to estimate intra-class
correlation (ICC) across paraphrases within each item — quantifying whether the projection is
stable under rewording.  High ICC indicates the vectors are measuring something about the moral
scenario rather than surface form.

---

## Stage 4D — Reliability / Generalizability Analysis (RQ2)

**Purpose**: answers RQ2 — are persona-vector projections dependable under paraphrase variation?

**Method**: one-way ANOVA variance decomposition separates stable between-item variance (universe
score variance) from wording noise (error variance = within-item variance across paraphrases).
From these components we derive:

- **ICC (intra-class correlation)** — the proportion of total projection variance attributable to
  stable item differences.  Equivalent to G-coefficient at k=1.
- **G-coefficient** — generalizes ICC to k averaged paraphrases: G(k) = σ²_between / (σ²_between + σ²_within / k).
- **D-study** — shows how reliability improves as more paraphrases are averaged (k = 1…5).
- **Universe score variance** — stable variance due to real differences between moral scenarios.
- **Error variance** — wording noise; ideally small relative to universe score variance.

**Key terms**:
| Term | Meaning |
|------|---------|
| ICC (intra-class correlation) | Reliability for a single paraphrase variant |
| G-coefficient | Reliability for the mean of k paraphrase variants |
| D-study (decision study) | How G-coefficient changes with number of averaged variants |
| Universe score variance | Between-item variance; stable signal across wordings |
| Error variance | Within-item variance; wording noise |

**Commands**:
```bash
python scripts/run_reliability_analysis.py
python scripts/plot_reliability_analysis.py  # optional, requires matplotlib
```

**Outputs** (in `outputs/reliability_analysis/`):
- `reliability_summary.csv` — reliability_1 through reliability_5 per (layer, trait)
- `variance_components.csv` — between/within variance per (layer, trait)
- `d_study_results.csv` — G-coefficient for k=1…5, all layers and traits
- `item_level_reliability_long.csv` — per-item mean and SD of projections
- `reliability_analysis_report.md` — full Markdown report with tables and interpretation
- `figures/` — three diagnostic plots (requires matplotlib)

---

## Fallback: Manual Annotation if Modal Is Unavailable

If you cannot use Modal (no account, billing issues, network restrictions), you can still annotate by reviewing the auto-labeling prompts manually or using the Anthropic API path.

### Option A — Anthropic API (requires `ANTHROPIC_API_KEY`)

```bash
pip install anthropic

export ANTHROPIC_API_KEY=sk-ant-...

# Dry-run: inspect prompts without calling the API
python scripts/auto_annotate_items.py --dry-run --limit 5

# Annotate first 10 rows
python scripts/auto_annotate_items.py --limit 10

# Full run
python scripts/auto_annotate_items.py

# Resume after interruption
python scripts/auto_annotate_items.py --resume
```

### Option B — Fully manual annotation

Open `data/processed/ethics_annotation_sheet.csv` in a spreadsheet editor and fill in each row directly using the rubric in [`configs/traits.yaml`](configs/traits.yaml).

Validate after annotating:

```bash
python scripts/validate_annotations.py
```

Then export as in Stage 1e above.

---

## Controls

The controls suite validates that the observed persona-vector results are
specific to the moral constructs under study and are not artifacts of the
analysis pipeline, random directions, label assignments, or preprocessing choices.

### Negative controls (what should NOT work)

| Control | Question | Expected finding |
|---------|----------|-----------------|
| **Random vectors** | Would arbitrary directions produce similar structure/reliability to moral persona vectors? | Real vectors should outperform random directions on structure and trait-label alignment |
| **Shuffled labels** | Is the observed trait-label alignment better than chance? | Real label alignment should exceed the shuffled-label null (p < 0.05) |
| **Permuted item-variant grouping** | Is reliability driven by stable item identity, not coincidental grouping? | True grouping should produce higher G-coefficient than permuted grouping |

### Positive controls (what SHOULD work)

| Control | Question | Expected finding |
|---------|----------|-----------------|
| **Exact duplicates** | Does the reliability pipeline behave correctly when wording variation is zero? | G(k=1) ≈ 1.0 — validates the reliability analysis has no bugs |
| **Contrast-validation responses** | Do vectors still reproduce their calibration signal? | AUC ≥ 0.75 for all trait × layer combinations |
| **Synthetic obvious moral scenarios** | Do vectors work on clear, unambiguous, format-controlled cases? | Strong within-trait discrimination and cross-trait diagonal dominance on a confound-free item bank (see below) |

### Synthetic Confound-Controlled Item Bank (Extended Positive Control)

The ETHICS-based analysis throughout this project is limited by a structural
confound: item *format* is nearly collinear with *trait* (deontology is ~96%
honesty, justice is ~72% fairness — see Stage 2.3). This item bank removes
that confound by construction: one uniform format across all four traits,
authored independently of both ETHICS and the persona-vector construction
pipeline.

**Design**: 4 traits × 20 matched pairs × 2 labels = 160 single-sentence,
first-person items. Within each pair, the "upheld" and "violated" versions
share the same context and differ only in the trait-relevant action (e.g.
*told the cashier about extra change* vs. *kept it silently*). Zero literal
trait-label words are used anywhere (checked programmatically against the
same word list `src/vectors/artifact_quality.py`'s leakage audit uses). Full
rationale and every design decision: [`docs/synthetic_item_bank_guidelines.md`](docs/synthetic_item_bank_guidelines.md).

This supersedes the earlier `data/processed/synthetic_moral_scenarios.csv`
scaffold, which had no ground-truth label column and was never wired into
an extraction/projection script.

```bash
# Build the item bank (CPU-only, deterministic, includes a leakage self-check)
python scripts/build_synthetic_trait_bank.py
```

Produces `data/processed/synthetic_trait_bank.{csv,parquet}`
(`item_id`, `scenario_text`, `label`, `label_semantics`, `primary_trait`,
`source_split="synthetic"`, `pair_id`, ... — schema mirrors
`ethics_curated_mvp.parquet` for drop-in compatibility with the Stage 3
extraction/projection scripts).

**Status: item bank built; GPU extraction/projection not yet run.** Next
steps: extract last-prompt-token activations for these 160 items via Modal
(reusing `extract_ethics_activations.py`/`compute_ethics_projections.py`
pointed at the new file), then project onto both the 4 original and 4
synonym persona vectors and compare within-trait/cross-trait discrimination
against the ETHICS-derived results.

### Convergent-validity controls

| Control | Question | Expected finding |
|---------|----------|-----------------|
| **Synonym/construct-neighbor vectors** | Are results robust to exact trait wording? | Synonym vectors cosine-closest to their intended parent trait; ETHICS projections correlated with original |

### Robustness controls

| Control | Question | Expected finding |
|---------|----------|-----------------|
| **Raw vs centered preprocessing** | Do conclusions depend on centering? | Key findings stable across both preprocessing choices |
| **Layer comparison** | Are vectors directionally stable across layers 32, 40, 47? | Similar structure and label alignment across layers |

**Important scope note on preprocessing robustness:** the original ETHICS projections
(204 items, one row per item, layer 32 only) and the reliability-variant projections
(761 paraphrase/framing variants × 3 layers = 2283 rows) are **different datasets** and
must be analyzed separately. Stage 4A's structure numbers (`outputs/structure_analysis/structure_summary.csv`)
come from the **centered original ETHICS** projections (`ethics_trait_projections_centered_wide.parquet`),
computed on the real Gemma-3-12B activations: effective dimensionality ≈ 1.13 at layer 32 and
≈ 1.15 at layer 47 (both single-dominant-dimension), and ≈ 2.46 at layer 40 (partial separation).

**Data integrity note:** an initial Stage 4A run in June used stale mock activations
(`dim=64`, written before real GPU extraction) and reported effective dimensionality ≈ 3.87
at layer 32 — an artifact of random projections onto 4 unit vectors being naturally
near-orthogonal. That mock result has been fully superseded: all tables, figures, and the
numbers above come from the corrected real Gemma-3-12B activations (re-extracted June 29,
`dim=3840`). An earlier version of `scripts/run_preprocessing_controls.py` separately
mistakenly loaded the reliability-variant files for the "preprocessing" comparison, which
pools many highly-correlated paraphrase variants per item and produces a misleadingly low
effective dimensionality and inflated inter-trait correlation if conflated with the
original-item-level numbers above. The preprocessing control has been fixed to load the
correct per-scope files, compute metrics strictly per-layer, and label each row with a
`source_dataset` column (`ethics_original` vs `reliability_variants`) so the two scopes are
never conflated again.

### Running the controls

```bash
# Run all CPU-safe controls
python scripts/run_all_controls.py

# Individual controls
python scripts/run_random_vector_controls.py
python scripts/run_shuffled_label_controls.py
python scripts/run_reliability_grouping_controls.py
python scripts/run_positive_controls.py
python scripts/run_preprocessing_controls.py

# Synonym controls (requires synonym vectors to be built first via Stage 2B pipeline)
python scripts/inspect_synonym_vector_artifacts.py
python scripts/run_synonym_vector_controls.py --mvp-only
```

**Synonym vector GPU work**: The synonym vector analysis (`configs/synonym_vector_artifacts.yaml`)
requires running the Stage 2B pipeline with the synonym artifacts to generate new persona vectors.
This is a GPU job on Modal. The analysis scripts (cosine similarity, projection agreement) are
fully CPU-only and run once the synonym vectors exist.

All control outputs are saved to `outputs/controls/`. The final report is at
`outputs/controls/controls_report.md`.

---

## Limitations and Future Work

- **Single model.** All results are for Gemma-3-12B-IT. Whether the single-axis collapse
  generalises across model families, sizes, or RLHF regimes is untested.
- **Trait-specific measurement requires different methods.** Orthogonalisation (projecting
  out the shared axis before extracting a second direction), supervised probing on
  trait-labelled items, or causal steering experiments would be needed to determine whether
  finer-grained trait directions exist beneath the shared axis.
- **ETHICS format–trait confound.** Even with the confound-free synthetic bank as a
  cross-check, a larger balanced benchmark with fully controlled item format would allow
  cleaner interpretation of what the shared axis predicts.
- **Cause of the collapse is still open.** The generic-alignment (`virtue_axis`) explanation
  was directly tested and rejected (see [`configs/virtue_axis_control.yaml`](configs/virtue_axis_control.yaml)
  and the paper §3.9); what the shared axis actually represents remains unresolved.

See [`outputs/persona_vectors_paper.pdf`](outputs/persona_vectors_paper.pdf) §4.4 for the full discussion.

---

## Citation

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021).
*Aligning AI With Shared Human Values.* ICLR 2021.
