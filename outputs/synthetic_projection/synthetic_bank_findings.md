# Synthetic Confound-Controlled Item Bank — Findings

Draft findings for later merge into the paper. Covers the 4 original trait
vectors and the 4 synonym vectors. See "Still pending" at the bottom before
treating this as final.

Design and rationale: [`docs/synthetic_item_bank_guidelines.md`](../../docs/synthetic_item_bank_guidelines.md).
Item bank: [`data/processed/synthetic_trait_bank.csv`](../../data/processed/synthetic_trait_bank.csv).

## Setup

- 160 items: 4 traits × 20 matched pairs × {upheld, violated}.
- Single sentence, first-person, zero literal trait-label leakage (checked
  programmatically against the same word list `src/vectors/artifact_quality.py`
  uses), one uniform format across all four traits — removes the format–trait
  confound present throughout the ETHICS-based analysis.
- Real Gemma-3-12B-IT extraction (last-prompt-token, layers 32/40/47) via
  Modal, `google/gemma-3-12b-it`, hidden_dim=3840. Verified post-hoc: 480/480
  expected (item × layer) combinations present, zero duplicates, zero NaNs,
  zero all-zero or exact-duplicate-content activations, norms in a sensible
  range (54k–167k) consistent with the ~61k shared-component norm already
  reported for this model.
- Projected onto the 4 existing trait vectors (same vectors used throughout
  the ETHICS analysis — not refit on this bank).

## Finding 1 — Cross-trait structure replicates the ETHICS collapse

Full Stage 4A structure analysis (PCA, effective dimensionality, parallel
analysis) run on this bank at all three layers
(`outputs/synthetic_structure_analysis/`), directly comparable to Table 2 in
the paper:

| Layer | Metric | ETHICS (204 items) | Synthetic bank (160 items) |
|---|---|---|---|
| 32 | Eff. Dim | 1.130 | 1.214 |
| 32 | PC1 variance | 94.0% | 90.5% |
| 32 | Mean \|r\| | 0.919 | 0.873 |
| 32 | Top correlated pair | harmlessness–fairness | **harmlessness–fairness** |
| 40 | Eff. Dim | 2.461 | 2.091 |
| 40 | PC1 variance | 53.4% | 60.7% |
| 40 | Mean \|r\| | 0.339 | 0.466 |
| 40 | Top correlated pair | honesty–fairness | **honesty–fairness** |
| 47 | Eff. Dim | 1.151 | 1.389 |
| 47 | PC1 variance | 93.0% | 83.6% |
| 47 | Mean \|r\| | 0.906 | 0.769 |
| 47 | Top correlated pair | honesty–harmlessness | harmlessness–compassion |

The layer-32/47-collapse vs. layer-40-partial-separation pattern replicates
directionally at every layer. At layers 32 and 40, the **same specific pair**
of traits is most correlated in both independent datasets (harmlessness–
fairness at 32, honesty–fairness at 40) — stronger evidence than "collapse
happens somewhere": the *same* two traits collapse together, on a dataset
sharing zero vocabulary or format with ETHICS or the construction prompts.
This directly answers the earlier open question (raised in the paper's own
critique) of whether the layer-40 anomaly was a 3-layer multiple-comparisons
artifact — it now replicates on an independent dataset, which a pure fluke
would not be expected to do.

Diagonal dominance at layer 32 (from `diagnose_ethics_projections.py`):
**33.8%** (vs. chance 25%, vs. 27.5% pooled on ETHICS).

## Finding 2 — Within-trait discrimination (own vector, upheld vs. violated)

**Label convention verified**: `label_semantics` = `1=trait_violated,
0=trait_upheld` throughout, and cross-checked directly against every
`item_id` suffix (`_upheld` / `_violated`) — 0/160 mismatches. Not a labeling
bug.

AUC computed both directions per trait×layer (they're complementary,
AUC(upheld>violated) = 1 − AUC(violated>upheld); reporting both removes any
ambiguity about which direction a number describes), with a two-sided
permutation test (10,000 label shuffles per trait×layer, labels shuffled
against fixed projection scores). Only significant results shown (p<0.05);
all omitted trait×layer combinations were non-significant (p>0.1):

| Layer | Trait | Direction | AUC | p (perm.) |
|---|---|---|---|---|
| 32 | **fairness** | upheld scores higher | 0.728 | 0.012 |
| 40 | **harmlessness** | violated scores higher | 0.930 | <0.0001 |
| 40 | **compassion** | violated scores higher | 0.725 | 0.012 |
| 47 | **fairness** | upheld scores higher | 0.858 | <0.0001 |
| 47 | **compassion** | violated scores higher | 0.715 | 0.019 |

5/12 trait×layer combinations significant at p<0.05; 2/12 survive a strict
Bonferroni correction for 12 comparisons (α=0.0042): harmlessness@40 and
fairness@47.

**Robustness check (fairness, the two significant layers)**: removing the 3
most extreme-magnitude items (regardless of direction) and recomputing —
layer 32: AUC 0.273→0.260 (p 0.012→0.011); layer 47: AUC 0.142→0.168
(p<0.0001→0.0005). Barely moves. The effect is broad-based across most of
the 40 items, not driven by a handful of outliers. Qualitative read of the
full sorted item list confirms this: the most extreme low-scoring items are
almost all clean, unambiguous violations ("gave one applicant easier
interview questions," "cut in front of people," "approved a loan for someone
I knew personally"), and the most extreme high-scoring items are almost all
clean upholding examples — no mislabeled or ambiguous items driving the
result.

**Interpretation:**

- **Fairness consistently goes the "expected" direction** (upheld scores
  higher than violated) at both layers where it's significant (32, 47) —
  matching the vector's `high_persona − low_persona` construction convention,
  where a higher projection should mean more of the virtuous trait. This is
  a clean, real, robust, non-circular signal on items sharing no vocabulary
  or format with ETHICS or the construction pipeline.
- **Harmlessness and compassion consistently go the opposite direction**
  (violated scores *higher* than upheld) at every layer where they're
  significant (harmlessness@40, compassion@40, compassion@47) — the reverse
  of the naive expectation from how the vectors were built. This is a clean
  trait-level split, not noise scattered randomly across traits: fairness
  reliably behaves as expected, harmlessness/compassion reliably behave
  backwards. Worth flagging as a real, specific property of how these three
  traits' vectors generalize, rather than an error to correct.
- **Layer 40 / harmlessness is the strongest single result in this analysis**
  (AUC=0.930 toward "violated scores higher," p<0.0001) — real,
  non-circular discrimination, just in the counter-intuitive direction.
- **Honesty shows no signal at any layer** (p>0.4 throughout, neither
  direction). In a clean, confound-free, single-format bank, the honesty
  vector does not distinguish honesty-upheld from honesty-violated sentences
  at all — notable because honesty was the strongest, most narratively
  convenient trait in the original ETHICS analysis (96% of the
  EXCUSE-format items).

## Finding 3 — Synonym vectors replicate both the collapse and the direction pattern

Projected the 4 synonym vectors (truthfulness/harm_avoidance/impartiality/
empathy — convergent-validity vectors for honesty/harmlessness/fairness/
compassion, already built via Stage 2B, no new GPU work needed) onto this
same bank's activations, via a newly-parameterized
`scripts/project_synonym_controls.py` (`--wide-path`, `--act-metadata`,
`--act-dir`, `--out-dir`; verified byte-identical output on the original
ETHICS case with default args before trusting it on new paths).

**At a glance:**

| Sub-finding | Result |
|---|---|
| 8-vector structure vs. ETHICS 8-vector result | Replicates (ED 1.28–1.68 here vs. 1.19 there) |
| Synonym pairs with matching parent direction | 3 of 4 (fairness/impartiality, harmlessness/harm_avoidance, honesty/truthfulness-null) |
| Synonym pairs diverging from parent | 1 of 4 (compassion/empathy) |
| New GPU work required | None — reused existing synonym vectors and bank activations |

**8-vector structure (extends Finding 1, mirrors paper Figure 3):**

| Layer | ED (8-vec) | PC1 variance | Max \|corr\| | Top pair |
|---|---|---|---|---|
| 32 | 1.283 | 87.9% | 0.993 | impartiality–empathy |
| 40 | 1.557 | 78.1% | 0.998 | honesty–truthfulness |
| 47 | 1.683 | 75.2% | 0.971 | harmlessness–impartiality |

Matches the paper's ETHICS-based 8-vector result closely (ED=1.19, PC1=92%
at layer 32). The collapse is not specific to the four original trait labels
or to ETHICS — it reproduces with independently-constructed synonym vectors
on an independent item bank. One nuance: not every pairwise correlation
exceeds 0.75 here (unlike the ETHICS version) — truthfulness is the most
independent vector of the eight (r=0.51–0.76 with the others, vs. 0.82–0.99
among the rest); still substantial, but the least collapsed pairing in this
dataset.

**Within-trait discrimination for synonym vectors (extends Finding 2)** —
AUC computed on each synonym vector against its *own parent trait's*
upheld/violated items (e.g. does `truthfulness` discriminate *honesty*
items?), same permutation methodology (10,000 shuffles):

| Layer | Synonym | Parent | AUC(violated>upheld) | p |
|---|---|---|---|---|
| 32 | **harm_avoidance** | harmlessness | 0.770 | 0.003 |
| 40 | **impartiality** | fairness | 0.140 | <0.0001 |
| 47 | **impartiality** | fairness | 0.290 | 0.024 |
| — | truthfulness | honesty | n.s. (p=0.06–0.68 all layers) | — |
| — | empathy | compassion | n.s. (p=0.20–0.36 all layers) | — |

**Convergent-validity comparison — direction matches even where the specific
significant layer doesn't:**

| Trait pair | Original vector (Finding 2) | Synonym vector (Finding 3) | Direction match? |
|---|---|---|---|
| honesty / truthfulness | null at all layers | null at all layers | ✅ consistent null |
| harmlessness / harm_avoidance | significant @40, violated higher | significant @32, violated higher | ✅ same direction, different layer |
| fairness / impartiality | significant @32,47, upheld higher | significant @40,47, upheld higher | ✅ same direction, overlaps @47 |
| compassion / empathy | significant @40,47, violated higher | null at all layers | ❌ diverges |

Given two independently-constructed vector pairs (different contrastive
prompts, different elicitation questions, same underlying trait concept)
reproduce the *same sign* of discrimination wherever both are significant,
that's real convergent-validity evidence — direction agreeing across
independent constructions is much less likely under a pure-noise account
than either result alone. Compassion/empathy is the one case where the
synonym doesn't replicate its parent's within-trait result — worth flagging
as a limitation rather than omitting.

## Finding 4 — Preprocessing robustness: centering doesn't change structure, but is required for diagonal dominance

Reran Stage 4A structure analysis on the **raw** (uncentered) projections
(`outputs/synthetic_structure_analysis_raw/`, already generated alongside
the centered files back when the projection step ran with
`--preprocessing both` — no new extraction needed) and compared against the
centered results in Finding 1.

**Correlation/PCA-derived metrics are exactly identical, at all 3 layers:**

| Layer | ED (raw) | ED (centered) | Mean \|r\| (raw) | Mean \|r\| (centered) |
|---|---|---|---|---|
| 32 | 1.2145 | 1.2145 | 0.8729 | 0.8729 |
| 40 | 2.0906 | 2.0906 | 0.4657 | 0.4657 |
| 47 | 1.3886 | 1.3886 | 0.7687 | 0.7687 |

This matches the paper's ETHICS-based §3.7 result closely. Worth being
precise about *why*: Pearson correlation (and everything derived from it —
PCA, effective dimensionality) is mathematically invariant to per-variable
additive shifts, and mean-centering is exactly that kind of shift. So this
identity is close to a mathematical guarantee for these particular metrics,
not fresh empirical evidence — it mainly confirms the pipeline isn't doing
anything unexpected, rather than revealing something new about the model.

**Diagonal dominance is not shift-invariant, and here centering matters a
lot:**

| Preprocessing | Diagonal dominance (layer 32) |
|---|---|
| Raw | 0.250 (exactly chance) |
| Centered | 0.338 |

Raw projections sit at *exactly* chance — the shared baseline component
(the same large common direction discussed throughout the paper, ~60k
ℓ₂-norm) swamps the item-specific signal enough that "which vector has the
highest raw value" carries no information at all. Centering is what reveals
the real, above-chance trait-label signal. This validates the project's
default of using centered projections for all diagonal-dominance-style
diagnostics — it's not just convention, it's necessary for this particular
metric on this bank.

## Caveats

- n=20/20 per trait is small; individual AUCs (especially the more modest
  significant ones — fairness@32, compassion@40/47) should be treated as
  suggestive pending a closer look at which specific items drive them.
- Item bank is single-author-drafted with no independent inter-rater
  validation (see guidelines doc, "What this does not resolve").
- Synonym vectors were validated in Stage 2B against their own contrastive
  construction set, not against this bank — same caveat as the original 4
  vectors applies (generalization to a novel item bank, not construction-set
  performance).

## Still pending before this section is complete

1. Optional: `plot_structure_analysis.py` figures for this bank (matplotlib,
   mirrors Figures 1–2 in the paper) — not run yet, cosmetic only.
2. Separately: the generic virtuous/unethical-AI control vector has not been
   built — a bigger, independent task (its own Stage 2B-style construction
   round), not blocking this section.

Done: real GPU extraction (verified), projection, full Stage 4A structure
analysis, within-trait AUC + permutation test for the 4 original vectors
(both directions, label convention verified, robustness-checked against
outlier items), the same for the 4 synonym vectors (8-vector structure
analysis + within-trait AUC against parent-trait labels), and raw-vs-centered
preprocessing robustness.
