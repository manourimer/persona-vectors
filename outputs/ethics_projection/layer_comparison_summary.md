# Stage 3: Layer Comparison Summary

Preprocessing: **mean_centered**

## Metrics by Layer

| layer | diagonal_dominance | matching_margin | max_inter_trait_correlation | diagonal_honesty | diagonal_harmlessness | diagonal_fairness | diagonal_compassion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32.0000 | 0.3578 | 0.4214 | 0.1863 | 0.3428 | 0.5900 | 0.5072 | 0.3687 |
| 40.0000 | 0.3676 | 0.3939 | 0.2300 | 0.1980 | 0.7143 | 0.4556 | 0.6213 |
| 47.0000 | 0.3922 | 0.4160 | 0.2714 | 0.3978 | 0.4739 | 0.4470 | 0.3643 |

## Layer Selection

- **Contrast-validation-selected layer**: 32 (chosen by held-out AUC on contrast artifacts in Stage 2B)
- **Best downstream ETHICS layer**: 47 (highest diagonal dominance on mean-centered ETHICS projections)
- **Layers agree**: ❌ No

## Interpretation

Layer 32 was selected by contrast-prompt validation (diagonal dominance on ETHICS = 0.358), but layer 47 produced stronger downstream ETHICS trait-structure diagnostics (diagonal dominance = 0.392, Δ = +0.034). This suggests contrast-prompt validation and downstream measurement validity can diverge — the layer that best separates elicitation artifacts may not be the layer that best captures novel moral scenarios. Layer 32 remains the methodologically primary layer (selected pre-ETHICS); layer 47 is reported as a comparison.

## ⚠ Warnings

- Best downstream ETHICS layer (47) differs from contrast-validation-selected layer (32). See layer_comparison_summary for details.

## Notes

- Diagonal dominance = fraction of items where the annotated trait projects highest on the matching vector (chance = 0.25 for 4 traits).
- Matching margin = mean(matching projection) − mean(non-matching projections).
- Weak diagonal dominance does not mean the data is invalid — it may indicate that the four traits share latent structure in Gemma's representations (investigate in Stage 5: factor analysis).
- Layer 32 remains the primary layer for all downstream analyses unless explicitly overridden.
