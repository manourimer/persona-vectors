# Stage 3: ETHICS Projection Diagnostics (raw)

## Overview

- Preprocessing: **raw**
- Items projected: 204
- Trait vectors applied: 4
- Layers: 3
- Missing activations: 0

## ⚠ Warnings

- Diagonal dominance (0.157) is at or below four-way chance (0.25). Items' annotated traits are not projecting highest on the matching vector. Investigate trait specificity.
- Trait projections 'harmlessness' and 'fairness' are extremely correlated (r=-0.968). Vectors may not discriminate distinct traits.
- One vector dominates raw projections (mean spread=48723.8). This is expected with large residual-stream activations — use mean-centered projections for interpretation.

## Projection Distribution (target layer)

| projected_trait | mean | std | min | max | n |
| --- | --- | --- | --- | --- | --- |
| honesty | -20804.4293 | 1049.6565 | -23633.1309 | -18005.6934 | 204 |
| harmlessness | 26087.9835 | 1407.6192 | 22151.9277 | 29701.2031 | 204 |
| fairness | -22635.8208 | 1271.5735 | -25785.7012 | -19307.8867 | 204 |
| compassion | -18633.2105 | 880.5801 | -21240.1699 | -16000.7148 | 204 |

## Mean Projection by primary_trait × projected_trait

Rows = annotated trait of item. Columns = trait vector projected onto.
Diagonal entries should be higher than off-diagonal if vectors discriminate.

| primary_trait | compassion | fairness | harmlessness | honesty |
| --- | --- | --- | --- | --- |
| compassion | -18760.8582 | -23269.4909 | 26580.0638 | -21206.5714 |
| fairness | -18817.8381 | -22637.2762 | 26129.7138 | -20975.9174 |
| harmlessness | -18742.3994 | -22605.3234 | 26277.0571 | -20938.8920 |
| honesty | -18477.8930 | -22465.7971 | 25868.7261 | -20569.2802 |

## Diagonal Dominance

Fraction of items where annotated trait projects highest: **0.157** (chance = 0.25)  ❌

## Trait Projection Correlation Matrix

| trait | honesty | harmlessness | fairness | compassion |
| --- | --- | --- | --- | --- |
| honesty | 1.0000 | -0.9359 | 0.9349 | 0.9160 |
| harmlessness | -0.9359 | 1.0000 | -0.9675 | -0.8953 |
| fairness | 0.9349 | -0.9675 | 1.0000 | 0.8662 |
| compassion | 0.9160 | -0.8953 | 0.8662 | 1.0000 |

## Notes

- These diagnostics are sanity checks, not the final reliability/validity study.
- Paraphrase and framing reliability analysis comes in Stage 4.
- Diagonal dominance > chance (0.25) is necessary but not sufficient for validity.
- High inter-trait correlations may indicate a general moral valence factor —
  investigate with factor analysis in Stage 5.
- Weak diagonal dominance does not mean the data is invalid; it may reflect
  genuine overlap in how Gemma represents these moral traits.
