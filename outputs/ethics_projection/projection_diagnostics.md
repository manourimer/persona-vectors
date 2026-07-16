# Stage 3: ETHICS Projection Diagnostics

## Overview

- Items projected: 204
- Trait vectors applied: 4
- Layers: 3
- Missing activations: 0
- Missing vectors: 0

## ⚠ Warnings

- Trait projections 'harmlessness' and 'fairness' are extremely correlated (r=-0.968). Vectors may not be discriminating distinct traits.

## Projection Distribution (target layer)

| projected_trait | mean | std | min | max | n |
| --- | --- | --- | --- | --- | --- |
| honesty | -0.0001 | 1049.6565 | -2828.7012 | 2798.7339 | 204 |
| harmlessness | 0.0001 | 1407.6191 | -3936.0554 | 3613.2190 | 204 |
| fairness | -0.0001 | 1271.5733 | -3149.8777 | 3327.9314 | 204 |
| compassion | -0.0000 | 880.5801 | -2606.9565 | 2632.4941 | 204 |

## Mean Projection by primary_trait × projected_trait

Rows = annotated trait of item. Columns = trait vector projected onto.

Diagonal entries should be higher than off-diagonal if vectors discriminate.

| primary_trait | compassion | fairness | harmlessness | honesty |
| --- | --- | --- | --- | --- |
| compassion | -127.6477 | -633.6702 | 492.0807 | -402.1418 |
| fairness | -184.6276 | -1.4555 | 41.7304 | -171.4886 |
| harmlessness | -109.1882 | 30.4973 | 189.0737 | -134.4623 |
| honesty | 155.3171 | 170.0236 | -219.2573 | 235.1489 |

## Diagonal Dominance

Fraction of items where annotated trait projects highest: **0.275**

❌ Matching trait is NOT most-projected for the majority of items.

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
- Diagonal dominance > 0.5 is a necessary (not sufficient) condition for validity.
- High inter-trait correlations may indicate a general moral valence factor
  rather than trait-specific measurement — investigate in Stage 5 (factor analysis).
