# Stage 3: ETHICS Projection Diagnostics (mean_centered)

## Overview

- Preprocessing: **mean_centered**
- Items projected: 204
- Trait vectors applied: 4
- Layers: 3
- Missing activations: 0

## ✅ No warnings

## Projection Distribution (target layer)

| projected_trait | mean | std | min | max | n |
| --- | --- | --- | --- | --- | --- |
| honesty | 0.0000 | 0.9656 | -2.6526 | 2.4560 | 204 |
| harmlessness | -0.0000 | 0.9942 | -3.2550 | 2.3561 | 204 |
| fairness | -0.0000 | 1.0281 | -2.7579 | 2.3903 | 204 |
| compassion | -0.0000 | 1.0195 | -2.5400 | 2.7706 | 204 |

## Mean Projection by primary_trait × projected_trait

Rows = annotated trait of item. Columns = trait vector projected onto.
Diagonal entries should be higher than off-diagonal if vectors discriminate.

| primary_trait | compassion | fairness | harmlessness | honesty |
| --- | --- | --- | --- | --- |
| compassion | 0.2534 | -0.1159 | 0.1485 | -0.3785 |
| fairness | -0.0686 | 0.4038 | -0.0080 | -0.2338 |
| harmlessness | -0.0907 | -0.2677 | 0.4231 | -0.1424 |
| honesty | -0.0111 | -0.0642 | -0.1751 | 0.2594 |

## Diagonal Dominance

Fraction of items where annotated trait projects highest: **0.358** (chance = 0.25)  ✅

## Trait Projection Correlation Matrix

| trait | honesty | harmlessness | fairness | compassion |
| --- | --- | --- | --- | --- |
| honesty | 1.0000 | -0.1863 | 0.0031 | 0.0592 |
| harmlessness | -0.1863 | 1.0000 | 0.1502 | 0.0711 |
| fairness | 0.0031 | 0.1502 | 1.0000 | 0.0375 |
| compassion | 0.0592 | 0.0711 | 0.0375 | 1.0000 |

## Notes

- These diagnostics are sanity checks, not the final reliability/validity study.
- Paraphrase and framing reliability analysis comes in Stage 4.
- Diagonal dominance > chance (0.25) is necessary but not sufficient for validity.
- High inter-trait correlations may indicate a general moral valence factor —
  investigate with factor analysis in Stage 5.
- Weak diagonal dominance does not mean the data is invalid; it may reflect
  genuine overlap in how Gemma represents these moral traits.
