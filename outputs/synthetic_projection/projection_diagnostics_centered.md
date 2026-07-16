# Stage 3: ETHICS Projection Diagnostics (mean_centered)

## Overview

- Preprocessing: **mean_centered**
- Items projected: 160
- Trait vectors applied: 4
- Layers: 3
- Missing activations: 0

## ⚠ Warnings

- Trait projections 'harmlessness' and 'fairness' are extremely correlated (r=-0.971). Vectors may not discriminate distinct traits.

## Projection Distribution (target layer)

| projected_trait | mean | std | min | max | n |
| --- | --- | --- | --- | --- | --- |
| honesty | -0.0005 | 770.3991 | -2226.7148 | 2686.7815 | 160 |
| harmlessness | 0.0006 | 964.3692 | -3878.4905 | 2503.1514 | 160 |
| fairness | -0.0006 | 749.3661 | -1954.3732 | 3206.2151 | 160 |
| compassion | -0.0005 | 680.8414 | -2163.9102 | 2806.7581 | 160 |

## Mean Projection by primary_trait × projected_trait

Rows = annotated trait of item. Columns = trait vector projected onto.
Diagonal entries should be higher than off-diagonal if vectors discriminate.

| primary_trait | compassion | fairness | harmlessness | honesty |
| --- | --- | --- | --- | --- |
| compassion | 4.7608 | -132.2812 | 200.5060 | -253.6920 |
| fairness | -334.9820 | -0.9327 | 47.3500 | -126.6817 |
| harmlessness | 77.1117 | -71.9341 | 165.8265 | -16.8173 |
| honesty | 253.1075 | 205.1457 | -413.6800 | 397.1889 |

## Diagonal Dominance

Fraction of items where annotated trait projects highest: **0.338** (chance = 0.25)  ✅

## Trait Projection Correlation Matrix

| trait | honesty | harmlessness | fairness | compassion |
| --- | --- | --- | --- | --- |
| honesty | 1.0000 | -0.8698 | 0.8386 | 0.9028 |
| harmlessness | -0.8698 | 1.0000 | -0.9706 | -0.8325 |
| fairness | 0.8386 | -0.9706 | 1.0000 | 0.8228 |
| compassion | 0.9028 | -0.8325 | 0.8228 | 1.0000 |

## Notes

- These diagnostics are sanity checks, not the final reliability/validity study.
- Paraphrase and framing reliability analysis comes in Stage 4.
- Diagonal dominance > chance (0.25) is necessary but not sufficient for validity.
- High inter-trait correlations may indicate a general moral valence factor —
  investigate with factor analysis in Stage 5.
- Weak diagonal dominance does not mean the data is invalid; it may reflect
  genuine overlap in how Gemma represents these moral traits.
