# Stage 4D: Reliability / Generalizability Analysis Report

## Dataset Summary

- Total items in bank: 204
- Items used in analysis: 188
- Items dropped (too few variants): 16
- Total variant observations: 9132
- Layers analyzed: [32, 40, 47]
- Projected traits: ['honesty', 'harmlessness', 'fairness', 'compassion']

## Reliability by Layer × Projected Trait

Single variant (k=1) and average of 3 variants (k=3).

|   layer | projected_trait   |   reliability_1 |   reliability_3 |
|--------:|:------------------|----------------:|----------------:|
|      32 | honesty           |           0.552 |           0.787 |
|      32 | harmlessness      |           0.585 |           0.808 |
|      32 | fairness          |           0.594 |           0.815 |
|      32 | compassion        |           0.498 |           0.748 |
|      40 | honesty           |           0.516 |           0.762 |
|      40 | harmlessness      |           0.68  |           0.865 |
|      40 | fairness          |           0.559 |           0.792 |
|      40 | compassion        |           0.701 |           0.876 |
|      47 | honesty           |           0.595 |           0.815 |
|      47 | harmlessness      |           0.602 |           0.819 |
|      47 | fairness          |           0.67  |           0.859 |
|      47 | compassion        |           0.587 |           0.81  |

## Best / Worst at Primary Layer (32)

- **Best**: `fairness` = 0.594
- **Worst**: `compassion` = 0.498

## Layer 32 vs Layer 40 Comparison

- Mean reliability (k=1) at layer 32: 0.557
- Mean reliability (k=1) at layer 40: 0.614

## D-Study: Reliability Improves with More Paraphrases

Primary layer (32), all projected traits.

|   n_paraphrases |   compassion |   fairness |   harmlessness |   honesty |
|----------------:|-------------:|-----------:|---------------:|----------:|
|               1 |        0.498 |      0.594 |          0.585 |     0.552 |
|               2 |        0.665 |      0.746 |          0.738 |     0.711 |
|               3 |        0.748 |      0.815 |          0.808 |     0.787 |
|               4 |        0.799 |      0.854 |          0.849 |     0.831 |
|               5 |        0.832 |      0.88  |          0.876 |     0.86  |

## Interpretation

Reliability (ICC/G-coefficient) measures the proportion of total projection variance attributable to stable between-item differences versus wording noise. Values > 0.70 indicate adequate generalizability; values > 0.85 indicate good generalizability. The D-study shows how reliability improves as more paraphrases are averaged.

## Caveats

- This is a one-facet G-theory analysis (facet = paraphrase). Framing effects are not modeled.
- Centering removes the cross-item mean activation; all projections are deviation scores.
- Negative between-item variance estimates are clamped to zero (can occur when within-item noise dominates).
- Results depend on the quality and diversity of the generated paraphrases.
