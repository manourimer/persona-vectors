# Contrast Validation Positive Control
AUC threshold: 0.75

All pass: False
Warnings: 6

## AUC Table

| trait        |   layer |      auc | passes_threshold   |   auc_threshold |
|:-------------|--------:|---------:|:-------------------|----------------:|
| honesty      |      16 | 0.603785 | False              |            0.75 |
| honesty      |      24 | 0.754274 | True               |            0.75 |
| honesty      |      28 | 0.962302 | True               |            0.75 |
| honesty      |      32 | 0.935745 | True               |            0.75 |
| honesty      |      40 | 0.83837  | True               |            0.75 |
| honesty      |      47 | 0.973291 | True               |            0.75 |
| harmlessness |      16 | 0.705329 | False              |            0.75 |
| harmlessness |      24 | 0.730369 | False              |            0.75 |
| harmlessness |      28 | 0.755208 | True               |            0.75 |
| harmlessness |      32 | 0.887821 | True               |            0.75 |
| harmlessness |      40 | 1        | True               |            0.75 |
| harmlessness |      47 | 0.869992 | True               |            0.75 |
| fairness     |      16 | 0.59451  | False              |            0.75 |
| fairness     |      24 | 0.681307 | False              |            0.75 |
| fairness     |      28 | 0.703791 | False              |            0.75 |
| fairness     |      32 | 0.808366 | True               |            0.75 |
| fairness     |      40 | 0.788235 | True               |            0.75 |
| fairness     |      47 | 0.916863 | True               |            0.75 |
| compassion   |      16 | 0.93715  | True               |            0.75 |
| compassion   |      24 | 0.952597 | True               |            0.75 |
| compassion   |      28 | 0.984855 | True               |            0.75 |
| compassion   |      32 | 0.985461 | True               |            0.75 |
| compassion   |      40 | 1        | True               |            0.75 |
| compassion   |      47 | 0.999394 | True               |            0.75 |

## Warnings

          trait  layer       auc              warning
0       honesty     16  0.603785  AUC below threshold
1  harmlessness     16  0.705329  AUC below threshold
2  harmlessness     24  0.730369  AUC below threshold
3      fairness     16  0.594510  AUC below threshold
4      fairness     24  0.681307  AUC below threshold
5      fairness     28  0.703791  AUC below threshold
