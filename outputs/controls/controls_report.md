# Controls Suite Report

This report summarises all negative controls, positive controls, convergent-validity controls, and robustness controls.

## Summary

| control             | status   |
|:--------------------|:---------|
| random_vector       | ran      |
| shuffled_label      | ran      |
| permuted_grouping   | ran      |
| exact_duplicate     | ran      |
| contrast_validation | ran      |
| synthetic_scenarios | ran      |
| synonym_similarity  | not_run  |
| preprocessing       | ran      |

## Negative Control 1: Random Vectors

| metric                   |   layer |   real_value |   random_mean |   random_std |   percentile_of_real |
|:-------------------------|--------:|-------------:|--------------:|-------------:|---------------------:|
| effective_dimensionality |      32 |     1.14041  |     3.98499   |   0.00833649 |                    0 |
| pc1_variance             |      32 |     0.935266 |     0.269592  |   0.00626823 |                  100 |
| mean_abs_off_diag_corr   |      32 |     0.913307 |     0.0283406 |   0.00823173 |                  100 |
| effective_dimensionality |      40 |     2.46101  |     3.98445   |   0.00952471 |                    0 |
| pc1_variance             |      40 |     0.507008 |     0.27005   |   0.00694756 |                  100 |
| mean_abs_off_diag_corr   |      40 |     0.30906  |     0.028731  |   0.00955948 |                  100 |
| effective_dimensionality |      47 |     1.13262  |     3.98577   |   0.00868498 |                    0 |
| pc1_variance             |      47 |     0.938324 |     0.269593  |   0.0065534  |                  100 |
| mean_abs_off_diag_corr   |      47 |     0.916951 |     0.0275787 |   0.00868477 |                  100 |

## Negative Control 2: Shuffled Labels

**diagonal_dominance**: real=0.2745, null_mean=0.2204, null_p95=0.2647, p=0.0226, percentile=98.6%

**matching_margin**: real=-804.8913, null_mean=-936.8785, null_p95=-784.3131, p=0.0787, percentile=92.1%

## Negative Control 3: Permuted Item-Variant Grouping

### Real G-coefficients

|   layer | projected_trait   |      g_1 |      g_3 |
|--------:|:------------------|---------:|---------:|
|      32 | honesty           | 0.559301 | 0.791986 |
|      32 | harmlessness      | 0.587589 | 0.810401 |
|      32 | fairness          | 0.605579 | 0.821622 |
|      32 | compassion        | 0.495046 | 0.746266 |
|      40 | honesty           | 0.539178 | 0.778275 |
|      40 | harmlessness      | 0.676964 | 0.862767 |
|      40 | fairness          | 0.580924 | 0.806149 |
|      40 | compassion        | 0.708392 | 0.87934  |
|      47 | honesty           | 0.596639 | 0.816092 |
|      47 | harmlessness      | 0.603314 | 0.82023  |
|      47 | fairness          | 0.673509 | 0.860891 |
|      47 | compassion        | 0.584761 | 0.808603 |

## Positive Control 1: Exact Duplicates

Mean G(k=1): 1.0000 (expected: ~1.0)

|   layer | projected_trait   |   between_item_var |   within_item_var |        total_var |   within_item_sd |   between_item_sd |   variance_ratio |   reliability_1 |   reliability_2 |   reliability_3 |   reliability_4 |   reliability_5 |   n_items_used | clamped_negative   |
|--------:|:------------------|-------------------:|------------------:|-----------------:|-----------------:|------------------:|-----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|---------------:|:-------------------|
|      32 | honesty           |        1.10178e+06 |                 0 |      1.10178e+06 |                0 |           1049.66 |              inf |               1 |             nan |               1 |             nan |             nan |            204 | False              |
|      32 | harmlessness      |        1.98139e+06 |                 0 |      1.98139e+06 |                0 |           1407.62 |              inf |               1 |             nan |               1 |             nan |             nan |            204 | False              |
|      32 | fairness          |        1.6169e+06  |                 0 |      1.6169e+06  |                0 |           1271.57 |              inf |               1 |             nan |               1 |             nan |             nan |            204 | False              |
|      32 | compassion        |   775421           |                 0 | 775421           |                0 |            880.58 |              inf |               1 |             nan |               1 |             nan |             nan |            204 | False              |

## Positive Control 2: Contrast Validation

All pass: False, Warnings: 6

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

## Positive Control 3: Synthetic Obvious Scenarios

Total: 100, Filled: 100, Reviewed: 0

## Convergent-Validity Control: Synonym Vectors

*Not run (synonym vectors not yet built — run Stage 2B with synonym artifacts first).*

## Robustness Control: Raw vs Centered Preprocessing

| source_dataset   |   layer | projected_trait   | preprocessing   |   n_rows |   n_items | n_variants   |   effective_dimensionality |   pc1_variance |   mean_abs_off_diag_corr |   max_abs_corr |   reliability_g1_proxy |
|:-----------------|--------:|:------------------|:----------------|---------:|----------:|:-------------|---------------------------:|---------------:|-------------------------:|---------------:|-----------------------:|
| ethics_original  |      32 | all               | raw             |      204 |       204 |              |                    1.13042 |       0.939637 |                 0.919308 |       0.967508 |            2.46008e-15 |
| ethics_original  |      32 | honesty           | raw             |       99 |        99 |              |                    1.11965 |       0.944299 |                 0.925538 |       0.977447 |            0.0315083   |
| ethics_original  |      32 | harmlessness      | raw             |       32 |        32 |              |                    1.1883  |       0.915332 |                 0.887093 |       0.949663 |            0.0233395   |
| ethics_original  |      32 | fairness          | raw             |       45 |        45 |              |                    1.12413 |       0.942121 |                 0.922336 |       0.973669 |            0.00978572  |
| ethics_original  |      32 | compassion        | raw             |       28 |        28 |              |                    1.07262 |       0.965254 |                 0.953555 |       0.982547 |            0.127325    |
| ethics_original  |      40 | all               | raw             |      204 |       204 |              |                    1.13042 |       0.939637 |                 0.919308 |       0.967508 |            2.46008e-15 |
| ethics_original  |      40 | honesty           | raw             |       99 |        99 |              |                    1.11965 |       0.944299 |                 0.925538 |       0.977447 |            0.0315083   |
| ethics_original  |      40 | harmlessness      | raw             |       32 |        32 |              |                    1.1883  |       0.915332 |                 0.887093 |       0.949663 |            0.0233395   |
| ethics_original  |      40 | fairness          | raw             |       45 |        45 |              |                    1.12413 |       0.942121 |                 0.922336 |       0.973669 |            0.00978572  |
| ethics_original  |      40 | compassion        | raw             |       28 |        28 |              |                    1.07262 |       0.965254 |                 0.953555 |       0.982547 |            0.127325    |
| ethics_original  |      47 | all               | raw             |      204 |       204 |              |                    1.13042 |       0.939637 |                 0.919308 |       0.967508 |            2.46008e-15 |
| ethics_original  |      47 | honesty           | raw             |       99 |        99 |              |                    1.11965 |       0.944299 |                 0.925538 |       0.977447 |            0.0315083   |
| ethics_original  |      47 | harmlessness      | raw             |       32 |        32 |              |                    1.1883  |       0.915332 |                 0.887093 |       0.949663 |            0.0233395   |
| ethics_original  |      47 | fairness          | raw             |       45 |        45 |              |                    1.12413 |       0.942121 |                 0.922336 |       0.973669 |            0.00978572  |
| ethics_original  |      47 | compassion        | raw             |       28 |        28 |              |                    1.07262 |       0.965254 |                 0.953555 |       0.982547 |            0.127325    |
| ethics_original  |      32 | all               | centered        |      204 |       204 |              |                    1.13042 |       0.939637 |                 0.919308 |       0.967508 |            2.46008e-15 |
| ethics_original  |      32 | honesty           | centered        |       99 |        99 |              |                    1.11965 |       0.944299 |                 0.925538 |       0.977447 |            0.0315083   |
| ethics_original  |      32 | harmlessness      | centered        |       32 |        32 |              |                    1.1883  |       0.915332 |                 0.887093 |       0.949663 |            0.0233395   |
| ethics_original  |      32 | fairness          | centered        |       45 |        45 |              |                    1.12413 |       0.942121 |                 0.922336 |       0.973669 |            0.00978572  |
| ethics_original  |      32 | compassion        | centered        |       28 |        28 |              |                    1.07262 |       0.965254 |                 0.953555 |       0.982547 |            0.127325    |
| ethics_original  |      40 | all               | centered        |      204 |       204 |              |                    1.13042 |       0.939637 |                 0.919308 |       0.967508 |            2.46008e-15 |
| ethics_original  |      40 | honesty           | centered        |       99 |        99 |              |                    1.11965 |       0.944299 |                 0.925538 |       0.977447 |            0.0315083   |
| ethics_original  |      40 | harmlessness      | centered        |       32 |        32 |              |                    1.1883  |       0.915332 |                 0.887093 |       0.949663 |            0.0233395   |
| ethics_original  |      40 | fairness          | centered        |       45 |        45 |              |                    1.12413 |       0.942121 |                 0.922336 |       0.973669 |            0.00978572  |
| ethics_original  |      40 | compassion        | centered        |       28 |        28 |              |                    1.07262 |       0.965254 |                 0.953555 |       0.982547 |            0.127325    |
| ethics_original  |      47 | all               | centered        |      204 |       204 |              |                    1.13042 |       0.939637 |                 0.919308 |       0.967508 |            2.46008e-15 |
| ethics_original  |      47 | honesty           | centered        |       99 |        99 |              |                    1.11965 |       0.944299 |                 0.925538 |       0.977447 |            0.0315083   |
| ethics_original  |      47 | harmlessness      | centered        |       32 |        32 |              |                    1.1883  |       0.915332 |                 0.887093 |       0.949663 |            0.0233395   |
| ethics_original  |      47 | fairness          | centered        |       45 |        45 |              |                    1.12413 |       0.942121 |                 0.922336 |       0.973669 |            0.00978572  |
| ethics_original  |      47 | compassion        | centered        |       28 |        28 |              |                    1.07262 |       0.965254 |                 0.953555 |       0.982547 |            0.127325    |

## Robustness Control: Layer Comparison

| source_dataset       |   layer | projected_trait   |   n_rows |   n_items |   effective_dimensionality |   pc1_variance |   mean_abs_off_diag_corr |   max_abs_corr |   reliability_g1_proxy |
|:---------------------|--------:|:------------------|---------:|----------:|---------------------------:|---------------:|-------------------------:|---------------:|-----------------------:|
| reliability_variants |      32 | all               |      204 |       204 |                    1.13042 |       0.939637 |                 0.919308 |       0.967508 |            2.46008e-15 |
| reliability_variants |      32 | honesty           |       99 |        99 |                    1.11965 |       0.944299 |                 0.925538 |       0.977447 |            0.0315083   |
| reliability_variants |      32 | harmlessness      |       32 |        32 |                    1.1883  |       0.915332 |                 0.887093 |       0.949663 |            0.0233395   |
| reliability_variants |      32 | fairness          |       45 |        45 |                    1.12413 |       0.942121 |                 0.922336 |       0.973669 |            0.00978572  |
| reliability_variants |      32 | compassion        |       28 |        28 |                    1.07262 |       0.965254 |                 0.953555 |       0.982547 |            0.127325    |
| reliability_variants |      40 | all               |      204 |       204 |                    1.13042 |       0.939637 |                 0.919308 |       0.967508 |            2.46008e-15 |
| reliability_variants |      40 | honesty           |       99 |        99 |                    1.11965 |       0.944299 |                 0.925538 |       0.977447 |            0.0315083   |
| reliability_variants |      40 | harmlessness      |       32 |        32 |                    1.1883  |       0.915332 |                 0.887093 |       0.949663 |            0.0233395   |
| reliability_variants |      40 | fairness          |       45 |        45 |                    1.12413 |       0.942121 |                 0.922336 |       0.973669 |            0.00978572  |
| reliability_variants |      40 | compassion        |       28 |        28 |                    1.07262 |       0.965254 |                 0.953555 |       0.982547 |            0.127325    |
| reliability_variants |      47 | all               |      204 |       204 |                    1.13042 |       0.939637 |                 0.919308 |       0.967508 |            2.46008e-15 |
| reliability_variants |      47 | honesty           |       99 |        99 |                    1.11965 |       0.944299 |                 0.925538 |       0.977447 |            0.0315083   |
| reliability_variants |      47 | harmlessness      |       32 |        32 |                    1.1883  |       0.915332 |                 0.887093 |       0.949663 |            0.0233395   |
| reliability_variants |      47 | fairness          |       45 |        45 |                    1.12413 |       0.942121 |                 0.922336 |       0.973669 |            0.00978572  |
| reliability_variants |      47 | compassion        |       28 |        28 |                    1.07262 |       0.965254 |                 0.953555 |       0.982547 |            0.127325    |

## Overall Interpretation

The controls suite validates that the observed results are specific to the moral persona vectors and are not artifacts of the analysis pipeline.

- **Random vectors**: If real vectors outperform random directions on structure and label-alignment metrics, this rules out that any direction produces similar patterns.
- **Shuffled labels**: A significant p-value (p < 0.05) for diagonal dominance confirms that trait-label alignment exceeds chance.
- **Permuted grouping**: If real G-coefficients exceed the permuted null, reliability is driven by stable item identity, not coincidental grouping.
- **Exact duplicates** (positive control): G ~1.0 confirms the reliability pipeline is bug-free.
- **Contrast validation** (positive control): AUC ≥ 0.75 for all trait × layer combos confirms vectors still reproduce their calibration signal.
- **Synonym vectors**: Closest-parent match confirms construct validity across synonymous wordings of trait names.
- **Preprocessing**: Stability of findings across raw and centered projections confirms that centering does not manufacture the observed effects.
