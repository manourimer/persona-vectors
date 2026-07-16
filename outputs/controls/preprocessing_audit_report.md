# Preprocessing Control Audit Report

## Candidate file metrics

| source               |   layer | path                                                                                                                                            |   n_rows |   n_items |   n_variants |   effective_dimensionality |   pc1_variance |   mean_abs_off_diag_corr |   max_abs_corr |
|:---------------------|--------:|:------------------------------------------------------------------------------------------------------------------------------------------------|---------:|----------:|-------------:|---------------------------:|---------------:|-------------------------:|---------------:|
| ethics_raw_wide      |      32 | /Users/manouchehrrimer/Desktop/BlueDot_Perona_Vector_Project/outputs/ethics_projection/ethics_trait_projections_raw_wide.parquet                |      204 |       204 |          nan |                    3.86987 |       0.310822 |                 0.084579 |       0.186332 |
| ethics_centered_wide |      32 | /Users/manouchehrrimer/Desktop/BlueDot_Perona_Vector_Project/outputs/ethics_projection/ethics_trait_projections_centered_wide.parquet           |      204 |       204 |          nan |                    3.86987 |       0.310822 |                 0.084579 |       0.186332 |
| rel_raw_wide         |      32 | /Users/manouchehrrimer/Desktop/BlueDot_Perona_Vector_Project/outputs/reliability_projection/reliability_trait_projections_wide_raw.parquet      |      761 |       204 |          761 |                    1.14041 |       0.935266 |                 0.913307 |       0.978061 |
| rel_raw_wide         |      40 | /Users/manouchehrrimer/Desktop/BlueDot_Perona_Vector_Project/outputs/reliability_projection/reliability_trait_projections_wide_raw.parquet      |      761 |       204 |          761 |                    2.46101 |       0.507008 |                 0.30906  |       0.988784 |
| rel_raw_wide         |      47 | /Users/manouchehrrimer/Desktop/BlueDot_Perona_Vector_Project/outputs/reliability_projection/reliability_trait_projections_wide_raw.parquet      |      761 |       204 |          761 |                    1.13262 |       0.938324 |                 0.916951 |       0.986104 |
| rel_centered_wide    |      32 | /Users/manouchehrrimer/Desktop/BlueDot_Perona_Vector_Project/outputs/reliability_projection/reliability_trait_projections_wide_centered.parquet |      761 |       204 |          761 |                    1.14041 |       0.935266 |                 0.913307 |       0.978061 |
| rel_centered_wide    |      40 | /Users/manouchehrrimer/Desktop/BlueDot_Perona_Vector_Project/outputs/reliability_projection/reliability_trait_projections_wide_centered.parquet |      761 |       204 |          761 |                    2.46101 |       0.507008 |                 0.30906  |       0.988784 |
| rel_centered_wide    |      47 | /Users/manouchehrrimer/Desktop/BlueDot_Perona_Vector_Project/outputs/reliability_projection/reliability_trait_projections_wide_centered.parquet |      761 |       204 |          761 |                    1.13262 |       0.938324 |                 0.916951 |       0.986104 |


## Stage 4A reference (centered ETHICS originals)

|   layer |   effective_dimensionality |   first_pc_variance |   mean_abs_off_diag_corr |
|--------:|---------------------------:|--------------------:|-------------------------:|
|      32 |                    3.86987 |            0.310822 |                 0.084579 |
|      40 |                    3.82706 |            0.326994 |                 0.101596 |
|      47 |                    3.78464 |            0.341564 |                 0.100046 |


## Bug diagnosis

The `run_preprocessing_controls.py` script loads `reliability_trait_projections_wide_centered.parquet`
and `reliability_trait_projections_wide_raw.parquet` for BOTH the preprocessing comparison
AND the layer robustness analysis.  These files contain 2283 rows (761 variants × 3 layers).

When filtered to layer 32 (~761 rows of reliability variants), the 4 trait projections are
highly inter-correlated (all variants of the same item cluster together), yielding:
  ED ≈ 1.14, mean|r| ≈ 0.91 — matching the broken output exactly.

The correct files for original ETHICS preprocessing comparison are:
  - ethics_trait_projections_raw_wide.parquet (204 rows, no layer column)
  - ethics_trait_projections_centered_wide.parquet (204 rows, no layer column)
These yield ED ≈ 3.87, mean|r| ≈ 0.085 at layer 32 — matching Stage 4A.
