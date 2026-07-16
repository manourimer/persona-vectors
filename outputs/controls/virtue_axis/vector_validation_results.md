# Persona Vector Held-Out Validation Results

> These results use CONTRAST ARTIFACT validation responses (not ETHICS items).
> Proceed to Stage 3 (ETHICS projection) only after all trait vectors pass AUC threshold.

## Virtue_axis

| Layer | AUC | Accuracy | Mean Pos Proj | Mean Neg Proj | Cohen's d | Pass? |
|---|---|---|---|---|---|---|
| 16 | 0.898 | 0.803 | 21548.527 | 20685.604 | 1.800 | ✅ |
| 24 | 0.927 | 0.820 | 51146.953 | 48747.898 | 2.053 | ✅ |
| 28 | 1.000 | 0.978 | 24831.289 | 22025.350 | 4.118 | ✅ |
| 32 | 1.000 | 0.995 | 2118.656 | -1158.839 | 6.310 | ✅ |
| 40 | 0.941 | 0.847 | -57979.668 | -64125.328 | 2.407 | ✅ |
| 47 | 1.000 | 0.995 | 5016.992 | -4417.071 | 6.416 | ✅ |

## Recommended Layer

Layer **32** has the highest mean AUC across all traits.

Update `model.target_layer` in `configs/mvp_experiment.yaml` to this value.