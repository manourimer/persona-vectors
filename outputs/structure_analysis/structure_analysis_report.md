# Stage 4A — Projection Structure Analysis

> **RQ1**: Do the four morally relevant persona-vector projections behave like
> one latent 'morality' dimension, or several separable dimensions?

**Methodology**: PCA and correlation analysis on mean-centered ETHICS projection
matrices (Stage 3). Parallel analysis with permutation estimates random baseline.
Factor analysis skipped if `factor_analyzer` is not installed.

> ⚠ **Factor-analysis caution**: With only four observed variables at most
> two factors are estimable without Heywood cases. Treat factor analysis
> outputs as tentative; PCA, correlation structure, and effective dimensionality
> are the primary evidence.

**Layer notes**:
- Layer 32: contrast-validation-selected (Stage 2B AUC on held-out contrastive prompts)
- Layer 40: strongest downstream ETHICS diagonal dominance

---

## Layer 32 (contrast-validation-selected)

**Items**: 204  |  **Effective dimensionality**: 1.13  |  **Parallel analysis retains**: 1 component(s)

### Correlation matrix

| trait | honesty | harmlessness | fairness | compassion |
|---|---|---|---|---|
| honesty | 1.000 | -0.936 | 0.935 | 0.916 |
| harmlessness | -0.936 | 1.000 | -0.968 | -0.895 |
| fairness | 0.935 | -0.968 | 1.000 | 0.866 |
| compassion | 0.916 | -0.895 | 0.866 | 1.000 |

Mean absolute off-diagonal correlation: **0.919**  
Maximum absolute trait correlation: **0.968** (harmlessness – fairness)

### PCA explained variance

| Component | Eigenvalue | Variance explained | Cumulative |
|---|---|---|---|
| PC1 | 3.759 | 94.0% | 94.0% |
| PC2 | 0.151 | 3.8% | 97.7% |
| PC3 | 0.062 | 1.6% | 99.3% |
| PC4 | 0.029 | 0.7% | 100.0% |

Components for 80% variance: **1**  |  Components for 90% variance: **1**

### PCA loadings

| Variable | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| honesty | -0.504 | 0.065 | 0.846 | 0.164 |
| harmlessness | 0.506 | 0.319 | 0.411 | -0.688 |
| fairness | -0.502 | -0.516 | -0.127 | -0.683 |
| compassion | -0.489 | 0.792 | -0.317 | -0.180 |

### Parallel analysis

| Component | Observed λ | Random 95th pct | Retained? |
|---|---|---|---|
| PC1 | 3.759 | 1.248 | ✅ Yes |
| PC2 | 0.151 | 1.100 | ❌ No |
| PC3 | 0.062 | 1.001 | ❌ No |
| PC4 | 0.029 | 0.918 | ❌ No |

### Interpretation

Single dominant dimension: PC1 explains 94% of variance, effective dimensionality 1.13. The four trait projections largely collapse onto a single latent moral-valence axis at this layer.

---

## Layer 40 (downstream ETHICS best)

**Items**: 204  |  **Effective dimensionality**: 2.46  |  **Parallel analysis retains**: 2 component(s)

### Correlation matrix

| trait | honesty | harmlessness | fairness | compassion |
|---|---|---|---|---|
| honesty | 1.000 | 0.115 | 0.990 | -0.260 |
| harmlessness | 0.115 | 1.000 | 0.060 | 0.290 |
| fairness | 0.990 | 0.060 | 1.000 | -0.319 |
| compassion | -0.260 | 0.290 | -0.319 | 1.000 |

Mean absolute off-diagonal correlation: **0.339**  
Maximum absolute trait correlation: **0.990** (honesty – fairness)

### PCA explained variance

| Component | Eigenvalue | Variance explained | Cumulative |
|---|---|---|---|
| PC1 | 2.138 | 53.4% | 53.4% |
| PC2 | 1.251 | 31.3% | 84.7% |
| PC3 | 0.603 | 15.1% | 99.8% |
| PC4 | 0.007 | 0.2% | 100.0% |

Components for 80% variance: **2**  |  Components for 90% variance: **3**

### PCA loadings

| Variable | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| honesty | -0.662 | -0.170 | -0.200 | -0.702 |
| harmlessness | -0.017 | -0.782 | 0.622 | 0.028 |
| fairness | -0.671 | -0.107 | -0.185 | 0.710 |
| compassion | 0.335 | -0.590 | -0.734 | 0.036 |

### Parallel analysis

| Component | Observed λ | Random 95th pct | Retained? |
|---|---|---|---|
| PC1 | 2.138 | 1.249 | ✅ Yes |
| PC2 | 1.251 | 1.092 | ✅ Yes |
| PC3 | 0.603 | 0.998 | ❌ No |
| PC4 | 0.007 | 0.922 | ❌ No |

### Interpretation

Near-single-dimensional structure: PC1 explains 53% of variance (effective dimensionality 2.46). A weak secondary dimension is present but the dominant direction accounts for most projection variance.

---

## Layer 47

**Items**: 204  |  **Effective dimensionality**: 1.15  |  **Parallel analysis retains**: 1 component(s)

### Correlation matrix

| trait | honesty | harmlessness | fairness | compassion |
|---|---|---|---|---|
| honesty | 1.000 | 0.982 | 0.886 | 0.929 |
| harmlessness | 0.982 | 1.000 | 0.872 | 0.960 |
| fairness | 0.886 | 0.872 | 1.000 | 0.807 |
| compassion | 0.929 | 0.960 | 0.807 | 1.000 |

Mean absolute off-diagonal correlation: **0.906**  
Maximum absolute trait correlation: **0.982** (honesty – harmlessness)

### PCA explained variance

| Component | Eigenvalue | Variance explained | Cumulative |
|---|---|---|---|
| PC1 | 3.722 | 93.0% | 93.0% |
| PC2 | 0.209 | 5.2% | 98.3% |
| PC3 | 0.058 | 1.4% | 99.7% |
| PC4 | 0.012 | 0.3% | 100.0% |

Components for 80% variance: **1**  |  Components for 90% variance: **1**

### PCA loadings

| Variable | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| honesty | -0.511 | 0.068 | 0.655 | -0.552 |
| harmlessness | -0.513 | 0.212 | 0.247 | 0.794 |
| fairness | -0.478 | -0.831 | -0.285 | 0.002 |
| compassion | -0.497 | 0.510 | -0.654 | -0.254 |

### Parallel analysis

| Component | Observed λ | Random 95th pct | Retained? |
|---|---|---|---|
| PC1 | 3.722 | 1.250 | ✅ Yes |
| PC2 | 0.209 | 1.102 | ❌ No |
| PC3 | 0.058 | 0.999 | ❌ No |
| PC4 | 0.012 | 0.917 | ❌ No |

### Interpretation

Single dominant dimension: PC1 explains 93% of variance, effective dimensionality 1.15. The four trait projections largely collapse onto a single latent moral-valence axis at this layer.

---

## Layer 32 vs layer 40 comparison

| Metric | Layer 32 (contrast-selected) | Layer 40 (downstream best) |
|---|---|---|
| PC1 explained variance | 94.0% | 53.4% |
| Effective dimensionality | 1.13 | 2.46 |
| Parallel analysis components | 1 | 2 |
| Mean off-diagonal |corr| | 0.919 | 0.339 |
| Max trait correlation | 0.968 | 0.990 |

Layer 32 was selected by contrast-prompt validation AUC; layer 40 showed stronger ETHICS diagonal dominance. If the two layers show meaningfully different structure, report both and note the divergence.
