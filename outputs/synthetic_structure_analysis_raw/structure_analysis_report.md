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

**Items**: 160  |  **Effective dimensionality**: 1.21  |  **Parallel analysis retains**: 1 component(s)

### Correlation matrix

| trait | honesty | harmlessness | fairness | compassion |
|---|---|---|---|---|
| honesty | 1.000 | -0.870 | 0.839 | 0.903 |
| harmlessness | -0.870 | 1.000 | -0.971 | -0.832 |
| fairness | 0.839 | -0.971 | 1.000 | 0.823 |
| compassion | 0.903 | -0.832 | 0.823 | 1.000 |

Mean absolute off-diagonal correlation: **0.873**  
Maximum absolute trait correlation: **0.971** (harmlessness – fairness)

### PCA explained variance

| Component | Eigenvalue | Variance explained | Cumulative |
|---|---|---|---|
| PC1 | 3.619 | 90.5% | 90.5% |
| PC2 | 0.257 | 6.4% | 96.9% |
| PC3 | 0.097 | 2.4% | 99.3% |
| PC4 | 0.026 | 0.7% | 100.0% |

Components for 80% variance: **1**  |  Components for 90% variance: **1**

### PCA loadings

| Variable | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| honesty | -0.499 | 0.423 | -0.740 | 0.160 |
| harmlessness | 0.508 | 0.450 | 0.072 | 0.731 |
| fairness | -0.502 | -0.532 | 0.177 | 0.658 |
| compassion | -0.491 | 0.580 | 0.645 | -0.079 |

### Parallel analysis

| Component | Observed λ | Random 95th pct | Retained? |
|---|---|---|---|
| PC1 | 3.619 | 1.278 | ✅ Yes |
| PC2 | 0.257 | 1.120 | ❌ No |
| PC3 | 0.097 | 1.002 | ❌ No |
| PC4 | 0.026 | 0.908 | ❌ No |

### Interpretation

Single dominant dimension: PC1 explains 90% of variance, effective dimensionality 1.21. The four trait projections largely collapse onto a single latent moral-valence axis at this layer.

---

## Layer 40 (downstream ETHICS best)

**Items**: 160  |  **Effective dimensionality**: 2.09  |  **Parallel analysis retains**: 2 component(s)

### Correlation matrix

| trait | honesty | harmlessness | fairness | compassion |
|---|---|---|---|---|
| honesty | 1.000 | 0.248 | 0.962 | 0.466 |
| harmlessness | 0.248 | 1.000 | 0.085 | 0.718 |
| fairness | 0.962 | 0.085 | 1.000 | 0.314 |
| compassion | 0.466 | 0.718 | 0.314 | 1.000 |

Mean absolute off-diagonal correlation: **0.466**  
Maximum absolute trait correlation: **0.962** (honesty – fairness)

### PCA explained variance

| Component | Eigenvalue | Variance explained | Cumulative |
|---|---|---|---|
| PC1 | 2.428 | 60.7% | 60.7% |
| PC2 | 1.301 | 32.5% | 93.2% |
| PC3 | 0.249 | 6.2% | 99.5% |
| PC4 | 0.022 | 0.5% | 100.0% |

Components for 80% variance: **2**  |  Components for 90% variance: **2**

### PCA loadings

| Variable | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| honesty | -0.580 | 0.361 | -0.092 | -0.724 |
| harmlessness | -0.381 | -0.643 | -0.660 | 0.068 |
| fairness | -0.522 | 0.499 | -0.114 | 0.682 |
| compassion | -0.496 | -0.454 | 0.736 | 0.076 |

### Parallel analysis

| Component | Observed λ | Random 95th pct | Retained? |
|---|---|---|---|
| PC1 | 2.428 | 1.276 | ✅ Yes |
| PC2 | 1.301 | 1.115 | ✅ Yes |
| PC3 | 0.249 | 1.004 | ❌ No |
| PC4 | 0.022 | 0.906 | ❌ No |

### Interpretation

Near-single-dimensional structure: PC1 explains 61% of variance (effective dimensionality 2.09). A weak secondary dimension is present but the dominant direction accounts for most projection variance.

---

## Layer 47

**Items**: 160  |  **Effective dimensionality**: 1.39  |  **Parallel analysis retains**: 1 component(s)

### Correlation matrix

| trait | honesty | harmlessness | fairness | compassion |
|---|---|---|---|---|
| honesty | 1.000 | 0.964 | 0.616 | 0.931 |
| harmlessness | 0.964 | 1.000 | 0.607 | 0.971 |
| fairness | 0.616 | 0.607 | 1.000 | 0.522 |
| compassion | 0.931 | 0.971 | 0.522 | 1.000 |

Mean absolute off-diagonal correlation: **0.769**  
Maximum absolute trait correlation: **0.971** (harmlessness – compassion)

### PCA explained variance

| Component | Eigenvalue | Variance explained | Cumulative |
|---|---|---|---|
| PC1 | 3.345 | 83.6% | 83.6% |
| PC2 | 0.575 | 14.4% | 98.0% |
| PC3 | 0.062 | 1.5% | 99.5% |
| PC4 | 0.018 | 0.5% | 100.0% |

Components for 80% variance: **1**  |  Components for 90% variance: **2**

### PCA loadings

| Variable | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| honesty | -0.532 | 0.156 | 0.782 | 0.286 |
| harmlessness | -0.537 | 0.196 | -0.107 | -0.813 |
| fairness | -0.395 | -0.911 | -0.108 | 0.056 |
| compassion | -0.522 | 0.329 | -0.605 | 0.504 |

### Parallel analysis

| Component | Observed λ | Random 95th pct | Retained? |
|---|---|---|---|
| PC1 | 3.345 | 1.275 | ✅ Yes |
| PC2 | 0.575 | 1.109 | ❌ No |
| PC3 | 0.062 | 0.998 | ❌ No |
| PC4 | 0.018 | 0.909 | ❌ No |

### Interpretation

Single dominant dimension: PC1 explains 84% of variance, effective dimensionality 1.39. The four trait projections largely collapse onto a single latent moral-valence axis at this layer.

---

## Layer 32 vs layer 40 comparison

| Metric | Layer 32 (contrast-selected) | Layer 40 (downstream best) |
|---|---|---|
| PC1 explained variance | 90.5% | 60.7% |
| Effective dimensionality | 1.21 | 2.09 |
| Parallel analysis components | 1 | 2 |
| Mean off-diagonal |corr| | 0.873 | 0.466 |
| Max trait correlation | 0.971 | 0.962 |

Layer 32 was selected by contrast-prompt validation AUC; layer 40 showed stronger ETHICS diagonal dominance. If the two layers show meaningfully different structure, report both and note the divergence.
