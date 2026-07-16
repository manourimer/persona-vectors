# Stage 4C: Reliability Variant Projection Diagnostics

## Summary

- Items: 204
- Total variants: 761
- Originals: 612
- Paraphrases: 1671
- Missing variants (below median): 22

## Count Breakdowns

**By variant type:**
  - paraphrase: 557
  - original: 204

**By paraphrase_id:**
  - original: 204
  - p2: 186
  - p3: 186
  - p1: 185

**By framing:**
  - neutral: 761

**By primary_trait:**
  - honesty: 365
  - fairness: 171
  - harmlessness: 128
  - compassion: 97

## Projection Statistics (mean/std/min/max by trait × layer)

  - compassion layer 32: mean=-0.0000, std=890.5409, min=-3137.2455, max=2839.8795
  - compassion layer 40: mean=0.0000, std=468.2293, min=-1271.7309, max=1940.1831
  - compassion layer 47: mean=-0.0000, std=2070.5064, min=-7535.8727, max=6350.6976
  - fairness layer 32: mean=0.0000, std=1217.4076, min=-3556.1770, max=3409.5144
  - fairness layer 40: mean=-0.0000, std=3055.8726, min=-12227.7904, max=8801.0534
  - fairness layer 47: mean=-0.0000, std=1734.1612, min=-6240.7191, max=4306.1559
  - harmlessness layer 32: mean=-0.0000, std=1404.1697, min=-4083.7278, max=3901.1491
  - harmlessness layer 40: mean=-0.0000, std=630.0108, min=-1857.2551, max=1888.0143
  - harmlessness layer 47: mean=-0.0000, std=4393.9736, min=-16054.6082, max=12158.5911
  - honesty layer 32: mean=-0.0000, std=1040.9868, min=-3071.2905, max=3516.4009
  - honesty layer 40: mean=0.0000, std=3957.1656, min=-16333.1767, max=11981.0850
  - honesty layer 47: mean=0.0000, std=3616.9859, min=-13038.2561, max=9720.5643

## Correlation Matrices

### Layer 32

|                         |   projection_honesty |   projection_harmlessness |   projection_fairness |   projection_compassion |
|:------------------------|---------------------:|--------------------------:|----------------------:|------------------------:|
| projection_honesty      |                1     |                    -0.935 |                 0.937 |                   0.901 |
| projection_harmlessness |               -0.935 |                     1     |                -0.978 |                  -0.869 |
| projection_fairness     |                0.937 |                    -0.978 |                 1     |                   0.86  |
| projection_compassion   |                0.901 |                    -0.869 |                 0.86  |                   1     |

### Layer 40

|                         |   projection_honesty |   projection_harmlessness |   projection_fairness |   projection_compassion |
|:------------------------|---------------------:|--------------------------:|----------------------:|------------------------:|
| projection_honesty      |                1     |                     0.098 |                 0.989 |                  -0.105 |
| projection_harmlessness |                0.098 |                     1     |                 0.015 |                   0.469 |
| projection_fairness     |                0.989 |                     0.015 |                 1     |                  -0.179 |
| projection_compassion   |               -0.105 |                     0.469 |                -0.179 |                   1     |

### Layer 47

|                         |   projection_honesty |   projection_harmlessness |   projection_fairness |   projection_compassion |
|:------------------------|---------------------:|--------------------------:|----------------------:|------------------------:|
| projection_honesty      |                1     |                     0.986 |                 0.898 |                   0.946 |
| projection_harmlessness |                0.986 |                     1     |                 0.874 |                   0.97  |
| projection_fairness     |                0.898 |                     0.874 |                 1     |                   0.828 |
| projection_compassion   |                0.946 |                     0.97  |                 0.828 |                   1     |


## Warnings

- Trait 'honesty' has 85 item(s) with within-item std > 2× median (2739.4336) — potentially unstable.
- Trait 'harmlessness' has 162 item(s) with within-item std > 2× median (1448.8115) — potentially unstable.
- Trait 'fairness' has 88 item(s) with within-item std > 2× median (1687.3647) — potentially unstable.
- Trait 'compassion' has 120 item(s) with within-item std > 2× median (1003.5990) — potentially unstable.
