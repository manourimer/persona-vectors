# Synonym Vector Convergent-Validity Control

## Cosine Similarity to Original Vectors

Closest-parent match: 2/4

| synonym_id     | parent_trait   | closest_parent   | closest_matches_parent   |   cosine_honesty |   cosine_harmlessness |   cosine_fairness |   cosine_compassion |
|:---------------|:---------------|:-----------------|:-------------------------|-----------------:|----------------------:|------------------:|--------------------:|
| harm_avoidance | harmlessness   | compassion       | False                    |        0.0298562 |              0.168883 |          0.134859 |            0.408851 |
| impartiality   | fairness       | harmlessness     | False                    |        0.0570557 |              0.181312 |          0.134534 |           -0.447745 |
| truthfulness   | honesty        | honesty          | True                     |        0.777974  |              0.131506 |          0.146749 |           -0.126739 |
| empathy        | compassion     | compassion       | True                     |        0.085888  |             -0.103548 |          0.061418 |            0.720929 |

## Projection Agreement with Parent Trait

| synonym_id     | parent_trait   |   layer |   pearson_r |   spearman_r |   mean_abs_dev |
|:---------------|:---------------|--------:|------------:|-------------:|---------------:|
| harm_avoidance | harmlessness   |      32 |   -0.862571 |    -0.837342 |       17787.2  |
| impartiality   | fairness       |      32 |   -0.952626 |    -0.944529 |       38621.9  |
| truthfulness   | honesty        |      32 |    0.91211  |     0.90291  |        8180.89 |
| empathy        | compassion     |      32 |    0.967203 |     0.958526 |       43836    |
