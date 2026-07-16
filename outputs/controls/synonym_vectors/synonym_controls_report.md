# Synonym Vector Convergent-Validity Control

## Cosine Similarity to Original Vectors

Closest-parent match: 2/4

| synonym_id     | parent_trait   | closest_parent   | closest_matches_parent   |   cosine_honesty |   cosine_harmlessness |   cosine_fairness |   cosine_compassion |
|:---------------|:---------------|:-----------------|:-------------------------|-----------------:|----------------------:|------------------:|--------------------:|
| impartiality   | fairness       | harmlessness     | False                    |        0.0570557 |              0.181312 |          0.134534 |           -0.447745 |
| empathy        | compassion     | compassion       | True                     |        0.085888  |             -0.103548 |          0.061418 |            0.720929 |
| harm_avoidance | harmlessness   | compassion       | False                    |        0.0298562 |              0.168883 |          0.134859 |            0.408851 |
| truthfulness   | honesty        | honesty          | True                     |        0.777974  |              0.131506 |          0.146749 |           -0.126739 |

## Projection Agreement with Parent Trait

| synonym_id     | parent_trait   |   layer |   pearson_r |   spearman_r |   mean_abs_dev |
|:---------------|:---------------|--------:|------------:|-------------:|---------------:|
| impartiality   | fairness       |      32 |   0.0646642 |    0.0437724 |       38621.9  |
| empathy        | compassion     |      32 |  -0.0596335 |   -0.093213  |       43836    |
| harm_avoidance | harmlessness   |      32 |  -0.0214676 |   -0.054477  |       17787.2  |
| truthfulness   | honesty        |      32 |   0.131891  |    0.100177  |        8180.89 |
