# Results summary

## E1/E2/E3 main (mean over seeds)

| domain     | method           |   acc |    f1 |   auc |   ece | rule_f1   | rule_rec   | logic_fid   |
|:-----------|:-----------------|------:|------:|------:|------:|:----------|:-----------|:------------|
| industrial | DecisionTree     | 0.905 | 0.567 | 0.784 | 0.018 | 0.340     | 0.333      | 0.859       |
| industrial | ESS(discrete)    | 0.903 | 0.605 | 0.748 | 0.086 | 0.333     | 0.333      | 0.972       |
| industrial | ESS(soft)        | 0.916 | 0.623 | 0.757 | 0.071 | n/a       | n/a        | n/a         |
| industrial | GradientBoosting | 0.911 | 0.581 | 0.778 | 0.024 | 0.278     | 0.333      | 0.819       |
| industrial | LogReg           | 0.838 | 0.435 | 0.754 | 0.017 | 0.000     | 0.000      | 0.794       |
| industrial | MLP              | 0.879 | 0.514 | 0.773 | 0.022 | 0.111     | 0.111      | 0.879       |
| industrial | NEXS(proposed)   | 0.88  | 0.537 | 0.776 | 0.019 | 0.229     | 0.222      | 0.859       |
| industrial | RandomForest     | 0.915 | 0.624 | 0.788 | 0.016 | 0.162     | 0.222      | 0.657       |
| industrial | SemanticLoss     | 0.878 | 0.529 | 0.761 | 0.018 | 0.190     | 0.222      | 0.871       |
| medical    | DecisionTree     | 0.901 | 0.741 | 0.893 | 0.029 | 0.340     | 0.333      | 0.929       |
| medical    | ESS(discrete)    | 0.883 | 0.748 | 0.866 | 0.094 | 0.333     | 0.333      | 0.953       |
| medical    | ESS(soft)        | 0.903 | 0.771 | 0.878 | 0.076 | n/a       | n/a        | n/a         |
| medical    | GradientBoosting | 0.938 | 0.842 | 0.907 | 0.018 | 0.265     | 0.333      | 0.843       |
| medical    | LogReg           | 0.763 | 0.554 | 0.812 | 0.04  | 0.000     | 0.000      | 0.770       |
| medical    | MLP              | 0.892 | 0.733 | 0.873 | 0.028 | 0.222     | 0.222      | 0.896       |
| medical    | NEXS(proposed)   | 0.875 | 0.706 | 0.88  | 0.029 | 0.333     | 0.333      | 0.908       |
| medical    | RandomForest     | 0.938 | 0.839 | 0.901 | 0.022 | 0.278     | 0.333      | 0.844       |
| medical    | SemanticLoss     | 0.88  | 0.703 | 0.866 | 0.028 | 0.111     | 0.111      | 0.872       |

## Key deltas (NEXS vs baselines), mean over domains

- **NEXS - MLP**: acc=-0.008, auc=+0.005, ece=-0.001, f1=-0.002, logic_fid=-0.004, rule_f1=+0.114
- **NEXS - SemanticLoss**: acc=-0.001, auc=+0.014, ece=+0.001, f1=+0.006, logic_fid=+0.012, rule_f1=+0.130
- **NEXS - RandomForest**: acc=-0.049, auc=-0.017, ece=+0.005, f1=-0.110, logic_fid=+0.134, rule_f1=+0.061
- **NEXS - GradientBoosting**: acc=-0.047, auc=-0.015, ece=+0.003, f1=-0.090, logic_fid=+0.053, rule_f1=+0.010

## E4 data efficiency

| domain     | method         |   150 |   300 |   600 |   1200 |   2400 |
|:-----------|:---------------|------:|------:|------:|-------:|-------:|
| industrial | ESS(discrete)  | 0.624 | 0.624 | 0.624 |  0.624 |  0.624 |
| industrial | MLP            | 0.413 | 0.408 | 0.35  |  0.435 |  0.476 |
| industrial | NEXS(proposed) | 0.428 | 0.548 | 0.54  |  0.497 |  0.561 |
| industrial | RandomForest   | 0.491 | 0.509 | 0.585 |  0.605 |  0.616 |
| industrial | SemanticLoss   | 0.356 | 0.364 | 0.348 |  0.483 |  0.531 |
| medical    | ESS(discrete)  | 0.742 | 0.742 | 0.742 |  0.742 |  0.742 |
| medical    | MLP            | 0.544 | 0.61  | 0.585 |  0.623 |  0.643 |
| medical    | NEXS(proposed) | 0.581 | 0.647 | 0.643 |  0.661 |  0.698 |
| medical    | RandomForest   | 0.691 | 0.748 | 0.734 |  0.793 |  0.836 |
| medical    | SemanticLoss   | 0.536 | 0.638 | 0.616 |  0.597 |  0.671 |

## E5 shift

| domain     | shift     |   ESS(discrete) |   MLP |   NEXS(proposed) |   RandomForest |   SemanticLoss |
|:-----------|:----------|----------------:|------:|-----------------:|---------------:|---------------:|
| industrial | covariate |           0.946 | 0.884 |            0.866 |          0.966 |          0.87  |
| industrial | novel     |           0.976 | 0.954 |            0.945 |          0.988 |          0.942 |
| medical    | covariate |           0.948 | 0.92  |            0.905 |          0.958 |          0.918 |
| medical    | novel     |           0.932 | 0.939 |            0.932 |          0.993 |          0.936 |

## E6 ablation

| domain     | variant                |   acc |    f1 |   logic_fid |
|:-----------|:-----------------------|------:|------:|------------:|
| industrial | NEXS(full)             | 0.877 | 0.509 |       0.866 |
| industrial | NEXS(no expert prior)  | 0.88  | 0.533 |       0.859 |
| industrial | NEXS(no rule layer)    | 0.879 | 0.514 |       0.879 |
| industrial | NEXS(no semantic loss) | 0.88  | 0.514 |       0.871 |
| industrial | NEXS(rule layer only)  | 0.883 | 0.56  |       0.843 |
| medical    | NEXS(full)             | 0.879 | 0.708 |       0.899 |
| medical    | NEXS(no expert prior)  | 0.882 | 0.716 |       0.901 |
| medical    | NEXS(no rule layer)    | 0.892 | 0.733 |       0.896 |
| medical    | NEXS(no semantic loss) | 0.88  | 0.718 |       0.902 |
| medical    | NEXS(rule layer only)  | 0.861 | 0.68  |       0.912 |

## E7 latency

| domain     | method         |   us_per_sample |
|:-----------|:---------------|----------------:|
| industrial | ESS(discrete)  |           0.729 |
| industrial | MLP            |           0.293 |
| industrial | NEXS(proposed) |           0.618 |
| industrial | RandomForest   |          71.364 |
| industrial | SemanticLoss   |           0.266 |
| medical    | ESS(discrete)  |           0.754 |
| medical    | MLP            |           0.25  |
| medical    | NEXS(proposed) |           0.659 |
| medical    | RandomForest   |          72.967 |
| medical    | SemanticLoss   |           0.249 |

---

## Corrected protocol (leak-free nested CV, row-level stats)

Generated by `experiments/nested_*.py`. Every row is predicted exactly once
out-of-fold; preprocessing, tuning, early stopping and threshold selection are
all inside the training folds. These supersede the seed-resampled numbers above
for any inferential claim.

### Accuracy / F1 / AUC

**acc**

| method           |   heart |   monk1 |   monk3 |   pima |    ttt |    wbc |
|:-----------------|--------:|--------:|--------:|-------:|-------:|-------:|
| DecisionTree     |  0.7426 |  0.6385 |  0.9368 | 0.6927 | 0.8006 | 0.9084 |
| GradientBoosting |  0.7888 |  0.982  |  0.9856 | 0.7383 | 0.9854 | 0.9557 |
| LogReg           |  0.8119 |  0.6007 |  0.852  | 0.7474 | 0.9614 | 0.9714 |
| MLP              |  0.7921 |  0.9928 |  0.9765 | 0.7539 | 0.9885 | 0.9685 |
| NEXS_Corrupt     |  0.7888 |  0.9676 |  0.9801 | 0.7474 | 0.9906 | 0.9714 |
| NEXS_Discovered  |  0.8086 |  0.9802 |  0.9819 | 0.75   | 1      | 0.9642 |
| NEXS_Expert      |  0.7921 |  0.9802 |  0.9856 | 0.7422 | 0.9948 | 0.97   |
| RandomForest     |  0.8284 |  0.9856 |  0.9819 | 0.7591 | 0.9812 | 0.9685 |
| SemanticLoss     |  0.7921 |  0.9928 |  0.9765 | 0.7539 | 0.9885 | 0.9685 |

**f1**

| method           |   heart |   monk1 |   monk3 |   pima |    ttt |    wbc |
|:-----------------|--------:|--------:|--------:|-------:|-------:|-------:|
| DecisionTree     |  0.7562 |  0.7295 |  0.9414 | 0.655  | 0.8667 | 0.873  |
| GradientBoosting |  0.773  |  0.9823 |  0.9863 | 0.6667 | 0.9889 | 0.9374 |
| LogReg           |  0.8068 |  0.6487 |  0.8464 | 0.6734 | 0.971  | 0.96   |
| MLP              |  0.7864 |  0.9929 |  0.9774 | 0.6813 | 0.9913 | 0.956  |
| NEXS_Corrupt     |  0.7962 |  0.9685 |  0.981  | 0.6891 | 0.9929 | 0.9598 |
| NEXS_Discovered  |  0.8067 |  0.9806 |  0.9829 | 0.6903 | 1      | 0.9499 |
| NEXS_Expert      |  0.7893 |  0.9806 |  0.9863 | 0.6722 | 0.996  | 0.9579 |
| RandomForest     |  0.8143 |  0.9857 |  0.9828 | 0.6942 | 0.9857 | 0.956  |
| SemanticLoss     |  0.7864 |  0.9929 |  0.9774 | 0.6813 | 0.9913 | 0.956  |

**auc**

| method           |   heart |   monk1 |   monk3 |   pima |    ttt |    wbc |
|:-----------------|--------:|--------:|--------:|-------:|-------:|-------:|
| DecisionTree     |  0.848  |  0.8401 |  0.9866 | 0.8042 | 0.8965 | 0.974  |
| GradientBoosting |  0.8806 |  1      |  0.9935 | 0.8124 | 0.9994 | 0.9829 |
| LogReg           |  0.9064 |  0.7366 |  0.9    | 0.8385 | 0.9791 | 0.9943 |
| MLP              |  0.8955 |  1      |  0.9958 | 0.8371 | 0.9962 | 0.9931 |
| NEXS_Corrupt     |  0.8959 |  0.9984 |  0.9957 | 0.8357 | 0.9999 | 0.9927 |
| NEXS_Discovered  |  0.8802 |  1      |  0.9955 | 0.8398 | 1      | 0.9912 |
| NEXS_Expert      |  0.8988 |  1      |  0.9954 | 0.8365 | 1      | 0.9931 |
| RandomForest     |  0.9079 |  0.9993 |  0.9924 | 0.8317 | 0.9965 | 0.9888 |
| SemanticLoss     |  0.8955 |  1      |  0.9958 | 0.8371 | 0.9962 | 0.9931 |

### Paired row-level statistics (Holm-corrected)

| dataset   | method_a        | method_b         |   mean_diff |   ci_lo |   ci_hi |   p_raw |   p_holm |   significant_holm |
|:----------|:----------------|:-----------------|------------:|--------:|--------:|--------:|---------:|-------------------:|
| heart     | NEXS_Expert     | MLP              |      0      | -0.0297 |  0.0297 |  1      |   1      |                  0 |
| heart     | NEXS_Discovered | MLP              |      0.0165 | -0.0264 |  0.0561 |  0.5327 |   1      |                  0 |
| heart     | NEXS_Expert     | NEXS_Discovered  |     -0.0165 | -0.0561 |  0.0231 |  0.5224 |   1      |                  0 |
| heart     | NEXS_Expert     | NEXS_Corrupt     |      0.0033 | -0.0231 |  0.033  |  1      |   1      |                  0 |
| heart     | NEXS_Expert     | GradientBoosting |      0.0033 | -0.0429 |  0.0495 |  1      |   1      |                  0 |
| pima      | NEXS_Expert     | MLP              |     -0.0117 | -0.0286 |  0.0052 |  0.211  |   1      |                  0 |
| pima      | NEXS_Discovered | MLP              |     -0.0039 | -0.0221 |  0.0143 |  0.7754 |   1      |                  0 |
| pima      | NEXS_Expert     | NEXS_Discovered  |     -0.0078 | -0.0234 |  0.0078 |  0.4177 |   1      |                  0 |
| pima      | NEXS_Expert     | NEXS_Corrupt     |     -0.0052 | -0.0208 |  0.0104 |  0.6271 |   1      |                  0 |
| pima      | NEXS_Expert     | GradientBoosting |      0.0039 | -0.0234 |  0.0312 |  0.8496 |   1      |                  0 |
| wbc       | NEXS_Expert     | MLP              |      0.0014 |  0      |  0.0043 |  1      |   1      |                  0 |
| wbc       | NEXS_Discovered | MLP              |     -0.0043 | -0.0114 |  0.0014 |  0.375  |   1      |                  0 |
| wbc       | NEXS_Expert     | NEXS_Discovered  |      0.0057 |  0.0014 |  0.0114 |  0.125  |   0.5    |                  0 |
| wbc       | NEXS_Expert     | NEXS_Corrupt     |     -0.0014 | -0.0072 |  0.0029 |  1      |   1      |                  0 |
| wbc       | NEXS_Expert     | GradientBoosting |      0.0143 |  0.0057 |  0.0243 |  0.0063 |   0.0317 |                  1 |
| ttt       | NEXS_Expert     | MLP              |      0.0063 |  0.0021 |  0.0115 |  0.0312 |   0.0938 |                  0 |
| ttt       | NEXS_Discovered | MLP              |      0.0115 |  0.0052 |  0.0188 |  0.001  |   0.0049 |                  1 |
| ttt       | NEXS_Expert     | NEXS_Discovered  |     -0.0052 | -0.0104 | -0.001  |  0.0625 |   0.125  |                  0 |
| ttt       | NEXS_Expert     | NEXS_Corrupt     |      0.0042 |  0      |  0.0094 |  0.2188 |   0.2188 |                  0 |
| ttt       | NEXS_Expert     | GradientBoosting |      0.0094 |  0.0031 |  0.0167 |  0.0117 |   0.0469 |                  1 |
| monk1     | NEXS_Expert     | MLP              |     -0.0126 | -0.0252 | -0.0018 |  0.0654 |   0.3271 |                  0 |
| monk1     | NEXS_Discovered | MLP              |     -0.0126 | -0.0252 | -0.0018 |  0.0654 |   0.3271 |                  0 |
| monk1     | NEXS_Expert     | NEXS_Discovered  |      0      |  0      |  0      |  1      |   1      |                  0 |
| monk1     | NEXS_Expert     | NEXS_Corrupt     |      0.0126 | -0.0054 |  0.0306 |  0.2478 |   0.7434 |                  0 |
| monk1     | NEXS_Expert     | GradientBoosting |     -0.0018 | -0.018  |  0.0144 |  1      |   1      |                  0 |
| monk3     | NEXS_Expert     | MLP              |      0.009  |  0      |  0.0181 |  0.125  |   0.625  |                  0 |
| monk3     | NEXS_Discovered | MLP              |      0.0054 | -0.0054 |  0.0181 |  0.5488 |   1      |                  0 |
| monk3     | NEXS_Expert     | NEXS_Discovered  |      0.0036 | -0.0054 |  0.0126 |  0.6875 |   1      |                  0 |
| monk3     | NEXS_Expert     | NEXS_Corrupt     |      0.0054 | -0.0036 |  0.0144 |  0.4531 |   1      |                  0 |
| monk3     | NEXS_Expert     | GradientBoosting |      0      | -0.0072 |  0.0072 |  1      |   1      |                  0 |

### Structural rule recovery (out-of-fold, vs. published ground truth)

| dataset   |   seed |   fold |   n_train |   n_truth |   n_discovered |   exact_recovery |   exact_recovery_semantic |   rule_precision |   rule_recall |   rule_f1 |   rule_precision_semantic |   rule_recall_semantic |   rule_f1_semantic |   mean_literals |   logical_equivalence |   decision_agreement |
|:----------|-------:|-------:|----------:|----------:|---------------:|-----------------:|--------------------------:|-----------------:|--------------:|----------:|--------------------------:|-----------------------:|-------------------:|----------------:|----------------------:|---------------------:|
| monk1     |      1 |      2 |     355.8 |         4 |            4   |                1 |                         1 |           1      |        1      |    1      |                    1      |                 1      |             1      |          1.75   |                1      |               1      |
| monk3     |      1 |      2 |     354.2 |         7 |           10.8 |                0 |                         0 |           0.1865 |        0.2857 |    0.2253 |                    0.6444 |                 0.2857 |             0.3952 |          2.3582 |                0.6654 |               0.6654 |
| ttt       |      1 |      2 |     612.4 |         8 |            8   |                0 |                         1 |           0      |        0      |    0      |                    1      |                 1      |             1      |          3      |                1      |               1      |

### Robustness sweeps (accuracy, out-of-fold)

|                                 |    MLP |   NEXS |
|:--------------------------------|-------:|-------:|
| ('monk1', 'completeness', 0.0)  | 0.9964 | 0.4676 |
| ('monk1', 'completeness', 0.25) | 0.9964 | 0.9173 |
| ('monk1', 'completeness', 0.5)  | 0.9964 | 0.9478 |
| ('monk1', 'completeness', 0.75) | 0.9964 | 0.9982 |
| ('monk1', 'completeness', 1.0)  | 0.9964 | 1      |
| ('monk1', 'corrupt', 0.0)       | 0.9964 | 1      |
| ('monk1', 'corrupt', 0.25)      | 0.9964 | 0.9676 |
| ('monk1', 'corrupt', 0.5)       | 0.9964 | 0.9335 |
| ('monk1', 'corrupt', 0.75)      | 0.9964 | 0.8183 |
| ('monk1', 'corrupt', 1.0)       | 0.9964 | 0.4676 |
| ('monk1', 'lowdata', 0.1)       | 0.6727 | 0.9964 |
| ('monk1', 'lowdata', 0.25)      | 0.804  | 1      |
| ('monk1', 'lowdata', 0.5)       | 0.8669 | 1      |
| ('monk1', 'lowdata', 0.75)      | 0.8597 | 1      |
| ('monk1', 'lowdata', 1.0)       | 0.9964 | 1      |
| ('monk3', 'completeness', 0.0)  | 0.9711 | 0.4007 |
| ('monk3', 'completeness', 0.25) | 0.9711 | 0.944  |
| ('monk3', 'completeness', 0.5)  | 0.9711 | 0.9495 |
| ('monk3', 'completeness', 0.75) | 0.9711 | 0.9747 |
| ('monk3', 'completeness', 1.0)  | 0.9711 | 0.9892 |
| ('monk3', 'corrupt', 0.0)       | 0.9711 | 0.9892 |
| ('monk3', 'corrupt', 0.25)      | 0.9711 | 0.9838 |
| ('monk3', 'corrupt', 0.5)       | 0.9711 | 0.9693 |
| ('monk3', 'corrupt', 0.75)      | 0.9711 | 0.9314 |
| ('monk3', 'corrupt', 1.0)       | 0.9711 | 0.4007 |
| ('monk3', 'lowdata', 0.1)       | 0.8014 | 0.9783 |
| ('monk3', 'lowdata', 0.25)      | 0.8718 | 0.9838 |
| ('monk3', 'lowdata', 0.5)       | 0.9549 | 0.9892 |
| ('monk3', 'lowdata', 0.75)      | 0.9513 | 0.9892 |
| ('monk3', 'lowdata', 1.0)       | 0.9693 | 0.9892 |
| ('ttt', 'completeness', 0.0)    | 0.9896 | 0.6555 |
| ('ttt', 'completeness', 0.25)   | 0.9896 | 0.9864 |
| ('ttt', 'completeness', 0.5)    | 0.9896 | 0.9937 |
| ('ttt', 'completeness', 0.75)   | 0.9896 | 0.9979 |
| ('ttt', 'completeness', 1.0)    | 0.9896 | 0.9979 |
| ('ttt', 'corrupt', 0.0)         | 0.9896 | 0.9979 |
| ('ttt', 'corrupt', 0.25)        | 0.9896 | 0.999  |
| ('ttt', 'corrupt', 0.5)         | 0.9896 | 0.9958 |
| ('ttt', 'corrupt', 0.75)        | 0.9896 | 0.9916 |
| ('ttt', 'corrupt', 1.0)         | 0.9896 | 0.6555 |
| ('ttt', 'lowdata', 0.1)         | 0.7568 | 0.9238 |
| ('ttt', 'lowdata', 0.25)        | 0.8445 | 0.9624 |
| ('ttt', 'lowdata', 0.5)         | 0.9342 | 0.9927 |
| ('ttt', 'lowdata', 0.75)        | 0.9729 | 0.9948 |
| ('ttt', 'lowdata', 1.0)         | 0.9885 | 0.9969 |
| ('wbc', 'completeness', 0.0)    | 0.97   | 0.97   |
| ('wbc', 'completeness', 0.25)   | 0.97   | 0.9671 |
| ('wbc', 'completeness', 0.5)    | 0.97   | 0.9685 |
| ('wbc', 'completeness', 0.75)   | 0.97   | 0.9671 |
| ('wbc', 'completeness', 1.0)    | 0.97   | 0.9685 |
| ('wbc', 'corrupt', 0.0)         | 0.97   | 0.9685 |
| ('wbc', 'corrupt', 0.25)        | 0.97   | 0.97   |
| ('wbc', 'corrupt', 0.5)         | 0.97   | 0.97   |
| ('wbc', 'corrupt', 0.75)        | 0.97   | 0.9671 |
| ('wbc', 'corrupt', 1.0)         | 0.97   | 0.97   |
| ('wbc', 'lowdata', 0.1)         | 0.9628 | 0.9642 |
| ('wbc', 'lowdata', 0.25)        | 0.9614 | 0.9657 |
| ('wbc', 'lowdata', 0.5)         | 0.9685 | 0.9714 |
| ('wbc', 'lowdata', 0.75)        | 0.9671 | 0.9685 |
| ('wbc', 'lowdata', 1.0)         | 0.97   | 0.9671 |

### Ablation (accuracy, out-of-fold)

| method         |   monk1 |   monk3 |    ttt |    wbc |
|:---------------|--------:|--------:|-------:|-------:|
| fused          |  1      |  0.9892 | 0.9979 | 0.9685 |
| fused_no_prior |  1      |  0.9874 | 0.9979 | 0.9685 |
| neural_only    |  0.9964 |  0.9711 | 0.9896 | 0.97   |
| symbolic_only  |  0.7464 |  0.5307 | 1      | 0.9399 |

### Faithfulness (obedience and over-firing on counterfactuals)

|                          |   obedience |   over_firing |
|:-------------------------|------------:|--------------:|
| ('monk1', 'MLP')         |           1 |        0.4485 |
| ('monk1', 'NEXS_Expert') |           1 |        0.353  |
| ('monk3', 'MLP')         |           1 |        0.5825 |
| ('monk3', 'NEXS_Expert') |           1 |        0.482  |
| ('ttt', 'MLP')           |           1 |        1      |
| ('ttt', 'NEXS_Expert')   |           1 |        0.998  |
| ('wbc', 'MLP')           |           1 |        1      |
| ('wbc', 'NEXS_Expert')   |           1 |        1      |
