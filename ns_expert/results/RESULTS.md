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
