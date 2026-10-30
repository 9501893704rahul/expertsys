
## Accuracy-improvement round (2026-10-30)

Training knobs added to `src/models.py` (`_TorchClassifier` / `NEXS`):
* `pos_weight="auto"` — positive-class BCE weight from the training labels; used
  uniformly by MLP/SemanticLoss/NEXS and `class_weight="balanced"` for sklearn
  baselines (GradientBoosting has no such option).
* `select="auc"` — early stopping on validation ROC-AUC instead of the
  (regularised) training loss. Note: the early-stopping checkpoint now saves and
  restores **all** learnable params, not just `self.model`, which was a bug for
  NEXS (`_cf`/`alpha`/gate were previously left at the last epoch).
* `gated=True` — learned per-sample gate on the symbolic logit
  (`z = f(x) + sigmoid(gate(x)) * alpha * logit(p_sat)`).

Evaluation: `_evaluate` now also reports `acc_opt` / `threshold_acc` — accuracy at
the accuracy-optimal validation threshold, because the F1-optimal threshold
systematically understates accuracy on imbalanced data (e.g. pima 0.696 -> 0.740).
`_best_threshold(p, y, metric="f1"|"acc")`.

Honest outcome: class weighting + the two operating points are the real gains.
Gating and AUC early stopping did **not** improve accuracy (within seed noise);
the gate is kept as an ablation option, off by default.

## Statistical / factorial round (2026-10-30)

New scripts:
* `experiments/run_rule_conditions.py` — grid over KB corruption {0,.25,.5,.75,1}
  x training fraction {.25,.5,1} x {ESS, MLP, NEXS} x datasets x seeds; writes
  `results/conditions_5seed/all_runs.csv` (675 rows at 3x5x15x3).
* `experiments/advanced_analysis.py` — `metric_overview.csv` (mean/std/95% CI),
  `paired_statistics.csv` (Wilcoxon signed-rank + paired bootstrap CI + Holm),
  and an optional tic-tac-toe rule-behaviour probe.
* `experiments/make_stats_figures.py` — `fig_conditions.png`,
  `table_conditions.tex`, `table_stats.tex`, `table_behavior.tex`.
* `run_real_experiments.py` gained `--tag` so a 20-seed run writes
  `real_e1_main_20seed.csv` without clobbering the 5-seed files.

Findings (evidence-backed):
* Expert prior is monotone in KB quality: NEXS-MLP accuracy +0.025 (kb=0) ->
  +0.016 (kb=.75) -> -0.089 (kb=1, no knowledge).
* Data-efficiency: on ttt, NEXS with clean KB at 25% data ~= MLP at 100% data.
* 20 seeds: 83/162 comparisons Holm-significant; all ttt NEXS-vs-{MLP,SemanticLoss}
  metric gaps significant at p<0.01. heart/pima NOT significant -> report as
  "comparable accuracy, higher logic fidelity", not superiority.
* Rule-behaviour probe (5 seeds): valid line 1.000, broken 0.747, random 0.476.
  Functional check only, not structural rule recovery.

Paper: new Section "When Do Expert Rules Help?" (label sec:whenhelp), abstract
extended, threats clarified. Compiles 24 pages, 0 errors, 0 undefined refs.
