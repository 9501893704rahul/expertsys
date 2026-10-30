
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
