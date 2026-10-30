
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

## Rigor round 2 (2026-10-30): independence, generalisation, structural recovery

New scripts:
* `experiments/paired_validation.py` — three protocols for NEXS vs MLP:
  seed-resampled McNemar (pseudo-replicated), out-of-fold McNemar (each row
  predicted exactly once; the trustworthy one), and StratifiedGroupKFold folds.
* `experiments/structural_recovery.py` — enumerates the full input domain
  (3^9 boards; the 432 MONK tuples) and compares the model's OWN rule layer with
  the ground-truth rule set, plus oracle/MLP decision agreement on that domain.
* `src/real_data.py` — added MONK-1/MONK-3 (loader, exact KBs, ground_truth_rules).

Load-bearing findings (these changed the paper's claims):
* Seeds are pseudo-replicates: over 20 seeds each row is tested up to 11 times
  and the test-set union covers ~99% of the data. Under a clean out-of-fold
  McNemar the seed-level significance vanishes (heart p=1.00, pima p=0.15,
  ttt p=0.25). Claim parity + higher logic fidelity, NOT significant superiority.
* The distilled-tree rule-F1 is INVALID for OR-of-conjunctions: it returns 0 even
  for the exact expert KB on ttt and MONK-3. Use structural_recovery.py instead.
* Structural result: NEXS's rule layer == true rules on all 3 logical domains
  (exact set equality, P=R=1.0). But fused decisions obey the rules only 63% on
  the full ttt domain (neural head overrides off-support); MONK is 97-98%.
* MONK KBs verified against the published formulas: MONK-1 100%, MONK-3 ~98%
  (the gap is the documented 5% class noise). Note: MONK file column order is
  class,a1..a6,Id; the input space is 3*3*2*3*4*2=432 tuples.

## Rigor round 3 (2026-10-30): non-circular rule discovery

Critique: comparing NEXS's rule layer to the ground truth is CIRCULAR because
`models.py` injects it (`self.rules = list(domain.expert_rules)`). Correct.

* `experiments/rule_discovery.py` — genuine, non-circular discovery. A
  candidate-generation (CN2-style) learner gets only literal memberships + labels
  (no expert rules) and its discovered DNF is compared to the hidden ground truth
  by exact set equality and scored on the full domain.
  Result: exact recovery on MONK-1 (4 rules) and MONK-3 (7 rules), P=R=1.0,
  full-domain agreement 1.0; tic-tac-toe 0.85 P/R, 0.95 agreement (exact on some
  seeds). Also contains `DiscoveryLayer`, a differentiable structure-free rule
  layer, kept as a (weaker, local-optima-prone) comparison.
* Two real bugs found while building this and fixed in BOTH scripts:
  (1) MONK attribute ranges were wrong (a1..a6 = 3,3,2,3,4,2 -> 432 tuples, not
      288); fixed in `structural_recovery.MONK_RANGES` and `rule_discovery`.
  (2) Membership literals are defined in RAW units, so the normalised frames must
      be inverted using the SAME per-seed bounds load_real used
      (`_train_split_bounds`), not min/max of the already-normalised frame.
* `structural_recovery.py` now labels its injected-KB comparison as an
  implementation check and reports the discovery metrics alongside.
* Paper: abstract, contributions, results ("two questions, one of which we can
  answer"), and threats rewritten to distinguish injection from discovery; new
  Table tab:discovery.
* README updated. New tables: figures/table_discovery.tex.

Tests added: injected_rule_layer_mirrors_kb (renamed from structural_recovery_exact),
genuine_rule_discovery_from_data. full_monk_domain_labels now expects 432.
All pass. Paper: 26 pages, 0 errors, 0 undefined refs.

## Phase 1 (2026-10-30): corrected nested protocol

An audit of the nested-CV harness against the actual code found four real defects
(plus one latent) that the earlier write-up did not reflect. All are fixed.

1. **Fold-specific threshold selection.** `nested_cv.run_nested` used to pool all
   out-of-fold probabilities and apply one *averaged* threshold, so one fold's
   operating point decided another fold's labels. Now each fold keeps its own
   threshold (`thr_min`/`thr_max` in `nested_cv_metrics.csv` show the real
   spread, e.g. LogReg on MONK-1 0.33-0.62), metrics are computed per fold and
   averaged, and `nested_cv_oof_proba.csv` records `fold` + `threshold` per row so
   `yhat` uses the row's own fold threshold.
2. **Inner cross-validation.** The old `_tune` used a single validation block.
   Replaced by `_inner_cv`: 3-fold stratified CV over the outer *training* rows
   only, hyperparameters chosen by mean inner AUC, threshold from the F1-optimal
   cut of the pooled inner predictions.
3. **MONK categorical encoding.** MONK attributes are categorical; per-fold
   min-max scaling drifted the encoding with the fold and collapsed `a5=3`/`a5=4`
   onto one value. The loader now maps every level through a *declared* codebook
   (`MONK_NORM`, `(j-1)/(k-1)`), independent of the fold. `DATASETS[name]
   ["preprocess"]` selects `none` for ttt/monk1/monk3 and `minmax` elsewhere;
   `Preprocessor` gained `scale="none"`.
4. **Train-only preprocessing** verified explicitly: imputation medians and
   scaling bounds equal the outer-training statistics (not full-dataset ones),
   folds are disjoint, every row is tested exactly once.
5. **Latent bug (found while fixing 3).** `rule_discovery.monk_candidates` used
   `sigmoid((1 - 2|x-v|)/tau)` instead of `src.models`'
   `sigmoid(1 - 2|x-v|/tau)`, which fires on neighbouring levels once the spacing
   drops below `tau`; and both legacy scripts un-scaled features with
   `_train_split_bounds`, invalid once MONK stopped being min-max scaled. Fixed;
   MONK-1/3/ttt now recover exactly across seeds.

Single source of truth: `src.real_data.monk_full_domain` / `monk_truth_dnf` /
`monk_candidate_literals`; `nested_structural.full_domain`, `structural_recovery
.full_monk_domain` and `rule_discovery._full_domain` all delegate to them.
`_check_codebook` cross-checks the enumerated grid against the observed rows
(tol 0 for MONK-1, 5% for MONK-3's documented label noise).

Corrected nested results (5 folds, seeds 0--4, `results/nested_cv`): NEXS-Expert is
1.000 on ttt/MONK-1 and 0.987 on MONK-3 (MLP 0.988/1.000/0.974); the significant
paired wins are all on exact-rule or corrupted-KB comparisons (ttt, MONK-1,
MONK-3, pima-vs-GB); the clinical MLP comparisons remain non-significant.
Structural: MONK-3 full-domain equivalence 0.998 (the extra rules are harmless
refinements, set F1 still 0.393). Tests: 105 pass.

6. **Inner-fold refit (NEXS v5).** The nested protocol previously reused the
   outer-fold-fitted preprocessor *and* the outer-fold-discovered rules inside the
   inner CV that selects hyperparameters and the threshold --- a leak of the
   outer training statistics into model selection. Fixed: `src/real_data.py`
   gained `fit_preprocessor(name, Xraw, rows, y) -> (prep, domain)` (fit
   imputation/scaling on `rows` only, map expert thresholds through those
   bounds), and `experiments/nested_cv.py` now refits *both* preprocessing and
   rule discovery inside every inner training split via the `_variant()` helper.
   `make_oof_folds` also yields `Xraw`/`y` so callers can refit on a smaller
   partition. Regression tests `test_inner_fold_preprocessing_is_independent`
   and `test_inner_fold_discovery_is_independent` assert the inner prep median and
   bounds equal the inner-training statistics and exclude inner-validation rows.
   `make_stats_figures.table_nested_seed_summary` adds a seed-stability table.
   Multi-seed runs (`--seeds 0 1 2 3 4`) write
   `results/nested_cv/nested_cv_seed_summary.csv`; inference still uses the
   primary seed only (`nested_stats.py` filters to the first seed).
