# Repository notes for agents

## Project
`ns_expert/` is a self-contained research codebase for a neuro-symbolic expert
system (NEXS) study. It is designed to be run end-to-end on CPU.

## Environment
- Python 3.13, no GPU. Deps: numpy, pandas, scikit-learn, matplotlib, scipy,
  xgboost, shap, torch (CPU). Install torch with
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Scripts must be run with `PYTHONPATH=.` from inside `ns_expert/` (the package
  is `src`, imported as `from src...`).

## Key commands
```bash
cd ns_expert
PYTHONPATH=. python experiments/run_experiments.py --quick      # 1 seed, n=2000
PYTHONPATH=. python experiments/run_experiments.py --seeds 0 1 2 --n 4000
PYTHONPATH=. python experiments/make_figures.py                 # figures + LaTeX tables
PYTHONPATH=. python experiments/report.py                       # results/RESULTS.md
PYTHONPATH=. python experiments/tests.py                        # invariant tests
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Real-data extension
- `src/real_data.py` loads heart (UCI Cleveland), pima, and tic-tac-toe from
  `ns_expert/data/` (committed, offline-reproducible) and builds expert KBs from
  published clinical cut-offs. Run `experiments/run_real_experiments.py` then
  `experiments/make_real_figures.py`.
- On heart/pima there is no ground-truth rule base, so rule recovery is measured
  against the expert KB and logic fidelity against the KB oracle. Only ttt has a
  genuine ground truth (the 8 winning lines).
- Literal matching and rule extraction must be **support-aware** on discrete
  features: pass `support=real_data.feature_support(X)` to
  `rules.rule_set_match` / `rules.extract_rules`. Otherwise `>0.5` and `>=0.75`
  look different on `{0, 0.5, 1}` and the extraction tree splits at interpolated
  points like `c00<=0.549`.
- `rules._parse_literal` must accept feature names with digits (ttt cells are
  `c00`..`c22`).

## Conventions / gotchas
- Data are generated from a **known Horn rule base** with **hidden concepts**.
  Rule recovery must be measured against `rules.ground_truth_dnf(...)`, the
  concept-expanded feature-only DNF, NOT the raw rule bodies (which reference
  hidden concepts and are unrepresentable by feature-based models).
- Literal matching in `rules._literal_match` is semantic (partition-equivalence
  over a feature grid), because extracted trees emit `>`/`<=` while rules use
  `>=`/`==`.
- The rule-extraction reference must be representable by a depth-limited tree,
  so `extract_rules(depth=3)` is the default.
- All models share a `fit(X, y, X_val, y_val)` / `predict_proba(X)` interface;
  probability outputs are isotonic-calibrated on validation and thresholded at
  the F1-optimal validation threshold.
- Everything is deterministic given the seed; results live in `results/*.csv`
  and are consumed by `make_figures.py` and `report.py`.

## Findings that recur
- NEXS (forward-pass differentiable rule layer + expert prior) beats MLP and
  semantic-loss on F1 and logic fidelity.
- The rule-layer-only ablation has the highest logic fidelity of any learned
  model — the rule layer, not the neural head, aligns the model with true logic.
- The discrete certainty-factor expert system is best calibrated and most robust
  under shift.
