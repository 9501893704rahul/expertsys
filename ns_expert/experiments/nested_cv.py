"""Nested cross-validation with independent, non-replicated testing.

Methodological contract
-----------------------
* **Outer loop** = 5 stratified folds.  Each row falls in the test block of
  exactly one fold, so every observation is predicted exactly once.  Test rows
  are never reused, so seed-level pseudo-replication cannot occur.
* **Inner loop** = 3-fold cross-validation over the outer *training* rows only,
  used for hyperparameter search and threshold selection.  The threshold comes
  from the pooled inner-fold predictions; the outer test block never influences
  the choice.
* **Per-fold operating points.**  Each fold carries its own threshold, and a
  row's label is decided at *its own fold's* threshold.  Metrics are computed
  per fold and averaged, so one fold's operating point never decides another
  fold's labels.
* **Preprocessing** (imputation, scaling, feature selection) is fit on the outer
  training rows only (see :mod:`src.pipeline`).
* **Rule discovery** for ``NEXS_Discovered`` runs on the outer training rows
  only, so the discovered KB never sees the test block.

The harness writes per-row out-of-fold probabilities.  Downstream statistics run
on those pooled predictions, so the unit of observation is a data row, not a
random seed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import baselines, discovery, models, real_data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

NEXS_GRID = [{"lr": 2e-3, "weight_decay": 1e-4},
             {"lr": 1e-3, "weight_decay": 1e-3}]


def build_methods(seed: int) -> dict:
    """Every model configuration compared under the corrected protocol.

    Each value is ``dict(make=..., torch=...)`` where ``make(domain, hp)``
    returns an unfitted model.  ``torch`` marks the configs that accept the
    neural hyperparameter grid (learning rate / weight decay).

    Requested NEXS configurations:
      * ``MLP``             - neural only, no symbolic pathway
      * ``NEXS_Expert``     - NEXS with the original expert rules
      * ``NEXS_Discovered`` - NEXS with rules discovered from training rows only
      * ``NEXS_Corrupt``    - NEXS with corrupted / incomplete expert rules
    plus classical baselines and a semantic-loss neuro-symbolic baseline.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    def torch_cfg(cls):
        return dict(torch=True,
                    make=lambda d, hp: cls(d, seed=seed, pos_weight="auto", **hp))

    def sk_cfg(name, est):
        return dict(torch=False,
                    make=lambda d, hp: baselines.SklearnModel(name, est))

    return {
        "LogReg": sk_cfg("LogReg", LogisticRegression(
            max_iter=5000, C=1.0, class_weight="balanced")),
        "DecisionTree": sk_cfg("DecisionTree", DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20, class_weight="balanced",
            random_state=seed)),
        "RandomForest": sk_cfg("RandomForest", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            n_jobs=-1, random_state=seed)),
        "GradientBoosting": sk_cfg("GradientBoosting", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            random_state=seed)),
        "MLP": torch_cfg(models.MLPModel),
        "SemanticLoss": torch_cfg(models.SemanticLossModel),
        "NEXS_Expert": torch_cfg(models.NEXS),
        "NEXS_Discovered": torch_cfg(models.NEXS),
        "NEXS_Corrupt": torch_cfg(models.NEXS),
    }


def _inner_cv(cfg, domain, tr, seed, epochs, patience, n_inner=3):
    """Inner cross-validation over the outer *training* rows.

    Returns ``(best_hp, threshold)``.  Every inner fold holds out its own
    validation block from the training rows, so the hyperparameter choice and
    the decision threshold are both estimated without touching the outer test
    block.  The threshold comes from the *pooled* inner-fold predictions, which
    is why it is a genuine training-side estimate rather than a value read off
    the test rows.

    A single inner split (as the earlier version used) makes the choice depend
    on one arbitrary partition; averaging AUC over ``n_inner`` folds is the
    standard nested-CV design and is much less noisy.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    from experiments.run_experiments import _best_threshold

    grid = NEXS_GRID if cfg["torch"] else [{}]
    Xtr, ytr = tr.X.to_numpy(), np.asarray(tr.y)
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    splits = list(skf.split(Xtr, ytr))

    scored = []
    for hp in grid:
        pooled = np.full(len(ytr), np.nan)
        for k, (itr, iva) in enumerate(splits):
            m = cfg["make"](domain, hp)
            if hasattr(m, "kw"):
                m.kw.update(epochs=epochs, patience=patience, seed=seed + k)
            m.fit(Xtr[itr], ytr[itr], Xtr[iva], ytr[iva])
            pooled[iva] = m.predict_proba(Xtr[iva])
        assert not np.isnan(pooled).any()
        auc = roc_auc_score(ytr, pooled) if len(np.unique(ytr)) > 1 else 0.5
        thr = _best_threshold(pooled, ytr, metric="f1")
        scored.append((float(auc), hp, float(thr)))

    scored.sort(key=lambda r: r[0], reverse=True)
    _, best_hp, best_thr = scored[0]
    return best_hp, best_thr


def run_nested(name: str, n_folds: int, seed: int, epochs: int, patience: int,
               corrupt_level: float = 0.4) -> tuple[pd.DataFrame, dict, list]:
    """Nested CV for one dataset; returns (metrics, oof, fold_info, per-fold data)."""
    from experiments.run_experiments import _metrics

    _, y = real_data.DATASETS[name]["loader"]()
    n = len(y)
    methods = build_methods(seed)
    oof = {m: np.full(n, np.nan) for m in methods}
    thr_of = {m: [] for m in methods}
    hp_of = {m: [] for m in methods}
    fold_of = np.full(n, -1, dtype=int)
    fold_te = []
    fold_info = []

    for f in real_data.make_oof_folds(name, n_folds=n_folds, seed=seed):
        fold, domain = f["fold"], f["domain"]
        tr, va, te, te_i = f["train"], f["val"], f["test"], f["te_i"]
        fold_of[te_i] = fold
        fold_te.append(te_i)

        disc_rules = discovery.discover_rules(domain, tr.X, tr.y)
        dom_disc = discovery.with_discovered_rules(domain, disc_rules)
        dom_corr = _degrade(domain, corrupt_level, seed=seed + fold)
        fold_info.append(dict(
            dataset=name, seed=seed, fold=fold, n_train=len(tr.y),
            n_val=len(va.y), n_test=len(te.y),
            n_expert_rules=len(domain.expert_rules),
            n_discovered_rules=len(disc_rules),
            n_corrupt_rules=len(dom_corr.expert_rules)))

        for mname, cfg in methods.items():
            d = {"NEXS_Discovered": dom_disc, "NEXS_Corrupt": dom_corr}.get(
                mname, domain)
            hp, thr = _inner_cv(cfg, d, tr, seed + fold, epochs, patience)
            m = cfg["make"](d, hp)
            if hasattr(m, "kw"):
                m.kw.update(epochs=epochs, patience=patience, seed=seed + fold)
            m.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
            oof[mname][te_i] = m.predict_proba(te.X.to_numpy())
            thr_of[mname].append(thr)
            hp_of[mname].append(hp)

    # Metrics are computed *per fold at that fold's own threshold*, then averaged.
    # Pooling all rows and applying one global threshold would let one fold's
    # operating point decide another fold's labels.
    rows = []
    for mname in oof:
        proba = oof[mname]
        assert not np.isnan(proba).any(), f"{mname}: incomplete OOF coverage"
        per_fold = [_metrics(y[te_i], proba[te_i], thr=thr_of[mname][k])
                    for k, te_i in enumerate(fold_te)]
        agg = {k: float(np.mean([m[k] for m in per_fold]))
               for k in ("acc", "f1", "auc", "brier", "ece")}
        rows.append(dict(dataset=name, method=mname, seed=seed, n=n,
                         threshold=float(np.mean(thr_of[mname])),
                         thr_min=float(np.min(thr_of[mname])),
                         thr_max=float(np.max(thr_of[mname])),
                         hp=str(hp_of[mname][0]), **agg))
    return pd.DataFrame(rows), oof, fold_info, thr_of, fold_of


def _degrade(domain, level, seed, kind="corrupt"):
    from dataclasses import replace

    rules = domain.expert_rules
    if kind == "incomplete":
        rules = discovery.incomplete_rules(rules, 1.0 - level, seed=seed)
    else:
        rules = discovery.corrupt_rules(rules, level, seed=seed)
    return replace(domain, expert_rules=tuple(rules),
                   meta={**domain.meta, "kb_source": f"{kind}:{level}"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["heart", "pima", "wbc", "ttt", "monk1", "monk3"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--corrupt-level", type=float, default=0.4)
    ap.add_argument("--out", default="results/nested_cv")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    metrics, oof_rows, folds = [], [], []
    for name in args.datasets:
        for seed in args.seeds:
            print(f"== nested CV {name} seed={seed} ==", flush=True)
            t0 = time.time()
            df, oof, info, thr_of, fold_of = run_nested(
                name, args.folds, seed, args.epochs, args.patience,
                args.corrupt_level)
            df["seconds"] = time.time() - t0
            metrics.append(df)
            folds += info
            for mname, proba in oof.items():
                for i, p in enumerate(proba):
                    # yhat uses *this row's own fold* threshold, never another
                    # fold's operating point
                    thr = thr_of[mname][fold_of[i]]
                    oof_rows.append(dict(dataset=name, seed=seed, method=mname,
                                         row=i, fold=int(fold_of[i]),
                                         threshold=float(thr), proba=float(p),
                                         yhat=int(p >= thr)))
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_csv(out / "nested_cv_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out / "nested_cv_oof_proba.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "nested_cv_folds.csv", index=False)
    print(metrics.round(4).to_string(index=False))
    print("written ->", out)


if __name__ == "__main__":
    main()
