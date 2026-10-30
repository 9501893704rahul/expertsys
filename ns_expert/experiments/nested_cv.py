"""Nested cross-validation with independent, non-replicated testing.

Methodological contract
-----------------------
* **Outer loop** = 5 stratified folds.  Each row falls in the test block of
  exactly one fold, so every observation is predicted exactly once.  Test rows
  are never reused, so seed-level pseudo-replication cannot occur.
* **Inner loop** = a validation block carved from the outer *training* rows
  only, used for hyperparameter search, early stopping, and threshold selection.
  The test block never influences any of those choices.
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


def _tune(cfg, domain, tr, va, seed, epochs, patience):
    """Inner-loop search over NEXS hyperparameters; fixed config otherwise."""
    from sklearn.metrics import roc_auc_score

    from experiments.run_experiments import _best_threshold

    grid = NEXS_GRID if cfg["torch"] else [{}]
    best, best_auc, best_hp = None, -np.inf, {}
    for hp in grid:
        m = cfg["make"](domain, hp)
        if hasattr(m, "kw"):
            m.kw.update(epochs=epochs, patience=patience, seed=seed)
        m.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
        p = m.predict_proba(va.X.to_numpy())
        auc = roc_auc_score(va.y, p) if len(np.unique(va.y)) > 1 else 0.5
        if auc > best_auc:
            best, best_auc, best_hp = m, float(auc), hp
    thr = _best_threshold(best.predict_proba(va.X.to_numpy()), va.y, metric="f1")
    return best, best_hp, float(thr)


def _degrade(domain, level, seed, kind="corrupt"):
    from dataclasses import replace

    rules = domain.expert_rules
    if kind == "incomplete":
        rules = discovery.incomplete_rules(rules, 1.0 - level, seed=seed)
    else:
        rules = discovery.corrupt_rules(rules, level, seed=seed)
    return replace(domain, expert_rules=tuple(rules),
                   meta={**domain.meta, "kb_source": f"{kind}:{level}"})


def run_nested(name: str, n_folds: int, seed: int, epochs: int, patience: int,
               corrupt_level: float = 0.4) -> tuple[pd.DataFrame, dict, list]:
    """Nested CV for one dataset; returns (metrics, oof, fold_info)."""
    from experiments.run_experiments import _metrics

    _, y = real_data.DATASETS[name]["loader"]()
    n = len(y)
    methods = build_methods(seed)
    oof = {m: np.full(n, np.nan) for m in methods}
    thr_of = {m: [] for m in methods}
    hp_of = {m: [] for m in methods}
    fold_info = []

    for f in real_data.make_oof_folds(name, n_folds=n_folds, seed=seed):
        fold, domain = f["fold"], f["domain"]
        tr, va, te, te_i = f["train"], f["val"], f["test"], f["te_i"]

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
            m, hp, thr = _tune(cfg, d, tr, va, seed + fold, epochs, patience)
            oof[mname][te_i] = m.predict_proba(te.X.to_numpy())
            thr_of[mname].append(thr)
            hp_of[mname].append(hp)

    rows = []
    for mname in oof:
        proba = oof[mname]
        assert not np.isnan(proba).any(), f"{mname}: incomplete OOF coverage"
        thr = float(np.mean(thr_of[mname]))
        m = _metrics(y, proba, thr=thr)
        rows.append(dict(dataset=name, method=mname, seed=seed, n=n,
                         threshold=thr, hp=str(hp_of[mname][0]),
                         **{k: float(v) for k, v in m.items()}))
    return pd.DataFrame(rows), oof, fold_info, thr_of


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
            df, oof, info, thr_of = run_nested(name, args.folds, seed,
                                               args.epochs, args.patience,
                                               args.corrupt_level)
            df["seconds"] = time.time() - t0
            metrics.append(df)
            folds += info
            thr_by_method = {m: float(np.mean(thr_of[m])) for m in oof}
            for mname, proba in oof.items():
                thr = thr_by_method[mname]
                for i, p in enumerate(proba):
                    oof_rows.append(dict(dataset=name, seed=seed, method=mname,
                                         row=i, proba=float(p),
                                         yhat=int(p >= thr)))
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_csv(out / "nested_cv_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out / "nested_cv_oof_proba.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "nested_cv_folds.csv", index=False)
    print(metrics.round(4).to_string(index=False))
    print("written ->", out)


if __name__ == "__main__":
    main()
