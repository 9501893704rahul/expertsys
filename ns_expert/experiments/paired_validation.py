"""Independent-sample validation of the NEXS-vs-MLP comparison.

Why this exists
---------------
The 20-seed study in ``run_real_experiments.py`` re-permutes the *same* finite
dataset for every seed. Over 20 seeds each row is held out up to ~11 times and the
union of test sets covers ~99% of the data, so the seeds are *pseudo-replicates*:
the observations are correlated and a paired test over seeds treats them as
independent, which inflates significance. This module fixes that in two ways.

1. **Row-level paired test.** Pool the per-row correctness of NEXS and the MLP
   across all test appearances, then run an exact McNemar test on the discordant
   pairs. This makes each (row, model) comparison one observation and is the
   standard test for paired binary predictions. Its ``n`` is the number of test
   rows, not the number of seeds.

2. **Repeated grouped splits.** Use ``StratifiedGroupKFold`` so a given row is
   held out in *exactly one* fold; report per-fold accuracy and a paired test
   over folds. This is the cleanest small-data protocol: every row is tested
   once, so there is no reuse at all.

Both are reported; neither assumes seed-level independence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import models, real_data  # noqa: E402
from experiments.run_experiments import _decisions, _evaluate  # noqa: E402

METHODS = {"MLP": models.MLPModel, "NEXS": models.NEXS}


def _fit_predict(cls, domain, tr, va, te, seed, epochs, patience):
    m = cls(domain, seed=seed, pos_weight="auto", epochs=epochs, patience=patience)
    m.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
    mm, _ = _evaluate(m, va.X.to_numpy(), va.y, te.X.to_numpy(), te.y, calibrate=True)
    thr = mm.get("threshold", 0.5)
    return np.asarray(_decisions(m, thr)(te.X), dtype=int)


def mcnemar_rowlevel(name: str, seeds: list[int], epochs: int, patience: int):
    """Pool per-row correctness over seeds; exact McNemar on discordant pairs."""
    rows = []
    correct = {k: [] for k in METHODS}
    truth = []
    for s in seeds:
        domain, tr, va, te = real_data.load_real(name, seed=s)
        truth.append(np.asarray(te.y, dtype=int))
        for k, cls in METHODS.items():
            correct[k].append(_fit_predict(cls, domain, tr, va, te, s, epochs, patience))
    truth = np.concatenate(truth)
    preds = {k: np.concatenate(v) for k, v in correct.items()}
    a = preds["NEXS"] == truth
    b = preds["MLP"] == truth
    n01 = int(np.sum(~b & a))   # NEXS right, MLP wrong
    n10 = int(np.sum(b & ~a))   # MLP right, NEXS wrong
    if n01 + n10 == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
    rows.append(dict(dataset=name, test="mcnemar_rowlevel", n_obs=len(truth),
                     nexs_acc=float(a.mean()), mlp_acc=float(b.mean()),
                     nexs_only=n01, mlp_only=n10, p_value=p))
    return rows


def mcnemar_oof(name: str, n_folds: int, seed: int, epochs: int, patience: int):
    """Out-of-fold McNemar: every row predicted exactly once, by a model that
    never saw it. Full sample size, no row reuse, no seed pseudo-replication."""
    Xraw, y = real_data.DATASETS[name]["loader"]()
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = {k: np.full(len(y), -1, dtype=int) for k in METHODS}
    for fold, (tr_i, te_i) in enumerate(skf.split(Xraw, y)):
        lo, hi = Xraw.iloc[tr_i].min(), Xraw.iloc[tr_i].max()
        Xn = real_data.normalize(Xraw, lo, hi)
        domain = real_data.build_domain(name, lo, hi)
        from src.data import Dataset
        tr = Dataset(Xn.iloc[tr_i].reset_index(drop=True), y[tr_i],
                     pd.DataFrame(index=range(len(tr_i))), domain)
        te = Dataset(Xn.iloc[te_i].reset_index(drop=True), y[te_i],
                     pd.DataFrame(index=range(len(te_i))), domain)
        for k, cls in METHODS.items():
            oof[k][te_i] = _fit_predict(cls, domain, tr, tr, te,
                                        seed * 100 + fold, epochs, patience)
    a = oof["NEXS"] == y
    b = oof["MLP"] == y
    n01, n10 = int(np.sum(~b & a)), int(np.sum(b & ~a))
    p = 1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
    return [dict(dataset=name, test=f"mcnemar_oof_{n_folds}fold", n_obs=len(y),
                 nexs_acc=float(a.mean()), mlp_acc=float(b.mean()),
                 nexs_only=n01, mlp_only=n10, p_value=p)]


def grouped_folds(name: str, n_folds: int, seed: int, epochs: int, patience: int):
    """Each row is held out exactly once (StratifiedGroupKFold on row id)."""
    Xraw, y = real_data.DATASETS[name]["loader"]()
    groups = np.arange(len(y))
    rows = []
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_fold = {k: [] for k in METHODS}
    for fold, (tr_i, te_i) in enumerate(sgkf.split(Xraw, y, groups)):
        lo, hi = Xraw.iloc[tr_i].min(), Xraw.iloc[tr_i].max()
        Xn = real_data.normalize(Xraw, lo, hi)
        domain = real_data.build_domain(name, lo, hi)
        from src.data import Dataset
        tr = Dataset(Xn.iloc[tr_i].reset_index(drop=True), y[tr_i],
                     pd.DataFrame(index=range(len(tr_i))), domain)
        te = Dataset(Xn.iloc[te_i].reset_index(drop=True), y[te_i],
                     pd.DataFrame(index=range(len(te_i))), domain)
        for k, cls in METHODS.items():
            pred = _fit_predict(cls, domain, tr, tr, te, seed * 100 + fold, epochs, patience)
            per_fold[k].append(float((pred == y[te_i]).mean()))
    a = np.array(per_fold["NEXS"])
    b = np.array(per_fold["MLP"])
    d = a - b
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    return [dict(dataset=name, test=f"grouped_{n_folds}fold", n_obs=n_folds,
                 nexs_acc=float(a.mean()), mlp_acc=float(b.mean()),
                 mean_diff=float(d.mean()), p_value=p)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["heart", "pima", "ttt"])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out", default="results/paired_validation")
    args = ap.parse_args()

    rows = []
    for name in args.datasets:
        print(f"== {name}: row-level McNemar (seed-resampled) ==", flush=True)
        rows += mcnemar_rowlevel(name, args.seeds, args.epochs, args.patience)
        print(f"== {name}: out-of-fold McNemar (clean) ==", flush=True)
        rows += mcnemar_oof(name, args.folds, 0, args.epochs, args.patience)
        print(f"== {name}: {args.folds}-fold grouped ==", flush=True)
        rows += grouped_folds(name, args.folds, 0, args.epochs, args.patience)
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "independent_validation.csv", index=False)
    print(df.round(4).to_string(index=False))
    print("written ->", out)


if __name__ == "__main__":
    main()
