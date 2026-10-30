"""Robustness sweeps under the corrected (nested-CV) protocol.

Three prespecified sweeps, each evaluated out-of-fold so no row is reused:

* **rule corruption** (0 -> 100%): degrade the expert KB and measure accuracy.
  This tests how gracefully the neuro-symbolic model degrades as its knowledge
  goes wrong.
* **rule completeness** (0 -> 100%): keep a fraction of the expert rules.  This
  tests sensitivity to missing knowledge.
* **low data** (10 -> 100%): subsample the *training* rows only; the test block
  is always the full held-out fold, so the learning curve is honest.

Every point reports mean out-of-fold accuracy and the paired difference against
the neural-only MLP on the same rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import discovery, models, real_data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _fit_predict(domain, tr, va, te, seed, epochs, patience, kind):
    cls = {"MLP": models.MLPModel, "NEXS": models.NEXS}[kind]
    m = cls(domain, seed=seed, pos_weight="auto", epochs=epochs,
            patience=patience)
    m.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
    return m.predict_proba(te.X.to_numpy())


def _degrade(domain, level, seed, mode):
    from dataclasses import replace
    if mode == "corrupt":
        rules = discovery.corrupt_rules(domain.expert_rules, level, seed=seed)
    else:  # completeness: level is the fraction *kept*
        rules = discovery.incomplete_rules(domain.expert_rules, level, seed=seed)
    return replace(domain, expert_rules=tuple(rules))


def sweep(name: str, mode: str, levels: list[float], n_folds: int, seed: int,
          epochs: int, patience: int, low_data: bool = False) -> list[dict]:
    from sklearn.metrics import accuracy_score

    _, y = real_data.DATASETS[name]["loader"]()
    rows = []
    for level in levels:
        preds = {"MLP": np.full(len(y), np.nan), "NEXS": np.full(len(y), np.nan)}
        for f in real_data.make_oof_folds(name, n_folds=n_folds, seed=seed):
            tr, va, te, te_i = f["train"], f["val"], f["test"], f["te_i"]
            if low_data:
                frac = level
                rng = np.random.default_rng(seed * 1000 + int(frac * 1000))
                k = max(5, int(frac * len(tr.y)))
                idx = rng.choice(len(tr.y), size=k, replace=False)
                tr = tr.split(idx)
            dom = f["domain"]
            if not low_data:
                dom = _degrade(dom, level, seed=seed + f["fold"], mode=mode)
            for kind in preds:
                preds[kind][te_i] = _fit_predict(dom, tr, va, te, seed + f["fold"],
                                                 epochs, patience, kind)
        for kind in preds:
            yhat = (preds[kind] >= 0.5).astype(int)
            rows.append(dict(dataset=name, sweep=mode, level=level, method=kind,
                             acc=float(accuracy_score(y, yhat))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["wbc", "ttt", "monk1", "monk3"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out", default="results/nested_robustness")
    args = ap.parse_args()

    corrupt_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    completeness_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    data_levels = [0.1, 0.25, 0.5, 0.75, 1.0]

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.datasets:
        for seed in args.seeds:
            print(f"== corruption {name} seed={seed} ==", flush=True)
            rows += sweep(name, "corrupt", corrupt_levels, args.folds, seed,
                          args.epochs, args.patience)
            print(f"== completeness {name} seed={seed} ==", flush=True)
            rows += sweep(name, "completeness", completeness_levels, args.folds,
                          seed, args.epochs, args.patience)
            print(f"== low data {name} seed={seed} ==", flush=True)
            rows += sweep(name, "lowdata", data_levels, args.folds, seed,
                          args.epochs, args.patience, low_data=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "robustness.csv", index=False)
    print(df.groupby(["dataset", "sweep", "level", "method"])["acc"].mean()
          .round(4).to_string())
    print("written ->", out)


if __name__ == "__main__":
    main()
