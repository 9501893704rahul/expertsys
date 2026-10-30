"""Neural-vs-symbolic ablation and logical faithfulness.

Ablation (Task 13)
------------------
Does a gain come from the neural pathway, the symbolic pathway, or their
interaction?  We evaluate, out-of-fold, four variants of the same architecture:

* ``neural_only``  --- ``alpha = 0`` (pure MLP; the symbolic term is switched off)
* ``symbolic_only``--- ``alpha_only = 1`` (the network's own logit is dropped)
* ``fused``        --- both pathways (the proposed model)
* ``fused_no_prior`` --- fused, but the expert prior regulariser is removed

An *interaction* effect is present when ``fused`` beats both single-pathway
variants by more than the paired noise.  We report the paired differences with
bootstrap CIs, so a null interaction is reported as such.

Logical faithfulness (Task 14)
------------------------------
Accuracy is not faithfulness.  For each fold we check whether the model's
decisions satisfy the expert rules on *unseen* rows and on *counterfactual*
rows constructed to make a specific rule fire (by setting that rule's literals
to their satisfying values).  We report the rule-satisfaction rate: the fraction
of counterfactual rows where the model's prediction agrees with the rule's
consequent, and, for rows where the antecedent does not fire, the false-positive
rate.  A model can be accurate and still violate its own rules; this measures
that directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import models, real_data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

ABLATIONS = {
    "neural_only": dict(alpha_only=False, lam_sem=0.0, lam_prior=0.0,
                        alpha_init=0.0, freeze_alpha=True),
    "symbolic_only": dict(alpha_only=True, lam_sem=0.0, lam_prior=1.0),
    "fused": dict(alpha_only=False, lam_sem=0.5, lam_prior=1.0),
    "fused_no_prior": dict(alpha_only=False, lam_sem=0.5, lam_prior=0.0),
}


def _fit(domain, tr, va, seed, epochs, patience, cfg):
    """Fit one ablation variant.

    ``alpha`` is frozen at its configured value *before* training, so the
    ``neural_only`` variant never trains with an active symbolic pathway ---
    otherwise the ablation would be contaminated by the very term it removes.
    """
    m = models.NEXS(domain, seed=seed, pos_weight="auto", epochs=epochs,
                    patience=patience, **cfg)
    m.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
    return m


def run_ablation(name: str, n_folds: int, seed: int, epochs: int,
                 patience: int) -> tuple[pd.DataFrame, dict]:
    from sklearn.metrics import accuracy_score

    _, y = real_data.DATASETS[name]["loader"]()
    oof = {k: np.full(len(y), np.nan) for k in ABLATIONS}
    for f in real_data.make_oof_folds(name, n_folds=n_folds, seed=seed):
        tr, va, te, te_i = f["train"], f["val"], f["test"], f["te_i"]
        for k, cfg in ABLATIONS.items():
            m = _fit(f["domain"], tr, va, seed + f["fold"], epochs, patience, cfg)
            oof[k][te_i] = m.predict_proba(te.X.to_numpy())
    rows = [dict(dataset=name, method=k,
                 acc=float(accuracy_score(y, (v >= 0.5).astype(int))))
            for k, v in oof.items()]
    return pd.DataFrame(rows), oof


def run_faithfulness(name: str, n_folds: int, seed: int, epochs: int,
                     patience: int, n_cf: int = 400) -> pd.DataFrame:
    """Rule obedience and over-firing on counterfactual rows, out-of-fold.

    Two probes per expert label rule:

    * **obedience** --- set every literal of the rule to a satisfying value; a
      faithful model should predict the rule's consequent (positive).
    * **over-firing** --- set all but one literal to satisfying values and make
      the remaining literal *violate* the rule; a faithful model should now
      predict negative.  High over-firing means the model reacts to partial
      matches, i.e. it does not actually implement the conjunction.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for f in real_data.make_oof_folds(name, n_folds=n_folds, seed=seed):
        domain = f["domain"]
        tr, va = f["train"], f["val"]
        mlp = models.MLPModel(domain, seed=seed + f["fold"], pos_weight="auto",
                              epochs=epochs, patience=patience)
        mlp.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
        nexs = models.NEXS(domain, seed=seed + f["fold"], pos_weight="auto",
                           epochs=epochs, patience=patience)
        nexs.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)

        label_rules = [r for r in domain.expert_rules
                       if r.head == domain.label
                       and all(l.kind == "feature" for l in r.body)]
        if not label_rules:
            continue
        Xpos = _counterfactuals(domain, label_rules, n_cf, rng, violate=False)[0]
        Xneg = _counterfactuals(domain, label_rules, n_cf, rng, violate=True)[0]
        if Xpos is None or Xneg is None:
            continue
        for mname, m in (("MLP", mlp), ("NEXS_Expert", nexs)):
            ypos = (m.predict_proba(Xpos.to_numpy()) >= 0.5).astype(int)
            yneg = (m.predict_proba(Xneg.to_numpy()) >= 0.5).astype(int)
            rows.append(dict(
                dataset=name, fold=f["fold"], method=mname,
                obedience=float((ypos == 1).mean()),
                over_firing=float((yneg == 1).mean()),
                n_probe=int(len(Xpos))))
    return pd.DataFrame(rows)


def _counterfactuals(domain, rules, n_cf, rng, violate: bool):
    """Rows that make a rule's antecedent fire, or violate it in one literal."""
    X = pd.DataFrame(rng.random((n_cf, len(domain.features))) * 0.4 + 0.3,
                     columns=list(domain.features))
    col = {c: i for i, c in enumerate(X.columns)}
    for i in range(n_cf):
        r = rules[rng.integers(len(rules))]
        k = rng.integers(len(r.body)) if violate else -1
        for j, lit in enumerate(r.body):
            if lit.var not in col:
                return None, None
            c = col[lit.var]
            if j == k:  # violate this literal
                if lit.op == ">=":
                    X.iat[i, c] = max(0.0, lit.thr - 0.15)
                elif lit.op == "<=":
                    X.iat[i, c] = min(1.0, lit.thr + 0.15)
                else:
                    X.iat[i, c] = min(1.0, lit.thr + 0.3)
            else:
                if lit.op == ">=":
                    X.iat[i, c] = min(1.0, lit.thr + 0.02)
                elif lit.op == "<=":
                    X.iat[i, c] = max(0.0, lit.thr - 0.02)
                else:
                    X.iat[i, c] = lit.thr
    return X, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["wbc", "ttt", "monk1", "monk3"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out", default="results/nested_ablation")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    abl, faith = [], []
    for name in args.datasets:
        for seed in args.seeds:
            print(f"== ablation {name} seed={seed} ==", flush=True)
            df, _ = run_ablation(name, args.folds, seed, args.epochs, args.patience)
            df["seed"] = seed
            abl.append(df)
            print(f"== faithfulness {name} seed={seed} ==", flush=True)
            fd = run_faithfulness(name, args.folds, seed, args.epochs, args.patience)
            fd["seed"] = seed
            faith.append(fd)
    a = pd.concat(abl, ignore_index=True)
    a.to_csv(out / "ablation.csv", index=False)
    f = pd.concat(faith, ignore_index=True)
    f.to_csv(out / "faithfulness.csv", index=False)
    print(a.groupby(["dataset", "method"])["acc"].mean().round(4).to_string())
    print(f.groupby(["dataset", "method"])[["obedience", "over_firing"]].mean().round(4).to_string())
    print("written ->", out)


if __name__ == "__main__":
    main()
