"""Multi-condition rule-quality / data-efficiency protocol.

Grid: ``datasets x seeds x conditions x methods`` where a *condition* is a pair
of (knowledge-base corruption, training-data fraction).

* ``kb_corruption`` in {0, .25, .5, .75, 1}: fraction of expert rules whose
  thresholds are perturbed (1.0 removes the KB entirely, i.e. no knowledge);
* ``data_frac`` in {.25, .5, 1}: fraction of the training split used.

This is the protocol that answers the paper's central question --- *when do
expert rules help neural learning?* --- by crossing how good the knowledge is
against how much data there is. Methods: the symbolic expert system (ESS), a
plain MLP, and NEXS.

Writes a long-format CSV (one row per run) consumable by
``advanced_analysis.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import _evaluate  # noqa: E402
from run_real_experiments import kb_oracle  # noqa: E402

from src import models, real_data, rules  # noqa: E402
from src.expert_system import run as symbolic_run  # noqa: E402

CORRUPTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
FRACTIONS = [0.25, 0.5, 1.0]


def corrupt_domain(domain, level: float, seed: int):
    """Perturb ``level`` of the expert rules' thresholds; 1.0 empties the KB."""
    if level <= 0:
        return domain
    if level >= 1.0:
        return dataclasses.replace(domain, expert_rules=tuple())
    rng = np.random.default_rng(seed)
    rules_ = list(domain.expert_rules)
    n_corrupt = int(round(level * len(rules_)))
    idx = rng.choice(len(rules_), size=n_corrupt, replace=False)
    for i in idx:
        r = rules_[i]
        body = []
        for lit in r.body:
            if lit.kind == "feature":
                jitter = float(rng.uniform(-0.35, 0.35))
                body.append(dataclasses.replace(lit, thr=float(np.clip(lit.thr + jitter, 0.0, 1.0))))
            else:
                body.append(lit)
        rules_[i] = dataclasses.replace(r, body=tuple(body))
    return dataclasses.replace(domain, expert_rules=tuple(rules_))


def run_grid(datasets, seeds, epochs, patience, out: Path) -> pd.DataFrame:
    rows = []
    for name in datasets:
        for seed in seeds:
            domain0, tr, va, te = real_data.load_real(name, seed=seed)
            Xte, yte = te.X.to_numpy(), te.y
            Xva, yva = va.X.to_numpy(), va.y
            Xtr_full, ytr_full = tr.X.to_numpy(), tr.y
            n_full = len(ytr_full)
            sup = real_data.feature_support(pd.concat([tr.X, va.X, te.X]))
            gt = real_data.ground_truth_rules(name)
            ref = ([frozenset(str(l) for l in r.body) for r in gt] if gt is not None
                   else [frozenset(str(l) for l in r.body) for r in domain0.expert_rules
                         if r.head == domain0.label])
            for frac in FRACTIONS:
                n = max(40, int(frac * n_full))
                idx = np.random.default_rng(seed + n).permutation(n_full)[:n]
                Xtr, ytr = Xtr_full[idx], ytr_full[idx]
                for level in CORRUPTIONS:
                    cond = f"kb{int(level*100):03d}_f{int(frac*100):03d}"
                    dom = corrupt_domain(domain0, level, seed)
                    kb = kb_oracle(dom, te.X) if dom.expert_rules else np.zeros(len(yte), int)
                    # symbolic expert system
                    sym = symbolic_run(dom, te.X, soft=False)
                    from run_experiments import _metrics
                    m = _metrics(yte, sym["label_conf"])
                    m["logic_fid"] = float((sym["y_hat"] == kb).mean()) if dom.expert_rules else float("nan")
                    rows.append(dict(dataset=name, seed=seed, condition=cond,
                                     kb_corruption=level, data_frac=frac,
                                     method="ESS", **m))
                    cands = {
                        "MLP": models.MLPModel(dom, seed=seed, pos_weight="auto",
                                               epochs=epochs, patience=patience),
                        "NEXS": models.NEXS(dom, seed=seed, pos_weight="auto",
                                            epochs=epochs, patience=patience),
                    }
                    for mname, model in cands.items():
                        t0 = time.time()
                        model.fit(Xtr, ytr, Xva, yva)
                        mm, _ = _evaluate(model, Xva, yva, Xte, yte, calibrate=True)
                        thr = mm.get("threshold", 0.5)
                        from run_experiments import _decisions
                        pred = rules.extract_rules(_decisions(model, thr), dom,
                                                   depth=real_data.DATASETS[name]["depth"],
                                                   seed=1, support=sup)
                        rr = rules.rule_set_match(pred, ref, support=sup)
                        mm.update(dataset=name, seed=seed, condition=cond,
                                  kb_corruption=level, data_frac=frac, method=mname,
                                  train_seconds=time.time() - t0,
                                  rule_f1=rr["f1"], rule_prec=rr["precision"],
                                  rule_rec=rr["recall"])
                        mm["logic_fid"] = (float((_decisions(model, thr)(te.X) == kb).mean())
                                           if dom.expert_rules else float("nan"))
                        rows.append(mm)
            print(f"  {name} seed={seed} done ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_runs.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["heart", "pima", "ttt"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    df = run_grid(args.datasets, args.seeds, args.epochs, args.patience, Path(args.output))
    print(f"records: {len(df)}  ->  {args.output}/all_runs.csv")


if __name__ == "__main__":
    main()
