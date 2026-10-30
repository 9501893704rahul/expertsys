"""Real-data experiments (heart, pima, tic-tac-toe).

Mirrors the synthetic study but on real data.  The differences that matter:

* There is no synthetic shift, so robustness (E5) is measured on **natural
  subgroups** of the test set (e.g. older patients, obese patients), which is
  the realistic analogue of a covariate shift.
* On ``ttt`` the eight winning lines are a genuine ground truth, so rule
  recovery (E2) is measured against the true logic.  On ``heart``/``pima`` there
  is no ground truth, so E2 measures recovery of the *expert knowledge base*
  itself, and E3 measures agreement with the KB oracle --- i.e. how much of the
  expert's logic the learned model preserves.

Writes ``results/real_*.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import (  # noqa: E402
    _best_threshold, _decisions, _evaluate, _fit_calibrator, _metrics, _rule_recovery,
)

from src import baselines, models, real_data, rules  # noqa: E402
from src.expert_system import run as symbolic_run  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

DATASETS = ["heart", "pima", "ttt"]


# --------------------------------------------------------------------------- #
def kb_oracle(domain, X):
    """Deterministic forward chaining with the expert KB (the deployed logic)."""
    return rules.oracle(domain, X, rules=domain.expert_rules)["label"]


def _subgroup(name, X_raw_cols):
    """Natural subgroup masks for the robustness experiment."""
    if name == "heart":
        return "age", X_raw_cols["age"] >= X_raw_cols["age"].median()
    if name == "pima":
        return "bmi>=30", X_raw_cols["bmi"] >= 0.5
    return "x-majority", X_raw_cols[[c for c in X_raw_cols.columns]].sum(axis=1) > 4.5


def e1_e3(name, seed):
    domain, tr, va, te = real_data.load_real(name, seed=seed)
    Xtr, ytr = tr.X.to_numpy(), tr.y
    Xva, yva = va.X.to_numpy(), va.y
    Xte, yte = te.X.to_numpy(), te.y
    rows = []

    # expert system + KB oracle
    sym = symbolic_run(domain, te.X, soft=False)
    sym_soft = symbolic_run(domain, te.X, soft=True)
    kb = kb_oracle(domain, te.X)
    # reference for rule recovery: the genuine winning-line logic on ttt, the
    # expert KB elsewhere (where no ground truth exists).
    gt = real_data.ground_truth_rules(name)
    if gt is not None:
        ref = [frozenset(str(l) for l in r.body) for r in gt]
    else:
        ref = [frozenset(str(l) for l in r.body) for r in domain.expert_rules
               if r.head == domain.label]
    sup = real_data.feature_support(pd.concat([tr.X, va.X, te.X]))
    rr = rules.rule_set_match(
        [frozenset(str(l) for l in r.body) for r in domain.expert_rules
         if r.head == domain.label], ref, support=sup)
    for tag, out in (("ESS(discrete)", sym), ("ESS(soft)", sym_soft)):
        m = _metrics(yte, out["label_conf"])
        m.update(method=tag, dataset=name, seed=seed, family="symbolic",
                 rule_f1=rr["f1"], rule_prec=rr["precision"], rule_rec=rr["recall"],
                 logic_fid=float((out["y_hat"] == kb).mean()))
        rows.append(m)

    built = {
        **baselines.build_baselines(seed),
        "MLP": models.MLPModel(domain, seed=seed),
        "SemanticLoss": models.SemanticLossModel(domain, seed=seed),
        "NEXS(proposed)": models.NEXS(domain, seed=seed),
    }
    for mname, model in built.items():
        t0 = time.time()
        model.fit(Xtr, ytr, Xva, yva)
        train_t = time.time() - t0
        m, _ = _evaluate(model, Xva, yva, Xte, yte, calibrate=True)
        thr = m.get("threshold", 0.5)
        pred = rules.extract_rules(_decisions(model, thr), domain,
                                   depth=real_data.DATASETS[name]["depth"], seed=1,
                                   support=sup)
        r = rules.rule_set_match(pred, ref, support=sup)
        m.update(method=mname, dataset=name, seed=seed,
                 family="neuro-symbolic" if mname in ("SemanticLoss", "NEXS(proposed)")
                 else ("neural" if mname == "MLP" else "ml"),
                 train_seconds=train_t,
                 rule_f1=r["f1"], rule_prec=r["precision"], rule_rec=r["recall"])
        m["logic_fid"] = float((_decisions(model, thr)(te.X) == kb).mean())
        rows.append(m)
    return pd.DataFrame(rows)


def e4_efficiency(name, seed, fractions=(0.25, 0.5, 0.75, 1.0)):
    domain, tr, va, te = real_data.load_real(name, seed=seed)
    Xte, yte = te.X.to_numpy(), te.y
    Xva, yva = va.X.to_numpy(), va.y
    n_full = len(tr.y)
    rows = []
    for frac in fractions:
        n = max(40, int(frac * n_full))
        rng = np.random.default_rng(seed + n)
        idx = rng.permutation(n_full)[:n]
        Xtr, ytr = tr.X.to_numpy()[idx], tr.y[idx]
        cands = {
            "RandomForest": baselines.build_baselines(seed)["RandomForest"],
            "MLP": models.MLPModel(domain, seed=seed),
            "SemanticLoss": models.SemanticLossModel(domain, seed=seed),
            "NEXS(proposed)": models.NEXS(domain, seed=seed),
        }
        for mname, model in cands.items():
            model.fit(Xtr, ytr, Xva, yva)
            m, _ = _evaluate(model, Xva, yva, Xte, yte, calibrate=True)
            rows.append(dict(dataset=name, seed=seed, n_train=n, frac=frac,
                             method=mname, **m))
        sym = symbolic_run(domain, te.X, soft=False)
        rows.append(dict(dataset=name, seed=seed, n_train=n, frac=frac,
                         method="ESS(discrete)", **_metrics(yte, sym["label_conf"])))
    return pd.DataFrame(rows)


def e5_subgroup(name, seed):
    domain, tr, va, te = real_data.load_real(name, seed=seed)
    Xtr, ytr = tr.X.to_numpy(), tr.y
    Xva, yva = va.X.to_numpy(), va.y
    tag, mask = _subgroup(name, te.X)
    mask = np.asarray(mask)
    rows = []
    cands = {
        "RandomForest": baselines.build_baselines(seed)["RandomForest"],
        "MLP": models.MLPModel(domain, seed=seed),
        "SemanticLoss": models.SemanticLossModel(domain, seed=seed),
        "NEXS(proposed)": models.NEXS(domain, seed=seed),
    }
    fitted = {k: m.fit(Xtr, ytr, Xva, yva) for k, m in cands.items()}
    kb = kb_oracle(domain, te.X)
    for grp, sel in ((f"{tag}=low", ~mask), (f"{tag}=high", mask)):
        if sel.sum() < 10:
            continue
        Xg, yg = te.X.to_numpy()[sel], te.y[sel]
        for mname, model in fitted.items():
            thr = _best_threshold(model.predict_proba(Xva), yva)
            m = _metrics(yg, model.predict_proba(Xg), thr=thr)
            rows.append(dict(dataset=name, seed=seed, subgroup=grp, method=mname, **m))
        out = symbolic_run(domain, te.X.iloc[sel], soft=False)
        m = _metrics(yg, out["label_conf"])
        rows.append(dict(dataset=name, seed=seed, subgroup=grp,
                         method="ESS(discrete)", **m))
    return pd.DataFrame(rows)


def e6_ablation(name, seed):
    domain, tr, va, te = real_data.load_real(name, seed=seed)
    Xtr, ytr = tr.X.to_numpy(), tr.y
    Xva, yva = va.X.to_numpy(), va.y
    Xte, yte = te.X.to_numpy(), te.y
    kb = kb_oracle(domain, te.X)
    variants = {
        "NEXS(full)": ("nexs", {}),
        "NEXS(no rule layer)": ("mlp", {}),
        "NEXS(rule layer only)": ("nexs", dict(alpha_only=True)),
        "NEXS(no semantic loss)": ("nexs", dict(lam_sem=0.0)),
        "NEXS(no expert prior)": ("nexs", dict(lam_prior=0.0)),
    }
    rows = []
    for vname, (kind, kw) in variants.items():
        model = models.MLPModel(domain, seed=seed) if kind == "mlp" else models.NEXS(domain, seed=seed, **kw)
        model.fit(Xtr, ytr, Xva, yva)
        m, _ = _evaluate(model, Xva, yva, Xte, yte, calibrate=True)
        thr = m.get("threshold", 0.5)
        m.update(dataset=name, seed=seed, variant=vname)
        m["logic_fid"] = float((_decisions(model, thr)(te.X) == kb).mean())
        rows.append(m)
    return pd.DataFrame(rows)


def e7_latency(name, seed, reps=30):
    domain, tr, va, te = real_data.load_real(name, seed=seed)
    Xtr, ytr = tr.X.to_numpy(), tr.y
    Xva, yva = va.X.to_numpy(), va.y
    Xte = te.X.to_numpy()
    rows = []
    cands = {
        "RandomForest": baselines.build_baselines(seed)["RandomForest"],
        "MLP": models.MLPModel(domain, seed=seed),
        "SemanticLoss": models.SemanticLossModel(domain, seed=seed),
        "NEXS(proposed)": models.NEXS(domain, seed=seed),
    }
    for mname, model in cands.items():
        model.fit(Xtr, ytr, Xva, yva)
        t0 = time.time()
        for _ in range(reps):
            model.predict_proba(Xte)
        rows.append(dict(dataset=name, method=mname,
                         us_per_sample=(time.time() - t0) / reps / len(Xte) * 1e6))
    t0 = time.time()
    for _ in range(reps):
        symbolic_run(domain, te.X, soft=False)
    rows.append(dict(dataset=name, method="ESS(discrete)",
                     us_per_sample=(time.time() - t0) / reps / len(Xte) * 1e6))
    return pd.DataFrame(rows)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()

    main_rows, eff_rows, sub_rows, abl_rows, lat_rows = [], [], [], [], []
    for name in args.datasets:
        for s in args.seeds:
            print(f"== real {name} seed={s} ==", flush=True)
            main_rows.append(e1_e3(name, s))
            sub_rows.append(e5_subgroup(name, s))
            abl_rows.append(e6_ablation(name, s))
            lat_rows.append(e7_latency(name, s))
        eff_rows.append(e4_efficiency(name, args.seeds[0]))

    main_df = pd.concat(main_rows, ignore_index=True)
    eff_df = pd.concat(eff_rows, ignore_index=True)
    sub_df = pd.concat(sub_rows, ignore_index=True)
    abl_df = pd.concat(abl_rows, ignore_index=True)
    lat_df = pd.concat(lat_rows, ignore_index=True)

    main_df.to_csv(RESULTS / "real_e1_main.csv", index=False)
    eff_df.to_csv(RESULTS / "real_e4_data_efficiency.csv", index=False)
    sub_df.to_csv(RESULTS / "real_e5_subgroup.csv", index=False)
    abl_df.to_csv(RESULTS / "real_e6_ablation.csv", index=False)
    lat_df.to_csv(RESULTS / "real_e7_latency.csv", index=False)

    (RESULTS / "real_summary.json").write_text(json.dumps({
        "seeds": args.seeds, "datasets": args.datasets,
        "main_mean": main_df.groupby(["dataset", "method"]).mean(numeric_only=True)
        .reset_index().to_dict("records"),
    }, indent=2, default=float))
    print("\nWrote real-data results to", RESULTS)
    print(main_df.groupby(["dataset", "method"])[["acc", "f1", "auc", "ece"]].mean().round(3))


if __name__ == "__main__":
    main()
