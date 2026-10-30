"""Main experiment driver.

Runs the full benchmark for every (domain, method) pair and writes tidy CSV
results plus a JSON summary to ``results/``.  Experiments:

  E1  accuracy / F1 / AUC / calibration on held-out data
  E2  rule recovery vs. the ground-truth rule base (rule-set F1)
  E3  logic fidelity to the deterministic oracle on a uniform grid
  E4  data efficiency (accuracy vs. training-set size)
  E5  distribution shift (covariate shift + novel-regime robustness)
  E6  expert-knowledge ablation for the proposed model
  E7  inference latency

Everything is CPU-only and deterministic given the seed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import baselines, models, rules
from src.data import make_splits
from src.expert_system import run as symbolic_run
from src.knowledge import DOMAINS

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
def _metrics(y, proba, thr=0.5) -> dict:
    yhat = (proba >= thr).astype(int)
    out = {
        "acc": accuracy_score(y, yhat),
        "f1": f1_score(y, yhat, zero_division=0),
        "auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(np.mean((proba - y) ** 2)),
    }
    # expected calibration error (10 bins)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        m = idx == b
        if m.sum():
            ece += m.mean() * abs(proba[m].mean() - y[m].mean())
    out["ece"] = float(ece)
    return out


def _fit_calibrator(p_val, y_val):
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds="clip").fit(p_val, y_val)


def _best_threshold(p_val, y_val):
    """F1-optimal decision threshold chosen on the validation split."""
    thrs = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thrs:
        f = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def _evaluate(model, Xva, yva, Xte, yte, calibrate=True):
    """Calibrate on validation, pick the F1-optimal threshold, score on test."""
    p_va = model.predict_proba(Xva)
    p_te = model.predict_proba(Xte)
    if calibrate:
        iso = _fit_calibrator(p_va, yva)
        p_va, p_te = iso.predict(p_va), iso.predict(p_te)
    thr = _best_threshold(p_va, yva)
    m = _metrics(yte, p_te, thr=thr)
    m["threshold"] = thr
    return m, p_te


def _decisions(model, thr: float = 0.5):
    """Wrap a model's thresholded decision as a plain function on a DataFrame."""
    def fn(X):
        arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        return (model.predict_proba(arr) >= thr).astype(int)
    return fn


def _rule_recovery(model, domain, thr: float = 0.5) -> dict:
    pred = rules.extract_rules(_decisions(model, thr), domain, seed=1)
    ref = rules.ground_truth_dnf(domain, domain.label)   # feature-only DNF
    return rules.rule_set_match(pred, ref)


# --------------------------------------------------------------------------- #
def e1_e3_main(domain, seed, n=4000, calib=True) -> pd.DataFrame:
    tr, va, te = make_splits(domain, n=n, seed=seed)
    Xtr, ytr = tr.X.to_numpy(), tr.y
    Xva, yva = va.X.to_numpy(), va.y
    Xte, yte = te.X.to_numpy(), te.y

    rows = []

    # symbolic expert system
    sym = symbolic_run(domain, te.X, soft=False)
    sym_soft = symbolic_run(domain, te.X, soft=True)
    for tag, out in (("ESS(discrete)", sym), ("ESS(soft)", sym_soft)):
        m = _metrics(yte, out["label_conf"])
        m.update(method=tag, domain=domain.name, seed=seed, family="symbolic")
        rows.append(m)
    # rule recovery / fidelity for the discrete ESS
    rr = rules.rule_set_match(
        [frozenset(str(l) for l in r.body) for r in domain.expert_rules if r.head == domain.label],
        rules.ground_truth_dnf(domain, domain.label),
    )
    fid = rules.logic_fidelity(lambda X: symbolic_run(domain, X)["y_hat"], domain)
    rows[-2].update(rule_f1=rr["f1"], rule_prec=rr["precision"], rule_rec=rr["recall"], logic_fid=fid)

    built = {
        **{k: v for k, v in baselines.build_baselines(seed).items()},
        "MLP": models.MLPModel(domain, seed=seed),
        "SemanticLoss": models.SemanticLossModel(domain, seed=seed),
        "NEXS(proposed)": models.NEXS(domain, seed=seed),
    }

    for name, model in built.items():
        t0 = time.time()
        model.fit(Xtr, ytr, Xva, yva)
        train_t = time.time() - t0
        m, _ = _evaluate(model, Xva, yva, Xte, yte, calibrate=calib)
        m.update(method=name, domain=domain.name, seed=seed,
                 family="neuro-symbolic" if name in ("SemanticLoss", "NEXS(proposed)") else
                 ("neural" if name == "MLP" else "ml"),
                 train_seconds=train_t)
        thr = m.get("threshold", 0.5)
        rr = _rule_recovery(model, domain, thr)
        m.update(rule_f1=rr["f1"], rule_prec=rr["precision"], rule_rec=rr["recall"])
        m["logic_fid"] = rules.logic_fidelity(_decisions(model, thr), domain)
        rows.append(m)

    return pd.DataFrame(rows)


def e4_data_efficiency(domain, seed, sizes=(150, 300, 600, 1200, 2400)) -> pd.DataFrame:
    rows = []
    full = make_splits(domain, n=4000, seed=seed)
    Xte, yte = full[2].X.to_numpy(), full[2].y
    for n in sizes:
        tr, va, _ = make_splits(domain, n=n + 800, seed=seed + 100 + n)
        Xtr, ytr, Xva, yva = tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y
        cands = {
            "RandomForest": baselines.build_baselines(seed)["RandomForest"],
            "MLP": models.MLPModel(domain, seed=seed),
            "SemanticLoss": models.SemanticLossModel(domain, seed=seed),
            "NEXS(proposed)": models.NEXS(domain, seed=seed),
        }
        for name, model in cands.items():
            model.fit(Xtr, ytr, Xva, yva)
            m, _ = _evaluate(model, Xva, yva, Xte, yte, calibrate=True)
            rows.append(dict(domain=domain.name, seed=seed, n_train=n, method=name, **m))
        sym = symbolic_run(domain, full[2].X, soft=False)
        m = _metrics(yte, sym["label_conf"])
        rows.append(dict(domain=domain.name, seed=seed, n_train=n, method="ESS(discrete)", **m))
    return pd.DataFrame(rows)


def _shifted(domain, n, seed, kind) -> pd.DataFrame:
    from src.data import generate

    ds = generate(domain, n=n, seed=seed)
    X = ds.X.copy()
    rng = np.random.default_rng(seed + 999)
    if kind == "covariate":  # mild covariate shift: stretch continuous features
        for f in domain.features:
            if f not in domain.binary_features:
                X[f] = np.clip(X[f] * 1.15 + 0.05 + rng.normal(0, 0.02, len(X)), 0, 1)
    elif kind == "novel":  # novel regime: swap tails of two key features
        cols = [f for f in domain.features if f not in domain.binary_features][:2]
        X[cols[0]] = 1.0 - X[cols[0]]
    return X


def e5_shift(domain, seed, n=4000) -> pd.DataFrame:
    tr, va, te = make_splits(domain, n=n, seed=seed)
    Xtr, ytr, Xva, yva = tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y
    rows = []
    cands = {
        "RandomForest": baselines.build_baselines(seed)["RandomForest"],
        "MLP": models.MLPModel(domain, seed=seed),
        "SemanticLoss": models.SemanticLossModel(domain, seed=seed),
        "NEXS(proposed)": models.NEXS(domain, seed=seed),
    }
    fitted = {name: m.fit(Xtr, ytr, Xva, yva) for name, m in cands.items()}
    for kind in ("covariate", "novel"):
        Xs = _shifted(domain, len(te.y), seed, kind)
        # labels under the true rules on the shifted inputs
        ys = rules.oracle(domain, Xs)["label"]
        for name, model in fitted.items():
            p_va = model.predict_proba(Xva)
            thr = _best_threshold(p_va, yva)  # threshold fixed on the clean distribution
            m = _metrics(ys, model.predict_proba(Xs.to_numpy()), thr=thr)
            rows.append(dict(domain=domain.name, seed=seed, shift=kind, method=name, **m))
        sym = symbolic_run(domain, Xs, soft=False)
        m = _metrics(ys, sym["label_conf"])
        rows.append(dict(domain=domain.name, seed=seed, shift=kind, method="ESS(discrete)", **m))
    return pd.DataFrame(rows)


def e6_ablation(domain, seed, n=4000) -> pd.DataFrame:
    tr, va, te = make_splits(domain, n=n, seed=seed)
    Xtr, ytr, Xva, yva = tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y
    Xte, yte = te.X.to_numpy(), te.y
    variants = {
        "NEXS(full)": dict(),
        "NEXS(no rule layer)": dict(),
        "NEXS(rule layer only)": dict(),
        "NEXS(no semantic loss)": dict(lam_sem=0.0),
        "NEXS(no expert prior)": dict(lam_prior=0.0),
    }
    rows = []
    for name, kw in variants.items():
        if name == "NEXS(no rule layer)":
            model = models.MLPModel(domain, seed=seed)
        elif name == "NEXS(rule layer only)":
            model = models.NEXS(domain, seed=seed, alpha_only=True)
        else:
            model = models.NEXS(domain, seed=seed, **kw)
        model.fit(Xtr, ytr, Xva, yva)
        m, _ = _evaluate(model, Xva, yva, Xte, yte, calibrate=True)
        thr = m.get("threshold", 0.5)
        m.update(domain=domain.name, seed=seed, variant=name)
        m["logic_fid"] = rules.logic_fidelity(_decisions(model, thr), domain)
        rows.append(m)
    return pd.DataFrame(rows)


def e7_latency(domain, seed, n=4000, reps=30) -> pd.DataFrame:
    tr, va, te = make_splits(domain, n=n, seed=seed)
    Xtr, ytr, Xva, yva = tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y
    Xte = te.X.to_numpy()
    rows = []
    cands = {
        "RandomForest": baselines.build_baselines(seed)["RandomForest"],
        "MLP": models.MLPModel(domain, seed=seed),
        "SemanticLoss": models.SemanticLossModel(domain, seed=seed),
        "NEXS(proposed)": models.NEXS(domain, seed=seed),
    }
    for name, model in cands.items():
        model.fit(Xtr, ytr, Xva, yva)
        t0 = time.time()
        for _ in range(reps):
            model.predict_proba(Xte)
        dt = (time.time() - t0) / reps / len(Xte) * 1e6  # microseconds / sample
        rows.append(dict(domain=domain.name, method=name, us_per_sample=dt))
    t0 = time.time()
    for _ in range(reps):
        symbolic_run(domain, te.X, soft=False)
    rows.append(dict(domain=domain.name, method="ESS(discrete)",
                     us_per_sample=(time.time() - t0) / reps / len(Xte) * 1e6))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--quick", action="store_true", help="smaller/faster run")
    args = ap.parse_args()
    seeds = args.seeds[:1] if args.quick else args.seeds
    n = 2000 if args.quick else args.n

    all_main, all_eff, all_shift, all_abl, all_lat = [], [], [], [], []
    for dname, domain in DOMAINS.items():
        for s in seeds:
            print(f"== {dname} seed={s} ==", flush=True)
            all_main.append(e1_e3_main(domain, s, n=n))
            all_shift.append(e5_shift(domain, s, n=n))
            all_abl.append(e6_ablation(domain, s, n=n))
            all_lat.append(e7_latency(domain, s, n=n))
        all_eff.append(e4_data_efficiency(domain, seeds[0]))

    main_df = pd.concat(all_main, ignore_index=True)
    eff_df = pd.concat(all_eff, ignore_index=True)
    shift_df = pd.concat(all_shift, ignore_index=True)
    abl_df = pd.concat(all_abl, ignore_index=True)
    lat_df = pd.concat(all_lat, ignore_index=True)

    main_df.to_csv(RESULTS / "e1_main.csv", index=False)
    eff_df.to_csv(RESULTS / "e4_data_efficiency.csv", index=False)
    shift_df.to_csv(RESULTS / "e5_shift.csv", index=False)
    abl_df.to_csv(RESULTS / "e6_ablation.csv", index=False)
    lat_df.to_csv(RESULTS / "e7_latency.csv", index=False)

    summary = {
        "n": n,
        "seeds": seeds,
        "main_mean": main_df.groupby(["domain", "method"]).mean(numeric_only=True).reset_index().to_dict("records"),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote results to", RESULTS)
    print(main_df.groupby(["domain", "method"])[["acc", "f1", "auc", "ece"]].mean().round(3))


if __name__ == "__main__":
    main()
