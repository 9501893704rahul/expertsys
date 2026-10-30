"""Structural rule recovery, measured out-of-fold and against hidden truth.

The discovery learner in :mod:`src.discovery` sees only training rows, so this
is the non-circular test: the rules are found, not injected.  For every fold we
compare the discovered DNF with the published ground truth on three levels:

1. **Literal-set overlap** --- rule precision, recall, F1 over conjunctive rules
   (a rule matches if its literal set matches, up to threshold tolerance).
2. **Exact recovery** --- the discovered rule set equals the ground-truth set.
3. **Logical equivalence** --- the discovered DNF and the ground truth agree on
   *every* point of the complete input domain.  This is the strongest check and
   does not require the discovered rules to be syntactically identical, only
   logically equivalent.

We also report rule complexity (mean literals per rule, rule count) and the
model's decision agreement with the true label over the full domain.

Ground truth exists for ``ttt`` (eight winning lines), ``monk1`` and ``monk3``.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import discovery, real_data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# full discrete domains (for logical equivalence)
# --------------------------------------------------------------------------- #
def full_domain(name: str):
    """Enumerate the complete input domain as a normalized DataFrame + labels."""
    if name == "ttt":
        vals = [1.0, 0.0, 0.5]  # x, o, blank
        rows = list(itertools.product(vals, repeat=9))
        X = pd.DataFrame(rows, columns=real_data.TTT_CELLS)
        y = np.zeros(len(X), dtype=int)
        A = X.to_numpy()
        for line in real_data.TTT_LINES:
            y |= (A[:, list(line)] == 1.0).all(axis=1).astype(int)
        return X, y
    # MONK: enumerate the attribute domains in normalized units
    v = int(name[-1])
    ranges = [[1, 2, 3], [1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2]]
    norm = {1: 0.0, 2: 0.5, 3: 1.0, 4: 1.0}
    combos = list(itertools.product(*ranges))
    X = pd.DataFrame(combos, columns=real_data.MONK_COLS).astype(float)
    for c in X.columns:
        X[c] = X[c].map(norm).astype(float)
    a1, a2, a3, a4, a5, a6 = (X[c].to_numpy() for c in real_data.MONK_COLS)
    # in normalized units the codebook values are 1->0.0, 2->0.5, 3->1.0, 4->1.0
    if v == 1:
        y = ((a1 == a2) | (a5 == 0.0)).astype(int)
    else:
        y = (((a5 == 1.0) & (a4 == 0.0)) | ((a5 != 1.0) & (a2 != 1.0))).astype(int)
    return X, y


def truth_literals(name: str) -> list[frozenset]:
    """Ground-truth rule set as literal sets, in normalized units."""
    if name == "ttt":
        return [frozenset((real_data.TTT_CELLS[i], ">=", 0.75) for i in line)
                for line in real_data.TTT_LINES]
    v = int(name[-1])
    a1, a2, a3, a4, a5, a6 = real_data.MONK_COLS
    norm = {1: 0.0, 2: 0.5, 3: 1.0}
    if v == 1:
        rules = [frozenset({(a1, "==", norm[t]), (a2, "==", norm[t])})
                 for t in (1, 2, 3)]
        rules.append(frozenset({(a5, "==", norm[1])}))
        return rules
    rules = [frozenset({(a5, "==", norm[3]), (a4, "==", norm[1])})]
    rules += [frozenset({(a5, "==", norm[s]), (a2, "==", norm[t])})
              for s in (1, 2, 3) for t in (1, 2)]
    return rules


def _canon(rule) -> frozenset:
    """Canonicalize a discovered Rule into a comparable literal set."""
    return frozenset((lit.var, lit.op, round(float(lit.thr), 3)) for lit in rule.body)


def _truth_canon(s: frozenset) -> frozenset:
    return frozenset((v, op, round(float(t), 3)) for v, op, t in s)


def _semantic_key(lits: frozenset, Xd: pd.DataFrame) -> tuple:
    """A syntactic-independent key for a conjunction: the set of domain points
    it selects.  Two rules that fire on exactly the same points are equivalent
    even if written with different literals."""
    A = Xd.to_numpy(dtype=float)
    col = {c: i for i, c in enumerate(Xd.columns)}
    m = np.ones(len(Xd), dtype=bool)
    for v, op, t in lits:
        x = A[:, col[v]]
        if op == ">=":
            m &= x >= t
        elif op == "<=":
            m &= x <= t
        else:
            m &= np.isclose(x, t, atol=1e-6)
    return tuple(np.flatnonzero(m).tolist())


def _rule_decision(rules, X: pd.DataFrame) -> np.ndarray:
    """Crisp DNF decision: OR over rules of AND over literals."""
    if not rules:
        return np.zeros(len(X), dtype=int)
    A = X.to_numpy(dtype=float)
    col = {c: i for i, c in enumerate(X.columns)}
    out = np.zeros(len(X), dtype=bool)
    for r in rules:
        m = np.ones(len(X), dtype=bool)
        for lit in r.body:
            v = A[:, col[lit.var]]
            if lit.op == ">=":
                m &= v >= lit.thr
            elif lit.op == "<=":
                m &= v <= lit.thr
            else:
                m &= np.isclose(v, lit.thr, atol=1e-6)
        out |= m
    return out.astype(int)


def evaluate_fold(name: str, rules, Xd: pd.DataFrame, yd: np.ndarray) -> dict:
    truth_syn = {_truth_canon(s) for s in truth_literals(name)}
    disc_syn = {_canon(r) for r in rules}
    inter = len(truth_syn & disc_syn)
    prec = inter / max(1, len(disc_syn))
    rec = inter / max(1, len(truth_syn))
    f1 = 2 * prec * rec / max(1e-12, prec + rec)

    # semantic (point-set) comparison: rewards logically equivalent rules even
    # when the literal syntax differs (e.g. `cell >= 0.75` vs `cell = 1`).
    truth_sem = {_semantic_key(s, Xd) for s in truth_literals(name)}
    disc_sem = {_semantic_key(_canon(r), Xd) for r in rules}
    inter_s = len(truth_sem & disc_sem)
    prec_s = inter_s / max(1, len(disc_sem))
    rec_s = inter_s / max(1, len(truth_sem))
    f1_s = 2 * prec_s * rec_s / max(1e-12, prec_s + rec_s)

    pred = _rule_decision(rules, Xd)
    equiv = float((pred == yd).mean())
    return dict(
        n_truth=len(truth_syn), n_discovered=len(disc_syn),
        exact_recovery=int(truth_syn == disc_syn),
        exact_recovery_semantic=int(truth_sem == disc_sem),
        rule_precision=prec, rule_recall=rec, rule_f1=f1,
        rule_precision_semantic=prec_s, rule_recall_semantic=rec_s,
        rule_f1_semantic=f1_s,
        mean_literals=float(np.mean([len(r.body) for r in rules])) if rules else 0.0,
        logical_equivalence=equiv,
        decision_agreement=equiv)


def run(name: str, n_folds: int, seed: int, max_lits: int = 3,
        min_precision: float = 0.85) -> pd.DataFrame:
    Xd, yd = full_domain(name)
    rows = []
    for f in real_data.make_oof_folds(name, n_folds=n_folds, seed=seed):
        rules = discovery.discover_rules(
            f["domain"], f["train"].X, f["train"].y,
            max_lits=max_lits, min_precision=min_precision)
        r = evaluate_fold(name, rules, Xd, yd)
        rows.append(dict(dataset=name, seed=seed, fold=f["fold"],
                         n_train=len(f["train"].y), **r))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["ttt", "monk1", "monk3"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-lits", type=int, default=3)
    ap.add_argument("--min-precision", type=float, default=0.85)
    ap.add_argument("--out", default="results/nested_structural")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in args.datasets:
        for seed in args.seeds:
            print(f"== structural recovery {name} seed={seed} ==", flush=True)
            frames.append(run(name, args.folds, seed, args.max_lits,
                              args.min_precision))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "structural_recovery_oof.csv", index=False)
    agg = df.groupby("dataset")[
        ["n_truth", "n_discovered", "exact_recovery", "exact_recovery_semantic",
         "rule_precision_semantic", "rule_recall_semantic", "rule_f1_semantic",
         "mean_literals", "logical_equivalence"]].mean()
    print(agg.round(4).to_string())
    print("written ->", out)


if __name__ == "__main__":
    main()
