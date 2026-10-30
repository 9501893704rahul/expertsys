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
    """Enumerate the complete input domain as a normalized DataFrame + labels.

    The MONK attributes are enumerated over their *declared* codebooks
    (:data:`real_data.MONK_NORM`) and labelled by the published formula, so the
    domain is the true 432-point input space rather than a 144-point grid with
    collapsed levels.  The labels are cross-checked against the observed rows:
    a mismatch would mean the codebook and the dataset disagree, which would
    invalidate the structural comparison.
    """
    if name == "ttt":
        vals = [1.0, 0.0, 0.5]  # x, o, blank
        rows = list(itertools.product(vals, repeat=9))
        X = pd.DataFrame(rows, columns=real_data.TTT_CELLS)
        y = np.zeros(len(X), dtype=int)
        A = X.to_numpy()
        for line in real_data.TTT_LINES:
            y |= (A[:, list(line)] == 1.0).all(axis=1).astype(int)
        return X, y

    # MONK: full cartesian product of the declared attribute codebooks
    v = int(name[-1])
    X, y = real_data.monk_full_domain(v)
    # MONK-3's training file carries 5% label noise, so a handful of observed
    # rows legitimately disagree with the clean formula; MONK-1 must be exact.
    _check_codebook(name, X, y, tol=0.0 if v == 1 else 0.05)
    return X, y


def _check_codebook(name: str, X: pd.DataFrame, y: np.ndarray,
                    tol: float = 0.0) -> None:
    """Verify the enumerated domain reproduces the observed labels.

    ``tol`` is the fraction of observed rows allowed to disagree, used for
    MONK-3's documented 5% class noise.  Both sides are keyed through the same
    rounding, so a formatting mismatch cannot silently skip the comparison.
    """
    Xraw, yraw = real_data.DATASETS[name]["loader"]()
    cols = list(X.columns)

    def key(row) -> str:
        return "|".join(f"{float(row[c]):.6f}" for c in cols)

    dom = {key(X.iloc[i]): int(y[i]) for i in range(len(X))}
    n_bad, n_seen = 0, 0
    for i in range(len(Xraw)):
        k = key(Xraw.iloc[i])
        if k in dom:
            n_seen += 1
            if dom[k] != int(yraw[i]):
                n_bad += 1
    if n_seen == 0:
        raise AssertionError(
            f"{name}: enumerated domain matched none of the observed rows --- "
            f"encoding mismatch between the loader and the domain grid")
    frac = n_bad / n_seen
    if frac > tol + 1e-9:
        raise AssertionError(
            f"{name}: enumerated ground truth disagrees with the dataset on "
            f"{n_bad} of {n_seen} observed rows ({frac:.1%} > tol {tol:.1%}) "
            f"--- codebook or formula is wrong")


def truth_literals(name: str) -> list[frozenset]:
    """Ground-truth rule set as literal sets, in normalized units."""
    if name == "ttt":
        return [frozenset((real_data.TTT_CELLS[i], ">=", 0.75) for i in line)
                for line in real_data.TTT_LINES]
    return real_data.monk_truth_dnf(int(name[-1]))


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
