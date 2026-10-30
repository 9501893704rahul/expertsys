"""Ground-truth logic oracle, rule extraction, and rule-recovery metrics.

The point of this module is to answer a question that accuracy cannot: *did the
model learn the underlying logic?*  We compare every method's decision function
against a deterministic oracle built from the true rule base, both on the data
distribution and on a uniform grid over the feature cube (which probes the
logic away from the training support).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from .knowledge import Domain, Literal, Rule


# --------------------------------------------------------------------------- #
# deterministic oracle
# --------------------------------------------------------------------------- #
def _sat(X: pd.DataFrame, lit: Literal, concepts: pd.DataFrame | None) -> np.ndarray:
    if lit.kind == "concept":
        return concepts[lit.var].to_numpy() >= 0.5
    v = X[lit.var].to_numpy(dtype=float)
    if lit.op == ">=":
        return v >= lit.thr
    if lit.op == "<=":
        return v <= lit.thr
    return np.isclose(v, lit.thr)


def oracle(domain: Domain, X: pd.DataFrame, rules: tuple[Rule, ...] | None = None) -> dict:
    """Crisp forward chaining with the (true) rule base."""
    rules = rules if rules is not None else domain.true_rules
    concepts = pd.DataFrame(index=X.index)
    for c in domain.concepts:
        fired = np.zeros(len(X), dtype=bool)
        for r in rules:
            if r.head != c:
                continue
            m = np.ones(len(X), dtype=bool)
            for lit in r.body:
                m &= _sat(X, lit, None)
            fired |= m
        concepts[c] = fired.astype(int)
    label = np.zeros(len(X), dtype=bool)
    for r in rules:
        if r.head != domain.label:
            continue
        m = np.ones(len(X), dtype=bool)
        for lit in r.body:
            m &= _sat(X, lit, concepts)
        label |= m
    return {"concepts": concepts, "label": label.astype(int)}


def uniform_grid(domain: Domain, n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for f in domain.features:
        cols[f] = rng.random(n) if f not in domain.binary_features else (rng.random(n) < 0.5).astype(float)
    return pd.DataFrame(cols)


def _support_grid(domain: Domain, support: dict, n: int, seed: int = 7) -> pd.DataFrame:
    """Grid that samples low-cardinality features only on their real values."""
    rng = np.random.default_rng(seed)
    cols = {}
    for f in domain.features:
        vals = support.get(f)
        if vals is not None and len(vals) > 0:
            vals = np.asarray(sorted(set(np.asarray(vals, dtype=float))))
            cols[f] = vals[rng.integers(0, len(vals), n)]
        elif f in domain.binary_features:
            cols[f] = (rng.random(n) < 0.5).astype(float)
        else:
            cols[f] = rng.random(n)
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- #
# rule extraction from a black-box decision function
# --------------------------------------------------------------------------- #
def extract_rules(decision_fn, domain: Domain, n: int = 4000, depth: int = 3,
                  seed: int = 0, support: dict | None = None,
                  grid_n: int = 4000) -> list[frozenset[str]]:
    """Distil a depth-limited decision tree from any binary decision function.

    The tree is fit on uniform samples so that the extracted rules describe the
    model's global logic rather than the training support.

    ``support`` optionally maps low-cardinality features to their observed values.
    Without it a continuous grid is used, which lets the tree split at
    interpolated points (e.g. ``cell <= 0.549``) and hides that a model learned
    the exact discrete rule.  With it those features are sampled only on their
    real values, so the extracted literals line up with the true logic.
    """
    if support:
        Xg = _support_grid(domain, support, grid_n, seed)
    else:
        Xg = uniform_grid(domain, n, seed=seed)
    yg = decision_fn(Xg)
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=max(20, n // 200),
                                  random_state=seed).fit(Xg.to_numpy(), yg)
    return _tree_to_rules(tree, domain.features)


def _tree_to_rules(tree: DecisionTreeClassifier, features: tuple[str, ...]) -> list[frozenset[str]]:
    t = tree.tree_
    rules: list[frozenset[str]] = []

    def recurse(node: int, lits: set[str]):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            counts = t.value[node][0]
            if counts.size > 1 and counts[1] > counts[0]:  # majority class = 1
                rules.append(frozenset(lits))
            return
        name = features[t.feature[node]]
        thr = t.threshold[node]
        recurse(t.children_left[node], lits | {f"{name}<={thr:.3f}"})
        recurse(t.children_right[node], lits | {f"{name}>{thr:.3f}"})

    recurse(0, set())
    return rules


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _parse_literal(s: str):
    import re

    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(<=|>=|>|<|=)(-?[0-9.]+)", s)
    return (m.group(1), m.group(2), float(m.group(3))) if m else (s, "", np.nan)


def _indicator(var: str, op: str, thr: float, grid: np.ndarray) -> np.ndarray:
    if op == ">=":
        return grid >= thr
    if op == ">":
        return grid > thr
    if op == "<=":
        return grid <= thr
    if op == "<":
        return grid < thr
    return np.isclose(grid, thr, atol=1e-9)


def _literal_match(a: str, b: str, tol: float = 0.06,
                   support: dict[str, np.ndarray] | None = None) -> bool:
    """Semantic literal match: same variable and (near-)identical partition.

    Comparing operators textually fails on equivalent forms (``x>=0.5`` vs
    ``x>0.4``, ``smoking=1`` vs ``smoking>0.5``). We instead compare the
    indicator sets of the two literals and accept the match when they differ on
    at most ``tol`` of the evaluated points.

    ``support`` maps a variable to the values on which the partition is judged.
    On continuous data the default is a fine grid over ``[0, 1]``. On data with
    discrete support (e.g. the tic-tac-toe cells in ``{0, 0.5, 1}``) the grid
    would wrongly separate ``>0.5`` from ``>=0.75`` even though they select
    exactly the same rows; passing the observed values fixes that.
    """
    va, oa, ta = _parse_literal(a)
    vb, ob, tb = _parse_literal(b)
    if va != vb or np.isnan(ta) or np.isnan(tb):
        return False
    if support is not None and va in support:
        grid = np.asarray(support[va], dtype=float)
    else:
        grid = np.linspace(0.0, 1.0, 201)
    if grid.size == 0:
        return False
    ia, ib = _indicator(va, oa, ta, grid), _indicator(vb, ob, tb, grid)
    return float(np.mean(ia != ib)) <= tol


def rule_set_match(pred: list[frozenset[str]], ref: list[frozenset[str]],
                   tol: float = 0.06,
                   support: dict[str, np.ndarray] | None = None) -> dict:
    """Greedy bipartite matching of predicted vs reference rules.

    A match requires that every literal of the shorter rule is matched by a
    literal in the longer rule with the same variable and a close threshold.
    Returns precision / recall / F1 over rule bodies plus the matched pairs.
    """
    used = set()
    tp = 0
    for p in pred:
        best, best_j = None, -1
        for j, r in enumerate(ref):
            if j in used:
                continue
            short, long = (p, r) if len(p) <= len(r) else (r, p)
            if all(any(_literal_match(a, b, tol, support) for b in long) for a in short):
                jac = len(short) / len(set(p) | set(r))
                if best is None or jac > best:
                    best, best_j = jac, j
        if best_j >= 0:
            used.add(best_j)
            tp += 1
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(ref) if ref else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp,
            "n_pred": len(pred), "n_ref": len(ref)}


def true_rules_as_literals(domain: Domain, head: str) -> list[frozenset[str]]:
    out = []
    for r in domain.true_rules:
        if r.head != head:
            continue
        out.append(frozenset(str(l) for l in r.body))
    return out


def ground_truth_dnf(domain: Domain, head: str | None = None) -> list[frozenset[str]]:
    """Ground-truth label logic expanded to a DNF over *observable features*.

    The true label rules reference hidden concepts, which no model that only
    sees features can express directly.  Substituting each concept by its own
    rule bodies and distributing conjunctions yields an equivalent
    feature-only DNF: the canonical target for rule recovery.  A depth-limited
    tree can represent each resulting conjunction, so this is a well-posed
    reference for any feature-based decision function.
    """
    head = head or domain.label
    concept_rules = {c: [r.body for r in domain.true_rules if r.head == c] for c in domain.concepts}
    out: list[frozenset[str]] = []
    for r in domain.true_rules:
        if r.head != head:
            continue
        acc: list[frozenset[str]] = [frozenset()]
        for lit in r.body:
            if lit.kind == "concept":
                options = concept_rules.get(lit.var, [])
                acc = [s | frozenset(str(l) for l in opt) for s in acc for opt in options]
            else:
                acc = [s | {str(lit)} for s in acc]
        out.extend(acc)
    # de-duplicate
    seen, uniq = set(), []
    for r in out:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def logic_fidelity(decision_fn, domain: Domain, n: int = 6000, seed: int = 11) -> float:
    """Agreement between a decision function and the oracle on a uniform grid."""
    Xg = uniform_grid(domain, n, seed=seed)
    y_true = oracle(domain, Xg)["label"]
    y_hat = decision_fn(Xg)
    return float((np.asarray(y_hat) == y_true).mean())
