"""General rule discovery: learn Horn-style rules from training data alone.

The specialized learners in ``experiments/rule_discovery.py`` are hard-wired to
the tic-tac-toe and MONK candidate sets.  For the nested-CV study we need to
discover rules on *any* dataset, including heart and pima, and hand them to
``NEXS`` as a drop-in replacement for the expert knowledge base.

The pipeline is:

1. propose candidate literals from the training features only (equality
   literals on discrete support, threshold literals at training quantiles);
2. build the literal membership matrix on the training rows;
3. run the CN2-style candidate-generation learner (``learn_dnf_cover``);
4. convert the selected conjunctions into :class:`~src.knowledge.Rule` objects.

Nothing here reads ``domain.expert_rules`` or the ground truth, so a NEXS built
on the output is measuring discovery, not injection.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .knowledge import Literal, Rule


def candidate_literals(X: pd.DataFrame, n_thresholds: int = 3,
                       max_unique: int = 12) -> list[tuple[str, str, float]]:
    """Propose literals from training data only.

    Low-cardinality features get equality literals on their observed values
    (``cell = x`` is semantically the same as ``cell >= 0.75``, and the semantic
    comparison in the structural metrics recognises that).  Continuous features
    get ``>=`` literals at training quantiles.  No label or expert information
    is used.
    """
    lits: list[tuple[str, str, float]] = []
    for c in X.columns:
        v = X[c].to_numpy(dtype=float)
        u = np.unique(v)
        if len(u) <= max_unique:
            for val in u:
                lits.append((c, "==", float(val)))
        else:
            for q in np.linspace(0.25, 0.75, n_thresholds):
                lits.append((c, ">=", float(np.quantile(v, q))))
    return lits


def membership_matrix(X: pd.DataFrame, lits: list[tuple[str, str, float]]) -> np.ndarray:
    """Crisp membership of each literal on each row (normalized feature space)."""
    cols = {c: i for i, c in enumerate(X.columns)}
    A = X.to_numpy(dtype=float)
    M = np.zeros((len(X), len(lits)), dtype=bool)
    for j, (f, op, t) in enumerate(lits):
        v = A[:, cols[f]]
        if op == ">=":
            M[:, j] = v >= t
        elif op == "<=":
            M[:, j] = v <= t
        else:
            M[:, j] = np.isclose(v, t, atol=1e-6)
    return M


def _cover(M: np.ndarray, y: np.ndarray, max_lits: int, min_precision: float,
           min_laplace: float = 0.70):
    """Separate-and-conquer over candidate conjunctions.

    Rules are ranked by Laplace-corrected accuracy ``(P+1)/(P+N+2)`` rather than
    by raw coverage.  Ranking by coverage alone favours broad, imprecise
    conjunctions (a 85%-pure rule covering 200 positives beats a 100%-pure rule
    covering 20), which costs logical equivalence; the Laplace objective prefers
    pure rules and is the standard FOIL/CN2 criterion.
    """
    N, L = M.shape
    pos, neg = (y == 1), (y == 0)
    cands = []
    for k in range(1, max_lits + 1):
        for combo in itertools.combinations(range(L), k):
            cov = M[:, list(combo)].all(axis=1)
            P = int((cov & pos).sum())
            if P == 0:
                continue
            Nn = int((cov & neg).sum())
            prec = P / (P + Nn)
            if prec < min_precision:
                continue
            cands.append(((P + 1) / (P + Nn + 2), prec, -Nn, combo, cov))
    covered = np.zeros(N, bool)
    chosen = []
    while (pos & ~covered).any():
        best, best_key = None, None
        for laplace, prec, negP, combo, cov in cands:
            new = int((cov & pos & ~covered).sum())
            if new == 0:
                continue
            key = (laplace, new, prec, negP)
            if best_key is None or key > best_key:
                best_key, best = key, (combo, cov, laplace)
        if best is None:
            break
        combo, cov, laplace = best
        if laplace < min_laplace:
            break
        chosen.append(combo)
        covered |= cov & pos
    return chosen


def discover_rules(domain, X_train: pd.DataFrame, y_train: np.ndarray,
                   label: str | None = None, max_lits: int = 3,
                   min_precision: float = 0.85, n_thresholds: int = 3,
                   max_rules: int = 12) -> list[Rule]:
    """Return discovered rules in the domain's normalized feature space."""
    label = label or domain.label
    y = np.asarray(y_train, dtype=int)
    if y.sum() == 0:
        return []
    lits = candidate_literals(X_train, n_thresholds=n_thresholds)
    if not lits:
        return []
    M = membership_matrix(X_train, lits)
    bodies = _cover(M, y, max_lits=max_lits, min_precision=min_precision)[:max_rules]
    rules = []
    for b in bodies:
        body = tuple(Literal(f, op, float(t), "feature") for f, op, t in (lits[i] for i in b))
        rules.append(Rule(label, body, 1.0))
    return rules


def with_discovered_rules(domain, rules: list[Rule]):
    """Copy a domain, replacing the expert KB with discovered rules."""
    from dataclasses import replace

    return replace(domain, expert_rules=tuple(rules),
                   meta={**domain.meta, "kb_source": "discovered"})


# --------------------------------------------------------------------------- #
# rule corruption / incompleteness (for the ablation configurations)
# --------------------------------------------------------------------------- #
def corrupt_rules(rules: tuple[Rule, ...], level: float, seed: int = 0,
                  domain=None) -> tuple[Rule, ...]:
    """Degrade a KB: drop rules, then perturb surviving thresholds.

    ``level`` in ``[0, 1]`` is the expected fraction of rules dropped plus the
    threshold-jitter fraction.  ``level=0`` returns the KB unchanged; ``level=1``
    removes all rules.
    """
    from dataclasses import replace

    if level <= 0 or not rules:
        return tuple(rules)
    rng = np.random.default_rng(seed)
    keep = [r for r in rules if rng.random() >= level]
    if not keep:
        return ()
    out = []
    for r in keep:
        body = []
        for lit in r.body:
            if rng.random() < level:
                # jitter the threshold by up to +/-20% of its magnitude
                delta = rng.uniform(-0.2, 0.2) * max(abs(lit.thr), 0.1)
                body.append(replace(lit, thr=float(np.clip(lit.thr + delta, 0.0, 1.0))))
            else:
                body.append(lit)
        out.append(replace(r, body=tuple(body)))
    return tuple(out)


def incomplete_rules(rules: tuple[Rule, ...], completeness: float,
                     seed: int = 0) -> tuple[Rule, ...]:
    """Keep a random ``completeness`` fraction of the rules."""
    if completeness >= 1.0 or not rules:
        return tuple(rules)
    rng = np.random.default_rng(seed)
    k = int(round(completeness * len(rules)))
    if k <= 0:
        return ()
    idx = np.sort(rng.choice(len(rules), size=k, replace=False))
    return tuple(rules[i] for i in idx)
