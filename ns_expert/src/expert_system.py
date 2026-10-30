"""A classical rule-based expert system: forward chaining with certainty factors.

Concepts are inferred first (layer 1), then the label is derived from concepts
and features (layer 2).  Each rule contributes ``cf * prod(body_membership)`` and
contributions to the same head are combined with the MYCIN-style combination
function, which is the standard way expert systems propagate uncertain evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .knowledge import Domain, Literal, Rule


def _membership(X: pd.DataFrame, lit: Literal, concepts: pd.DataFrame | None,
                soft: bool, width: float = 0.08) -> np.ndarray:
    if lit.kind == "concept":
        assert concepts is not None
        return concepts[lit.var].to_numpy(dtype=float)
    v = X[lit.var].to_numpy(dtype=float)
    if not soft:
        if lit.op == ">=":
            return (v >= lit.thr).astype(float)
        if lit.op == "<=":
            return (v <= lit.thr).astype(float)
        return (v == lit.thr).astype(float)
    # soft sigmoid-style membership, smooth in a band of +/- `width`
    z = (v - lit.thr) / max(width, 1e-6)
    if lit.op == "<=":
        z = -z
    if lit.op == "==":
        z = 1.0 - 2.0 * np.abs(z)
    return np.clip(0.5 + 0.5 * np.tanh(z), 0.0, 1.0)


def _combine(a: float, b: float) -> float:
    """MYCIN certainty-factor combination."""
    return a + b * (1.0 - a)


def infer_layer(rules: list[Rule], X: pd.DataFrame, concepts: pd.DataFrame | None,
                head: str, soft: bool) -> np.ndarray:
    n = len(X)
    conf = np.zeros(n, dtype=float)
    for r in rules:
        if r.head != head:
            continue
        body = np.ones(n, dtype=float)
        for lit in r.body:
            body *= _membership(X, lit, concepts, soft)
        contrib = r.cf * body
        conf = _combine(conf, contrib)  # vectorised over samples
    return conf


def run(domain: Domain, X: pd.DataFrame, rules: list[Rule] | None = None,
        soft: bool = False, concept_threshold: float = 0.5) -> dict:
    """Forward-chain the rule base over ``X``.

    Returns concept confidences, label confidence, and the hard prediction.
    """
    rules = rules if rules is not None else list(domain.expert_rules)
    concept_conf = pd.DataFrame(index=X.index)
    for c in domain.concepts:
        concept_conf[c] = infer_layer(rules, X, None, c, soft)

    # label layer sees binarised concepts (classical expert systems are discrete)
    concept_hard = (concept_conf >= concept_threshold).astype(float)
    label_conf = infer_layer(rules, X, concept_hard, domain.label, soft)
    y_hat = (label_conf >= concept_threshold).astype(int)

    return {
        "concept_conf": concept_conf.to_numpy(),
        "label_conf": label_conf,
        "y_hat": y_hat,
    }
