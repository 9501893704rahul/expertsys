"""Synthetic benchmark generation with ground-truth rules.

Data are produced by the *true* rule base in :mod:`knowledge`.  Intermediate
concepts are computed from the rules but are **hidden** from every model; only
the observed features and the final label are released.  This gives us a clean
ground truth for two quantities that are rarely measurable together:

  1. task accuracy on the observed features, and
  2. whether a model recovers the underlying logical rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .knowledge import Domain, Literal, Rule


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    concepts: pd.DataFrame
    domain: Domain

    def split(self, idx: np.ndarray) -> "Dataset":
        return Dataset(
            self.X.iloc[idx].reset_index(drop=True),
            self.y[idx],
            self.concepts.iloc[idx].reset_index(drop=True),
            self.domain,
        )


def _sample_features(rng: np.random.Generator, domain: Domain, n: int) -> pd.DataFrame:
    """Mixture of a 'healthy' and an 'at-risk' regime per continuous feature.

    Correlating the regimes makes the task non-trivial (a naive single-feature
    threshold rule is insufficient) while keeping the marginal distributions
    realistic.
    """
    cols: dict[str, np.ndarray] = {}
    # shared latent propensity drives which regime a sample falls into
    propensity = rng.beta(2.2, 2.6, size=n)
    for f in domain.features:
        if f in domain.binary_features:
            p = 0.12 + 0.55 * propensity
            cols[f] = (rng.random(n) < p).astype(float)
        else:
            mu = 0.32 + 0.42 * propensity
            x = rng.normal(mu, 0.16, size=n)
            cols[f] = np.clip(x, 0.0, 1.0)
    return pd.DataFrame(cols)


def _eval_literal(row: pd.Series, lit: Literal) -> bool:
    v = row[lit.var]
    if lit.op == ">=":
        return v >= lit.thr
    if lit.op == "<=":
        return v <= lit.thr
    return v == lit.thr


def _concept_matrix(rng: np.random.Generator, domain: Domain, X: pd.DataFrame) -> pd.DataFrame:
    """Realise hidden concepts from the true concept rules."""
    out = pd.DataFrame(index=X.index)
    for c in domain.concepts:
        rules = [r for r in domain.true_rules if r.head == c]
        fired = np.zeros(len(X), dtype=bool)
        for r in rules:
            m = np.ones(len(X), dtype=bool)
            for lit in r.body:
                m &= X.apply(lambda row, l=lit: _eval_literal(row, l), axis=1).to_numpy()
            fired |= m
        # small stochastic realisation error so concepts are not deterministic
        flip = rng.random(len(X)) < domain.label_noise
        fired = np.where(flip, ~fired, fired)
        out[c] = fired.astype(int)
    return out


def _labels(rng: np.random.Generator, domain: Domain, X: pd.DataFrame, C: pd.DataFrame) -> np.ndarray:
    aug = pd.concat([X, C], axis=1)
    rules = [r for r in domain.true_rules if r.head == domain.label]
    fired = np.zeros(len(X), dtype=bool)
    for r in rules:
        m = np.ones(len(X), dtype=bool)
        for lit in r.body:
            m &= aug.apply(lambda row, l=lit: _eval_literal(row, l), axis=1).to_numpy()
        fired |= m
    flip = rng.random(len(X)) < domain.label_noise
    y = np.where(flip, ~fired, fired).astype(int)
    return y


def generate(domain: Domain, n: int = 4000, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    X = _sample_features(rng, domain, n)
    C = _concept_matrix(rng, domain, X)
    y = _labels(rng, domain, X, C)
    return Dataset(X=X, y=y, concepts=C, domain=domain)


def make_splits(domain: Domain, n: int = 4000, seed: int = 0,
                fractions=(0.6, 0.2, 0.2)) -> tuple[Dataset, Dataset, Dataset]:
    ds = generate(domain, n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(ds.y))
    a, b = int(fractions[0] * len(idx)), int((fractions[0] + fractions[1]) * len(idx))
    return ds.split(idx[:a]), ds.split(idx[a:b]), ds.split(idx[b:])


def concept_ground_truth(ds: Dataset, rule: Rule, feature_cols) -> np.ndarray:
    """Boolean firing indicator of a *true* rule on the raw features (no concepts).

    Only usable for rules whose body references observed features; used for
    rule-recovery evaluation.
    """
    m = np.ones(len(ds.X), dtype=bool)
    for lit in rule.body:
        if lit.kind == "concept":
            m &= ds.concepts[lit.var].to_numpy() >= lit.thr
        else:
            m &= ds.X[lit.var].apply(lambda v, l=lit: _eval_literal(pd.Series({l.var: v}), l)).to_numpy()
    return m
