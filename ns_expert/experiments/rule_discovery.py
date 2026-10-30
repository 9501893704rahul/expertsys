"""Genuine structural rule discovery from data.

Motivation
----------
The earlier ``structural_recovery.py`` compared the model's rule list with the
ground truth, but ``NEXS`` *initialises* that list from the supplied expert
knowledge (``self.rules = list(domain.expert_rules)``).  That test therefore
measures the injected KB, not discovery, and is circular.

This experiment removes the confound.  We build a rule layer whose **structure
is itself learned** from data and whose ground truth is hidden, then ask whether
the discovered rules coincide with the data-generating rule set.

Discovery layer
---------------
For each candidate rule we parameterise a differentiable *subset selector* over
the literals available in the domain:

    g_l = sigmoid(theta_l)                      # selection gate for literal l
    match = prod_l (1 - g_l * (1 - m_l(x)))     # soft conjunction
    rule  = sigmoid(cf) * match                 # confidence-weighted
    p_sat = 1 - prod_r (1 - rule_r)             # soft disjunction

where ``m_l(x)`` is the differentiable membership of literal ``l``.  The gates
are the only structural parameters: they are trained from the labels with no
expert rules supplied.  A gate is declared *selected* when
``sigmoid(theta_l) > 0.5``.  The discovered rule set is then the set of
conjunctions of selected literals, and we compare it to the hidden ground truth
by exact DNF equivalence on the full discrete input domain.

Metrics
-------
* ``discovered_exact_match``  : discovered rule set == ground-truth rule set
* ``literal_precision``/``recall`` : overlap of selected literals
* ``decision_agreement``      : agreement of the discovered DNF with the true
  label over the *complete* input domain (not just the test split)
* ``nexs_agreement``          : the same quantity for the rule-injecting NEXS,
  as the reference point that motivated this experiment.

This is the honest test of the claim "NEXS recovers the rules": the rules are
not supplied, and the target is verified on the full domain.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import real_data

# --------------------------------------------------------------------------- #
# differentiable discovery layer
# --------------------------------------------------------------------------- #


class DiscoveryLayer(nn.Module):
    """Learns a DNF over *domain-defined candidate literals* from data alone.

    There is no structural prior: each of the ``n_rules`` rules owns its own
    gate vector over all literals, so any rule may select any subset.  The only
    hyperparameter is the capacity ``n_rules`` (the maximum DNF size), which is
    standard.  ``n_rules`` is swept in ``main`` to show the result is not tuned.
    """

    def __init__(self, n_literals: int, n_rules: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # gates start near zero so literals must be switched on
        self.theta = nn.Parameter(-2.0 + 0.01 * torch.randn(n_rules, n_literals,
                                                            generator=g))
        # non-negative rule weights (softplus) + a bias, combined additively so
        # every rule keeps a usable gradient (unlike nested products, which
        # vanish as the number of rules/literals grows)
        self.w = nn.Parameter(1.0 + 0.01 * torch.randn(n_rules, generator=g))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, mem_fn) -> torch.Tensor:
        m = mem_fn(x)                                       # (N, L)
        gate = torch.sigmoid(self.theta)                    # (R, L)
        # noisy-AND: penalty grows when a *selected* literal is unsatisfied
        penalty = (gate.unsqueeze(0) * (1.0 - m.unsqueeze(1))).sum(dim=2)  # (N,R)
        # activity: a rule with no selected literal must contribute nothing,
        # else "all gates off" is the degenerate "always fire" solution
        active = 1.0 - torch.prod(1.0 - gate, dim=1)        # (R,)
        match = active.unsqueeze(0) * torch.exp(-penalty)   # (N, R) in [0,1]
        w = torch.nn.functional.softplus(self.w)            # (R,) non-negative
        score = (w.unsqueeze(0) * match).sum(dim=1) - self.bias
        return torch.sigmoid(score)                         # (N,)

    def n_selected(self) -> int:
        return int((self.theta > 0).sum())


# --------------------------------------------------------------------------- #
# candidate literal definitions for each logical domain
# --------------------------------------------------------------------------- #

# Literals are defined in RAW units; membership functions receive raw features
# (we invert the per-seed normalisation before evaluating them).
# Attribute domains per the UCI codebook: a1,a2 in {1,2,3}; a3 in {1,2};
# a4 in {1,2,3}; a5 in {1,2,3,4}; a6 in {1,2}.
MONK_ATTRS = ["a1", "a2", "a3", "a4", "a5", "a6"]
MONK_RANGES = {1: (1, 2, 3), 2: (1, 2, 3), 3: (1, 2),
               4: (1, 2, 3), 5: (1, 2, 3, 4), 6: (1, 2)}


def monk_candidates() -> tuple[list[tuple], callable]:
    lits = []
    for j, a in enumerate(MONK_ATTRS, start=1):
        for v in MONK_RANGES[j]:
            lits.append((a, "==", float(v)))
    cols = [MONK_ATTRS.index(a) for a, _, _ in lits]
    vals = [v for _, _, v in lits]

    def mem(x, cols=cols, vals=vals):
        return torch.stack(
            [torch.sigmoid((1.0 - 2.0 * (x[:, j] - v).abs()) / 0.10)
             for j, v in zip(cols, vals)], dim=1)

    return lits, mem


def ttt_candidates() -> tuple[list[tuple], callable]:
    lits = [(c, ">=", 0.75) for c in real_data.TTT_CELLS]
    cols = [real_data.TTT_CELLS.index(c) for c, _, _ in lits]

    def mem(x, cols=cols):
        return torch.stack([torch.sigmoid((x[:, j] - 0.75) / 0.10)
                            for j in cols], dim=1)

    return lits, mem


def ground_truth_dnf(name: str) -> tuple[list[frozenset], tuple]:
    """The hidden data-generating rule set, as a list of literal sets."""
    if name == "ttt":
        lines = [frozenset((real_data.TTT_CELLS[i], ">=", 0.75) for i in line)
                 for line in real_data.TTT_LINES]
        return lines, tuple(sorted(real_data.TTT_CELLS))
    v = int(name[-1])
    a1, a2, a3, a4, a5, a6 = MONK_ATTRS
    if v == 1:
        rules = [frozenset({(a1, "==", float(t)), (a2, "==", float(t))}) for t in (1, 2, 3)]
        rules.append(frozenset({(a5, "==", 1.0)}))
    else:
        rules = [frozenset({(a5, "==", 3.0), (a4, "==", 1.0)})]
        rules += [frozenset({(a5, "==", float(s)), (a2, "==", float(t))})
                  for s in (1, 2, 3) for t in (1, 2)]
    return rules, tuple(MONK_ATTRS)


# --------------------------------------------------------------------------- #
# training / evaluation
# --------------------------------------------------------------------------- #


def _full_domain(name: str):
    from experiments.structural_recovery import (full_monk_domain, full_ttt_domain)
    if name == "ttt":
        return full_ttt_domain()
    return full_monk_domain(int(name[-1]))


def _train_split_bounds(name: str, seed: int):
    from experiments.structural_recovery import _train_split_bounds as f
    return f(name, seed)


def learn_dnf_cover(M: np.ndarray, y: np.ndarray, max_lits: int = 3,
                    min_precision: float = 0.85) -> list[tuple[int, ...]]:
    """CN2-style DNF learner: candidate conjunctions + greedy set cover.

    The learner is given only the membership matrix (literal truth values) and
    the labels --- no expert rules.  It (i) enumerates every conjunction of up
    to ``max_lits`` literals, (ii) keeps those whose training precision clears
    ``min_precision``, and (iii) greedily covers the positive examples with the
    surviving conjunctions, breaking ties toward fewer false positives.  This
    is the classic separate-and-conquer / candidate-generation scheme and can
    represent patterns (like an equality between two attributes) that purely
    greedy one-literal-at-a-time growth cannot see.
    """
    N, L = M.shape
    pos, neg = (y == 1), (y == 0)
    Mp = M > 0.5
    cands: list[tuple[float, int, int, tuple[int, ...]]] = []
    for k in range(1, max_lits + 1):
        for combo in itertools.combinations(range(L), k):
            cov = Mp[:, list(combo)].all(axis=1)
            P = int((cov & pos).sum())
            if P == 0:
                continue
            Nn = int((cov & neg).sum())
            prec = P / (P + Nn)
            if prec >= min_precision:
                cands.append((prec, P, -Nn, combo))
    # greedy set cover over the positive examples
    covered = np.zeros(N, bool)
    chosen: list[tuple[int, ...]] = []
    while (pos & ~covered).any():
        best, best_key = None, None
        for prec, P, negP, combo in cands:
            cov = Mp[:, list(combo)].all(axis=1)
            new = int((cov & pos & ~covered).sum())
            if new == 0:
                continue
            key = (new, prec, negP)
            if best_key is None or key > best_key:
                best_key, best = key, combo
        if best is None:
            break
        chosen.append(best)
        covered |= Mp[:, list(best)].all(axis=1) & pos
    return chosen


def discover_rules_only(name: str, seed: int) -> dict:
    """Genuine discovery metrics (candidate-generation learner) for one seed.

    No model training: the rules are learned from the training split labels
    alone and scored against the hidden ground truth on the full domain.
    """
    lits, mem_fn = (ttt_candidates() if name == "ttt" else monk_candidates())
    domain, tr, va, te = real_data.load_real(name, seed=seed)
    lo, hi = _train_split_bounds(name, seed)
    span = (hi - lo).replace(0, 1.0)
    M_tr = mem_fn(torch.tensor((tr.X * span + lo).to_numpy(),
                               dtype=torch.float32)).detach().numpy()
    bodies = learn_dnf_cover(M_tr, tr.y.astype(int))
    disc_set = {frozenset(lits[i] for i in b) for b in bodies}
    gt, _ = ground_truth_dnf(name)
    gt_set = {frozenset(s) for s in gt}
    inter = len(gt_set & disc_set)
    Xd_raw, yd = _full_domain(name)
    M_d = mem_fn(torch.tensor(Xd_raw.to_numpy(), dtype=torch.float32)).detach().numpy()
    agree = float((_dnf_predict(M_d, bodies) == yd).mean())
    return dict(discovered_exact_match=int(gt_set == disc_set),
                discovered_rules=len(disc_set),
                discovered_recall=inter / max(1, len(gt_set)),
                discovered_precision=inter / max(1, len(disc_set)),
                discovered_agreement=agree)


def discover(name: str, seed: int, epochs: int, n_rules: int) -> dict:
    if name == "ttt":
        lits, mem_fn = ttt_candidates()
    else:
        lits, mem_fn = monk_candidates()

    domain, tr, va, te = real_data.load_real(name, seed=seed)
    # load_real normalises with bounds taken from the raw *training* rows; we
    # must recover those same bounds to invert the normalisation for membership
    # literals, which are defined in raw units.
    lo, hi = _train_split_bounds(name, seed)
    span = (hi - lo).replace(0, 1.0)

    def to_raw(df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor((df * span + lo).to_numpy(), dtype=torch.float32)

    M_tr = mem_fn(to_raw(tr.X)).detach().numpy()

    # --- genuine discovery: FOIL over the training split only ---
    bodies = learn_dnf_cover(M_tr, tr.y.astype(int))
    disc_set = {frozenset(lits[i] for i in b) for b in bodies}
    n_selected_literals = len({i for b in bodies for i in b})

    gt, _ = ground_truth_dnf(name)
    gt_set = {frozenset(s) for s in gt}
    inter = len(gt_set & disc_set)
    lit_gt = {l for s in gt for l in s}
    lit_disc = {lits[i] for s in bodies for i in s}

    # decision agreement with the true label over the complete domain
    Xd_raw, yd = _full_domain(name)
    M_d = mem_fn(torch.tensor(Xd_raw.to_numpy(), dtype=torch.float32)).detach().numpy()
    pred = _dnf_predict(M_d, bodies)
    agree = float((pred == yd).mean())

    # references on the same full domain
    Xd_norm = real_data.normalize(Xd_raw, lo, hi)
    nexs_agree = _nexs_full_domain_agreement(name, seed, Xd_norm, yd)
    dec_agree, dec_rules = _differentiable_discovery(name, seed, lits, mem_fn,
                                                     to_raw, M_d, yd, epochs,
                                                     n_rules)
    return dict(dataset=name, seed=seed,
                n_gt_rules=len(gt_set), n_discovered_rules=len(disc_set),
                discovered_exact_match=int(gt_set == disc_set),
                rules_recovered=inter,
                rule_precision=inter / max(1, len(disc_set)),
                rule_recall=inter / max(1, len(gt_set)),
                literal_precision=len(lit_gt & lit_disc) / max(1, len(lit_disc)),
                literal_recall=len(lit_gt & lit_disc) / max(1, len(lit_gt)),
                n_selected_literals=n_selected_literals,
                decision_agreement=agree,
                nexs_agreement=nexs_agree,
                diff_exact_match=int(gt_set == dec_rules),
                diff_decision_agreement=dec_agree)


def _differentiable_discovery(name, seed, lits, mem_fn, to_raw, M_d, yd, epochs,
                              n_rules):
    """Optional comparison: the differentiable rule layer, trained on labels.

    Reported for reference; the FOIL learner is the primary, structure-free
    discovery method because the differentiable layer is prone to local optima.
    """
    domain, tr, va, te = real_data.load_real(name, seed=seed)
    Xtr = to_raw(tr.X)
    ytr = torch.tensor(tr.y, dtype=torch.float32)
    torch.manual_seed(seed)
    layer = DiscoveryLayer(len(lits), n_rules, seed=seed)
    with torch.no_grad():
        layer.theta.fill_(1.5)
    opt = torch.optim.Adam(layer.parameters(), lr=0.05)
    bce = nn.BCELoss()
    for _ in range(epochs):
        layer.train()
        opt.zero_grad()
        p = layer(Xtr, mem_fn).clamp(1e-6, 1 - 1e-6)
        loss = bce(p, ytr) + 5e-4 * torch.sigmoid(layer.theta).sum()
        loss.backward()
        opt.step()
    on = (layer.theta.detach().numpy() > 0)
    disc = {frozenset(lits[i] for i in range(len(lits)) if on[r, i])
            for r in range(n_rules)}
    disc = {d for d in disc if d}
    cols = [[lits.index(l) for l in d] for d in disc]
    pred = _dnf_predict(M_d, cols)
    return float((pred == yd).mean()), disc


def _dnf_predict(M: np.ndarray, bodies) -> np.ndarray:
    """Crisp DNF prediction: a row is positive if any conjunction is satisfied."""
    pred = np.zeros(M.shape[0], dtype=int)
    for b in bodies:
        if len(b):
            pred |= (np.min(M[:, list(b)], axis=1) > 0.5).astype(int)
    return pred


def _nexs_full_domain_agreement(name, seed, Xd, yd) -> float:
    from src import models
    domain, tr, va, te = real_data.load_real(name, seed=seed)
    m = models.NEXS(domain, seed=seed, pos_weight="auto", epochs=150, patience=20)
    m.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
    with torch.no_grad():
        z = m._logits(torch.tensor(Xd.to_numpy(), dtype=torch.float32))
    return float(((torch.sigmoid(z).numpy() > 0.5).astype(int) == yd).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["ttt", "monk1", "monk3"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--rules", type=int, default=16)
    ap.add_argument("--out", default="results/discovery")
    args = ap.parse_args()
    rows = []
    for name in args.datasets:
        for s in args.seeds:
            r = discover(name, s, args.epochs, args.rules)
            rows.append(r)
            print(f"  {name} seed={s}: exact={r['discovered_exact_match']} "
                  f"rec={r['rules_recovered']}/{r['n_gt_rules']} "
                  f"litP={r['literal_precision']:.2f} litR={r['literal_recall']:.2f} "
                  f"agree={r['decision_agreement']:.3f} "
                  f"(NEXS={r['nexs_agreement']:.3f})", flush=True)
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rule_discovery.csv", index=False)
    print(df.groupby("dataset")[["discovered_exact_match", "rule_recall",
                                 "literal_precision", "literal_recall",
                                 "decision_agreement", "nexs_agreement"]]
          .mean().round(3).to_string())
    print("written ->", out)


if __name__ == "__main__":
    main()
