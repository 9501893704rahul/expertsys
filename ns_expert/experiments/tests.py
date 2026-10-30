"""Invariant tests for the symbolic and neuro-symbolic components.

Run with ``PYTHONPATH=. python experiments/tests.py``. These check the pieces
that the paper's claims depend on, without a test framework dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import models, rules
from src.data import make_splits
from src.expert_system import run as symbolic_run
from src.knowledge import DOMAINS, Literal


def check(name, cond):
    print(f"[{'ok' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_expert_system_matches_oracle():
    """The symbolic ESS with the *true* rules should equal the oracle."""
    for d in DOMAINS.values():
        tr, va, te = make_splits(d, n=1500, seed=3)
        out = symbolic_run(d, te.X, rules=list(d.true_rules), soft=False)
        ref = rules.oracle(d, te.X, rules=d.true_rules)["label"]
        agree = (out["y_hat"] == ref).mean()
        check(f"ESS(true rules) == oracle on {d.name} ({agree:.3f})", agree > 0.99)


def test_rule_extraction_recovers_expert_rules():
    """Distilling the expert system should recover its own label rules."""
    for d in DOMAINS.values():
        tr, va, te = make_splits(d, n=1500, seed=4)
        pred = rules.extract_rules(lambda X: symbolic_run(d, X)["y_hat"], d, seed=4)
        ref = rules.ground_truth_dnf(d, d.label)
        m = rules.rule_set_match(pred, ref)
        check(f"rule extraction recovers ESS rules on {d.name} (F1={m['f1']:.2f})", m["f1"] > 0.3)


def test_nexs_forward_and_gradients():
    """NEXS must produce valid probabilities and trainable rule confidences."""
    d = DOMAINS["medical"]
    tr, va, te = make_splits(d, n=1200, seed=5)
    model = models.NEXS(d, seed=5, epochs=40)
    model.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
    p = model.predict_proba(te.X.to_numpy())
    check("NEXS probabilities in [0,1]", np.all((p >= 0) & (p <= 1)))
    cf = model.rule_confidence()
    check("NEXS learns per-rule confidences", cf is not None and len(cf) == len(d.expert_rules))
    check("rule confidences are bounded (sigmoid)", np.all((cf > 0) & (cf < 1)))


def test_oracle_deterministic():
    d = DOMAINS["industrial"]
    X = rules.uniform_grid(d, 500, seed=9)
    a = rules.oracle(d, X)["label"]
    b = rules.oracle(d, X)["label"]
    check("oracle is deterministic", np.array_equal(a, b))


def test_membership_monotone():
    from src.expert_system import _membership

    X = pd.DataFrame({"age": np.linspace(0, 1, 50)})
    lit = Literal("age", ">=", 0.5, "feature")
    m = _membership(X, lit, None, soft=True)
    check("soft membership is monotone in the feature", np.all(np.diff(m) >= -1e-9))


def main():
    test_expert_system_matches_oracle()
    test_rule_extraction_recovers_expert_rules()
    test_nexs_forward_and_gradients()
    test_oracle_deterministic()
    test_membership_monotone()
    print("\nall tests passed")


if __name__ == "__main__":
    main()
