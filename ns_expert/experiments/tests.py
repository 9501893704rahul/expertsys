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


def test_real_datasets_load():
    from src import real_data

    for name in ["heart", "pima", "ttt"]:
        d, tr, va, te = real_data.load_real(name, seed=0)
        n = len(tr.y) + len(va.y) + len(te.y)
        check(f"real {name} splits are non-empty and disjoint", n > 0 and
              len(tr.y) > 0 and len(va.y) > 0 and len(te.y) > 0)
        check(f"real {name} features are normalised to [0,1]",
              float(pd.concat([tr.X, va.X, te.X]).to_numpy().min()) >= -1e-9 and
              float(pd.concat([tr.X, va.X, te.X]).to_numpy().max()) <= 1 + 1e-9)
        check(f"real {name} has expert rules", len(d.expert_rules) > 0)


def test_ttt_expert_system_is_exact():
    """The eight winning lines are the true logic, so the KB must be perfect."""
    from src import real_data

    d, tr, va, te = real_data.load_real("ttt", seed=0)
    out = symbolic_run(d, te.X, soft=False)
    check("ttt expert system reproduces the true winning-line logic exactly",
          float((out["y_hat"] == te.y).mean()) == 1.0)


def test_support_aware_literal_match():
    """`>0.5` and `>=0.75` are the same partition on the support {0,0.5,1}."""
    sup = {"cell": np.array([0.0, 0.5, 1.0])}
    check("literal match is support-aware on discrete features",
          rules._literal_match("cell>0.5", "cell>=0.75", support=sup))
    check("literal match still separates real differences",
          not rules._literal_match("cell>=1.0", "cell>=0.25", support=sup))


def test_literal_parse_handles_digit_names():
    """Feature names may contain digits (tic-tac-toe cells are c00..c22)."""
    var, op, thr = rules._parse_literal("c12>=0.750")
    check("literal parser accepts digits in feature names",
          var == "c12" and op == ">=" and abs(thr - 0.75) < 1e-9)


def test_ttt_rule_extraction_is_nonzero():
    """At least the proposed model recovers some of the true winning lines."""
    from src import real_data

    d, tr, va, te = real_data.load_real("ttt", seed=0)
    sup = real_data.feature_support(pd.concat([tr.X, va.X, te.X]))
    ref = [frozenset(str(l) for l in r.body)
           for r in real_data.ground_truth_rules("ttt")]
    m = models.NEXS(d, seed=0)
    m.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)

    def dec(X):
        return (m.predict_proba(np.asarray(X, dtype=float)) >= 0.5).astype(int)

    pred = rules.extract_rules(dec, d, depth=5, seed=1, support=sup)
    r = rules.rule_set_match(pred, ref, support=sup)
    check(f"ttt rule extraction recovers at least one true line (tp={r['tp']})",
          r["tp"] >= 1)


def test_training_knobs_are_available():
    """Class weighting, AUC early stopping, and the gated fusion all train."""
    from src import real_data

    d, tr, va, te = real_data.load_real("heart", seed=0)
    for kw in (dict(pos_weight="auto"), dict(select="auc"),
               dict(gated=True), dict(gated=True, pos_weight="auto", select="auc")):
        m = models.NEXS(d, seed=0, **kw)
        m.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
        p = m.predict_proba(te.X.to_numpy())
        check(f"NEXS trains with {sorted(kw)} and gives valid probabilities",
              p.shape == (len(te.y),) and np.all((p >= 0) & (p <= 1)))


def test_accuracy_optimal_threshold():
    """The accuracy-optimal threshold is not worse than the F1-optimal one."""
    from experiments.run_experiments import _best_threshold
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.4).astype(int)
    p = np.clip(y * 0.4 + rng.random(400), 0, 1)
    t_f1 = _best_threshold(p, y, metric="f1")
    t_acc = _best_threshold(p, y, metric="acc")
    check("accuracy-optimal threshold is at least as accurate as F1-optimal",
          accuracy_score(y, (p >= t_acc).astype(int))
          >= accuracy_score(y, (p >= t_f1).astype(int)) - 1e-12)


def main():
    test_expert_system_matches_oracle()
    test_rule_extraction_recovers_expert_rules()
    test_nexs_forward_and_gradients()
    test_oracle_deterministic()
    test_membership_monotone()
    test_real_datasets_load()
    test_ttt_expert_system_is_exact()
    test_support_aware_literal_match()
    test_literal_parse_handles_digit_names()
    test_ttt_rule_extraction_is_nonzero()
    test_training_knobs_are_available()
    test_accuracy_optimal_threshold()
    print("\nall tests passed")


if __name__ == "__main__":
    main()
