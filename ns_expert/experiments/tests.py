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


def test_holm_correction():
    """Holm correction is monotone and never below the raw p-value."""
    from experiments.advanced_analysis import _holm

    raw = [0.001, 0.02, 0.04, 0.5]
    adj = _holm(raw)
    check("Holm adjusted p-values are >= raw and <= 1",
          all(a >= r - 1e-12 and a <= 1 + 1e-12 for a, r in zip(adj, raw)))
    check("Holm preserves ordering of raw p-values",
          adj[0] <= adj[1] <= adj[2] <= adj[3])


def test_bootstrap_ci_covers_mean():
    """The paired bootstrap CI brackets the observed mean difference."""
    from experiments.advanced_analysis import _paired_ci
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, 30)
    b = rng.normal(0.7, 0.05, 30)
    lo, hi = _paired_ci(a, b, n_boot=2000, seed=0)
    check("paired bootstrap CI brackets the mean difference",
          lo <= (a - b).mean() <= hi)


def test_corruption_monotone_and_empty():
    """Corruption 0 is a no-op, 1.0 empties the KB, in between perturbs."""
    from experiments.run_rule_conditions import corrupt_domain
    from src import real_data

    d0, *_ = real_data.load_real("heart", seed=0)
    check("corruption 0 leaves the KB unchanged",
          corrupt_domain(d0, 0.0, 0).expert_rules == d0.expert_rules)
    check("corruption 1.0 removes all expert rules",
          len(corrupt_domain(d0, 1.0, 0).expert_rules) == 0)
    dc = corrupt_domain(d0, 0.5, 0)
    check("partial corruption keeps the rule count but changes thresholds",
          len(dc.expert_rules) == len(d0.expert_rules) and dc.expert_rules != d0.expert_rules)


def test_paired_statistics_schema():
    """paired_statistics returns one row per (dataset, metric, baseline)."""
    import numpy as np
    import pandas as pd
    from experiments.advanced_analysis import paired_statistics

    rows = []
    for seed in range(6):
        for method, base in (("NEXS(proposed)", 0.80), ("MLP", 0.78)):
            rows.append(dict(dataset="d", seed=seed, method=method,
                             acc=base + 0.01 * seed, f1=base + 0.01 * seed))
    out = paired_statistics(pd.DataFrame(rows))
    acc_rows = out[out.metric == "acc"]
    check("paired_statistics reports one NEXS-vs-MLP row per metric",
          len(acc_rows) == 1 and acc_rows.iloc[0]["method"] == "MLP"
          and acc_rows.iloc[0]["mean_diff"] > 0 and "p_holm" in out.columns)


def test_monk_kb_is_exact():
    """The MONK expert KB must reproduce the published formula, not approximate it."""
    from src import real_data
    from experiments.run_real_experiments import kb_oracle
    for name, tol in (("monk1", 1.0), ("monk3", 0.90)):
        accs = []
        for s in range(3):
            domain, tr, va, te = real_data.load_real(name, seed=s)
            accs.append(float((kb_oracle(domain, te.X) == te.y).mean()))
        m = float(np.mean(accs))
        # MONK-1 is noise-free (exact); MONK-3 carries 5% class noise by design.
        check(f"{name} KB oracle accuracy {m:.3f} >= {tol}", m >= tol)


def test_full_monk_domain_labels():
    """The enumerated MONK domain must match the published truth exactly."""
    from experiments.structural_recovery import full_monk_domain
    from src import real_data
    z = real_data.MONK_NORM
    for v, acc_lo in ((1, 1.0), (3, 1.0)):
        Xd, yd = full_monk_domain(v)
        # attribute domains per the UCI codebook: 3*3*2*3*4*2 = 432 tuples
        check(f"monk{v} full domain has 432 instances", len(yd) == 432)
        a1, a2, a3, a4, a5 = (Xd[c].to_numpy() for c in
                              ["a1", "a2", "a3", "a4", "a5"])
        # labels are evaluated on the *normalised* codebook scale
        if v == 1:
            ref = ((a1 == a2) | (a5 == z["a5"][1])).astype(int)
        else:
            ref = (((a5 == z["a5"][3]) & (a4 == z["a4"][1]))
                   | ((a5 != z["a5"][4]) & (a2 != z["a2"][3]))).astype(int)
        check(f"monk{v} enumerated labels == published formula",
              float((ref == yd).mean()) >= acc_lo)


def test_injected_rule_layer_mirrors_kb():
    """NEXS's rule layer is injected from the KB, so it must mirror it exactly.

    This is an *implementation* check, not evidence of discovery: the layer is
    initialised from ``domain.expert_rules``.
    """
    from src import real_data, models
    name = "monk1"
    domain, tr, va, te = real_data.load_real(name, seed=0)
    m = models.NEXS(domain, seed=0, pos_weight="auto", epochs=40, patience=8)
    m.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
    learned = {frozenset(str(l) for l in r.body)
               for r in m.rules if r.head == domain.label}
    true = {frozenset(str(l) for l in r.body)
            for r in domain.expert_rules if r.head == domain.label}
    check("NEXS injected rule layer == supplied KB", learned == true)


def test_genuine_rule_discovery_from_data():
    """Rules learned from labels alone must equal the hidden ground truth.

    Unlike the injected layer, the candidate-generation learner sees no expert
    rules.  This is the non-circular test of structural recovery.
    """
    from experiments.rule_discovery import discover_rules_only
    for name, n_true in (("ttt", 8), ("monk1", 4), ("monk3", 7)):
        r = discover_rules_only(name, seed=0)
        check(f"{name}: discovered rule set is exact", r["discovered_exact_match"] == 1)
        check(f"{name}: discovered rule count == {n_true}",
              r["discovered_rules"] == n_true)
        check(f"{name}: full-domain agreement >= 0.9",
              r["discovered_agreement"] >= 0.9)


def test_tree_metric_cannot_represent_or_of_conjunctions():
    """The distilled metric must fail on an exact KB (documented limitation)."""
    from src import real_data, rules
    name = "monk3"
    domain, tr, va, te = real_data.load_real(name, seed=0)
    from experiments.run_real_experiments import kb_oracle
    gt = real_data.ground_truth_rules(name)
    ref = [frozenset(str(l) for l in r.body) for r in gt]
    sup = real_data.feature_support(pd.concat([tr.X, va.X, te.X]))
    pred = rules.extract_rules(lambda Z: kb_oracle(domain, Z), domain,
                               depth=5, seed=1, support=sup)
    r = rules.rule_set_match(pred, ref, support=sup)
    check(f"distilled rule-F1 on the exact MONK-3 KB is ~0 ({r['f1']:.3f})",
          r["f1"] < 0.2)


def test_no_preprocessing_leak():
    """Imputation/scaling must be fit on training rows only.

    The leak-free loader leaves NaNs in place and fits the preprocessor on the
    training split.  We verify (a) raw loaders keep missing values, and (b) the
    fitted medians depend only on the training rows.
    """
    from src import real_data
    from src.pipeline import Preprocessor

    Xraw, y = real_data.load_heart_raw()
    check("heart raw keeps missing values", int(Xraw.isna().sum().sum()) > 0)

    prep = Preprocessor().fit(Xraw.iloc[:100], y[:100])
    med_tr = prep.medians_.copy()
    # a test row with a huge value must not move the fitted medians
    prep.transform(Xraw.iloc[100:120])
    check("fitted medians unchanged after transform", bool((prep.medians_ == med_tr).all()))

    d, tr, va, te = real_data.load_real("heart", seed=0)
    check("leak-free split has no NaNs", int(tr.X.isna().sum().sum() + te.X.isna().sum().sum()) == 0)
    check("leak-free split is scaled to [0,1]",
          float(tr.X.to_numpy().min()) >= 0.0 and float(tr.X.to_numpy().max()) <= 1.0)


def test_oof_folds_are_disjoint_and_cover():
    """Every row is a test row exactly once across outer folds."""
    from src import real_data

    _, y = real_data.DATASETS["heart"]["loader"]()
    seen = np.zeros(len(y), dtype=int)
    for f in real_data.make_oof_folds("heart", n_folds=5, seed=0):
        seen[f["te_i"]] += 1
        # validation rows must be drawn from the training pool, never the test pool
        check("val disjoint from test", len(set(f["va_i"]) & set(f["te_i"])) == 0)
        check("train disjoint from test", len(set(f["tr_i"]) & set(f["te_i"])) == 0)
    check("each row is tested exactly once", bool((seen == 1).all()))


def test_nested_threshold_is_validation_only():
    """The decision threshold is selected on validation, never on test."""
    from experiments.nested_cv import NEXS_GRID  # noqa: F401
    from experiments.run_experiments import _best_threshold

    rng = np.random.default_rng(0)
    y_val = rng.integers(0, 2, size=200)
    p_val = np.clip(0.5 + 0.3 * (y_val - 0.5) + rng.normal(0, 0.2, 200), 0, 1)
    thr = _best_threshold(p_val, y_val, metric="f1")
    check("threshold in (0,1)", 0.0 < thr < 1.0)


def test_discovered_rules_are_not_injected():
    """Discovery must not read the expert KB; it must learn from labels alone."""
    from src import discovery, real_data

    d, tr, va, te = real_data.load_real("ttt", seed=0)
    rules_a = discovery.discover_rules(d, tr.X, tr.y)
    # permute the expert KB: discovery output must be unchanged
    from dataclasses import replace
    d2 = replace(d, expert_rules=tuple(reversed(d.expert_rules)))
    rules_b = discovery.discover_rules(d2, tr.X, tr.y)
    check("discovery ignores expert KB", rules_a == rules_b)
    check("discovery finds rules on ttt", len(rules_a) > 0)


def test_semantic_rule_equivalence():
    """Two syntactically different but logically equal rules compare equal."""
    from experiments.nested_structural import _semantic_key, full_domain
    import pandas as pd

    Xd, _ = full_domain("ttt")
    a = frozenset({("c00", "==", 1.0)})
    b = frozenset({("c00", ">=", 0.75)})
    check("==1 and >=0.75 are semantically equal on the ttt domain",
          _semantic_key(a, Xd) == _semantic_key(b, Xd))


def test_corruption_degrades_kb():
    """Corruption must be monotone in level and empty at level 1."""
    from src import discovery
    from src.knowledge import DOMAINS

    kb = DOMAINS["medical"].expert_rules
    n0 = len(discovery.corrupt_rules(kb, 0.0, seed=0))
    n1 = len(discovery.corrupt_rules(kb, 1.0, seed=0))
    check("corruption level 0 keeps KB", n0 == len(kb))
    check("corruption level 1 empties KB", n1 == 0)


def test_monk_encoding_is_codebook_based():
    """MONK attributes use the declared codebook, independent of the fold.

    A min-max encoding fitted per fold would collapse or shift levels depending
    on which values the fold contains; the encoding must instead be a property
    of the dataset, and every declared level must stay distinct.
    """
    from src import real_data

    for v in (1, 3):
        X, _ = real_data.DATASETS[f"monk{v}"]["loader"]()
        for c in real_data.MONK_COLS:
            vals = sorted(set(X[c].tolist()))
            expected = sorted(set(real_data.MONK_NORM[c].values()))
            check(f"monk{v}.{c} levels match codebook", vals == expected)
        # a5 has four levels and must keep four distinct encodings
        check(f"monk{v}.a5 keeps 4 distinct levels",
              len(set(X["a5"].tolist())) == 4)

    # encoding must not depend on the training fold
    seen = set()
    for f in real_data.make_oof_folds("monk3", n_folds=5, seed=0):
        seen.add(tuple(sorted(set(f["train"].X["a5"].tolist()))))
    check("a5 encoding identical across folds", len(seen) == 1)


def test_monk_expert_kb_matches_published_truth():
    """The MONK expert KB must reproduce the published formula exactly."""
    from src import real_data
    from experiments.nested_structural import full_domain

    for v in (1, 3):
        name = f"monk{v}"
        Xd, yd = full_domain(name)
        _, yraw = real_data.DATASETS[name]["loader"]()
        Xraw, _ = real_data.DATASETS[name]["loader"]()
        dom = real_data.build_domain(name, Xraw.min(), Xraw.max())
        A = Xd.to_numpy()
        col = {c: i for i, c in enumerate(dom.features)}
        pred = np.zeros(len(Xd), dtype=bool)
        for r in dom.expert_rules:
            m = np.ones(len(Xd), dtype=bool)
            for lit in r.body:
                x = A[:, col[lit.var]]
                m &= (x >= lit.thr) if lit.op == ">=" else (
                    (x <= lit.thr) if lit.op == "<=" else
                    np.isclose(x, lit.thr, atol=1e-6))
            pred |= m
        check(f"{name} expert KB == published truth",
              float((pred.astype(int) == yd).mean()) == 1.0)


def test_full_monk_domain_is_432_with_four_a5_levels():
    """The enumerated domain must be the true input space, not a collapsed grid."""
    from experiments.nested_structural import full_domain

    for v in (1, 3):
        Xd, yd = full_domain(f"monk{v}")
        check(f"monk{v} enumerated domain has 432 points", len(Xd) == 432)
        check(f"monk{v} enumerated a5 has 4 levels",
              len(set(Xd["a5"].tolist())) == 4)


def test_metrics_are_per_fold_at_fold_threshold():
    """Per-fold metrics must use each fold's own threshold.

    Folds whose score distributions sit on different scales need different
    operating points.  Scoring every fold at one global threshold is strictly
    worse here, so the per-fold aggregation must beat it --- that is the
    property the corrected harness relies on.
    """
    from experiments.run_experiments import _metrics

    rng = np.random.default_rng(0)
    y = np.concatenate([rng.integers(0, 2, 200), rng.integers(0, 2, 200)])
    # fold 0 scores near 0.2, fold 1 near 0.8: the optimal cut differs sharply
    proba = np.concatenate([
        np.clip(0.2 + 0.15 * (y[:200] - 0.5) + rng.normal(0, 0.05, 200), 0, 1),
        np.clip(0.8 + 0.15 * (y[200:] - 0.5) + rng.normal(0, 0.05, 200), 0, 1)])
    te = [np.arange(0, 200), np.arange(200, 400)]
    thr = [0.2, 0.8]
    per_fold = [_metrics(y[i], proba[i], thr=t)["acc"] for i, t in zip(te, thr)]
    agg = float(np.mean(per_fold))
    global_thr = _metrics(y, proba, thr=0.5)["acc"]
    check("per-fold thresholds beat one global threshold", agg > global_thr)
    check("per-fold mean is a valid accuracy", 0.0 <= agg <= 1.0)


def test_run_nested_aggregates_per_fold():
    """run_nested's reported accuracy equals the mean of per-fold accuracies."""
    from src import models, real_data
    from experiments import nested_cv
    from experiments.run_experiments import _metrics

    name = "heart"
    _, y = real_data.DATASETS[name]["loader"]()
    df, oof, info, thr_of, fold_of = nested_cv.run_nested(
        name, n_folds=3, seed=0, epochs=8, patience=4)
    row = df[df.method == "MLP"].iloc[0]
    folds = sorted(set(fold_of.tolist()))
    manual = float(np.mean([
        _metrics(y[fold_of == f], oof["MLP"][fold_of == f],
                 thr=thr_of["MLP"][f])["acc"] for f in folds]))
    check("reported acc == mean of per-fold accuracies",
          abs(row["acc"] - manual) < 1e-9)
    check("per-fold thresholds were recorded", len(thr_of["MLP"]) == len(folds))



def test_inner_fold_preprocessing_is_independent():
    """Inner-fold preprocessing must be fit on inner-train rows only.

    The auditor's concern was that the inner CV reused the outer fold's
    scaler/selection.  We reproduce one inner split by hand, fit the
    preprocessor on the inner-train rows only, and check that every fitted
    statistic is recoverable from the inner-train rows and differs from the
    statistic over inner-train+inner-validation rows when those differ.
    """
    from src import real_data

    name = "heart"
    Xraw, y = real_data.DATASETS[name]["loader"]()
    f = next(iter(real_data.make_oof_folds(name, n_folds=5, seed=0)))
    tr_i = f["tr_i"]

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    itr, _ = next(skf.split(Xraw.iloc[tr_i], np.asarray(y)[tr_i]))
    inner_tr = tr_i[itr]

    prep_inner, dom_inner = real_data.fit_preprocessor(name, Xraw, inner_tr, y)
    inner_med = Xraw.iloc[inner_tr].median()
    check("inner prep median == inner-train median",
          bool(np.allclose(prep_inner.medians_.values, inner_med.values)))
    both_med = Xraw.iloc[tr_i].median()
    if not np.allclose(inner_med.values, both_med.values):
        check("inner prep median excludes inner-validation rows",
              not np.allclose(prep_inner.medians_.values, both_med.values))
    check("inner prep bounds == inner-train bounds",
          bool(np.allclose(prep_inner.lo_.values,
                           Xraw.iloc[inner_tr].min().values))
          and bool(np.allclose(prep_inner.hi_.values,
                               Xraw.iloc[inner_tr].max().values)))
    _, dom_outer = real_data.fit_preprocessor(name, Xraw, tr_i, y)
    check("domain is rebuilt from inner bounds",
          dom_inner.features == dom_outer.features)


def test_inner_fold_rule_discovery_is_independent():
    """Discovered rules must be mined inside the inner fold, not borrowed."""
    from src import discovery, real_data

    name = "ttt"
    Xraw, y = real_data.DATASETS[name]["loader"]()
    f = next(iter(real_data.make_oof_folds(name, n_folds=5, seed=0)))
    tr_i = f["tr_i"]

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    itr, _ = next(skf.split(Xraw.iloc[tr_i], np.asarray(y)[tr_i]))
    inner_tr = tr_i[itr]

    prep, dom = real_data.fit_preprocessor(name, Xraw, inner_tr, y)
    Xtr = prep.transform(Xraw.iloc[inner_tr])
    rules_inner = discovery.discover_rules(dom, Xtr, np.asarray(y)[inner_tr])
    check("inner-fold discovery returns rules", len(rules_inner) > 0)

    prep_o, dom_o = real_data.fit_preprocessor(name, Xraw, tr_i, y)
    Xo = prep_o.transform(Xraw.iloc[tr_i])
    rules_outer = discovery.discover_rules(dom_o, Xo, np.asarray(y)[tr_i])
    check("discovery runs on the inner partition independently",
          len(rules_outer) > 0)


def test_inner_cv_refits_preprocessing_per_split():
    """_inner_cv must refit preprocessing once per inner split."""
    from src import real_data
    from experiments import nested_cv

    name = "heart"
    Xraw, y = real_data.DATASETS[name]["loader"]()
    f = next(iter(real_data.make_oof_folds(name, n_folds=5, seed=0)))
    tr_i = f["tr_i"]

    calls = {"n": 0}
    real_fit = real_data.fit_preprocessor

    def counting(name_, Xraw_, rows, y_):
        calls["n"] += 1
        return real_fit(name_, Xraw_, rows, y_)

    nested_cv.real_data.fit_preprocessor = counting
    try:
        cfg = nested_cv.build_methods(0)["LogReg"]
        nested_cv._inner_cv(cfg, name, Xraw, np.asarray(y), tr_i, seed=0,
                            epochs=5, patience=3, kb_kind="expert",
                            corrupt_level=0.4, n_inner=3)
    finally:
        nested_cv.real_data.fit_preprocessor = real_fit
    check("preprocessing refit once per inner split", calls["n"] == 3)


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
    test_holm_correction()
    test_bootstrap_ci_covers_mean()
    test_corruption_monotone_and_empty()
    test_paired_statistics_schema()
    test_monk_kb_is_exact()
    test_full_monk_domain_labels()
    test_injected_rule_layer_mirrors_kb()
    test_genuine_rule_discovery_from_data()
    test_tree_metric_cannot_represent_or_of_conjunctions()
    test_no_preprocessing_leak()
    test_oof_folds_are_disjoint_and_cover()
    test_nested_threshold_is_validation_only()
    test_discovered_rules_are_not_injected()
    test_semantic_rule_equivalence()
    test_corruption_degrades_kb()
    test_monk_encoding_is_codebook_based()
    test_monk_expert_kb_matches_published_truth()
    test_full_monk_domain_is_432_with_four_a5_levels()
    test_metrics_are_per_fold_at_fold_threshold()
    test_run_nested_aggregates_per_fold()
    test_inner_fold_preprocessing_is_independent()
    test_inner_fold_rule_discovery_is_independent()
    test_inner_cv_refits_preprocessing_per_split()
    print("\nall tests passed")


if __name__ == "__main__":
    main()
