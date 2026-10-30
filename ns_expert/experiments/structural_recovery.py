"""Genuine structural rule-recovery test.

Motivation
----------
The distilled decision-tree metric used elsewhere floors at zero for
OR-of-conjunctions targets (e.g. the eight tic-tac-toe winning lines): a
depth-limited tree emits longer paths that never equal the 3-literal true rules,
so it reports ``rule-F1 = 0`` for *every* model even when the model's decisions
agree with the true logic. That makes the metric unable to answer the question it
is asked to answer.

This script measures structural recovery directly instead:

1. enumerate the finite domain on which the target logic is defined (for
   tic-tac-toe, all ``3^9`` boards over {x, o, blank});
2. check the model's decision function agrees with the ground-truth DNF on that
   whole domain (``decision_agreement``);
3. report whether the model's own rule layer *is* the ground-truth DNF ---
   exact set equality between the learned rule literals and the true lines
   (``structural_exact_match``), which is the honest, strong notion of recovery.

The expert knowledge base on tic-tac-toe *is* the ground truth, so this is a
fair test of whether the differentiable rule layer recovers it.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import models, real_data  # noqa: E402
from experiments.run_experiments import _decisions, _evaluate  # noqa: E402

X, O, B = 1.0, 0.0, 0.5


def full_ttt_domain() -> tuple[pd.DataFrame, np.ndarray]:
    """All 3^9 tic-tac-toe boards and their true win labels."""
    rows = list(itertools.product((X, O, B), repeat=9))
    Xd = pd.DataFrame(rows, columns=real_data.TTT_CELLS)
    labels = [int(any(r[i] == X and r[j] == X and r[k] == X
                      for i, j, k in real_data.TTT_LINES)) for r in rows]
    return Xd, np.asarray(labels, dtype=int)


# attribute ranges for the MONK domain, in file column order a1..a6
MONK_RANGES = [(1, 2), (1, 2, 3), (1, 2, 3), (1, 2), (1, 2, 3, 4), (1, 2)]


def full_monk_domain(version: int) -> tuple[pd.DataFrame, np.ndarray]:
    """All 432 MONK instances and their exact logical labels."""
    rows = list(itertools.product(*MONK_RANGES))
    Xd = pd.DataFrame(rows, columns=real_data.MONK_COLS)
    a1, a2, a3, a4, a5, a6 = (np.array([r[i] for r in rows]) for i in range(6))
    if version == 1:
        lab = (a1 == a2) | (a5 == 1)
    else:
        lab = ((a5 == 3) & (a4 == 1)) | ((a5 != 4) & (a2 != 3))
    return Xd, lab.astype(int)


def _full_domain(name: str):
    if name == "ttt":
        return full_ttt_domain()
    return full_monk_domain(int(name[-1]))


def _train_split_bounds(name: str, seed: int):
    """Reproduce load_real's per-seed normalisation bounds exactly."""
    Xraw, y = real_data.DATASETS[name]["loader"]()
    idx = np.random.default_rng(seed).permutation(len(y))
    tr_i = idx[: int(0.6 * len(idx))]
    return Xraw.iloc[tr_i].min(), Xraw.iloc[tr_i].max()


def evaluate_structural(name: str, seeds: list[int], epochs: int,
                        patience: int) -> pd.DataFrame:
    Xd_raw, yd = _full_domain(name)
    rows = []
    for seed in seeds:
        lo, hi = _train_split_bounds(name, seed)
        Xd = real_data.normalize(Xd_raw, lo, hi)
        domain, tr, va, te = real_data.load_real(name, seed=seed)
        # the true logic in the model's normalised space
        true_set = {frozenset(str(l) for l in r.body)
                    for r in domain.expert_rules if r.head == domain.label}

        model = models.NEXS(domain, seed=seed, pos_weight="auto",
                            epochs=epochs, patience=patience)
        model.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
        mm, _ = _evaluate(model, va.X.to_numpy(), va.y, te.X.to_numpy(), te.y,
                          calibrate=True)
        agree = float((np.asarray(_decisions(model, mm.get("threshold", 0.5))(Xd),
                                  dtype=int) == yd).mean())

        # the model's own rule layer, not a distilled surrogate
        learned_set = {frozenset(str(l) for l in r.body)
                       for r in model.rules if r.head == domain.label}
        overlap = len(learned_set & true_set)

        # reference points on the same full domain
        oracle_agree = float((_kb_predict(name, Xd, lo, hi) == yd).mean())
        mlp = models.MLPModel(domain, seed=seed, pos_weight="auto",
                              epochs=epochs, patience=patience)
        mlp.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
        mmm, _ = _evaluate(mlp, va.X.to_numpy(), va.y, te.X.to_numpy(), te.y,
                           calibrate=True)
        mlp_agree = float((np.asarray(_decisions(mlp, mmm.get("threshold", 0.5))(Xd),
                                      dtype=int) == yd).mean())

        rows.append(dict(dataset=name, seed=seed, n_true=len(true_set),
                         n_learned=len(learned_set),
                         structural_exact_match=int(learned_set == true_set),
                         rules_recovered=overlap,
                         rule_recall=overlap / max(1, len(true_set)),
                         rule_precision=overlap / max(1, len(learned_set)),
                         decision_agreement=agree,
                         oracle_agreement=oracle_agree,
                         mlp_agreement=mlp_agree))
        print(f"  {name} seed={seed}: NEXS={agree:.4f} oracle={oracle_agree:.4f} "
              f"MLP={mlp_agree:.4f} exact={int(learned_set == true_set)} "
              f"recovered={overlap}/{len(true_set)}", flush=True)
    return pd.DataFrame(rows)


def _kb_predict(name: str, Xd_norm: pd.DataFrame, lo, hi) -> np.ndarray:
    """Crisp expert-KB prediction on the full domain (normalised space)."""
    from src.expert_system import run as symbolic_run
    domain = real_data.build_domain(name, lo, hi)
    return np.asarray(symbolic_run(domain, Xd_norm, soft=False)["y_hat"], dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["ttt", "monk1", "monk3"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out", default="results/structural")
    args = ap.parse_args()
    frames = [evaluate_structural(n, args.seeds, args.epochs, args.patience)
              for n in args.datasets]
    df = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "structural_recovery.csv", index=False)
    print(df.groupby("dataset")[["decision_agreement", "oracle_agreement",
                                 "mlp_agreement", "structural_exact_match",
                                 "rule_recall", "rule_precision"]].mean().round(4).to_string())
    print("written ->", out)


if __name__ == "__main__":
    main()
