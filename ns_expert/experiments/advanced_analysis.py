"""Advanced statistical analysis for the NEXS study.

Consumes a long-format results CSV (one row per run, with ``dataset``,
``method``, ``seed`` and metric columns) and produces:

* ``metric_overview.csv``   - mean / std / 95% CI per (dataset, method, metric);
* ``paired_statistics.csv`` - paired NEXS-vs-baseline comparisons with Wilcoxon
  signed-rank tests, paired bootstrap confidence intervals and Holm-corrected
  p-values;
* ``ttt_rule_behavior_by_seed.csv`` (optional, ``--probes``) - how NEXS responds
  to valid vs. broken tic-tac-toe winning-line patterns.

The analysis is deliberately generic over the CSV schema so the same script
serves the real-data study and the expanded multi-condition protocol.

Statistical note: with n seeds the Wilcoxon signed-rank test is exact only up to
n = 25; beyond that we fall back to the normal approximation. All p-values are
Holm-corrected within each (dataset, metric) family.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METRICS = ["acc", "acc_opt", "f1", "auc", "ece", "logic_fid", "rule_f1"]
LOWER_IS_BETTER = {"ece", "brier"}
PROPOSED = "NEXS(proposed)"
N_BOOT = 10_000


def _ci_bootstrap(values: np.ndarray, n_boot: int = N_BOOT, alpha: float = 0.05,
                  seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
               alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI of the paired mean difference ``a - b`` (same seed order)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return list(adj)


def metric_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, method), g in df.groupby(["dataset", "method"]):
        for metric in METRICS:
            if metric not in g:
                continue
            vals = g[metric].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            lo, hi = _ci_bootstrap(vals, seed=hash((ds, method, metric)) % 2**31)
            rows.append(dict(dataset=ds, method=method, metric=metric, n=len(vals),
                             mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1))
                             if len(vals) > 1 else 0.0,
                             ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def paired_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Paired NEXS-vs-baseline tests per (dataset, metric)."""
    rows = []
    for ds, gd in df.groupby("dataset"):
        methods = [m for m in gd["method"].unique() if m != PROPOSED]
        if PROPOSED not in set(gd["method"]):
            continue
        for metric in METRICS:
            if metric not in gd:
                continue
            piv = gd.pivot_table(index="seed", columns="method", values=metric)
            if PROPOSED not in piv:
                continue
            fam_rows = []
            for other in methods:
                if other not in piv:
                    continue
                pair = piv[[PROPOSED, other]].dropna()
                if len(pair) < 4:
                    continue
                a = pair[PROPOSED].to_numpy(float)
                b = pair[other].to_numpy(float)
                diff = a - b
                if np.allclose(diff, 0):
                    stat, p = 0.0, 1.0
                else:
                    stat, p = stats.wilcoxon(a, b, zero_method="wilcox",
                                             alternative="two-sided")
                lo, hi = _paired_ci(a, b, seed=hash((ds, metric, other)) % 2**31)
                better = "NEXS" if ((np.mean(diff) > 0) != (metric in LOWER_IS_BETTER)) else other
                fam_rows.append(dict(dataset=ds, metric=metric, method=other,
                                     n_pairs=len(pair),
                                     mean_nexs=float(a.mean()), mean_other=float(b.mean()),
                                     mean_diff=float(diff.mean()),
                                     diff_ci_lo=lo, diff_ci_hi=hi,
                                     wilcoxon_stat=float(stat), p_value=float(p),
                                     winner=better))
            if fam_rows:
                adj = _holm([r["p_value"] for r in fam_rows])
                for r, pa in zip(fam_rows, adj):
                    r["p_holm"] = pa
                    r["significant_05"] = bool(pa < 0.05)
                rows.extend(fam_rows)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# tic-tac-toe rule-behaviour probe
# --------------------------------------------------------------------------- #
def rule_behavior_probe(project: Path, seeds: list[int], epochs: int,
                        patience: int, out: Path) -> pd.DataFrame:
    """Does NEXS respond to valid winning lines and not to broken ones?

    We train NEXS on the tic-tac-toe training split and then score it on
    synthetic boards constructed to be either a *valid* winning line for ``x``
    (all three cells ``x``), a *broken* line (two ``x`` and one ``o``/blank), or
    a random board. A logic-faithful model should assign high probability to
    valid lines, low to broken ones, and the gap should be large.
    """
    import torch

    from src import models, real_data

    torch.set_num_threads(2)
    rows = []
    for seed in seeds:
        domain, tr, va, te = real_data.load_real("ttt", seed=seed)
        model = models.NEXS(domain, seed=seed, pos_weight="auto",
                            epochs=epochs, patience=patience)
        model.fit(tr.X.to_numpy(), tr.y, va.X.to_numpy(), va.y)
        rng = np.random.default_rng(seed)

        def board_probs(cells: dict[int, float]) -> float:
            x = np.full((1, 9), 0.5)  # 0.5 == blank
            for i, v in cells.items():
                x[0, i] = v
            return float(model.predict_proba(x)[0])

        lines = real_data.TTT_LINES
        valid = [board_probs({i: 1.0 for i in ln}) for ln in lines]
        broken = []
        for ln in lines:
            cells = {ln[0]: 1.0, ln[1]: 1.0, ln[2]: 0.0}   # two x, one o
            broken.append(board_probs(cells))
        random_boards = []
        for _ in range(len(lines)):
            cells = {i: float(rng.choice([0.0, 0.5, 1.0])) for i in range(9)}
            random_boards.append(board_probs(cells))
        for kind, vals in (("valid_line", valid), ("broken_line", broken),
                           ("random_board", random_boards)):
            rows.append(dict(seed=seed, kind=kind,
                             agreement=float(np.mean(vals)),
                             mean_prob=float(np.mean(vals))))
    out_df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out / "ttt_rule_behavior_by_seed.csv", index=False)
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir or CSV with a long-format results file")
    ap.add_argument("--project", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--no-probes", dest="probes", action="store_false")
    ap.add_argument("--probe-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.set_defaults(probes=False)
    args = ap.parse_args()

    results = Path(args.results)
    csv = results / "real_e1_main.csv" if results.is_dir() else results
    if results.is_dir() and not csv.exists():
        cands = sorted(results.glob("*.csv"))
        csv = cands[0] if cands else None
    assert csv and csv.exists(), f"no results CSV under {results}"

    df = pd.read_csv(csv)
    df = df.rename(columns={"dataset": "dataset"})
    if "dataset" not in df:
        df["dataset"] = "all"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    overview = metric_overview(df)
    overview.to_csv(out / "metric_overview.csv", index=False)
    stats_df = paired_statistics(df)
    stats_df.to_csv(out / "paired_statistics.csv", index=False)

    print(f"records: {len(df)}  methods: {df['method'].nunique()}  "
          f"datasets: {df['dataset'].nunique()}")
    print(f"metric_overview rows: {len(overview)}")
    print(f"paired comparisons: {len(stats_df)}  "
          f"significant (Holm<0.05): {int(stats_df.get('significant_05', pd.Series()).sum())}")

    if args.probes:
        beh = rule_behavior_probe(Path(args.project), args.probe_seeds,
                                  args.epochs, args.patience, out)
        print(beh.groupby("kind")["mean_prob"].mean().round(3).to_string())
    print("written ->", out)


if __name__ == "__main__":
    main()
