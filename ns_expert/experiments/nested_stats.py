"""Independent statistical validation on pooled out-of-fold predictions.

Why this replaces the earlier seed-level tests
---------------------------------------------
Treating 20 random seeds as 20 independent samples is pseudo-replication: each
seed re-permutes the *same* finite dataset, so a row is tested many times and the
observations are correlated.  A paired test over seeds then has an inflated
sample size and anti-conservative p-values.

Here the unit of observation is a **data row**.  Every row is predicted exactly
once, out-of-fold, under the nested-CV protocol, so the n of each test is the
number of rows in the dataset and no observation is reused.  Comparisons are
paired (same rows, same folds), which is the correct design for two models
evaluated on identical data.

Reported per (dataset, method-A, method-B):
  * paired accuracy difference and its bootstrap CI (row-level resampling),
  * exact McNemar test on the discordant pairs,
  * Holm-adjusted p-values across the prespecified family of primary
    comparisons,
  * effect size (risk difference and Cohen's h where defined).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import real_data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Prespecified primary comparisons (fixed before looking at the results).
PRIMARY = [
    ("NEXS_Expert", "MLP"),           # does the expert rule layer help vs neural?
    ("NEXS_Discovered", "MLP"),       # does *discovered* knowledge help?
    ("NEXS_Expert", "NEXS_Discovered"),  # expert KB vs discovered KB
    ("NEXS_Expert", "NEXS_Corrupt"),  # does KB quality matter?
    ("NEXS_Expert", "GradientBoosting"),  # vs the strongest baseline
]


def _mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> tuple[int, int, float]:
    """Exact (binomial) McNemar test. Returns (a_only, b_only, p)."""
    n01 = int(np.sum(~b_correct & a_correct))   # A right, B wrong
    n10 = int(np.sum(b_correct & ~a_correct))   # B right, A wrong
    if n01 + n10 == 0:
        return n01, n10, 1.0
    p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
    return n01, n10, p


def _boot_ci(diff: np.ndarray, n_boot: int = 5000, seed: int = 0,
             alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference (row-level)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def _cohens_h(p1: float, p2: float) -> float:
    """Effect size for two proportions (Cohen's h)."""
    p1 = min(max(p1, 1e-6), 1 - 1e-6)
    p2 = min(max(p2, 1e-6), 1 - 1e-6)
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj.tolist()


def load_oof(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def analyse(dataset: str, oof: pd.DataFrame, y: np.ndarray,
            primary: list[tuple[str, str]] | None = None,
            n_boot: int = 5000, seed: int = 0) -> list[dict]:
    """Paired row-level comparisons for one dataset."""
    primary = primary or PRIMARY
    sub = oof[oof.dataset == dataset]
    use = "yhat" if "yhat" in sub.columns else "proba"
    piv = sub.pivot_table(index="row", columns="method", values=use)
    piv = piv.sort_index()
    rows = []
    raw_p = []
    for a, b in primary:
        if a not in piv.columns or b not in piv.columns:
            continue
        pa, pb = piv[a].to_numpy(), piv[b].to_numpy()
        if use == "yhat":
            ca, cb = pa.astype(int) == y, pb.astype(int) == y
        else:
            ca, cb = (pa >= 0.5).astype(int) == y, (pb >= 0.5).astype(int) == y
        diff = ca.astype(float) - cb.astype(float)
        lo, hi = _boot_ci(diff, n_boot=n_boot, seed=seed)
        n01, n10, p = _mcnemar_exact(ca, cb)
        raw_p.append(p)
        rows.append(dict(
            dataset=dataset, method_a=a, method_b=b, n=len(y),
            acc_a=float(ca.mean()), acc_b=float(cb.mean()),
            mean_diff=float(diff.mean()), ci_lo=lo, ci_hi=hi,
            a_only=n01, b_only=n10, p_raw=p,
            cohens_h=_cohens_h(float(ca.mean()), float(cb.mean()))))
    if rows:
        for r, padj in zip(rows, holm(raw_p)):
            r["p_holm"] = padj
            r["significant_holm"] = int(padj < 0.05)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="results/nested_cv/nested_cv_oof_proba.csv")
    ap.add_argument("--datasets", nargs="*",
                    default=["heart", "pima", "wbc", "ttt", "monk1", "monk3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", default="results/nested_cv")
    args = ap.parse_args()

    oof = load_oof(ROOT / args.oof)
    # one primary split: the first seed present, so no row is reused
    if "seed" in oof.columns:
        first = sorted(oof.seed.unique())[0]
        oof = oof[oof.seed == first]
    all_rows = []
    for ds in args.datasets:
        _, y = real_data.DATASETS[ds]["loader"]()
        all_rows += analyse(ds, oof, np.asarray(y), n_boot=args.boot, seed=args.seed)
    df = pd.DataFrame(all_rows)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "paired_statistics.csv", index=False)
    cols = ["dataset", "method_a", "method_b", "acc_a", "acc_b", "mean_diff",
            "ci_lo", "ci_hi", "p_raw", "p_holm", "significant_holm"]
    print(df[cols].round(4).to_string(index=False))
    print("written ->", out / "paired_statistics.csv")


if __name__ == "__main__":
    main()
