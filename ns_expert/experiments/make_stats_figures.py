"""Figures and LaTeX tables for the statistical / condition analysis.

Consumes ``results/conditions_5seed/all_runs.csv`` (from
``run_rule_conditions.py``) and the paired statistics from
``advanced_analysis.py``, and writes:

* ``fig_conditions.png``    - accuracy vs. data fraction for each KB-corruption
  level, plus rule recovery vs. corruption (the "when do rules help?" answer);
* ``table_conditions.tex``  - NEXS-minus-MLP accuracy by (corruption, data);
* ``table_stats.tex``       - paired NEXS-vs-baseline tests with Holm p-values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
COND = ROOT / "results" / "conditions_5seed" / "all_runs.csv"


def fig_conditions(df: pd.DataFrame) -> None:
    dsets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(2, len(dsets), figsize=(5 * len(dsets), 7),
                             squeeze=False)
    levels = sorted(df.kb_corruption.unique())
    cmap = plt.get_cmap("viridis")
    for j, ds in enumerate(dsets):
        gd = df[df.dataset == ds]
        ax = axes[0][j]
        for k, lv in enumerate(levels):
            gl = gd[(gd.kb_corruption == lv) & (gd.method == "NEXS")]
            cur = gl.groupby("data_frac")["acc"].mean()
            ax.plot(cur.index, cur.values, marker="o", color=cmap(k / max(1, len(levels) - 1)),
                    label=f"kb={lv:g}")
        ax.set_title(f"{ds}: NEXS accuracy vs data")
        ax.set_xlabel("training fraction")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, title="KB corruption")

        ax2 = axes[1][j]
        for m, ls in (("MLP", "--"), ("NEXS", "-")):
            gm = gd[gm_mask(gd, m)].groupby("kb_corruption")["rule_f1"].mean()
            ax2.plot(gm.index, gm.values, ls, marker="s", label=m)
        ax2.set_title(f"{ds}: rule recovery vs KB corruption")
        ax2.set_xlabel("KB corruption")
        ax2.set_ylabel("rule F1")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_conditions.png", dpi=150)
    plt.close(fig)


def gm_mask(gd: pd.DataFrame, m: str) -> pd.Series:
    return gd.method == m


def table_conditions(df: pd.DataFrame) -> None:
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Effect of expert-knowledge quality and data quantity. "
             r"Entries are NEXS $-$ MLP test accuracy (mean over 5 seeds); positive "
             r"means the expert prior helps. \emph{kb} is the fraction of expert rules "
             r"corrupted (0 = clean KB, 1 = no knowledge).}",
             r"\label{tab:conditions}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llccc}", r"\toprule",
             r"Dataset & KB corruption & 25\% data & 50\% data & 100\% data \\", r"\midrule"]
    for ds in sorted(df.dataset.unique()):
        gd = df[df.dataset == ds]
        for lv in sorted(gd.kb_corruption.unique()):
            cells = []
            for frac in sorted(gd.data_frac.unique()):
                sub = gd[(gd.kb_corruption == lv) & (gd.data_frac == frac)]
                piv = sub.pivot_table(index="seed", columns="method", values="acc")
                if {"NEXS", "MLP"} <= set(piv.columns):
                    d = (piv["NEXS"] - piv["MLP"]).mean()
                    cells.append(rf"\textbf{{{d:+.3f}}}" if d > 0.005 else f"{d:+.3f}")
                else:
                    cells.append("n/a")
            lines.append(f"{ds} & {lv:g} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table}"]
    (FIG / "table_conditions.tex").write_text("\n".join(lines))


def table_stats(stats: pd.DataFrame) -> None:
    keep = stats[stats.method.isin(["MLP", "SemanticLoss", "RandomForest"])]
    lines = [r"\begin{table}[t]\centering",
             f"\\caption{{Paired NEXS-vs-baseline comparisons ({int(stats.n_pairs.max())} seeds). "
             r"$\Delta$ is the mean paired difference (NEXS $-$ baseline); "
             r"$p_{\text{Holm}}$ is the Holm-corrected Wilcoxon signed-rank p-value. "
             r"CI is a 95\% paired bootstrap interval on $\Delta$.}",
             r"\label{tab:stats}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llccccc}", r"\toprule",
             r"Dataset & Metric & Baseline & $\Delta$ & 95\% CI & $p_{\text{Holm}}$ & Sig. \\",
             r"\midrule"]
    def esc(s: str) -> str:
        return str(s).replace("_", r"\_")

    for _, r in keep.iterrows():
        sig = r"\checkmark" if r["significant_05"] else "--"
        lines.append(f"{esc(r['dataset'])} & {esc(r['metric'])} & {esc(r['method'])} & "
                     f"{r['mean_diff']:+.3f} & "
                     f"[{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}] & "
                     f"{r['p_holm']:.3f} & {sig} \\\\")
    lines += [r"\end{tabular}}", r"\end{table}"]
    (FIG / "table_stats.tex").write_text("\n".join(lines))


def table_behavior(beh: pd.DataFrame) -> None:
    g = beh.groupby("kind")["mean_prob"].agg(["mean", "std"]).reset_index()
    order = {"valid_line": 0, "broken_line": 1, "random_board": 2}
    g = g.sort_values("kind", key=lambda s: s.map(order))
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Functional rule-behaviour probe on tic-tac-toe (5 seeds). "
             r"Mean predicted win probability for valid winning lines, lines with "
             r"one cell broken, and random boards. A logic-faithful model should "
             r"separate the first two.}",
             r"\label{tab:behavior}",
             r"\begin{tabular}{lcc}", r"\toprule",
             r"Board pattern & Mean prob. & Std \\", r"\midrule"]
    for _, r in g.iterrows():
        lines.append(f"{r['kind'].replace('_', ' ')} & {r['mean']:.3f} & {r['std']:.3f} \\\\")
    lines += [r"\end{tabular}", r"\end{table}"]
    (FIG / "table_behavior.tex").write_text("\n".join(lines))


def table_independent(val: pd.DataFrame) -> None:
    """NEXS vs MLP under three protocols, including the clean out-of-fold test."""
    lines = [r"% generated by make_stats_figures.py",
             r"\begin{table}[t]\centering\small",
             r"\caption{NEXS vs.\ MLP accuracy under three evaluation protocols. "
             r"The seed-resampled test reuses test rows across seeds; the "
             r"out-of-fold test predicts every row exactly once. Significance "
             r"largely disappears under the clean protocol, showing the "
             r"seed-level result is inflated by pseudo-replication.}",
             r"\label{tab:independent}",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"Dataset & Protocol & $n$ & NEXS & MLP & $p$ \\", r"\midrule"]
    pretty = {"mcnemar_rowlevel": "seed-resampled",
              "mcnemar_oof_5fold": "out-of-fold (clean)",
              "grouped_5fold": "5-fold grouped"}
    for _, r in val.iterrows():
        name = pretty.get(r["test"], r["test"])
        star = r"$^{*}$" if r["p_value"] < 0.05 else ""
        lines.append(f"{r['dataset']} & {name} & {int(r['n_obs'])} & "
                     f"{r['nexs_acc']:.3f} & {r['mlp_acc']:.3f} & "
                     f"{r['p_value']:.3f}{star} \\\\")
    lines += [r"\end{tabular}", r"\end{table}"]
    (FIG / "table_independent.tex").write_text("\n".join(lines))


def table_structural(struct: pd.DataFrame, monk: pd.DataFrame | None) -> None:
    """Rule recovery against genuine ground truth, on the full input domain."""
    lines = [r"% generated by make_stats_figures.py",
             r"\begin{table}[t]\centering\small",
             r"\caption{Rule recovery against \emph{ground-truth} logic on the "
             r"complete input domain. ``Exact'' is 1 when the model's rule layer "
             r"equals the true rule set. ``Agree'' is decision agreement with the "
             r"true label; ``oracle'' is the crisp expert KB and ``MLP'' a "
             r"baseline, both on the same domain.}",
             r"\label{tab:structural}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Domain & Rules & Exact & Agree & Oracle & MLP \\", r"\midrule"]
    agg = struct.groupby("dataset").mean(numeric_only=True)
    for ds, r in agg.iterrows():
        lines.append(f"{ds} & {int(r['n_true'])} & "
                     f"{r['structural_exact_match']:.2f} & "
                     f"{r['decision_agreement']:.3f} & "
                     f"{r['oracle_agreement']:.3f} & "
                     f"{r['mlp_agreement']:.3f} \\\\")
    lines += [r"\end{tabular}", r"\end{table}"]
    (FIG / "table_structural.tex").write_text("\n".join(lines))

    if monk is not None and len(monk):
        lines = [r"% generated by make_stats_figures.py",
                 r"\begin{table}[t]\centering\small",
                 r"\caption{Generalisation to a second rule-governed benchmark. "
                 r"MONK-1 and MONK-3 have published ground-truth formulas; the "
                 r"expert KB encodes them exactly. Accuracy (mean over seeds) "
                 r"with logic fidelity.}",
                 r"\label{tab:monk}",
                 r"\begin{tabular}{llcc}", r"\toprule",
                 r"Domain & Method & Acc. & Logic fid. \\", r"\midrule"]
        for ds in sorted(monk.dataset.unique()):
            g = monk[monk.dataset == ds].groupby("method")[
                ["acc", "logic_fid"]].mean().sort_values("acc", ascending=False)
            for m, r in g.iterrows():
                lines.append(f"{ds} & {m.replace('_', chr(92)+'_')} & "
                             f"{r['acc']:.3f} & {r['logic_fid']:.3f} \\\\")
        lines += [r"\end{tabular}", r"\end{table}"]
        (FIG / "table_monk.tex").write_text("\n".join(lines))


def main() -> None:
    if not COND.exists():
        print("missing", COND, "- run run_rule_conditions.py first")
        sys.exit(1)
    df = pd.read_csv(COND)
    fig_conditions(df)
    table_conditions(df)
    # Prefer the 20-seed analysis for inference; fall back to 5 seeds.
    for tag in ("stats_20seed", "stats_5seed"):
        stats_path = ROOT / "results" / tag / "paired_statistics.csv"
        if stats_path.exists():
            table_stats(pd.read_csv(stats_path))
            break
    for tag in ("stats_5seed", "stats_20seed"):
        beh_path = ROOT / "results" / tag / "ttt_rule_behavior_by_seed.csv"
        if beh_path.exists():
            table_behavior(pd.read_csv(beh_path))
            break
    val_path = ROOT / "results" / "paired_validation" / "independent_validation.csv"
    if val_path.exists():
        table_independent(pd.read_csv(val_path))
    struct_path = ROOT / "results" / "structural" / "structural_recovery.csv"
    monk_path = ROOT / "results" / "real_e1_main_monk.csv"
    if struct_path.exists():
        table_structural(pd.read_csv(struct_path),
                         pd.read_csv(monk_path) if monk_path.exists() else None)
    print("stats figures/tables ->", FIG)


if __name__ == "__main__":
    main()
