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
    """Decision fidelity on the full domain, plus genuine rule discovery.

    The ``NEXS rules'' column is the injected expert KB (an implementation
    check, not discovery); the ``Discovered'' columns come from a
    candidate-generation learner trained on labels alone.
    """
    lines = [r"% generated by make_stats_figures.py",
             r"\begin{table}[t]\centering\small",
             r"\caption{Logic fidelity on the complete input domain, and "
             r"non-circular rule discovery. ``Agree'' is decision agreement "
             r"with the true label over the whole domain (oracle = crisp KB, "
             r"MLP = baseline). ``Discovered'' rules are learned from labels "
             r"alone by a candidate-generation learner and compared to the "
             r"hidden ground truth (exact set equality, and rule precision / "
             r"recall). NEXS injects its rule layer, so it is not credited with "
             r"discovery.}",
             r"\label{tab:structural}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Domain & NEXS & Oracle & MLP & Discovered & Disc. P / R \\",
             r"\midrule"]
    agg = struct.groupby("dataset").mean(numeric_only=True)
    for ds, r in agg.iterrows():
        lines.append(f"{ds} & {r['decision_agreement']:.3f} & "
                     f"{r['oracle_agreement']:.3f} & {r['mlp_agreement']:.3f} & "
                     f"{'exact' if r['discovered_exact_match'] > 0.5 else 'partial'} & "
                     f"{r['discovered_precision']:.2f} / {r['discovered_recall']:.2f} \\\\")
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


def table_discovery(disc: pd.DataFrame) -> None:
    """Genuine structural rule discovery from labels alone (per domain, seeds)."""
    lines = [r"% generated by make_stats_figures.py",
             r"\begin{table}[t]\centering\small",
             r"\caption{Structural rule discovery with \emph{no} expert rules "
             r"supplied. A candidate-generation learner (CN2-style) is trained "
             r"on the training-split labels only; the discovered conjunction "
             r"set is compared to the hidden ground-truth rule set by exact "
             r"set equality. ``Agree'' is decision agreement of the discovered "
             r"DNF with the true label over the full domain. Mean over 5 seeds.}",
             r"\label{tab:discovery}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Domain & True rules & Exact & Precision / Recall & Agree \\",
             r"\midrule"]
    agg = disc.groupby("dataset").mean(numeric_only=True)
    for ds, r in agg.iterrows():
        lines.append(f"{ds} & {int(r['n_gt_rules'])} & "
                     f"{r['discovered_exact_match']:.2f} & "
                     f"{r['rule_precision']:.2f} / {r['rule_recall']:.2f} & "
                     f"{r['decision_agreement']:.3f} \\\\")
    lines += [r"\end{tabular}", r"\end{table}"]
    (FIG / "table_discovery.tex").write_text("\n".join(lines))


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_")


def table_nested_main(metrics: pd.DataFrame) -> None:
    """Accuracy / F1 / AUC per (dataset, method) under nested CV."""
    order = ["LogReg", "DecisionTree", "RandomForest", "GradientBoosting",
             "MLP", "SemanticLoss", "NEXS_Expert", "NEXS_Discovered",
             "NEXS_Corrupt"]
    methods = [m for m in order if m in set(metrics.method)]
    datasets = sorted(metrics.dataset.unique())
    acc = metrics.pivot_table(index="method", columns="dataset", values="acc")
    f1 = metrics.pivot_table(index="method", columns="dataset", values="f1")
    auc = metrics.pivot_table(index="method", columns="dataset", values="auc")
    lines = [
        r"\begin{table}[t]\centering",
        r"\small",
        r"\caption{Accuracy (top), F1 (middle) and AUC (bottom) under the corrected "
        r"nested-CV protocol (5 outer folds, every row predicted exactly once). "
        r"All preprocessing, tuning, early stopping and threshold selection happen "
        r"inside the training folds.}",
        r"\label{tab:nested-main}",
        r"\begin{tabular}{l" + "c" * len(datasets) + "}",
        r"\toprule",
        "Method & " + " & ".join(_esc(d) for d in datasets) + r" \\",
        r"\midrule",
    ]
    for block, name in ((acc, "Acc"), (f1, "F1"), (auc, "AUC")):
        lines.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(datasets) + 1, name))
        for m in methods:
            if m not in block.index:
                continue
            cells = " & ".join(f"{block.loc[m, d]:.3f}" for d in datasets)
            lines.append(f"{_esc(m)} & {cells} \\\\")
        if name != "AUC":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (FIG / "table_nested_main.tex").write_text("\n".join(lines))


def table_nested_seed_summary(metrics: pd.DataFrame) -> None:
    """Accuracy across independent outer-CV repetitions (stability check)."""
    order = ["LogReg", "DecisionTree", "RandomForest", "GradientBoosting",
             "MLP", "SemanticLoss", "NEXS_Expert", "NEXS_Discovered",
             "NEXS_Corrupt"]
    methods = [m for m in order if m in set(metrics.method)]
    datasets = sorted(metrics.dataset.unique())
    g = metrics.groupby(["dataset", "method"])["acc"]
    lines = [
        r"\begin{table}[t]\centering",
        r"\footnotesize",
        r"\caption{Stability of accuracy across independent outer-CV repetitions "
        r"(each repetition re-partitions the same rows into $5$ folds). Cells are "
        r"mean\,$\pm$\,std. These repetitions reuse the same observations, so they "
        r"are a stability check only; all inference runs on the pooled out-of-fold "
        r"rows of the primary repetition.}",
        r"\label{tab:nested-seed}",
        r"\begin{tabular}{l" + "c" * len(datasets) + "}",
        r"\toprule",
        "Method & " + " & ".join(_esc(d) for d in datasets) + r" \\",
        r"\midrule",
    ]
    for m in methods:
        cells = []
        for d in datasets:
            key = (d, m)
            if key in g.groups:
                mu, sd = g.get_group(key).mean(), g.get_group(key).std(ddof=0)
                cells.append(f"{mu:.3f}\\,$\\pm$\\,{sd:.3f}")
            else:
                cells.append("--")
        lines.append(f"{_esc(m)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (FIG / "table_nested_seed.tex").write_text("\n".join(lines))


def table_nested_stats(stats: pd.DataFrame) -> None:
    """Paired row-level comparisons with bootstrap CIs and Holm correction."""
    lines = [
        r"\begin{table}[t]\centering",
        r"\small",
        r"\caption{Paired comparisons on out-of-fold predictions. The unit of "
        r"observation is a data row, not a random seed. $\Delta$ is the mean "
        r"paired accuracy difference (A$-$B); CI is a 95\% row-level bootstrap "
        r"interval; $p$ is an exact McNemar test and $p_{\mathrm{Holm}}$ the "
        r"Holm-adjusted value across the prespecified family.}",
        r"\label{tab:nested-stats}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Comparison & $\Delta$ & CI$_{95}$ & $p$ & $p_{\mathrm{Holm}}$ & Sig. \\",
        r"\midrule",
    ]
    for _, r in stats.iterrows():
        comp = f"{_esc(r.method_a)} $-$ {_esc(r.method_b)}"
        sig = r"\checkmark" if r.get("significant_holm", 0) else "--"
        lines.append(
            f"{_esc(r.dataset)} & {comp} & {r.mean_diff:+.3f} & "
            f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {r.p_raw:.3f} & "
            f"{r.p_holm:.3f} & {sig} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (FIG / "table_nested_stats.tex").write_text("\n".join(lines))


def table_nested_structural(struct: pd.DataFrame) -> None:
    """Out-of-fold structural rule recovery (syntactic and semantic)."""
    agg = struct.groupby("dataset").mean(numeric_only=True)
    lines = [
        r"\begin{table}[t]\centering",
        r"\small",
        r"\caption{Structural rule recovery out-of-fold. Rules are discovered from "
        r"training rows only and compared with the published ground truth. "
        r"\emph{Semantic} matches are logical: two rules match if they select the "
        r"same points of the full input domain, even when written with different "
        r"literals. \emph{Equiv.} is the fraction of the complete input domain on "
        r"which the discovered DNF agrees with the true formula.}",
        r"\label{tab:nested-structural}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & \#true & \#disc. & Rule F1 (sem.) & Exact (sem.) & Equiv. \\",
        r"\midrule",
    ]
    for ds in sorted(agg.index):
        r = agg.loc[ds]
        lines.append(
            f"{_esc(ds)} & {r.n_truth:.0f} & {r.n_discovered:.1f} & "
            f"{r.rule_f1_semantic:.3f} & {r.exact_recovery_semantic:.2f} & "
            f"{r.logical_equivalence:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (FIG / "table_nested_structural.tex").write_text("\n".join(lines))


def table_robustness(rob: pd.DataFrame) -> None:
    """Accuracy vs KB corruption / completeness / training fraction."""
    piv = rob.pivot_table(index=["dataset", "sweep", "level"],
                          columns="method", values="acc").reset_index()
    lines = [
        r"\begin{table}[t]\centering",
        r"\small",
        r"\caption{Robustness under the corrected protocol. Each row is one "
        r"sweep point; accuracy is out-of-fold. \emph{corrupt} degrades the KB "
        r"(0 = intact, 1 = no rules); \emph{completeness} keeps a fraction of "
        r"the rules; \emph{lowdata} uses a fraction of the training rows only.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Sweep & Level & MLP & NEXS & $\Delta$ \\",
        r"\midrule",
    ]
    for _, r in piv.iterrows():
        mlp = r.get("MLP", np.nan)
        nexs = r.get("NEXS", np.nan)
        lines.append(
            f"{_esc(r.dataset)} & {_esc(r.sweep)} & {r.level:g} & {mlp:.3f} & "
            f"{nexs:.3f} & {nexs - mlp:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (FIG / "table_robustness.tex").write_text("\n".join(lines))


def table_ablation(abl: pd.DataFrame, faith: pd.DataFrame | None) -> None:
    """Neural-vs-symbolic ablation and rule-faithfulness on counterfactuals."""
    acc = abl.pivot_table(index="method", columns="dataset", values="acc")
    order = ["neural_only", "symbolic_only", "fused", "fused_no_prior"]
    methods = [m for m in order if m in acc.index]
    datasets = sorted(abl.dataset.unique())
    lines = [
        r"\begin{table}[t]\centering",
        r"\small",
        r"\caption{Neural-vs-symbolic ablation (accuracy, out-of-fold). "
        r"\emph{neural\_only} freezes the symbolic pathway off; "
        r"\emph{symbolic\_only} drops the neural logit; \emph{fused} is the full "
        r"model; \emph{fused\_no\_prior} removes the expert-prior regulariser.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l" + "c" * len(datasets) + "}",
        r"\toprule",
        "Variant & " + " & ".join(_esc(d) for d in datasets) + r" \\",
        r"\midrule",
    ]
    for m in methods:
        cells = " & ".join(f"{acc.loc[m, d]:.3f}" if d in acc.columns else "--"
                           for d in datasets)
        lines.append(f"{_esc(m)} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (FIG / "table_ablation_nested.tex").write_text("\n".join(lines))

    if faith is not None and len(faith):
        fg = faith.pivot_table(index="method", columns="dataset",
                               values="obedience")
        fg2 = faith.pivot_table(index="method", columns="dataset",
                                values="over_firing")
        fdatasets = sorted(faith.dataset.unique())
        flines = [
            r"\begin{table}[t]\centering",
            r"\small",
            r"\caption{Logical faithfulness (out-of-fold). \emph{Obed.}: fraction "
            r"of counterfactual rows satisfying a rule antecedent that the model "
            r"predicts positive. \emph{Over-fire}: fraction of near-miss rows "
            r"(all but one literal satisfied) that the model still predicts "
            r"positive --- a faithful conjunction should not fire on these.}",
            r"\label{tab:faithfulness}",
            r"\begin{tabular}{l" + "c" * len(fdatasets) + "}",
            r"\toprule",
            "Model & " + " & ".join(_esc(d) for d in fdatasets) + r" \\",
            r"\midrule",
            r"\multicolumn{%d}{l}{\textit{obedience}} \\" % (len(fdatasets) + 1),
        ]
        for m in fg.index:
            cells = " & ".join(f"{fg.loc[m, d]:.3f}" if d in fg.columns else "--"
                               for d in fdatasets)
            flines.append(f"{_esc(m)} & {cells} \\\\")
        flines.append(r"\midrule")
        flines.append(r"\multicolumn{%d}{l}{\textit{over-firing}} \\"
                      % (len(fdatasets) + 1))
        for m in fg2.index:
            cells = " & ".join(f"{fg2.loc[m, d]:.3f}" if d in fg2.columns else "--"
                               for d in fdatasets)
            flines.append(f"{_esc(m)} & {cells} \\\\")
        flines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (FIG / "table_faithfulness.tex").write_text("\n".join(flines))


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
    disc_path = ROOT / "results" / "discovery" / "rule_discovery.csv"
    if disc_path.exists():
        table_discovery(pd.read_csv(disc_path))

    # corrected-protocol tables (Phase 1)
    nested = ROOT / "results" / "nested_cv"
    met = nested / "nested_cv_metrics.csv"
    if met.exists():
        mdf = pd.read_csv(met)
        table_nested_main(mdf)
        if mdf.seed.nunique() > 1:
            table_nested_seed_summary(mdf)
    stat = nested / "paired_statistics.csv"
    if stat.exists():
        table_nested_stats(pd.read_csv(stat))
    nstruct = ROOT / "results" / "nested_structural" / "structural_recovery_oof.csv"
    if nstruct.exists():
        table_nested_structural(pd.read_csv(nstruct))
    rob = ROOT / "results" / "nested_robustness" / "robustness.csv"
    if rob.exists():
        table_robustness(pd.read_csv(rob))
    ablp = ROOT / "results" / "nested_ablation"
    if (ablp / "ablation.csv").exists():
        faith = (pd.read_csv(ablp / "faithfulness.csv")
                 if (ablp / "faithfulness.csv").exists() else None)
        table_ablation(pd.read_csv(ablp / "ablation.csv"), faith)
    print("stats figures/tables ->", FIG)


if __name__ == "__main__":
    main()
