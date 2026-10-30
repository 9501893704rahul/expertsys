"""Figures and LaTeX tables for the real-data experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 160, "savefig.bbox": "tight"})

PALETTE = {
    "ESS(discrete)": "#7f7f7f", "ESS(soft)": "#bcbd22", "LogReg": "#c7c7c7",
    "DecisionTree": "#aec7e8", "RandomForest": "#1f77b4", "GradientBoosting": "#2ca02c",
    "MLP": "#ff7f0e", "SemanticLoss": "#9467bd", "NEXS(proposed)": "#d62728",
}
ORDER = ["ESS(discrete)", "ESS(soft)", "LogReg", "DecisionTree", "RandomForest",
         "GradientBoosting", "MLP", "SemanticLoss", "NEXS(proposed)"]
DS = ["heart", "pima", "ttt"]


def _order(methods):
    return [m for m in ORDER if m in set(methods)]


def fig_main(main):
    agg = main.groupby(["dataset", "method"]).mean(numeric_only=True).reset_index()
    metrics = [("acc", "Accuracy"), ("f1", "F1"), ("auc", "ROC-AUC"), ("ece", "ECE (lower better)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, (col, title) in zip(axes, metrics):
        width = 0.26
        for k, d in enumerate(DS):
            sub = agg[agg.dataset == d].set_index("method")
            o = _order(sub.index)
            x = np.arange(len(o)) + (k - 1) * width
            ax.bar(x, [sub.loc[m, col] for m in o], width,
                   color=[PALETTE.get(m, "#333") for m in o], edgecolor="black",
                   linewidth=0.4, alpha=0.55 + 0.2 * k, label=d, hatch=["", "//", "xx"][k])
        ax.set_xticks(np.arange(len(o)))
        ax.set_xticklabels(o, rotation=55, ha="right", fontsize=7)
        ax.set_title(title)
    axes[0].legend(fontsize=7, title="dataset")
    fig.suptitle("Real-data benchmark: accuracy, F1, AUC, calibration (mean over 5 seeds)", y=1.04)
    fig.savefig(FIG / "fig_real_main.png")
    plt.close(fig)


def fig_logic(main):
    agg = main.groupby(["dataset", "method"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (col, title) in zip(axes, [("rule_f1", "Rule recovery (F1)"),
                                       ("logic_fid", "Logic fidelity to KB oracle")]):
        width = 0.26
        for k, d in enumerate(DS):
            sub = agg[agg.dataset == d].set_index("method").dropna(subset=[col])
            o = _order(sub.index)
            x = np.arange(len(o)) + (k - 1) * width
            ax.bar(x, [sub.loc[m, col] for m in o], width,
                   color=[PALETTE.get(m, "#333") for m in o], edgecolor="black",
                   linewidth=0.4, alpha=0.55 + 0.2 * k, label=d, hatch=["", "//", "xx"][k])
        ax.set_xticks(np.arange(len(o)))
        ax.set_xticklabels(o, rotation=55, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, title="dataset")
    fig.suptitle("Real data: is the expert logic recovered?", y=1.04)
    fig.savefig(FIG / "fig_real_logic.png")
    plt.close(fig)


def fig_efficiency(eff):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, d in zip(axes, DS):
        sub = eff[eff.dataset == d]
        for m in _order(sub.method.unique()):
            s = sub[sub.method == m].sort_values("n_train")
            ax.plot(s.n_train, s.f1, marker="o", ms=3.5, lw=1.4,
                    color=PALETTE.get(m, "#333"), label=m)
        ax.set_title(f"{d}: F1 vs. training size")
        ax.set_xlabel("# train")
        ax.set_ylabel("F1")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Real-data data efficiency", y=1.03)
    fig.savefig(FIG / "fig_real_efficiency.png")
    plt.close(fig)


def fig_subgroup(sub):
    agg = sub.groupby(["dataset", "subgroup", "method"]).mean(numeric_only=True).reset_index()
    keys = [(d, s) for d in DS for s in sorted(agg[agg.dataset == d].subgroup.unique())]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    o = _order(agg.method.unique())
    width = 0.8 / max(1, len(keys))
    for k, (d, s) in enumerate(keys):
        sb = agg[(agg.dataset == d) & (agg.subgroup == s)].set_index("method")
        x = np.arange(len(o)) + (k - len(keys) / 2 + 0.5) * width
        ax.bar(x, [sb.loc[m, "acc"] if m in sb.index else 0 for m in o], width,
               label=f"{d}/{s}", edgecolor="black", linewidth=0.3,
               color=plt.cm.tab20(k / max(1, len(keys))))
    ax.set_xticks(np.arange(len(o)))
    ax.set_xticklabels(o, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, ncol=3)
    ax.set_title("Accuracy on natural subgroups (robustness proxy)")
    fig.savefig(FIG / "fig_real_subgroup.png")
    plt.close(fig)


def fig_ablation(abl):
    agg = abl.groupby(["dataset", "variant"]).mean(numeric_only=True).reset_index()
    variants = ["NEXS(full)", "NEXS(no rule layer)", "NEXS(rule layer only)",
                "NEXS(no semantic loss)", "NEXS(no expert prior)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, (col, title) in zip(axes, [("f1", "F1"), ("logic_fid", "Logic fidelity")]):
        width = 0.26
        for k, d in enumerate(DS):
            sub = agg[agg.dataset == d].set_index("variant")
            x = np.arange(len(variants)) + (k - 1) * width
            ax.bar(x, [sub.loc[v, col] if v in sub.index else 0 for v in variants], width,
                   label=d, edgecolor="black", linewidth=0.4,
                   color=["#1f77b4", "#d62728", "#2ca02c"][k], hatch=["", "//", "xx"][k])
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels([v.replace("NEXS", "").strip("()") for v in variants],
                           rotation=35, ha="right", fontsize=7)
        ax.set_title(title)
    axes[0].legend(fontsize=7, title="dataset")
    fig.suptitle("Real-data ablation", y=1.04)
    fig.savefig(FIG / "fig_real_ablation.png")
    plt.close(fig)


def latex_tables(main, eff, sub, abl, lat):
    agg = main.groupby(["dataset", "method"]).mean(numeric_only=True).reset_index()
    o = _order(agg.method.unique())
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Real-data results (mean over 5 seeds). Best per dataset-metric in bold. "
             r"Acc is at the F1-optimal validation threshold; Acc$^\ast$ is at the accuracy-optimal "
             r"threshold (the expert system has no learnable operating point, so it is n/a).}",
             r"\label{tab:realmain}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llccccccc}", r"\toprule",
             r"Dataset & Method & Acc & Acc$^\ast$ & F1 & AUC & ECE & RuleF1 & LogicFid \\",
             r"\midrule"]
    for d in DS:
        sb = agg[agg.dataset == d].set_index("method")
        best = {c: sb[c].max() for c in ["acc", "acc_opt", "f1", "auc", "rule_f1", "logic_fid"]}
        best["ece"] = sb["ece"].min()
        for m in o:
            if m not in sb.index:
                continue
            r = sb.loc[m]

            def bf(c, fmt="{:.3f}"):
                v = r[c]
                s = "n/a" if pd.isna(v) else fmt.format(v)
                if not pd.isna(v) and abs(v - best[c]) < 1e-9:
                    s = r"\textbf{" + s + "}"
                return s

            lines.append(f"{d} & {m} & {bf('acc')} & {bf('acc_opt')} & {bf('f1')} & {bf('auc')} & "
                         f"{bf('ece')} & {bf('rule_f1')} & {bf('logic_fid')} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table}"]
    (FIG / "table_real_main.tex").write_text("\n".join(lines))

    ls = [r"\begin{table}[t]\centering",
          r"\caption{Real-data accuracy on natural subgroups.}",
          r"\label{tab:realsub}", r"\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{ll" + "c" * len(o) + r"}", r"\toprule",
          r"Dataset & Subgroup & " + " & ".join(o) + r" \\", r"\midrule"]
    aggs = sub.groupby(["dataset", "subgroup", "method"]).mean(numeric_only=True).reset_index()
    for d in DS:
        for s in sorted(aggs[aggs.dataset == d].subgroup.unique()):
            sb = aggs[(aggs.dataset == d) & (aggs.subgroup == s)].set_index("method")
            cells = " & ".join("n/a" if m not in sb.index else f"{sb.loc[m,'acc']:.3f}" for m in o)
            ls.append(f"{d} & {s} & {cells} \\\\")
    ls += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (FIG / "table_real_subgroup.tex").write_text("\n".join(ls))

    agga = abl.groupby(["dataset", "variant"]).mean(numeric_only=True).reset_index()
    variants = ["NEXS(full)", "NEXS(no rule layer)", "NEXS(rule layer only)",
                "NEXS(no semantic loss)", "NEXS(no expert prior)"]
    la = [r"\begin{table}[t]\centering", r"\caption{Real-data ablation.}",
          r"\label{tab:realabl}", r"\begin{tabular}{llccc}", r"\toprule",
          r"Dataset & Variant & Acc & F1 & LogicFid \\", r"\midrule"]
    for d in DS:
        sb = agga[agga.dataset == d].set_index("variant")
        for v in variants:
            if v not in sb.index:
                continue
            r = sb.loc[v]
            la.append(f"{d} & {v} & {r['acc']:.3f} & {r['f1']:.3f} & {r['logic_fid']:.3f} \\\\")
        la.append(r"\midrule")
    la[-1] = r"\bottomrule"
    la += [r"\end{tabular}", r"\end{table}"]
    (FIG / "table_real_ablation.tex").write_text("\n".join(la))

    le = [r"\begin{table}[t]\centering", r"\caption{Real-data data efficiency (F1).}",
          r"\label{tab:realeff}", r"\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{llcccc}", r"\toprule",
          r"Dataset & Method & 25\% & 50\% & 75\% & 100\% \\", r"\midrule"]
    piv = eff.pivot_table(index=["dataset", "method"], columns="frac", values="f1")
    fr = sorted(eff.frac.unique())
    for (d, m), row in piv.iterrows():
        cells = " & ".join("n/a" if pd.isna(row.get(c)) else f"{row.get(c):.3f}" for c in fr)
        le.append(f"{d} & {m} & {cells} \\\\")
    le += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (FIG / "table_real_efficiency.tex").write_text("\n".join(le))


def main():
    main_df = pd.read_csv(RES / "real_e1_main.csv")
    eff = pd.read_csv(RES / "real_e4_data_efficiency.csv")
    sub = pd.read_csv(RES / "real_e5_subgroup.csv")
    abl = pd.read_csv(RES / "real_e6_ablation.csv")
    lat = pd.read_csv(RES / "real_e7_latency.csv")
    fig_main(main_df)
    fig_logic(main_df)
    fig_efficiency(eff)
    fig_subgroup(sub)
    fig_ablation(abl)
    latex_tables(main_df, eff, sub, abl, lat)
    print("real figures ->", FIG)


if __name__ == "__main__":
    main()
