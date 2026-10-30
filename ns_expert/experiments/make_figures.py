"""Generate paper figures and LaTeX tables from the CSV results."""

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
    "ESS(discrete)": "#7f7f7f", "ESS(soft)": "#bcbd22",
    "LogReg": "#c7c7c7", "DecisionTree": "#aec7e8", "RandomForest": "#1f77b4",
    "GradientBoosting": "#2ca02c", "MLP": "#ff7f0e", "SemanticLoss": "#9467bd",
    "NEXS(proposed)": "#d62728",
}


def _order(methods):
    pref = ["ESS(discrete)", "ESS(soft)", "LogReg", "DecisionTree", "RandomForest",
            "GradientBoosting", "MLP", "SemanticLoss", "NEXS(proposed)"]
    return [m for m in pref if m in set(methods)]


def fig_main(main: pd.DataFrame):
    agg = main.groupby(["domain", "method"]).mean(numeric_only=True).reset_index()
    domains = sorted(agg.domain.unique())
    metrics = [("acc", "Accuracy"), ("f1", "F1 (minority)"), ("auc", "ROC-AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (col, title) in zip(axes, metrics):
        width = 0.38
        for k, d in enumerate(domains):
            sub = agg[agg.domain == d].set_index("method")
            order = _order(sub.index)
            x = np.arange(len(order)) + (k - 0.5) * width
            ax.bar(x, [sub.loc[m, col] for m in order], width,
                   color=[PALETTE.get(m, "#333") for m in order],
                   edgecolor="black", linewidth=0.4, alpha=0.7 + 0.3 * k,
                   label=d, hatch=["", "//"][k])
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=55, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=7, title="domain")
    fig.suptitle("Main benchmark: accuracy, minority-class F1, and ROC-AUC (mean over seeds)", y=1.04)
    fig.savefig(FIG / "fig_main_metrics.png")
    plt.close(fig)


def fig_rules(main: pd.DataFrame):
    agg = main.groupby(["domain", "method"]).mean(numeric_only=True).reset_index()
    domains = sorted(agg.domain.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (col, title) in zip(axes, [("rule_f1", "Rule-recovery F1 (vs. ground-truth rules)"),
                                       ("logic_fid", "Logic fidelity to deterministic oracle")]):
        width = 0.38
        for k, d in enumerate(domains):
            sub = agg[agg.domain == d].set_index("method").dropna(subset=[col])
            order = _order(sub.index)
            x = np.arange(len(order)) + (k - 0.5) * width
            ax.bar(x, [sub.loc[m, col] for m in order], width,
                   color=[PALETTE.get(m, "#333") for m in order],
                   edgecolor="black", linewidth=0.4, alpha=0.7 + 0.3 * k, label=d,
                   hatch=["", "//"][k])
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=55, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, title="domain")
    fig.suptitle("Interpretability: are the learned rules the true rules?", y=1.04)
    fig.savefig(FIG / "fig_rule_recovery.png")
    plt.close(fig)


def fig_efficiency(eff: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
    for ax, d in zip(axes, sorted(eff.domain.unique())):
        sub = eff[eff.domain == d]
        for m in _order(sub.method.unique()):
            s = sub[sub.method == m].sort_values("n_train")
            ax.plot(s.n_train, s.f1, marker="o", ms=3.5, lw=1.4,
                    color=PALETTE.get(m, "#333"), label=m)
        ax.set_xscale("log")
        ax.set_title(f"{d}: minority-class F1 vs. training size")
        ax.set_xlabel("# training samples (log)")
        ax.set_ylabel("F1")
    axes[0].legend(fontsize=7)
    fig.suptitle("Data efficiency", y=1.03)
    fig.savefig(FIG / "fig_data_efficiency.png")
    plt.close(fig)


def fig_shift(shift: pd.DataFrame):
    agg = shift.groupby(["domain", "shift", "method"]).mean(numeric_only=True).reset_index()
    keys = [(d, s) for d in sorted(agg.domain.unique()) for s in ["covariate", "novel"]]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    width = 0.19
    order = _order(agg.method.unique())
    for k, (d, s) in enumerate(keys):
        sub = agg[(agg.domain == d) & (agg.shift == s)].set_index("method")
        x = np.arange(len(order)) + (k - len(keys) / 2 + 0.5) * width
        ax.bar(x, [sub.loc[m, "acc"] if m in sub.index else 0 for m in order], width,
               label=f"{d}/{s}", edgecolor="black", linewidth=0.4,
               color=plt.cm.tab10(k / len(keys)))
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("accuracy under shift")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Robustness to covariate shift and novel regimes")
    fig.savefig(FIG / "fig_shift.png")
    plt.close(fig)


def fig_ablation(abl: pd.DataFrame):
    agg = abl.groupby(["domain", "variant"]).mean(numeric_only=True).reset_index()
    variants = ["NEXS(full)", "NEXS(no rule layer)", "NEXS(rule layer only)",
                "NEXS(no semantic loss)", "NEXS(no expert prior)"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (col, title) in zip(axes, [("f1", "F1"), ("logic_fid", "Logic fidelity")]):
        width = 0.38
        for k, d in enumerate(sorted(agg.domain.unique())):
            sub = agg[agg.domain == d].set_index("variant")
            x = np.arange(len(variants)) + (k - 0.5) * width
            ax.bar(x, [sub.loc[v, col] if v in sub.index else 0 for v in variants], width,
                   label=d, edgecolor="black", linewidth=0.4, alpha=0.7 + 0.3 * k,
                   color=["#1f77b4", "#d62728"][k], hatch=["", "//"][k])
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels([v.replace("NEXS", "").strip("()") for v in variants],
                           rotation=35, ha="right", fontsize=7)
        ax.set_title(title)
    axes[0].legend(fontsize=7, title="domain")
    fig.suptitle("Ablation of the proposed neuro-symbolic expert system", y=1.04)
    fig.savefig(FIG / "fig_ablation.png")
    plt.close(fig)


def fig_tradeoff(main: pd.DataFrame, lat: pd.DataFrame):
    agg = main.groupby(["domain", "method"]).mean(numeric_only=True).reset_index()
    l = lat.groupby(["domain", "method"]).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for d in sorted(agg.domain.unique()):
        for _, r in agg[agg.domain == d].iterrows():
            lr = l[(l.domain == d) & (l.method == r.method)]
            if lr.empty:
                continue
            ax.scatter(lr.us_per_sample.iloc[0], r.f1, s=60,
                       color=PALETTE.get(r.method, "#333"),
                       marker="o" if d == "medical" else "s",
                       edgecolor="black", linewidth=0.5)
    for _, r in l.iterrows():
        ax.annotate(r.method.replace("(proposed)", ""), (r.us_per_sample,
                    agg[(agg.domain == r.domain) & (agg.method == r.method)].f1.mean()),
                    fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("inference latency (µs / sample, log)")
    ax.set_ylabel("minority-class F1")
    ax.set_title("Accuracy-latency trade-off (o=medical, s=industrial)")
    fig.savefig(FIG / "fig_tradeoff.png")
    plt.close(fig)


def latex_tables(main, eff, shift, abl, lat):
    agg = main.groupby(["domain", "method"]).mean(numeric_only=True).reset_index()
    order = _order(agg.method.unique())
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Main results (mean over 3 seeds). Best per domain-metric in bold.}",
             r"\label{tab:main}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Domain & Method & Acc & F1 & AUC & ECE & RuleF1 & LogicFid \\", r"\midrule"]
    for d in sorted(agg.domain.unique()):
        sub = agg[agg.domain == d].set_index("method")
        best = {c: sub[c].max() for c in ["acc", "f1", "auc", "ece", "rule_f1", "logic_fid"]}
        for m in order:
            if m not in sub.index:
                continue
            r = sub.loc[m]

            def bf(c, fmt="{:.3f}"):
                v = r[c]
                s = "n/a" if pd.isna(v) else fmt.format(v)
                if c in best and not pd.isna(v) and abs(v - best[c]) < 1e-9:
                    s = r"\textbf{" + s + "}"
                return s

            lines.append(f"{d} & {m} & {bf('acc')} & {bf('f1')} & {bf('auc')} & "
                         f"{bf('ece')} & {bf('rule_f1')} & {bf('logic_fid')} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table}"]
    (FIG / "table_main.tex").write_text("\n".join(lines))

    # data efficiency
    le = [r"\begin{table}[t]\centering",
          r"\caption{Minority-class F1 vs. training-set size.}",
          r"\label{tab:eff}", r"\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{ll" + "c" * 5 + r"}", r"\toprule",
          r"Domain & Method & 150 & 300 & 600 & 1200 & 2400 \\", r"\midrule"]
    piv = eff.pivot_table(index=["domain", "method"], columns="n_train", values="f1")
    for (d, m), row in piv.iterrows():
        cells = " & ".join("n/a" if pd.isna(row.get(c)) else f"{row.get(c):.3f}"
                           for c in [150, 300, 600, 1200, 2400])
        le.append(f"{d} & {m} & {cells} \\\\")
    le += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (FIG / "table_efficiency.tex").write_text("\n".join(le))

    # shift
    aggs = shift.groupby(["domain", "shift", "method"]).mean(numeric_only=True).reset_index()
    ls = [r"\begin{table}[t]\centering",
          r"\caption{Accuracy under distribution shift (threshold fixed on the clean distribution).}",
          r"\label{tab:shift}",
          r"\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{ll" + "c" * len(order) + r"}", r"\toprule",
          r"Domain & Shift & " + " & ".join(order) + r" \\", r"\midrule"]
    for d in sorted(aggs.domain.unique()):
        for s in ["covariate", "novel"]:
            sub = aggs[(aggs.domain == d) & (aggs.shift == s)].set_index("method")
            cells = " & ".join("n/a" if m not in sub.index else f"{sub.loc[m,'acc']:.3f}"
                               for m in order)
            ls.append(f"{d} & {s} & {cells} \\\\")
    ls += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (FIG / "table_shift.tex").write_text("\n".join(ls))

    # ablation
    agga = abl.groupby(["domain", "variant"]).mean(numeric_only=True).reset_index()
    variants = ["NEXS(full)", "NEXS(no rule layer)", "NEXS(rule layer only)",
                "NEXS(no semantic loss)", "NEXS(no expert prior)"]
    la = [r"\begin{table}[t]\centering",
          r"\caption{Ablation of the proposed model (mean over seeds).}",
          r"\label{tab:abl}", r"\begin{tabular}{llccc}", r"\toprule",
          r"Domain & Variant & Acc & F1 & LogicFid \\", r"\midrule"]
    for d in sorted(agga.domain.unique()):
        sub = agga[agga.domain == d].set_index("variant")
        for v in variants:
            if v not in sub.index:
                continue
            r = sub.loc[v]
            la.append(f"{d} & {v} & {r['acc']:.3f} & {r['f1']:.3f} & {r['logic_fid']:.3f} \\\\")
        la.append(r"\midrule")
    la[-1] = r"\bottomrule"
    la += [r"\end{tabular}", r"\end{table}"]
    (FIG / "table_ablation.tex").write_text("\n".join(la))

    ll = [r"\begin{table}[t]\centering", r"\caption{Inference latency.}", r"\label{tab:lat}",
          r"\begin{tabular}{llc}", r"\toprule", r"Domain & Method & µs/sample \\", r"\midrule"]
    aggl = lat.groupby(["domain", "method"]).mean(numeric_only=True).reset_index()
    for _, r in aggl.iterrows():
        ll.append(f"{r['domain']} & {r['method']} & {r['us_per_sample']:.2f} \\\\")
    ll += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (FIG / "table_latency.tex").write_text("\n".join(ll))


def main():
    main_df = pd.read_csv(RES / "e1_main.csv")
    eff = pd.read_csv(RES / "e4_data_efficiency.csv")
    shift = pd.read_csv(RES / "e5_shift.csv")
    abl = pd.read_csv(RES / "e6_ablation.csv")
    lat = pd.read_csv(RES / "e7_latency.csv")
    fig_main(main_df)
    fig_rules(main_df)
    fig_efficiency(eff)
    fig_shift(shift)
    fig_ablation(abl)
    fig_tradeoff(main_df, lat)
    latex_tables(main_df, eff, shift, abl, lat)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
