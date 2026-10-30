"""Summarise results into a human-readable markdown report with exact numbers.

Used to keep the paper's reported values in sync with the artefacts and to make
every claim traceable to a CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "RESULTS.md"


def _fmt(df: pd.DataFrame, cols, index=None) -> str:
    d = df.copy()
    for c in cols:
        if c in d:
            d[c] = d[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
    return d.to_markdown(index=index)


def main():
    main_df = pd.read_csv(RES / "e1_main.csv")
    agg = main_df.groupby(["domain", "method"]).mean(numeric_only=True).reset_index()
    lines = ["# Results summary", ""]
    lines.append("## E1/E2/E3 main (mean over seeds)\n")
    lines.append(_fmt(agg[["domain", "method", "acc", "f1", "auc", "ece",
                           "rule_f1", "rule_rec", "logic_fid"]], 
                      ["acc", "f1", "auc", "ece", "rule_f1", "rule_rec", "logic_fid"]))
    lines.append("")

    lines.append("## Key deltas (NEXS vs baselines), mean over domains\n")
    piv = agg.pivot_table(index="method", values=["acc", "f1", "auc", "ece",
                                                  "rule_f1", "logic_fid"])
    for base in ["MLP", "SemanticLoss", "RandomForest", "GradientBoosting"]:
        if "NEXS(proposed)" in piv.index and base in piv.index:
            d = (piv.loc["NEXS(proposed)"] - piv.loc[base])
            lines.append(f"- **NEXS - {base}**: "
                         + ", ".join(f"{k}={v:+.3f}" for k, v in d.items()))
    lines.append("")

    for name, key, col in [("E4 data efficiency", "e4_data_efficiency.csv", "f1"),
                           ("E5 shift", "e5_shift.csv", "acc"),
                           ("E6 ablation", "e6_ablation.csv", "f1"),
                           ("E7 latency", "e7_latency.csv", "us_per_sample")]:
        p = RES / key
        if not p.exists():
            continue
        df = pd.read_csv(p)
        lines.append(f"## {name}\n")
        if key.startswith("e4"):
            piv2 = df.pivot_table(index=["domain", "method"], columns="n_train", values=col)
            lines.append(_fmt(piv2.reset_index(), list(piv2.columns)))
        elif key.startswith("e5"):
            piv2 = df.pivot_table(index=["domain", "shift"], columns="method", values=col)
            lines.append(_fmt(piv2.reset_index(), list(piv2.columns)))
        elif key.startswith("e6"):
            a = df.groupby(["domain", "variant"]).mean(numeric_only=True).reset_index()
            lines.append(_fmt(a[["domain", "variant", "acc", "f1", "logic_fid"]],
                              ["acc", "f1", "logic_fid"]))
        else:
            a = df.groupby(["domain", "method"]).mean(numeric_only=True).reset_index()
            lines.append(_fmt(a[["domain", "method", col]], [col]))
        lines.append("")

    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
