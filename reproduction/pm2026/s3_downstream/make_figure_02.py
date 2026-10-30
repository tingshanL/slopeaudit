#!/usr/bin/env python3
"""Downstream simulation (Supplement S3): main-text Table 1 and Figure 2. No fitting.

Reads the run folders written by downstream_simulation.py and writes
  s5_results_long.csv          every construction x c x method row from every run
  s5_table_main.csv / _rows.tex  main-text Table 1 (c = .5 cells)
  s5_boundary_table.csv        EB-omit results by boundary status of the selected fit
  figure_02_downstream.pdf/png Panel A: population prediction against the observed
                               known-parameter coefficient gap; Panel B: construction A,
                               common schedule, rejection of zero by covariate alignment c
                               (500 replications), with the N = 600 run.
(The s5_ prefix of the output names is kept from the original runs.)

Usage (from the repository root):
  python3 reproduction/pm2026/s3_downstream/make_figure_02.py --out-dir <dir> \
     --run common_80=<dir> --run common_50=<dir> --run irregular_80=<dir> --run irregular_50=<dir> \
     --dose <construction-A dose-response dir> --n600 <construction-A N=600 dir>
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"known": "#2a78d6", "est": "#eb6834", "one": "#1baf7a",
       "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
FS = 9
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.edgecolor": COL["ink2"],
                     "axes.linewidth": 0.6, "xtick.color": COL["ink2"], "ytick.color": COL["ink2"],
                     "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1, "savefig.dpi": 300})


def apa(x, d=3):
    if pd.isna(x):
        return ""
    t = f"{x:.{d}f}"
    if float(t) == 0:
        t = t.lstrip("-")
    t = t.replace("0.", ".", 1) if t.startswith(("0.", "-0.")) else t
    return t


def load_run(label, folder):
    folder = Path(folder)
    man = json.load(open(folder / "manifest.json"))
    s = pd.read_csv(folder / "summary_by_method.csv")
    pc = pd.read_csv(folder / "summary_paired_contrasts.csv")
    schedule, d2 = label.split("_")
    meta = dict(run=label, schedule=schedule, target_D2=man["target_d2"], N=man["n"], reps=man["reps"])
    return s.assign(**meta), pc.assign(**meta), folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=folder; label in {common,irregular}_{80,50}")
    ap.add_argument("--dose", type=Path, required=True)
    ap.add_argument("--n600", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    sums, pairs, bnd = [], [], []
    for item in a.run:
        label, folder = item.split("=", 1)
        s, pc, f = load_run(label, folder)
        sums.append(s); pairs.append(pc)
        b = pd.read_csv(f / "summary_EB_omit_by_boundary.csv")
        bnd.append(b.assign(run=label))
    S = pd.concat(sums, ignore_index=True)
    P = pd.concat(pairs, ignore_index=True)
    S.to_csv(a.out_dir / "s5_results_long.csv", index=False)
    pd.concat(bnd).to_csv(a.out_dir / "s5_boundary_table.csv", index=False)

    # ---- main table: c = .5
    rows = []
    order = [("common", .8), ("common", .5), ("irregular", .8), ("irregular", .5)]
    for sched, d2 in order:
        for cons in ["A", "M", "B"]:
            g = S[(S.schedule == sched) & (S.target_D2 == d2) & (S.construction == cons) & np.isclose(S.c, .5)]
            if g.empty:
                continue
            m = g.set_index("method")
            gap = P[(P.schedule == sched) & (P.target_D2 == d2) & (P.construction == cons) & np.isclose(P.c, .5)
                    & (P.contrast == "oracle_workflow_gap")].iloc[0]
            e = m.loc["est_EB_omit"]
            rows.append({
                "schedule": sched, "target_D2": d2, "construction": cons,
                "pred_beta_EB": e.workflow_target_beta,
                "known_gap": gap.mean_difference, "known_gap_mcse": gap.mc_se,
                "est_EB_mean_beta": e.mean_beta, "est_EB_se_ratio": e.se_ratio,
                "est_EB_cover_target": e.cover_workflow_target, "est_EB_reject_zero": e.reject_zero_05,
                "one_stage_reject": m.loc["est_one_stage"].reject_zero_05,
                "two_stage_reject": m.loc["est_two_stage_unshrunk"].reject_zero_05,
                "reps": int(e.usable)})
    T = pd.DataFrame(rows)
    T.to_csv(a.out_dir / "s5_table_main.csv", index=False)
    lines = []
    for (sched, d2), g in T.groupby(["schedule", "target_D2"], sort=False):
        schedule_label = "Common schedule" if sched == "common" else "Irregular schedules"
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{schedule_label}, "
                     f"target $D^2={apa(d2, 2)}$}}}}\\\\")
        for r in g.itertuples():
            lines.append(" & ".join([r.construction, apa(r.pred_beta_EB), f"{apa(r.known_gap)} ({apa(r.known_gap_mcse, 4)})",
                                     apa(r.est_EB_mean_beta), apa(r.est_EB_se_ratio, 2), apa(r.est_EB_cover_target, 3),
                                     apa(r.est_EB_reject_zero), apa(r.one_stage_reject), apa(r.two_stage_reject)]) + "\\\\")
    (a.out_dir / "s5_table_main_rows.tex").write_text("\n".join(lines).replace("& -.", "& $-$.") + "\n")

    # ---- Figure 2
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={"wspace": 0.38})
    ax = axes[0]
    gaps = P[P.contrast == "oracle_workflow_gap"]
    marker = {"A": "o", "M": "s", "B": "^"}
    for r in gaps.itertuples():
        filled = r.schedule == "common"
        ax.errorbar(r.predicted, r.mean_difference, yerr=1.96 * r.mc_se, fmt=marker[r.construction], ms=5.5,
                    mfc=COL["ink2"] if filled else "white", mec=COL["ink2"], ecolor=COL["ink2"], elinewidth=0.8,
                    capsize=0, zorder=3)
    lim = [-0.02, 0.24]
    ax.plot(lim, lim, color=COL["ink2"], lw=0.7, ls="--", zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Predicted population coefficient\n(slope units per SD of x)", fontsize=FS)
    ax.set_ylabel("Observed coefficient gap\n(known parameters)", fontsize=FS)
    ax.set_title("A  Population association", loc="left", fontsize=FS + 0.5, fontweight="bold")
    h = [mpl.lines.Line2D([], [], ls="", marker=marker[k], mfc=COL["ink2"], mec=COL["ink2"], label=f"Design {k}") for k in "AMB"]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=FS - 1.5, handletextpad=0.2, borderaxespad=0.2)
    ax.text(0.23, -0.005, "filled: common schedule\nopen: irregular schedules", ha="right", va="bottom",
            fontsize=FS - 1.5, color=COL["ink2"])
    tk = [0, .05, .10, .15, .20]
    ax.set_xticks(tk); ax.set_yticks(tk)
    ax.set_xticklabels([apa(t, 2) if t else "0" for t in tk]); ax.set_yticklabels([apa(t, 2) if t else "0" for t in tk])

    ax = axes[1]
    dose = pd.read_csv(a.dose / "summary_by_method.csv")
    n6 = pd.read_csv(a.n600 / "summary_by_method.csv")
    series = [("known_EB_omit", "known", "EB omitting x, known parameters", -0.012),
              ("est_EB_omit", "est", "EB omitting x, estimated parameters", 0.0),
              ("est_one_stage", "one", "One-stage model", 0.012)]
    for meth, ck, lab, dx in series:
        g = dose[(dose.construction == "A") & (dose.method == meth)].sort_values("c")
        p, n = g.reject_zero_05.to_numpy(), g.usable.to_numpy()
        se = np.sqrt(p * (1 - p) / n)
        ax.errorbar(g.c + dx, p, yerr=1.96 * se, color=COL[ck], marker="o", ms=4.5, lw=1.6, elinewidth=0.8, label=lab)
    g6 = n6[(n6.construction == "A") & (n6.method == "est_EB_omit")].iloc[0]
    se6 = np.sqrt(g6.reject_zero_05 * (1 - g6.reject_zero_05) / g6.usable)
    ax.errorbar(0.5 + 0.045, g6.reject_zero_05, yerr=1.96 * se6, color=COL["est"], marker="D", mfc="white", ms=5,
                elinewidth=0.8, ls="", label="EB omitting x, estimated, N = 600")
    ax.axhline(0.05, color=COL["ink2"], lw=0.7, ls="--")
    ax.set_xticks([0, .25, .5, .75]); ax.set_xticklabels(["0", ".25", ".50", ".75"])
    ax.set_yticks([0, .1, .2, .3]); ax.set_yticklabels(["0", ".10", ".20", ".30"])
    ax.set_ylim(0, 0.43)
    ax.set_xlabel("Alignment of x with level variation (c)", fontsize=FS)
    ax.set_ylabel("Rejection rate of zero", fontsize=FS)
    ax.set_title("B  Estimation-related miscalibration", loc="left", fontsize=FS + 0.5, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=FS - 1.5, handlelength=1.4, borderaxespad=0.2)
    for axx in axes:
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)
        axx.grid(True, color=COL["grid"], lw=0.5); axx.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(a.out_dir / f"figure_02_downstream.{ext}", bbox_inches="tight", facecolor="white")
    pd.set_option("display.width", 250)
    print(T.round(4).to_string(index=False))
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
