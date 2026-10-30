#!/usr/bin/env python3
"""Figure 4: empirical source decomposition and nuisance capacity.

Inputs are aggregate files only:
  s7_sleepstudy_summary.csv (from s7_sleepstudy_kappa.R)
  s7_aabc2_decomposition.csv, s7_aabc2_kappa_quantiles.csv (from s7_aabc2_aggregate.py)
Panel A: equal-participant means of R_direct and Delta_fit, stacked to mean D2.
Panel B: participant-level sqrt(kappa_i) within each fitted model (5/25/50/75/95th percentiles);
         sleepstudy shares one schedule, so it is a single value.
Usage: python3 make_figure_04.py --sleep <dir> --aabc2 <dir> --out-dir <dir>
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

COL = {"direct": "#2a78d6", "assist": "#eb6834", "box": "#6f6e69", "ink": "#0b0b0b",
       "ink2": "#52514e", "grid": "#e4e3df"}
FS = 9
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.edgecolor": COL["ink2"],
                     "axes.linewidth": 0.6, "xtick.color": COL["ink2"], "ytick.color": COL["ink2"],
                     "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 0.5, "savefig.dpi": 300})


def apa(x, d=2):
    t = f"{x:.{d}f}"
    return t[1:] if t.startswith("0.") else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=Path, required=True)
    ap.add_argument("--aabc2", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    sl = pd.read_csv(a.sleep / "s7_sleepstudy_summary.csv").iloc[0]
    dec = pd.read_csv(a.aabc2 / "s7_aabc2_decomposition.csv").set_index("outcome")
    kq = pd.read_csv(a.aabc2 / "s7_aabc2_kappa_quantiles.csv").set_index("outcome")
    chk = pd.read_csv(a.aabc2 / "s7_aabc2_checks.csv").set_index("outcome")
    assert chk.homoscedastic_RIRS_closed_form_valid.all()
    assert bool(sl.identical_across_subjects) and sl.max_abs_closed_form_error < 1e-10

    rows = [("sleepstudy", "sleepstudy\nreaction time", sl.mean_Rdirect, sl.mean_Delta_fit, int(sl.N)),
            ("lateral_ventricles", "AABC2\nventricles", dec.loc["lateral_ventricles", "mean_Rdirect"],
             dec.loc["lateral_ventricles", "mean_Delta_fit"], int(dec.loc["lateral_ventricles", "N"])),
            ("fluid_cognition", "AABC2\nfluid cognition", dec.loc["fluid_cognition", "mean_Rdirect"],
             dec.loc["fluid_cognition", "mean_Delta_fit"], int(dec.loc["fluid_cognition", "N"]))]

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={"wspace": 0.12, "width_ratios": [1.15, 1]})
    for i, (_, lab, r, dfit, n) in enumerate(rows):
        y = 2 - i
        ax.barh(y, r, height=0.5, color=COL["direct"], edgecolor="white", lw=1.5)
        ax.barh(y, dfit, left=r, height=0.5, color=COL["assist"], edgecolor="white", lw=1.5)
        if r > .12:
            ax.text(r / 2, y, apa(r), ha="center", va="center", color="white", fontsize=FS - 0.5)
        else:
            ax.text(r / 2, y + .34, apa(r), ha="center", va="center", color=COL["ink"], fontsize=FS - 0.5)
        if dfit > .08:
            ax.text(r + dfit / 2, y, apa(dfit), ha="center", va="center", color="white", fontsize=FS - 0.5)
        ax.text(r + dfit + .02, y, f"$D^2$ = {apa(r + dfit)}", va="center", fontsize=FS - 0.5, color=COL["ink"])
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([r[1] for r in rows])
    ax.set_xlim(0, 1.12); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_xlabel("Equal-participant mean", fontsize=FS)
    ax.set_title("A  Sources of determinacy", loc="left", fontsize=FS + 0.5, fontweight="bold")
    ax.legend(handles=[mpl.patches.Patch(color=COL["direct"], label="$R_{\\mathrm{direct}}$"),
                       mpl.patches.Patch(color=COL["assist"], label="$\\Delta_{\\mathrm{fit}}$")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=FS - 0.5)

    for i, (key, lab, *_rest) in enumerate(rows):
        y = 2 - i
        if key == "sleepstudy":
            axb.scatter([sl.mean_sqrt_kappa], [y], s=34, color=COL["ink"], zorder=3)
            axb.text(sl.mean_sqrt_kappa + .03, y, f"{apa(sl.mean_sqrt_kappa)} (all participants)", va="center",
                     fontsize=FS - 1.5, color=COL["ink2"])
            continue
        q = kq.loc[key]
        axb.plot([q.sqrt_kappa_q05, q.sqrt_kappa_q95], [y, y], color=COL["box"], lw=1.0)
        axb.add_patch(mpl.patches.Rectangle((q.sqrt_kappa_q25, y - .17), q.sqrt_kappa_q75 - q.sqrt_kappa_q25, .34,
                                            facecolor=COL["box"], alpha=.45, edgecolor=COL["box"], lw=.8))
        axb.plot([q.sqrt_kappa_q50] * 2, [y - .17, y + .17], color=COL["ink"], lw=1.6)
    axb.set_yticks([2, 1, 0]); axb.set_yticklabels([])
    axb.set_xlim(0, 1); axb.set_xticks([0, .25, .5, .75, 1]); axb.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    axb.set_ylim(-0.6, 2.6); ax.set_ylim(-0.6, 2.6)
    axb.set_xlabel("$\\sqrt{\\kappa_i}$ across participants", fontsize=FS)
    axb.set_title("B  Nuisance capacity", loc="left", fontsize=FS + 0.5, fontweight="bold")
    for axx in (ax, axb):
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)
        axx.tick_params(axis="y", length=0)
        axx.xaxis.grid(True, color=COL["grid"], lw=.5); axx.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(a.out_dir / f"figure_04_empirical.{ext}", bbox_inches="tight", facecolor="white")
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
