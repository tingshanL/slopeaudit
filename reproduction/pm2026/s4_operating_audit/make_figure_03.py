#!/usr/bin/env python3
"""Figure 3: fitted determinacy and heterogeneity testing when the generating slope variance is zero.

Reads the repaired operating-study replicate table (summarizer output after the consolidated
likelihood refit) and draws, for the four true-zero cells,
  A  distributions of participant-mean fitted D2_slope and R_direct (strict-valid fits), with the
     rank-one limits q0*n/(1+q0*n) (.600, .923) marked for the balanced three-wave cells only;
  B  heterogeneity-test false-positive rates on the evaluation halves (Wilson 95% intervals).
No fitting. Usage:
  python3 make_figure_03.py --operating-csv <summary/operating_study_replicate_results.csv> --out-dir <dir>
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

COL = {"total": "#6f6e69", "direct": "#2a78d6", "ink": "#0b0b0b", "ink2": "#52514e",
       "grid": "#e4e3df", "ref": "#eb6834"}
FS = 9
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.edgecolor": COL["ink2"],
                     "axes.linewidth": 0.6, "xtick.color": COL["ink2"], "ytick.color": COL["ink2"],
                     "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1, "savefig.dpi": 300})
CELLS = [("true_zero__balanced_three_wave__q0_0.5", "Balanced\n$q_0$ = .5", .5, 3),
         ("true_zero__balanced_three_wave__q0_4", "Balanced\n$q_0$ = 4", 4.0, 3),
         ("true_zero__empirical__fluid_cognition__q0_0.5", "AABC2\n$q_0$ = .5", .5, None),
         ("true_zero__empirical__fluid_cognition__q0_4", "AABC2\n$q_0$ = 4", 4.0, None)]
EXPECTED_KN = [(47, 999), (48, 997), (41, 998), (45, 999)]
EXPECTED_VALID = [1997, 1994, 1998, 1996]


def as_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def wilson(k, n, z=norm.ppf(.975)):
    p = k / n; d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def apa(x, d=2):
    t = f"{x:.{d}f}"
    return t[1:] if t.startswith("0.") else t


def violin(ax, vals, pos, color, width=0.36):
    v = ax.violinplot([np.clip(vals, 0, 1)], positions=[pos], widths=width, showextrema=False,
                      points=200, bw_method=0.15)
    for b in v["bodies"]:
        b.set_facecolor(color); b.set_edgecolor(color); b.set_alpha(0.45); b.set_linewidth(0.8)
    q1, med, q3 = np.quantile(vals, [.25, .5, .75])
    ax.plot([pos, pos], [q1, q3], color=color, lw=3.2, solid_capstyle="butt", zorder=4)
    ax.scatter([pos], [med], s=22, color="white", edgecolor=color, linewidth=1.2, zorder=5)
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operating-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.operating_csv, low_memory=False,
                    usecols=["cell_id", "strict_fit_valid", "mean_Pfull", "mean_Rdirect",
                             "LR_evaluation_replicate", "slope_heterogeneity_supported"])
    d = d[d.cell_id.str.startswith("true_zero")]
    d["valid"] = as_bool(d.strict_fit_valid)
    d["is_eval"] = as_bool(d.LR_evaluation_replicate)
    d["supp"] = as_bool(d.slope_heterogeneity_supported)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.5, 3.1), gridspec_kw={"width_ratios": [2.6, 1], "wspace": 0.32})
    rows = []
    for i, (cid, lab, q0, n) in enumerate(CELLS):
        g = d[(d.cell_id == cid) & d.valid]
        assert len(g) == EXPECTED_VALID[i], (cid, len(g))
        m1 = violin(ax, g.mean_Pfull.to_numpy(float), i - 0.2, COL["total"])
        m2 = violin(ax, g.mean_Rdirect.to_numpy(float), i + 0.2, COL["direct"])
        if n is not None:
            lim = q0 * n / (1 + q0 * n)
            ax.plot([i - 0.42, i + 0.02], [lim, lim], color=COL["ref"], lw=1.1, ls="--", zorder=6)
            ax.text(i + 0.05, lim, apa(lim, 3), ha="left", va="center", fontsize=FS - 1.5, color=COL["ref"])
        e = d[(d.cell_id == cid) & d.valid & d.is_eval]
        k, nn = int(e.supp.sum()), len(e)
        assert (k, nn) == EXPECTED_KN[i], (cid, k, nn)
        rows.append(dict(cell_id=cid, valid=len(g), D2_median=m1, Rdirect_median=m2, fp_k=k, fp_n=nn))
    ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in CELLS], fontsize=FS - 1)
    ax.set_xlim(-0.6, 3.5); ax.set_ylim(0, 1)
    ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.set_yticklabels(["0", ".2", ".4", ".6", ".8", "1"])
    ax.set_ylabel("Participant-mean fitted value", fontsize=FS)
    ax.set_title("A  Fitted determinacy and direct reliability", loc="left", fontsize=FS + 0.5, fontweight="bold")
    h = [mpl.patches.Patch(color=COL["total"], alpha=0.6, label="$D^2_{\\mathrm{slope}}$ (total)"),
         mpl.patches.Patch(color=COL["direct"], alpha=0.6, label="$R_{\\mathrm{direct}}$"),
         mpl.lines.Line2D([], [], color=COL["ref"], ls="--", lw=1.1, label="Rank-one limit (balanced)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False,
              fontsize=FS - 1, handlelength=1.4, columnspacing=1.2)

    for i, r in enumerate(rows):
        lo, hi = wilson(r["fp_k"], r["fp_n"])
        y = 3 - i
        axb.plot([lo, hi], [y, y], color=COL["ink2"], lw=1.2)
        axb.scatter([r["fp_k"] / r["fp_n"]], [y], s=26, color=COL["ink"], zorder=3)
        axb.text(0.105, y, f"{r['fp_k']}/{r['fp_n']}", va="center", ha="right", fontsize=FS - 1.5, color=COL["ink2"])
    axb.axvline(0.05, color=COL["ink2"], ls="--", lw=0.7)
    axb.set_yticks([3, 2, 1, 0]); axb.set_yticklabels([c[1].replace("\n", " ") for c in CELLS], fontsize=FS - 1.5)
    axb.set_xlim(0, 0.105); axb.set_xticks([0, .05, .10]); axb.set_xticklabels(["0", ".05", ".10"])
    axb.set_ylim(-0.6, 3.6)
    axb.set_xlabel("False-positive rate", fontsize=FS)
    axb.set_title("B  Heterogeneity test", loc="left", fontsize=FS + 0.5, fontweight="bold")
    for axx in (ax, axb):
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=COL["grid"], lw=0.5); axb.xaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True); axb.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(a.out_dir / f"figure_03_truezero.{ext}", bbox_inches="tight", facecolor="white")
    pd.DataFrame(rows).to_csv(a.out_dir / "figure_03_values.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
