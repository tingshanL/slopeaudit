#!/usr/bin/env python3
"""Supplement S7: design sensitivity under the random-intercept, random-slope model with
independent equal-variance residuals. No data; planning calculations only.

1. Checks the three-quantity reduction (lambda, m, rho_bar) against the matrix computation.
2. Writes S7_schedules_rows.tex: planned schedules under the fitted AABC2 cognition and
   ventricle models (aggregate fitted parameters from s7_aabc2_decomposition.csv).
3. Writes figure_S7_1.pdf: Delta_fit and sqrt(kappa) against lambda by rho_bar and m.
Usage: python3 make_S7.py --aabc2 <s7_aabc2_aggregate dir> --out-dir <dir>
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from s7_core import matrix_quantities, planning_quantities, from_planning

SCHEDULES = [("2 visits: 0, 5", [0, 5]), ("3 visits: 0, 2.5, 5", [0, 2.5, 5]),
             ("4 visits: 0, 1.67, 3.33, 5", [0, 5/3, 10/3, 5]), ("6 visits: annual, 0--5", [0, 1, 2, 3, 4, 5]),
             ("3 visits: 0, 0.5, 5", [0, .5, 5]), ("3 visits: 0, 4.5, 5", [0, 4.5, 5]),
             ("3 visits: 0, 1.25, 2.5", [0, 1.25, 2.5]), ("3 visits: 0, 5, 10$^a$", [0, 5, 10])]
LABEL = {"fluid_cognition": "Fluid cognition", "lateral_ventricles": "Lateral ventricles"}


def f(x, d=3):
    t = f"{x:.{d}f}"
    if float(t) == 0: t = t.lstrip("-")
    neg = t.startswith("-"); t = t.lstrip("-")
    t = t[1:] if t.startswith("0.") else t
    return ("$-$" if neg else "") + t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--aabc2", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True); a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rec = {}
    rng = np.random.default_rng(5); err = 0.0; nonfinite = 0; checked = 0
    for _ in range(20000):
        n = rng.integers(2, 8); t = np.sort(rng.uniform(-3, 3, n)); g00 = rng.uniform(.05, 5); tau2 = rng.uniform(.05, 3)
        r = rng.uniform(-.99, .99); c = r * np.sqrt(g00 * tau2); G = np.array([[g00, c], [c, tau2]]); s2 = rng.uniform(.1, 4)
        x = np.array(matrix_quantities(t, G, s2)); y = np.array(from_planning(*planning_quantities(t, G, s2)))
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            nonfinite += 1            # counted, never silently skipped by max()
            continue
        err = max(err, float(np.abs(x - y).max())); checked += 1
    rec.update(reduction_checked=checked, reduction_nonfinite=nonfinite, reduction_max_abs_error=err, tolerance=1e-10)
    if nonfinite or checked == 0 or not np.isfinite(err) or err > 1e-10:
        raise SystemExit(f"S7 reduction check FAILED: {rec}")

    dec = pd.read_csv(a.aabc2 / "s7_aabc2_decomposition.csv").set_index("outcome")
    rows = []
    for key in ("fluid_cognition", "lateral_ventricles"):
        d = dec.loc[key]; G = np.array([[d.G00, d.cov01], [d.cov01, d.tau2]]); s2 = d.sigma2
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{LABEL[key]}}}}}\\\\")
        for lab, u in SCHEDULES:
            t = (np.array(u, float) - 2.5) / 5          # five-year units centered at 2.5 years
            lam, m, rho = planning_quantities(t, G, s2); D2, R, Df, k = matrix_quantities(t, G, s2)
            assert abs(D2 - (R + Df)) < 1e-12
            rows.append(" & ".join([lab, f"{lam:.3f}", f"{m:.1f}", f(rho), f(D2), f(R), f(Df), f(np.sqrt(max(k, 0)))]) + "\\\\")
    (a.out_dir / "S7_schedules_rows.tex").write_text("\n".join(rows) + "\n")

    COL = ["#2a78d6", "#1f9e89", "#eb6834", "#7a3e9d"]
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": .6})
    lam = np.logspace(-2, 1.3, 300); ms = [1, 5, 20]; rhos = [0, .3, .6, .9]
    fig, ax = plt.subplots(2, 3, figsize=(6.5, 4.0), sharex=True, sharey="row")
    for j, m in enumerate(ms):
        for col, r in zip(COL, rhos):
            v = np.array([from_planning(L, m, r) for L in lam])
            ax[0, j].plot(lam, v[:, 2], color=col, lw=1.2, label=f"$\\bar\\rho$ = {r:g}".replace("0.", "."))
            ax[1, j].plot(lam, np.sqrt(np.clip(v[:, 3], 0, None)), color=col, lw=1.2)
        ax[0, j].set_title(f"$m$ = {m}", fontsize=8.5)
        for i in range(2):
            ax[i, j].set_xscale("log"); ax[i, j].grid(color="#e4e3df", lw=.5); ax[i, j].set_ylim(0, 1)
        ax[1, j].set_xlabel("$\\lambda$ (direct slope information)")
    ax[0, 0].set_ylabel("$\\Delta_{\\mathrm{fit}}$"); ax[1, 0].set_ylabel("$\\sqrt{\\kappa}$")
    for j in range(3):
        ax[0, j].plot(lam, lam / (1 + lam), color="#52514e", lw=.9, ls="--", label="$R_{\\mathrm{direct}}$ (all $m$, $\\bar\\rho$)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(a.out_dir / "figure_S7_1.pdf"); fig.savefig(a.out_dir / "figure_S7_1.png", dpi=200)
    json.dump(rec, open(a.out_dir / "S7_checks.json", "w"), indent=2)
    print(json.dumps(rec), len(rows))


if __name__ == "__main__":
    main()
