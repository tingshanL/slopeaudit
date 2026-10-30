#!/usr/bin/env python3
"""Supplement S3 tables for the downstream simulation. No fitting.

Reads the summary files written by downstream_simulation.py and
writes LaTeX table bodies (rows only; the table environments live in the supplement):
  S3_settings_rows.tex  construction settings at the common schedule (closed-form check)
  S3_c0_rows.tex        c = 0 controls: rejection rates and EB-omit SE ratio
  S3_known_rows.tex     known-parameter EB analyses at c = .5
  S3_paired_rows.tex    paired estimated-minus-known contrasts (EB omit, EB adjusted), c = 0 and .5
  S3_dose_rows.tex      construction A dose-response and N = 600 follow-up
  S3_boundary_rows.tex  construction B, EB-omit by boundary status of the selected fit
  S3_fit_rows.tex       fit-selection summary per run
Usage:
  python3 reproduction/pm2026/s3_downstream/make_S3_tables.py --out-dir <dir> --run common_80=<dir> --run common_50=<dir> \
      --run irregular_80=<dir> --run irregular_50=<dir> --dose <dir> --n600 <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONS = ["A", "M", "B"]
SHARE = {"A": 1.0, "M": 0.625, "B": 0.25}
LABEL = {"common_80": "Common, .80", "common_50": "Common, .50",
         "irregular_80": "Irregular, .80", "irregular_50": "Irregular, .50"}


def f(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    t = f"{x:.{d}f}"
    if float(t) == 0:
        t = t.lstrip("-")
    neg = t.startswith("-")
    t = t.lstrip("-")
    if t.startswith("0."):
        t = t[1:]
    return ("$-$" if neg else "") + t


def one(df, what):
    if len(df) != 1:
        raise SystemExit(f"expected exactly one row for {what}, found {len(df)}")
    return df.iloc[0]


def settings_rows():
    """Reproduce the construction settings (G00 = 100, tau2 = 1, t = 0..3 centered at 1.5)."""
    t = np.arange(4.0) - 1.5; n = 4; sst = float(t @ t); g00, tau2 = 100.0, 1.0
    rows = []
    for d2 in (.80, .50):
        for c in CONS:
            r = d2 * SHARE[c]; delta = d2 - r
            J = r / ((1 - r) * tau2); s2 = sst / J
            g0s = np.sqrt(delta * tau2 * (g00 + s2 / n) / ((1 - r) ** 2 + delta * r))
            G = np.array([[g00, g0s], [g0s, tau2]]); Z = np.column_stack([np.ones(n), t])
            V = Z @ G @ Z.T + s2 * np.eye(n); w = np.linalg.solve(V, Z @ G[:, 1])
            D2 = w @ Z @ G[:, 1] / tau2; S = g00 - g0s ** 2 / tau2
            kappa = (w.sum() ** 2) * S / (tau2 * D2)
            assert abs(D2 - d2) < 1e-9
            rows.append(f"{f(d2, 2)} & {c} & {s2:.2f} & {g0s:.3f} & {f(g0s / np.sqrt(g00 * tau2))} & {S:.2f} & "
                        f"{f(r)} & {f(D2 - r)} & {f(kappa)} & {f(np.sqrt(kappa))}\\\\")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--dose", type=Path, required=True)
    ap.add_argument("--n600", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    runs = dict(x.split("=", 1) for x in a.run)
    S, P, B, Q = {}, {}, {}, {}
    expect = {"common_80": ("common", .8), "common_50": ("common", .5),
              "irregular_80": ("irregular", .8), "irregular_50": ("irregular", .5)}
    for k, d in runs.items():
        d = Path(d)
        man = json.load(open(d / "manifest.json"))
        st, d2 = expect[k]
        if int(man["n"]) != 200 or abs(float(man["target_d2"]) - d2) > 1e-9 or man.get("stage", st) != st:
            raise SystemExit(f"{d}: manifest does not match label {k}")
        S[k] = pd.read_csv(d / "summary_by_method.csv")
        P[k] = pd.read_csv(d / "summary_paired_contrasts.csv")
        B[k] = pd.read_csv(d / "summary_EB_omit_by_boundary.csv")
        Q[k] = pd.read_csv(d / "summary_fit_quality.csv")
    out = {"S3_settings_rows.tex": settings_rows()}

    def m(k, cons, c, meth):
        g = S[k]
        return one(g[(g.construction == cons) & np.isclose(g.c, c) & (g.method == meth)], (k, cons, c, meth))

    rows = []
    for k in LABEL:
        for cons in CONS:
            e = m(k, cons, 0, "est_EB_omit")
            rows.append(" & ".join([LABEL[k] if cons == "A" else "", cons, f(e.reject_zero_05), f(e.se_ratio, 2),
                                    f(m(k, cons, 0, "est_EB_adjusted").reject_zero_05),
                                    f(m(k, cons, 0, "est_one_stage").reject_zero_05),
                                    f(m(k, cons, 0, "est_two_stage_unshrunk").reject_zero_05)]) + "\\\\")
    out["S3_c0_rows.tex"] = rows

    rows = []
    for k in LABEL:
        for cons in CONS:
            o, ad = m(k, cons, .5, "known_EB_omit"), m(k, cons, .5, "known_EB_adjusted")
            rows.append(" & ".join([LABEL[k] if cons == "A" else "", cons, f(o.workflow_target_beta), f(o.mean_beta),
                                    f(o.cover_workflow_target), f(o.reject_zero_05), f(o.se_ratio, 2),
                                    f(ad.mean_beta), f(ad.reject_zero_05), f(ad.se_ratio, 2)]) + "\\\\")
    out["S3_known_rows.tex"] = rows

    rows = []
    for k in LABEL:
        for cons in CONS:
            cells = []
            for c in (0.0, 0.5):
                for con in ("estimation_effect_EB_omit", "estimation_effect_EB_adjusted"):
                    g = P[k]; r = one(g[(g.construction == cons) & np.isclose(g.c, c) & (g.contrast == con)], (k, cons, c, con))
                    cells.append(f"{f(r.mean_difference, 4)} ({f(r.mc_se, 4)})")
            rows.append(" & ".join([LABEL[k] if cons == "A" else "", cons] + cells) + "\\\\")
    out["S3_paired_rows.tex"] = rows

    rows = []
    for path, nexp in ((a.dose, 200), (a.n600, 600)):
        man = json.load(open(path / "manifest.json"))
        if int(man["n"]) != nexp or abs(float(man["target_d2"]) - .5) > 1e-9 or man.get("stage", "common") != "common":
            raise SystemExit(f"{path}: manifest does not match N={nexp}, target D2=.50, common schedule")
        nlab = str(int(man["n"]))
        g = pd.read_csv(path / "summary_by_method.csv"); g = g[g.construction == "A"]
        for c in sorted(g.c.unique()):
            def r(meth):
                return one(g[np.isclose(g.c, c) & (g.method == meth)], (str(path), c, meth))
            ko, eo = r("known_EB_omit"), r("est_EB_omit")
            rows.append(" & ".join([nlab, f(c, 2), f(ko.reject_zero_05), f(eo.reject_zero_05), f(eo.se_ratio, 2),
                                    f(eo.cover_workflow_target), f(r("est_one_stage").reject_zero_05),
                                    f(r("est_two_stage_unshrunk").reject_zero_05), str(int(eo.usable))]) + "\\\\")
    out["S3_dose_rows.tex"] = rows

    rows = []
    for k in LABEL:
        g = B[k]; g = g[g.construction == "B"]
        for c in (0.0, 0.5):
            for bd in (False, True):
                r = g[np.isclose(g.c, c) & (g.EB_omit_best_boundary == bd)]
                if r.empty:
                    continue
                r = one(r, (k, c, bd))
                rows.append(" & ".join([LABEL[k], f(c, 1), "Boundary" if bd else "Interior", str(int(r.n)),
                                        f(r.mean_rho), f(r.plugin_kappa), f(r.plugin_D2), f(r.mean_beta),
                                        f(r.mean_known_beta), f(r.reject)]) + "\\\\")
    out["S3_boundary_rows.tex"] = rows

    rows = []
    for k in LABEL:
        g = Q[k]
        tot = int(g.n.sum()); corr = int((g.agreed + g.confirmed_on_restart).sum())
        rows.append(" & ".join([LABEL[k], str(tot), str(corr), str(int(g.confirmed_on_restart.sum())),
                                str(int(g.improved_by_restart.sum())),
                                str(int(g.unresolved.sum())), f"{g.boundary_rate.min():.3f}--{g.boundary_rate.max():.3f}".replace("0.", "."),
                                str(int(g.frozen_rule_below_best_gt_001.sum()))]) + "\\\\")
    out["S3_fit_rows.tex"] = rows

    for name, rws in out.items():
        (a.out_dir / name).write_text("\n".join(rws) + "\n")
    print(json.dumps({k: len(v) for k, v in out.items()}))


if __name__ == "__main__":
    main()
