#!/usr/bin/env python3
"""Supplement S4 tables for the operating study (true-zero simulation and its parent 46-cell study).

Inputs (no fitting):
  --summary   repaired summarizer output folder (operating_characteristics.csv,
              likelihood_ratio_calibration.csv, operating_study_replicate_results.csv)
  --step7     s4_step7_reporting folder (t2b_boundary_breakdown.csv, decision_flips_by_cell.csv,
              oc_changes_long.csv)
Writes LaTeX row files:
  S4_calibration_rows.tex  critical values and evaluation false-support rates
  S4_classes_rows.tex      true-zero fits by fitted covariance class (repaired)
  S4_cells_rows.tex        positive-variance cells: validity, Q1 support, recovery of mean D2/R_direct/Delta
  S4_repairs_rows.tex      refits and strict-invalid fits by family
  S4_changes_rows.tex      cells whose Q1 support rate changed by >= .005 after the refits, with flip counts
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FAM_ORDER = ["balanced_core", "critical_near_boundary", "empirical_AABC", "true_zero"]
FAM_LABEL = {"balanced_core": "Balanced core", "critical_near_boundary": "Critical (misspecified)",
             "empirical_AABC": "Empirical AABC2 anchors", "true_zero": "True zero"}
MISSPEC = {"late_visit_residual_variance_doubled": "Late-visit $\\sigma^2\\times2$",
           "residual_AR1_rho_0p4": "AR(1), .4", "mild_curvature_max_0p25_residual_SD": "Mild curvature"}
OUTCOME = {"lateral_ventricles": "Ventricles", "hippocampus": "Hippocampus",
           "global_mean_thickness": "Cortical thickness", "global_cortical_cbf": "Cortical CBF",
           "global_cortical_att": "Cortical ATT", "fluid_cognition": "Fluid cognition"}


def f(x, d=3):
    if x is None or not np.isfinite(x):
        return "--"
    t = f"{x:.{d}f}"
    if float(t) == 0:
        t = t.lstrip("-")
    neg = t.startswith("-"); t = t.lstrip("-")
    t = t[1:] if t.startswith("0.") else t
    return ("$-$" if neg else "") + t


def g(x):
    return f"{x:g}".replace("0.", ".")


def label(r):
    if r.cell_family == "balanced_core":
        return f"Core, $q_1$={g(r.q1)}, $q_0$={g(r.q0)}, $\\rho$={g(r.rho)}"
    if r.cell_family == "critical_near_boundary":
        return f"{MISSPEC[r.misspecification]}, $q_1$={g(r.q1)}, $q_0$={g(r.q0)}"
    if r.cell_family == "empirical_AABC":
        return OUTCOME[r.cell_id.replace("empirical__", "")]
    return r.cell_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--step7", type=Path, required=True)
    ap.add_argument("--design", type=Path, required=True, help="operating_study_design.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    oc = pd.read_csv(a.summary / "operating_characteristics.csv")
    cal = pd.read_csv(a.summary / "likelihood_ratio_calibration.csv")
    des = pd.read_csv(a.design).set_index("cell_id")
    if len(oc) != 46 or oc.cell_id.duplicated().any() or set(oc.cell_id) != set(des.index):
        raise SystemExit("operating characteristics do not match the 46-cell design")
    out = {}

    rows = []
    for r in cal.itertuples():
        lab = ("Balanced" if r.schedule_class == "balanced" else "AABC2") + f", $q_0$={g(r.q0)}"
        rows.append(f"{lab} & {r.calibration_strict_n} & {r.LR_95_critical_value:.3f} & {r.evaluation_strict_n} & "
                    f"{f(r.evaluation_false_support_rate)} ({f(r.evaluation_false_support_MCSE)})\\\\")
    out["S4_calibration_rows.tex"] = rows

    bd = pd.read_csv(a.step7 / "t2b_boundary_breakdown.csv")
    bd = bd[bd.version == "repaired"]
    cls = {"interior": "Interior", "near_rank_one": "Near rank one", "singular_proxy": "Singular proxy"}
    rows, prev = [], None
    for r in bd.itertuples():
        cl = ("Balanced" if "balanced" in r.cell_id else "AABC2") + ", $q_0$=" + ("4" if r.cell_id.endswith("_4") else ".5")
        med = r.D2_median_IQR.replace("0.", ".")
        rows.append(" & ".join([cl if cl != prev else "", cls[r.cov_class], str(r.n), f(r.share_of_cell),
                                f(r.abs_rho_median), med, f(r.D2_ge_040), f(r.Rdirect_median),
                                f"{r.Q1_evaluation_k}/{r.Q1_evaluation_n}"]) + "\\\\")
        prev = cl
    out["S4_classes_rows.tex"] = rows

    rows = []
    pos = oc[oc.cell_family != "true_zero"].copy()
    pos["fam_rank"] = pos.cell_family.map({k: i for i, k in enumerate(FAM_ORDER)})
    pos = pos.sort_values(["fam_rank", "misspecification", "q1", "q0", "rho"])
    prev = None
    for r in pos.itertuples():
        d = des.loc[r.cell_id]
        if r.cell_family != prev:
            rows.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{FAM_LABEL[r.cell_family]}}}}}\\\\")
            prev = r.cell_family
        rows.append(" & ".join([label(r), f(d.oracle_mean_Pfull), f(d.oracle_mean_Rdirect), f(r.strict_validity_rate),
                                f(r.slope_heterogeneity_support_rate), f(r.bias_mean_Pfull), f(r.RMSE_mean_Pfull),
                                f(r.bias_mean_Rdirect), f(r.bias_mean_Delta)]) + "\\\\")
    out["S4_cells_rows.tex"] = rows

    rep = pd.read_csv(a.summary / "operating_study_replicate_results.csv", low_memory=False,
                      usecols=["cell_family", "repair_method", "strict_fit_valid"])
    rows = []
    for fam in FAM_ORDER:
        x = rep[rep.cell_family == fam]
        m = x.repair_method.fillna("none")
        rows.append(" & ".join([FAM_LABEL[fam], f"{len(x):,}".replace(",", "{,}"),
                                str(int(m.eq("multistart_vectorized_direct_REML_LBFGSB_and_Powell").sum())),
                                f'{int(m.eq("startpoint_screen_likelihood_refit_v2").sum()):,}'.replace(",", "{,}"),
                                str(int((~x.strict_fit_valid.astype(str).str.lower().isin(["true", "1", "1.0"])).sum()))]) + "\\\\")
    out["S4_repairs_rows.tex"] = rows

    ch = pd.read_csv(a.step7 / "oc_changes_long.csv")
    ch = ch[(ch.metric == "slope_heterogeneity_support_rate") & (ch.change.abs() >= .005)]
    fl = pd.read_csv(a.step7 / "decision_flips_by_cell.csv")
    flips = fl.groupby("cell_id")[["0->1", "1->0"]].sum()
    rows = []
    for r in ch.sort_values("change").itertuples():
        lr = oc.set_index("cell_id").loc[r.cell_id]
        lr = pd.Series(lr); lr["cell_id"] = r.cell_id
        lab = label(lr) if lr.cell_family != "true_zero" else r.cell_id
        fr = flips.loc[r.cell_id] if r.cell_id in flips.index else pd.Series({"0->1": 0, "1->0": 0})
        rows.append(f"{lab} & {f(r.original)} & {f(r.repaired)} & {int(fr['0->1'])} & {int(fr['1->0'])}\\\\")
    out["S4_changes_rows.tex"] = rows

    for k, v in out.items():
        (a.out_dir / k).write_text("\n".join(v) + "\n")
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
