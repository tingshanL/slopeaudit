#!/usr/bin/env python3
"""Operating-study audit, reporting comparison (Supplement S4.6). No fitting.

Compares the summary of the original replicate table with the summary of the table after the
consolidated refit (both produced by reproduction/s4_operating_study/summarize_operating_study.py)
and writes:

  t2_true_zero_table.csv        true-zero cells, before and after: valid fits, D2 median [IQR],
                                D2 >= .40, R_direct median [IQR], evaluation false positives
  t2b_boundary_breakdown.csv    true-zero fits by fitted-covariance class (thresholds printed)
  t2c_repaired_true_zero.csv    every refit true-zero row: class and D2 before and after
  oc_changes_long.csv           every (cell, metric) of operating_characteristics.csv that changed
  lr_calibration_compare.csv    critical values and evaluation false-support rates, before and after
  decision_flips_by_cell.csv    per cell: heterogeneity-decision changes by direction, subset and
                                cause (likelihood-ratio change, cutoff change or both; descriptive)
  figure_02_contract.json       Panel-A values of the earlier (v0.1.0) Figure 2, recomputed
  make_figure_02_repaired.py    with --figure-script only: a copy of the v0.1.0 figure script with
                                its Panel-A assertions updated; not used by the 2026 manuscript

Usage (from the repository root):
  python3 reproduction/pm2026/s4_operating_audit/s4_step7_reporting.py \
      --orig-summary <original summary dir> --new-summary <summary after refits> --out-dir s4_step7_reporting
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["cell_id", "replicate"]
SING_TOL, NEAR_TOL = 1e-4, 1 - 0.95 ** 2           # same thresholds as step 2
CLASS_LABELS = {
    "singular_proxy": "boundary flag, missing rho, or 1 - rho^2 <= 1e-4 (|rho| >= .99995)",
    "near_rank_one": "1e-4 < 1 - rho^2 <= .0975 (.95 <= |rho| < .99995)",
    "interior": "1 - rho^2 > .0975 (|rho| < .95)",
}
CLASS_ORDER = ["interior", "near_rank_one", "singular_proxy"]
USE = KEY + ["cell_family", "schedule", "q0", "strict_fit_valid", "mean_Pfull", "mean_Rdirect",
             "intercept_slope_correlation", "boundary_flag", "restricted_LR", "matched_LR_critical_value",
             "slope_heterogeneity_supported", "LR_calibration_replicate", "LR_evaluation_replicate",
             "repair_attempted", "repair_method", "startpoint_refit"]


def as_bool(s):
    if s.dtype == bool:
        return s
    result = s.astype(str).str.strip().str.lower().map({
        "true": True, "1": True, "1.0": True,
        "false": False, "0": False, "0.0": False,
    })
    if result.isna().any():
        raise ValueError(f"Missing or unrecognized Boolean values in {s.name}")
    return result.astype(bool)


def load(summary_dir: Path):
    d = pd.read_csv(summary_dir / "operating_study_replicate_results.csv", low_memory=False,
                    usecols=lambda c: c in USE, float_precision="round_trip")
    missing = sorted(set(USE) - {"startpoint_refit"} - set(d.columns))
    if missing:
        raise ValueError(f"Missing columns in {summary_dir}: {missing}")
    if d[KEY].isna().any().any() or d.duplicated(KEY).any():
        raise ValueError(f"Missing or duplicate row keys in {summary_dir}")
    for c in ["strict_fit_valid", "boundary_flag", "slope_heterogeneity_supported",
              "LR_calibration_replicate", "LR_evaluation_replicate", "repair_attempted"]:
        d[c] = as_bool(d[c])
    if "startpoint_refit" in d:
        d["startpoint_refit"] = as_bool(d.startpoint_refit)
    else:
        d["startpoint_refit"] = False
    valid = d.loc[d.strict_fit_valid]
    numeric = ["mean_Pfull", "mean_Rdirect", "restricted_LR", "matched_LR_critical_value"]
    if not np.isfinite(valid[numeric].to_numpy(float)).all():
        raise ValueError(f"Nonfinite reporting metrics in valid rows of {summary_dir}")
    if not valid.slope_heterogeneity_supported.equals(
            valid.restricted_LR.ge(valid.matched_LR_critical_value)):
        raise ValueError(f"Stored Q1 decisions disagree with LR and cutoff in {summary_dir}")
    r = d.intercept_slope_correlation.astype(float)
    omr2 = 1 - r ** 2
    d["cov_class"] = np.where(d.boundary_flag | r.isna() | (omr2 <= SING_TOL), "singular_proxy",
                              np.where(omr2 <= NEAR_TOL, "near_rank_one", "interior"))
    return d.sort_values(KEY).reset_index(drop=True)


def mi(x):
    x = pd.Series(x).dropna()
    if not len(x):
        return ""
    return f"{x.median():.3f} [{x.quantile(.25):.3f}, {x.quantile(.75):.3f}]"


def kn(k, n):
    return f"{k}/{n} ({k / n:.1%})" if n else ""


def table2(d, tag):
    rows = []
    for cid, g in d[d.cell_family == "true_zero"].groupby("cell_id", sort=False):
        v = g[g.strict_fit_valid]
        ev = v[v.LR_evaluation_replicate]
        k40 = int((v.mean_Pfull >= .40).sum())
        rows.append({"version": tag, "cell_id": cid, "valid_fits": len(v),
                     "D2_median_IQR": mi(v.mean_Pfull), "D2_ge_040": kn(k40, len(v)),
                     "Rdirect_median_IQR": mi(v.mean_Rdirect),
                     "false_positive_evaluation_subset": kn(int(ev.slope_heterogeneity_supported.sum()), len(ev)),
                     "_k40": k40, "_kfp": int(ev.slope_heterogeneity_supported.sum()), "_nfp": len(ev)})
    return pd.DataFrame(rows)


def breakdown(d, tag):
    v = d[(d.cell_family == "true_zero") & d.strict_fit_valid]
    rows = []
    for (cid, cls), g in v.groupby(["cell_id", "cov_class"]):
        n_cell = int((v.cell_id == cid).sum())
        ev = g[g.LR_evaluation_replicate]
        rows.append({"version": tag, "cell_id": cid, "cov_class": cls, "n": len(g), "share_of_cell": len(g) / n_cell,
                     "abs_rho_median": g.intercept_slope_correlation.abs().median(),
                     "D2_median_IQR": mi(g.mean_Pfull), "D2_ge_040": (g.mean_Pfull >= .40).mean(),
                     "D2_ge_075_descriptive": (g.mean_Pfull >= .75).mean(),
                     "Rdirect_median": g.mean_Rdirect.median(),
                     "Q1_evaluation_n": len(ev),
                     "Q1_evaluation_k": int(ev.slope_heterogeneity_supported.sum()),
                     "Q1_supported_evaluation": ev.slope_heterogeneity_supported.mean()})
    out = pd.DataFrame(rows)
    out["cov_class"] = pd.Categorical(out.cov_class, CLASS_ORDER, ordered=True)
    return out.sort_values(["cell_id", "cov_class"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-summary", type=Path, required=True)
    ap.add_argument("--new-summary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--figure-script", type=Path, default=None)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 40, "display.max_colwidth", 60)

    old, new = load(a.orig_summary), load(a.new_summary)
    if not (len(old) == len(new) == 62_000 and old[KEY].equals(new[KEY])):
        raise ValueError("Expected identical sets of 62,000 unique row keys")
    if not old.strict_fit_valid.equals(new.strict_fit_valid):
        raise ValueError("Validity changed; comparison denominators need explicit review")
    print("== Repair accounting (repaired table) ==")
    print("repair_attempted:", int(new.repair_attempted.sum()), "| by method:",
          new.repair_method.value_counts().to_dict(), "| strict-invalid:", int((~new.strict_fit_valid).sum()),
          "(original: repair_attempted", int(old.repair_attempted.sum()), ", strict-invalid", int((~old.strict_fit_valid).sum()), ")")

    # ---- Table 2
    t2 = pd.concat([table2(old, "original"), table2(new, "repaired")])
    t2.drop(columns=[c for c in t2 if c.startswith("_")]).to_csv(a.out_dir / "t2_true_zero_table.csv", index=False)
    print("\n== Table 2 (true-zero cells; false positives from the independent evaluation subset) ==")
    print(t2.drop(columns=[c for c in t2 if c.startswith("_")]).to_string(index=False))

    # ---- boundary breakdown
    bd = pd.concat([breakdown(old, "original"), breakdown(new, "repaired")])
    bd.to_csv(a.out_dir / "t2b_boundary_breakdown.csv", index=False)
    print("\n== Covariance-class thresholds ==")
    for k in CLASS_ORDER:
        print(f"  {k:15s} {CLASS_LABELS[k]}")
    (a.out_dir / "covariance_class_definitions.json").write_text(
        json.dumps(CLASS_LABELS, indent=2) + "\n")
    print("\n== True-zero fits by covariance class (repaired; D2 >= .75 is descriptive only) ==")
    print(bd[bd.version == "repaired"].drop(columns="version").round(3).to_string(index=False))

    # ---- repaired true-zero rows
    m = old.merge(new, on=KEY, suffixes=("_old", "_new"), validate="one_to_one")
    tz = m[m.cell_family_new.eq("true_zero") & m.startpoint_refit_new]
    tzc = tz[KEY + ["cov_class_old", "cov_class_new", "mean_Pfull_old", "mean_Pfull_new", "intercept_slope_correlation_new",
                    "restricted_LR_old", "restricted_LR_new", "LR_calibration_replicate_new", "LR_evaluation_replicate_new",
                    "slope_heterogeneity_supported_old", "slope_heterogeneity_supported_new"]]
    tzc.to_csv(a.out_dir / "t2c_repaired_true_zero.csv", index=False)
    print(f"\n== Repaired true-zero rows: {len(tzc)} ==")
    print(tzc.groupby(["cell_id", "cov_class_old", "cov_class_new"]).agg(
        n=("replicate", "size"), D2_old_median=("mean_Pfull_old", "median"),
        D2_new_min=("mean_Pfull_new", "min"), D2_new_max=("mean_Pfull_new", "max")).round(3).to_string())
    for cid, g in new[(new.cell_family == "true_zero") & new.strict_fit_valid].groupby("cell_id"):
        nrep = int(g.startpoint_refit.sum())
        print(f"  {cid}: {nrep}/{len(g)} valid fits repaired ({nrep / len(g):.2%})")

    # ---- every changed operating-characteristic entry
    oo = pd.read_csv(a.orig_summary / "operating_characteristics.csv")
    on = pd.read_csv(a.new_summary / "operating_characteristics.csv")
    for frame in (oo, on):
        if frame.cell_id.isna().any() or frame.cell_id.duplicated().any():
            raise ValueError("Missing or duplicate operating-characteristic cell IDs")
    oo = oo.sort_values("cell_id").reset_index(drop=True)
    on = on.sort_values("cell_id").reset_index(drop=True)
    if oo.cell_id.tolist() != on.cell_id.tolist() or set(oo.columns) != set(on.columns):
        raise ValueError("Operating-characteristic schemas or cell IDs differ")
    rows = []
    for c in oo.columns:
        if c == "cell_id" or not pd.api.types.is_numeric_dtype(oo[c]) or oo[c].dtype == bool:
            if c != "cell_id" and (oo[c].astype(str) != on[c].astype(str)).any():
                for i in np.where(oo[c].astype(str) != on[c].astype(str))[0]:
                    rows.append({"cell_id": oo.cell_id[i], "metric": c, "original": oo[c][i], "repaired": on[c][i], "change": np.nan})
            continue
        x, y = oo[c].astype(float), on[c].astype(float)
        ch = ~np.isclose(x, y, rtol=0, atol=1e-12, equal_nan=True)
        for i in np.where(ch)[0]:
            rows.append({"cell_id": oo.cell_id[i], "metric": c, "original": x[i], "repaired": y[i], "change": y[i] - x[i]})
    oc = pd.DataFrame(rows, columns=["cell_id", "metric", "original", "repaired", "change"])
    oc.to_csv(a.out_dir / "oc_changes_long.csv", index=False)
    print(f"\n== operating_characteristics: {len(oc)} changed entries in {oc.cell_id.nunique()} cells ==")
    key_metrics = ["slope_heterogeneity_support_rate", "false_slope_heterogeneity_support_rate",
                   "false_high_Pfull_rate_when_oracle_Pfull_lt_0p20", "bias_mean_Pfull", "RMSE_mean_Pfull",
                   "bias_mean_Rdirect", "bias_mean_Delta", "boundary_rate",
                   "anchored_geometry_sensitivity", "anchored_geometry_specificity"]
    wide = oc[oc.metric.isin(key_metrics)].assign(
        val=lambda z: z.original.map("{:.3f}".format) + "->" + z.repaired.map("{:.3f}".format)).pivot(
        index="cell_id", columns="metric", values="val").fillna("")
    print(wide.to_string())

    # ---- calibration
    lo = pd.read_csv(a.orig_summary / "likelihood_ratio_calibration.csv")
    ln = pd.read_csv(a.new_summary / "likelihood_ratio_calibration.csv")
    if set(lo.cell_id) != set(ln.cell_id):
        raise ValueError("Calibration cell IDs differ")
    lc = lo.merge(ln, on="cell_id", suffixes=("_original", "_repaired"), validate="one_to_one")
    lc.to_csv(a.out_dir / "lr_calibration_compare.csv", index=False)
    print("\n== LR calibration ==")
    print(lc[["cell_id", "LR_95_critical_value_original", "LR_95_critical_value_repaired",
              "evaluation_false_support_rate_original", "evaluation_false_support_rate_repaired"]].round(4).to_string(index=False))

    # ---- decision flips
    m["flip"] = m.slope_heterogeneity_supported_old != m.slope_heterogeneity_supported_new
    m["lr_changed"] = ~np.isclose(m.restricted_LR_old.astype(float), m.restricted_LR_new.astype(float),
                                  rtol=0, atol=1e-9, equal_nan=True)
    m["cutoff_changed"] = ~np.isclose(m.matched_LR_critical_value_old.astype(float),
                                      m.matched_LR_critical_value_new.astype(float),
                                      rtol=0, atol=1e-9, equal_nan=True)
    f = m[m.flip & m.strict_fit_valid_new].assign(
        direction=lambda z: np.where(z.slope_heterogeneity_supported_new, "0->1", "1->0"),
        subset=lambda z: np.where(z.LR_calibration_replicate_new, "calibration",
                                  np.where(z.LR_evaluation_replicate_new, "evaluation", "neither")),
        change_pattern=lambda z: np.select(
            [z.lr_changed & z.cutoff_changed, z.lr_changed, z.cutoff_changed],
            ["LR_and_cutoff_changed", "LR_only_changed", "cutoff_only_changed"],
            default="below_numeric_tolerance"))
    fl = f.groupby(["cell_id", "subset", "change_pattern", "direction"]).size().unstack("direction", fill_value=0)
    fl.to_csv(a.out_dir / "decision_flips_by_cell.csv")
    print(f"\n== Q1 decision flips: {len(f)} rows in {f.cell_id.nunique()} cells ==")
    print(fl.to_string())

    # ---- Figure 2 Panel-A contract
    v = new[(new.cell_family == "true_zero") & new.strict_fit_valid]
    t2n = table2(new, "repaired")
    contract = {"strict_valid_true_zero_fits": int(len(v)),
                "score_counts": t2n.valid_fits.tolist(),
                "heterogeneity_k_over_n": [[int(k), int(n)] for k, n in zip(t2n._kfp, t2n._nfp)],
                "determinacy_proceeds_D2_ge_040": int((v.mean_Pfull >= .40).sum()),
                "direct_proceeds_Rdirect_ge_040": int((v.mean_Rdirect >= .40).sum()),
                "cell_order": t2n.cell_id.tolist()}
    (a.out_dir / "figure_02_contract.json").write_text(json.dumps(contract, indent=2))
    print("\n== Figure 2 Panel-A contract (repaired) ==\n" + json.dumps(contract))

    if a.figure_script:
        src = a.figure_script.read_text()
        tree = ast.parse(src)
        def constant_assignment(name):
            found = [node.value for node in ast.walk(tree)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
            if len(found) != 1:
                raise ValueError(f"Expected one figure assignment for {name}")
            return ast.literal_eval(found[0])
        for name, expected in [
            ("EXPECTED_SCORE_COUNTS", contract["score_counts"]),
            ("EXPECTED_HETEROGENEITY_KN", [tuple(x) for x in contract["heterogeneity_k_over_n"]]),
            ("SCORE_THRESHOLD", 0.40),
        ]:
            if constant_assignment(name) != expected:
                raise ValueError(f"Figure contract mismatch: {name}")
        checks = [node.test for node in ast.walk(tree) if isinstance(node, ast.Assert)]
        def require_assertion(left, expected):
            if not any(isinstance(test, ast.Compare) and len(test.ops) == 1
                       and isinstance(test.ops[0], ast.Eq)
                       and ast.dump(test.left) == ast.dump(ast.parse(left, mode="eval").body)
                       and isinstance(test.comparators[0], ast.Constant)
                       and test.comparators[0].value == expected for test in checks):
                raise ValueError(f"Figure assertion mismatch: {left} == {expected}")
        require_assertion("direct_proceeds", contract["direct_proceeds_Rdirect_ge_040"])
        require_assertion("len(true_zero)", contract["strict_valid_true_zero_fits"])
        pat = r"assert determinacy_proceeds == [0-9_]+"
        if len(re.findall(pat, src)) != 1:
            raise SystemExit("could not find exactly one determinacy_proceeds assertion")
        src = re.sub(pat, f"assert determinacy_proceeds == {contract['determinacy_proceeds_D2_ge_040']}", src)
        out = a.out_dir / "make_figure_02_repaired.py"
        out.write_text(src)
        print(f"wrote {out} (only the determinacy_proceeds assertion changed)")
    print(f"\nwrote {a.out_dir}/")


if __name__ == "__main__":
    main()
