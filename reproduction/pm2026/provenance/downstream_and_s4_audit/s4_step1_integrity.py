#!/usr/bin/env python3
"""S4 step 1: integrity of the operating-study files before any stratification.

Usage:
    python3 s4_step1_integrity.py <dir containing the five s4_completion_validation_v1 files> [--out step1_report.csv]

Checks
  A. Design vs results: 46 cells, per-cell counts = requested_reps, unique (cell_id, replicate),
     design columns in the results identical to the design file.
  B. Before vs after targeted refit: same keys and seeds; which rows changed; changes only where
     repair_attempted; what changed (LR, variance components, determinacy summaries, flags).
  C. Refit audit consistency: 758 audit rows = repair_attempted rows; success / remaining invalid.
  D. Fit-validity and boundary rates by cell family (denominators for later steps).
  E. Internal consistency of the determinacy summaries: ranges, mean_Delta = mean_Pfull - mean_Rdirect,
     and where they are missing.
Prints a report and writes the per-family / per-cell table to --out (send that file back).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["cell_id", "replicate"]
COMPARE = ["restricted_LR", "random_intercept_variance", "random_slope_variance", "intercept_slope_covariance",
           "intercept_slope_correlation", "residual_variance", "mean_Rdirect", "mean_Pfull", "mean_Delta",
           "fit_valid", "boundary_flag", "hessian_positive_definite"]


def section(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=Path("s4_step1_report.csv"))
    a = ap.parse_args()
    f = a.folder
    after = pd.read_csv(f / "operating_study_replicate_results.csv", low_memory=False)
    before = pd.read_csv(f / "operating_study_replicate_results_before_targeted_refit.csv", low_memory=False)
    design = pd.read_csv(f / "operating_study_design.csv")
    audit = pd.read_csv(f / "targeted_refit_audit.csv", low_memory=False)
    problems = []

    section("A. design vs results")
    print(f"design cells {len(design)}, requested reps {int(design.requested_reps.sum())}; result rows {len(after)}")
    print("cells by family:", design.cell_family.value_counts().to_dict())
    dup = after.duplicated(KEY).sum()
    counts = after.groupby("cell_id").size().rename("rows").to_frame().join(design.set_index("cell_id").requested_reps)
    bad_counts = counts[counts.rows != counts.requested_reps]
    missing_cells = set(design.cell_id) - set(after.cell_id)
    extra_cells = set(after.cell_id) - set(design.cell_id)
    print(f"duplicate keys {dup}; cells with count != requested {len(bad_counts)}; missing cells {len(missing_cells)}; extra cells {len(extra_cells)}")
    if dup or len(bad_counts) or missing_cells or extra_cells:
        problems.append("design/result counts")
    dcols = [c for c in design.columns if c != "cell_id" and c in after.columns]
    merged = after[["cell_id"] + dcols].drop_duplicates("cell_id").merge(design, on="cell_id", suffixes=("", "_design"))
    mism = []
    for c in dcols:
        x, y = merged[c], merged[c + "_design"]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            bad = ~(np.isclose(x, y, rtol=0, atol=1e-12, equal_nan=True))
        else:
            bad = x.astype(str) != y.astype(str)
        if bad.any():
            mism.append(c)
    print("design columns differing between results and design file:", mism or "none")
    if mism:
        problems.append("design column mismatch")
    if after.groupby("cell_id")[dcols].nunique(dropna=False).gt(1).any().any():
        problems.append("design columns vary within a cell")
        print("WARNING: design columns vary within a cell")

    section("B. before vs after targeted refit")
    m = before.merge(after, on=KEY, how="outer", suffixes=("_before", "_after"), indicator=True)
    print("key match:", m["_merge"].value_counts().to_dict())
    print("seed identical:", bool((m.seed_before == m.seed_after).all()))
    changed = pd.Series(False, index=m.index)
    per_col = {}
    for c in COMPARE:
        if c + "_before" in m and c + "_after" in m:
            x, y = m[c + "_before"], m[c + "_after"]
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                d = ~np.isclose(x.astype(float), y.astype(float), rtol=1e-10, atol=1e-12, equal_nan=True)
            else:
                d = x.astype(str) != y.astype(str)
            per_col[c] = int(d.sum())
            changed |= d
    print("rows changed by column:", per_col)
    rep = m["repair_attempted"].fillna(False).astype(bool) if "repair_attempted" in m else pd.Series(False, index=m.index)
    print(f"rows changed (any compared column): {int(changed.sum())}; of these repair_attempted: {int((changed & rep).sum())}; "
          f"changed WITHOUT repair_attempted: {int((changed & ~rep).sum())}")
    if (changed & ~rep).any():
        problems.append("rows changed without a recorded repair")
    if "random_slope_variance_before" in m:
        ch = m[changed]
        print("among changed rows: slope variance before->after median %.4g -> %.4g; boundary_flag before %.3f after %.3f; "
              "LR mean %.4g -> %.4g" % (ch.random_slope_variance_before.median(), ch.random_slope_variance_after.median(),
                                        ch.boundary_flag_before.astype(float).mean(), ch.boundary_flag_after.astype(float).mean(),
                                        ch.restricted_LR_before.mean(), ch.restricted_LR_after.mean()))
        print("changed rows by family:", m.loc[changed, "cell_family_after"].value_counts().to_dict())

    section("C. targeted refit audit")
    akeys = set(map(tuple, audit[KEY].values))
    rkeys = set(map(tuple, after.loc[after.repair_attempted.fillna(False).astype(bool), KEY].values))
    print(f"audit rows {len(audit)}; repair_attempted rows in results {len(rkeys)}; identical key sets: {akeys == rkeys}")
    print("repair_succeeded:", after.loc[after.repair_attempted.fillna(False).astype(bool), "repair_succeeded"].value_counts(dropna=False).to_dict())
    print("repair_method:", after.repair_method.value_counts(dropna=False).to_dict())
    print("strict_fit_valid False (all rows):", int((~after.strict_fit_valid.fillna(False).astype(bool)).sum()))
    if akeys != rkeys:
        problems.append("audit keys != repair_attempted keys")

    section("D. validity and boundary by family")
    b = after.assign(
        fit_valid=after.fit_valid.fillna(False).astype(bool),
        strict=after.strict_fit_valid.fillna(False).astype(bool),
        boundary=after.boundary_flag.fillna(False).astype(bool),
        repaired=after.repair_attempted.fillna(False).astype(bool),
        tau2_zero=after.random_slope_variance.fillna(np.nan) <= 0,
        has_Pfull=after.mean_Pfull.notna(), has_Rdirect=after.mean_Rdirect.notna())
    fam = b.groupby("cell_family").agg(cells=("cell_id", "nunique"), rows=("cell_id", "size"),
                                       fit_valid=("fit_valid", "mean"), strict_valid=("strict", "mean"),
                                       boundary=("boundary", "mean"), repaired=("repaired", "mean"),
                                       tau2_hat_le_0=("tau2_zero", "mean"),
                                       Pfull_available=("has_Pfull", "mean"), Rdirect_available=("has_Rdirect", "mean"))
    print(fam.round(4).to_string())

    section("E. determinacy summaries")
    for c in ("mean_Rdirect", "mean_Pfull", "mean_Rinnovation", "mean_Delta"):
        s = after[c]
        print(f"{c}: missing {int(s.isna().sum())}, min {s.min():.4g}, max {s.max():.4g}, outside [0,1]: {int(((s < -1e-9) | (s > 1 + 1e-9)).sum())}")
    dd = (after.mean_Delta - (after.mean_Pfull - after.mean_Rdirect)).abs()
    print(f"max |mean_Delta - (mean_Pfull - mean_Rdirect)| = {dd.max():.3g}")
    miss = after[after.mean_Pfull.isna()]
    if len(miss):
        print("mean_Pfull missing by family / boundary_flag:",
              miss.groupby(["cell_family", miss.boundary_flag.fillna(False).astype(bool)]).size().to_dict())
    print("columns renamed between files (before-only / after-only):",
          sorted(set(before.columns) - set(after.columns)), sorted(set(after.columns) - set(before.columns)))

    cell = b.groupby(["cell_family", "cell_id"]).agg(
        q1=("q1", "first"), q0=("q0", "first"), rho=("rho", "first"), schedule=("schedule", "first"),
        oracle_slope_variance=("oracle_slope_variance", "first"), rows=("cell_id", "size"),
        fit_valid=("fit_valid", "mean"), strict_valid=("strict", "mean"), boundary=("boundary", "mean"),
        repaired=("repaired", "mean"), tau2_hat_le_0=("tau2_zero", "mean"),
        mean_Pfull=("mean_Pfull", "mean"), mean_Rdirect=("mean_Rdirect", "mean"),
        oracle_mean_Pfull=("oracle_mean_Pfull", "first"), oracle_mean_Rdirect=("oracle_mean_Rdirect", "first")).reset_index()
    cell.to_csv(a.out, index=False)

    section("SUMMARY")
    print("problems:", problems or "none")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
