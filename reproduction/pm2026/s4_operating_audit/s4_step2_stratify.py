#!/usr/bin/env python3
"""Operating-study audit, descriptive stratification (Supplement S4.4, S5.3). No refitting.

Usage:
    python3 reproduction/pm2026/s4_operating_audit/s4_step2_stratify.py <original S4 output dir> \
        --out-dir s4_step2

Writes all-fit and strictly-valid summaries separately. Design cells are never merged in
per-cell tables; pooled tables are descriptive. q_direct_index = mean_Rdirect/(1-mean_Rdirect)
equals tau2_hat*I/sigma2_hat for a common schedule but is not mean(q_i) for heterogeneous
schedules. Correlation cutoffs are descriptive proxies, not a numerical rank test. At zero
true slope variance, oracle determinacy is undefined; the original run's zero convention is
kept only in the replicate export.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["cell_id", "replicate"]
SING_TOL = 1e-4
NEAR_RHO = 0.95
Q_BINS = [0, .005, .02, .1, .5, np.inf]
Q_LABELS = ["[0,.005)", "[.005,.02)", "[.02,.1)", "[.1,.5)", "[.5,inf)"]


def classify(df):
    tau = df.random_slope_variance.astype(float)
    rho = df.intercept_slope_correlation.astype(float)
    bad = ~np.isfinite(tau) | (tau < 0) | (rho.abs() > 1 + 1e-8)
    missing = ~np.isfinite(rho)
    singular = df.boundary_flag.fillna(False).astype(bool) | (1-rho**2 <= SING_TOL)
    cls = np.select(
        [bad, tau == 0, missing, singular, rho.abs() >= NEAR_RHO],
        ["invalid_parameter", "exact_zero_slope", "unknown_correlation",
         "boundary_or_singular_proxy", "near_correlation"],
        default="interior_correlation")
    R = df.mean_Rdirect.astype(float)
    valid = np.isfinite(R) & (R >= 0) & (R < 1)
    q = (R / (1-R)).where(valid)
    bins = pd.cut(q, Q_BINS, labels=Q_LABELS, right=False).astype(object)
    bins = bins.where(bins.notna(), "unknown_or_invalid")
    return pd.Series(cls, index=df.index), q, bins


def prepare(df):
    d = df.copy()
    d["sing"], d["q_direct_index"], d["q_index_bin"] = classify(d)
    # Do not reinterpret the archive's zero convention as defined oracle D2.
    positive = d.oracle_slope_variance > 0
    d["D2_oracle_defined"] = d.oracle_mean_Pfull.where(positive)
    d["signed_D2_error"] = d.mean_Pfull - d.D2_oracle_defined
    d["positive_D2_error"] = d.signed_D2_error.clip(lower=0)
    # Zero fitted slope variance also makes normalized fitted D2 undefined.
    d["D2_hat_defined"] = d.mean_Pfull.where(d.random_slope_variance > 0)
    d["archived_Q1_decision"] = d.slope_heterogeneity_supported.astype(float)
    eval_mask = d.LR_evaluation_replicate & d.strict_fit_valid
    d["Q1_evaluation_decision"] = d.archived_Q1_decision.where(eval_mask)
    d["optimizer_agreement_recorded"] = d.optimizer_agreement.astype(float)
    d["tau2_ratio"] = d.random_slope_variance / d.oracle_slope_variance.where(positive)
    return d


def summarize(d, by):
    out = d.groupby(by, observed=True, dropna=False).agg(
        n=("cell_id", "size"), strict_valid_n=("strict_fit_valid", "sum"),
        D2_available_n=("D2_hat_defined", "count"),
        D2_hat_mean=("D2_hat_defined", "mean"), D2_hat_median=("D2_hat_defined", "median"),
        D2_oracle_defined_n=("D2_oracle_defined", "count"),
        D2_oracle_mean=("D2_oracle_defined", "mean"),
        R_hat_mean=("mean_Rdirect", "mean"), R_hat_median=("mean_Rdirect", "median"),
        R_oracle_mean=("oracle_mean_Rdirect", "mean"), Delta_hat_mean=("mean_Delta", "mean"),
        q_direct_index_median=("q_direct_index", "median"),
        tau2_hat_median=("random_slope_variance", "median"), tau2_ratio_median=("tau2_ratio", "median"),
        abs_rho_hat_median=("absolute_intercept_slope_correlation", "median"),
        boundary_flag_rate=("boundary_flag", "mean"),
        signed_D2_error_mean=("signed_D2_error", "mean"),
        positive_D2_error_sum=("positive_D2_error", lambda x: x.sum(min_count=1)),
        archived_false_high_full_rate=("false_high_full_when_oracle_low", "mean"),
        archived_Q1_decision_n=("archived_Q1_decision", "count"),
        archived_Q1_decision_rate=("archived_Q1_decision", "mean"),
        Q1_evaluation_n=("Q1_evaluation_decision", "count"),
        Q1_evaluation_rate=("Q1_evaluation_decision", "mean"),
        optimizer_agreement_n=("optimizer_agreement_recorded", "count"),
        optimizer_agreement_rate=("optimizer_agreement_recorded", "mean"),
        repaired_n=("repair_attempted", "sum"))
    # Parent = all grouping columns except the final singularity class.
    # For the q table this is cell x q bin, not the whole cell.
    parent = out.groupby(level=list(range(len(by)-1)), observed=True, dropna=False)
    out["parent_n"] = parent.n.transform("sum")
    out["share_within_parent"] = out.n / out.parent_n
    denominator = parent.positive_D2_error_sum.transform(lambda x: x.sum(min_count=1))
    out["share_of_positive_D2_error_within_parent"] = out.positive_D2_error_sum / denominator.where(denominator > 0)
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("s4_step2"))
    a = ap.parse_args()
    after = pd.read_csv(a.folder / "operating_study_replicate_results.csv", low_memory=False)
    before = pd.read_csv(a.folder / "operating_study_replicate_results_before_targeted_refit.csv", low_memory=False)
    design = pd.read_csv(a.folder / "operating_study_design.csv")
    for name, d in [("after", after), ("before", before)]:
        if d.duplicated(KEY).any() or d[KEY].isna().any().any():
            raise ValueError("Duplicate or missing keys in " + name)
    if design.cell_id.duplicated().any():
        raise ValueError("Duplicate design cell IDs")
    match = before[KEY + ["seed"]].merge(after[KEY + ["seed"]], on=KEY,
            how="outer", suffixes=("_before", "_after"), indicator=True, validate="one_to_one")
    if not (match._merge.eq("both").all() and match.seed_before.eq(match.seed_after).all()):
        raise ValueError("Before/after key or seed mismatch")
    counts = after.groupby("cell_id").size()
    expected = design.set_index("cell_id").requested_reps
    if set(counts.index) != set(expected.index) or not counts.eq(expected.reindex(counts.index)).all():
        raise ValueError("Results do not match requested design counts")
    for col in ["strict_fit_valid", "repair_attempted", "boundary_flag",
                "LR_evaluation_replicate", "LR_calibration_replicate", "slope_heterogeneity_supported"]:
        if after[col].isna().any() or not pd.api.types.is_bool_dtype(after[col]):
            raise ValueError("Expected nonmissing Boolean column: " + col)
    if (after.LR_evaluation_replicate & after.LR_calibration_replicate).any():
        raise ValueError("Calibration and evaluation masks overlap")
    d = prepare(after)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_cols = ["cell_id", "cell_family", "q0", "q1", "rho", "schedule", "misspecification"]
    metadata = design[metadata_cols]
    denom = d.groupby("cell_id").agg(all_n=("replicate", "size"),
                strict_valid_n=("strict_fit_valid", "sum"),
                Q1_evaluation_n=("Q1_evaluation_decision", "count"),
                Q1_evaluation_rate=("Q1_evaluation_decision", "mean"),
                optimizer_agreement_n=("optimizer_agreement_recorded", "count"))
    denom["strict_invalid_n"] = denom.all_n - denom.strict_valid_n
    metadata.merge(denom, on="cell_id", validate="one_to_one").to_csv(a.out_dir / "cell_denominators.csv", index=False)
    for subset, data in [("all_fits", d), ("strict_valid", d[d.strict_fit_valid])]:
        folder = a.out_dir / subset
        folder.mkdir(exist_ok=True)
        tables = [
            ("t1_true_zero_by_cell_sing", data[data.cell_family == "true_zero"], ["cell_id", "sing"]),
            ("t2_true_zero_by_cell_q_sing", data[data.cell_family == "true_zero"], ["cell_id", "q_index_bin", "sing"]),
            ("t3_positive_by_cell_sing", data[data.oracle_slope_variance > 0], ["cell_id", "sing"]),
            ("t4_pooled_family_sing", data, ["cell_family", "sing"])]
        for name, source, by in tables:
            table = summarize(source, by)
            if "cell_id" in by:
                table = metadata.merge(table, on="cell_id", validate="one_to_many")
            table.to_csv(folder / (name + ".csv"), index=False)
        print("%s: %d rows, %d true-zero rows" % (subset, len(data), (data.cell_family == "true_zero").sum()))
    before["sing_before"], _, _ = classify(before)
    transitions = d[KEY + ["sing", "repair_attempted", "strict_fit_valid"]].merge(
        before[KEY + ["sing_before"]], on=KEY, validate="one_to_one")
    unrecorded = ((transitions.sing != transitions.sing_before) & ~transitions.repair_attempted).sum()
    if unrecorded:
        raise ValueError("Strata changed without recorded repair")
    rep = transitions[transitions.repair_attempted]
    pd.crosstab(rep.sing_before, rep.sing).to_csv(a.out_dir / "repair_class_transitions.csv")
    d[KEY + ["seed", "cell_family", "strict_fit_valid", "repair_attempted", "sing", "q_direct_index",
       "q_index_bin", "random_slope_variance", "boundary_flag", "mean_Pfull", "oracle_mean_Pfull",
       "LR_calibration_replicate", "LR_evaluation_replicate"]].to_csv(a.out_dir / "replicate_strata.csv", index=False)
    notes = """Interpretation notes
all_fits includes strict-invalid fits; strict_valid reproduces the archived strict-fit restriction.
The per-cell tables retain q0 and all other design distinctions. t4 pools designs descriptively.
q_direct_index is odds of mean direct reliability, not mean participant information.
boundary_or_singular_proxy means archive boundary flag OR 1-rho_hat^2 <= 1e-4;
near_correlation means remaining .95 <= abs(rho_hat); neither class proves exact numerical rank.
Missing correlations are unknown; exact zero fitted slope variance has its own class.
Normalized oracle D2 is undefined under true-zero variance. Archive zeros are not used as an oracle error target.
Positive-error shares use max(D2_hat-D2_oracle,0) and are undefined if the denominator is zero.
share_within_parent is within cell for t1/t3, within cell x q bin for t2, and within family for t4.
Q1_evaluation_rate uses only recorded strict-valid evaluation replicates (true-zero cells here).
archived_Q1_decision_rate describes all rows in a stratum and is not held-out null calibration.
Optimizer agreement rates are conditional on a recorded value; see optimizer_agreement_n.
These tables do not validate REML maxima or prove that boundary behavior is an optimizer artifact.
"""
    (a.out_dir / "README.txt").write_text(notes)
    print("\nTrue-zero evaluation denominators and rates:")
    print(denom.loc[d.loc[d.cell_family == "true_zero", "cell_id"].unique(),
                   ["all_n", "strict_valid_n", "Q1_evaluation_n", "Q1_evaluation_rate"]].to_string())
    print("\nWrote", a.out_dir)


if __name__ == "__main__":
    main()
