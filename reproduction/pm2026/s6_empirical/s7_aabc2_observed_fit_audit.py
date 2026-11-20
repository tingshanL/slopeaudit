#!/usr/bin/env python3
"""Compare saved AABC2 observed-model fits and cognition profile points.

Run alongside s7_aabc2_kappa_stability.py. The reliability audit identifies the
reported fit. The matching RLRT alternative rows supply full covariances.
Provide the original inputs to evaluate competing fits on the Figure 4 schedules.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd

import s7_aabc2_kappa_stability as capacity

OUTCOMES = {capacity.COGNITION: (691, 1805), capacity.VENTRICLES: (622, 1629)}
LL_TOL = 0.01
SAME_FIT_ATOL = 1e-6
SAME_FIT_RTOL = 1e-8
PROFILE_PARAMETER_RTOL = 1e-5  # allow reoptimization at the saved coordinate
CORROBORATED = "Corroborated by saved outputs"
FOLLOW_UP = "Targeted follow-up needed"
INSUFFICIENT = "Insufficient saved information"
FIT = capacity.FIT_COLUMNS
METRICS = capacity.METRICS
SCHEDULE_ERRORS = (OSError, KeyError, ValueError, TypeError, ImportError, RuntimeError, BadZipFile)


def read(path, columns=(), **kwargs):
    frame = pd.read_csv(path, float_precision="round_trip", **kwargs)
    missing = set(columns) - set(frame)
    if missing:
        raise KeyError(f"{path.name}: missing {', '.join(sorted(missing))}")
    return frame


def agrees(left, right, rtol=SAME_FIT_RTOL):
    return np.allclose(np.asarray(left, float), np.asarray(right, float),
                       atol=SAME_FIT_ATOL, rtol=rtol)


def true(series):
    return series.map(capacity.is_true)


def one(frame, label):
    if len(frame) != 1:
        raise ValueError(f"Expected one {label}, found {len(frame)}")
    return frame.iloc[0]


def optimizer_audit(outcome, reliability, rlrt, observed):
    # Each producer fits its optimizer panel on one frozen design with REML.
    panel = reliability.loc[reliability.outcome.eq(outcome) &
                            reliability.model_specification.eq("reference_unstructured_mid")].copy()
    if panel.empty:
        raise KeyError(f"{outcome}: missing reference optimizer panel")
    selected = one(panel.loc[true(panel.selected)], "reported reference fit")
    cohort_size = OUTCOMES[outcome]
    if not panel[["participants", "observations"]].eq(cohort_size).all().all():
        raise ValueError("Reference optimizer panel contains inconsistent cohort counts")
    if panel.optimizer.duplicated().any() or panel.fixed_design_columns.nunique() != 1:
        raise ValueError("Reference optimizer panel contains duplicate optimizers or different designs")
    if not agrees(selected[["slope_variance", "residual_variance"]],
                  observed[["tau2", "sigma2"]]):
        raise ValueError("Selected reference fit does not match the reported Figure 4 variances")

    finite = np.isfinite(panel.llf) & np.isfinite(panel.residual_variance)
    design_ok = true(panel.fixed_design_full_rank) & panel.fixed_design_rank.eq(panel.fixed_design_columns)
    converged = true(panel.optimizer_converged)
    admissible = (finite & design_ok & converged & true(panel.fixed_parameters_finite) &
                  true(panel.random_effect_covariance_psd) & panel.residual_variance.gt(0))
    panel["admissible"] = admissible
    panel["delta_loglik"] = panel.llf - selected.llf
    panel["higher_unresolved"] = (panel.delta_loglik.gt(LL_TOL) & ~admissible)
    usable = panel.loc[admissible]
    close_fits = usable.loc[usable.delta_loglik.abs().le(LL_TOL)]
    other = close_fits.loc[close_fits.optimizer.ne(selected.optimizer)]
    higher = panel.loc[panel.delta_loglik.gt(LL_TOL)]

    summary = {
        "outcome": outcome,
        "participants": cohort_size[0], "observations": cohort_size[1],
        "fixed_design_columns": selected.fixed_design_columns,
        "reported_optimizer": selected.optimizer,
        "reported_loglik": selected.llf,
        "reported_converged": capacity.is_true(selected.optimizer_converged),
        "reported_historical_agreement": capacity.is_true(selected.optimizer_agreement),
        "n_candidates_within_tolerance": len(close_fits),
        "corroborating_optimizers": ", ".join(other.optimizer),
        "higher_candidate_converged": bool(true(higher.optimizer_converged).any()) if len(higher) else pd.NA,
        "higher_unresolved_optimizers": ", ".join(panel.loc[panel.higher_unresolved, "optimizer"]),
    }
    issues, missing = [], []
    if not np.isfinite(selected.llf):
        missing.append("Reported REML likelihood unavailable")
    elif not bool(admissible.loc[selected.name]):
        issues.append("Reported fit fails saved admissibility checks")
    if len(usable):
        best = usable.loc[usable.llf.idxmax()]
        summary.update(best_saved_optimizer=best.optimizer, best_saved_loglik=best.llf,
                       loglik_improvement=best.delta_loglik)
        if best.delta_loglik > LL_TOL:
            issues.append("Higher admissible REML likelihood")
    if panel.higher_unresolved.any():
        issues.append("Higher saved likelihood with unresolved fit diagnostics")
    if other.empty and not issues:
        missing.append("No second admissible optimizer within 0.01 log-likelihood units")

    # The RLRT alternative panel supplies G00 and G01 omitted by the reference audit.
    if rlrt is None:
        missing.append("RLRT alternative covariance table unavailable")
        matched = pd.DataFrame(columns=["optimizer", *FIT]).set_index("optimizer")
    else:
        source = rlrt.loc[rlrt.outcome.eq(outcome) & rlrt.model.eq("RI_RS_alternative") &
                          rlrt.estimation.eq("REML")].copy()
        if source.optimizer.duplicated().any():
            raise ValueError("Duplicate optimizer in the RLRT alternative panel")
        joined = panel.merge(source, on="optimizer", suffixes=("", "_rlrt"))
        columns = [
            ("fixed_design_rank", "fixed_design_rank_rlrt"),
            ("fixed_design_columns", "fixed_design_columns_rlrt"),
            ("llf", "log_likelihood"), ("slope_variance", "random_slope_variance"),
            ("residual_variance", "residual_variance_rlrt"),
        ]
        matched_rows = np.logical_and.reduce([
            np.isclose(joined[left], joined[right], atol=SAME_FIT_ATOL, rtol=SAME_FIT_RTOL)
            for left, right in columns
        ])
        comparable = joined.loc[matched_rows].copy()
        designs = {tuple(sorted(json.loads(item))) for item in comparable.fixed_effects_json.dropna()}
        if len(designs) > 1 or (designs and len(next(iter(designs))) != selected.fixed_design_columns):
            raise ValueError("RLRT alternative panel shows inconsistent fixed-effect terms")
        comparable["residual_variance"] = comparable.residual_variance_rlrt
        matched = comparable.set_index("optimizer")[FIT]
        if selected.optimizer in matched.index:
            if not agrees(matched.loc[selected.optimizer, FIT],
                          observed[["G00", "tau2", "cov01", "sigma2"]]):
                issues.append("Selected optimizer covariance disagrees with Figure 4")
        else:
            missing.append("Full covariance could not be matched for the selected fit")

    panel = panel.merge(matched.add_prefix("matched_"), left_on="optimizer", right_index=True,
                        how="left", validate="one_to_one")
    return summary, issues, missing, panel


def anchored_d(G00, tau2, cov01):
    v0 = G00 - cov01 + tau2 / 4
    v5 = G00 + cov01 + tau2 / 4
    return (G00 * tau2 - cov01**2) / (v0 * v5) if min(v0, v5) > 0 else np.nan


def profile_audit(raw, evaluated, comparison, observed, reported_loglik):
    profile = raw.loc[raw.outcome.eq(capacity.COGNITION) & raw.dataset_role.eq("observed")].copy()
    if profile.empty:
        raise KeyError("Observed cognition likelihood profile unavailable")
    selected = one(profile.loc[profile.path_direction.eq("selected")], "selected profile coordinate")
    boundary = one(profile.loc[true(profile.exact_PSD_boundary)], "rank-one endpoint")
    issues, missing = [], []
    if (len(profile) != 43 or profile.d_0_5.nunique() != 43 or profile.dataset_id.nunique() != 1 or
            not true(profile.optimizer_agreement).all()):
        issues.append("Unexpected or uncorroborated profile grid")
    if abs(boundary.d_0_5) > SAME_FIT_ATOL or not np.isclose(boundary.relative_deviance, 1.10, atol=.005):
        issues.append("Saved boundary coordinate or deviance differs from the reported result")

    cov_fields = ["d_0_5", *FIT]
    expected = dict(zip(FIT, observed[["G00", "tau2", "cov01", "sigma2"]]))
    expected["d_0_5"] = anchored_d(*(expected[k] for k in FIT[:3]))
    if all(k in selected.index for k in ("selected_gamma_sign", "selected_gamma_0_5",
                                        "optimized_log_slope_sd", "optimized_log_sd5_over_sd0",
                                        "optimized_log_residual_sd")):
        parameters = dict(zip(FIT, capacity.profile_components(selected)[:4]))
        parameters["d_0_5"] = selected.d_0_5
        covariance_agrees = agrees([parameters[k] for k in cov_fields],
                                   [expected[k] for k in cov_fields], PROFILE_PARAMETER_RTOL)
        if not covariance_agrees:
            issues.append("Selected profile covariance differs from Figure 4")
        boundary_components = capacity.profile_components(boundary)[:4]
        if boundary_components[1] <= 0:
            issues.append("Rank-one profile endpoint lacks positive slope variance")
    else:
        covariance_agrees = pd.NA
        missing.append("Profile covariance parameters unavailable")

    result = {
        "profile_covariance_agrees": covariance_agrees,
        "profile_points": len(profile),
        "profile_boundary_deviance": boundary.relative_deviance,
        "profile_max_implied_improvement": -profile.relative_deviance.min() / 2,
    }
    if not np.isfinite(profile.relative_deviance).all():
        missing.append("Some relative deviances are unavailable")
    elif result["profile_max_implied_improvement"] > LL_TOL:
        issues.append("A profile point improves on the stored reference deviance")

    absolute = ["statsmodels_selected_log_likelihood", "reference_log_likelihood", "profile_log_likelihood"]
    if set(absolute).issubset(profile) and np.isfinite(profile[absolute]).all().all():
        source_ll = profile[absolute[0]]
        reference_ll = profile[absolute[1]]
        points_ll = profile[absolute[2]]
        errors = (profile.relative_deviance - 2 * (reference_ll - points_ll)).abs()
        result.update(
            profile_link_loglik_delta=float(selected[absolute[0]] - reported_loglik),
            profile_reference_loglik_delta=float(selected[absolute[1]] - reported_loglik),
            profile_selected_loglik_delta=float(selected[absolute[2]] - reported_loglik),
            profile_max_loglik_improvement=float(points_ll.max() - selected[absolute[1]]),
            profile_max_improvement_vs_reported=float(points_ll.max() - reported_loglik),
            profile_max_deviance_error=float(errors.max()),
        )
        result["profile_reference_agrees"] = bool(
            np.isfinite(reported_loglik) and (source_ll - reported_loglik).abs().max() <= LL_TOL
            and (reference_ll - reported_loglik).abs().max() <= LL_TOL
            and abs(result["profile_selected_loglik_delta"]) <= LL_TOL
            and agrees(reference_ll, selected[absolute[1]], rtol=0)
        )
        if not result["profile_reference_agrees"]:
            issues.append("Profile likelihood reference differs from the reported fit")
        if result["profile_max_loglik_improvement"] > LL_TOL or result["profile_max_improvement_vs_reported"] > LL_TOL:
            issues.append("A saved profile point has a higher REML likelihood")
        if not agrees(errors, 0, rtol=0):
            issues.append("Stored relative deviances disagree with the absolute likelihoods")
    else:
        result["profile_reference_agrees"] = pd.NA
        missing.append("Absolute profile likelihood comparison unavailable")

    if evaluated is None or comparison is None:
        missing.append("Saved evaluated profile or selected-point capacity comparison unavailable")
    else:
        ev_selected = one(evaluated.loc[evaluated.path_direction.eq("selected")], "evaluated selected point")
        ev_boundary = one(evaluated.loc[true(evaluated.exact_PSD_boundary)], "evaluated boundary")
        delta = comparison.set_index("quantity", verify_integrity=True).loc["median_sqrt_kappa"]
        result.update(profile_selected_capacity_delta=delta.profile_minus_original,
                      profile_boundary_capacity=ev_boundary.median_sqrt_kappa,
                      profile_boundary_slope_variance=ev_boundary.random_slope_variance)
        if (not agrees(ev_selected.relative_deviance, selected.relative_deviance) or
                not agrees(ev_boundary.relative_deviance, boundary.relative_deviance) or
                not agrees(ev_boundary.d_0_5, boundary.d_0_5) or
                (pd.notna(covariance_agrees) and not agrees(ev_selected[cov_fields],
                                      [parameters[k] for k in cov_fields], PROFILE_PARAMETER_RTOL)) or
                (pd.notna(covariance_agrees) and not agrees(ev_boundary[FIT], boundary_components,
                                                            PROFILE_PARAMETER_RTOL)) or
                not agrees(delta.original_fit, observed.median_sqrt_kappa) or
                not agrees(delta.selected_profile, ev_selected.median_sqrt_kappa) or
                not agrees(delta.profile_minus_original, delta.selected_profile - delta.original_fit) or
                abs(delta.profile_minus_original) > capacity.OBSERVED_TOL or
                ev_boundary.random_slope_variance <= 0 or
                not agrees(ev_boundary.median_sqrt_kappa, 0)):
            issues.append("Saved capacity comparison or rank-one endpoint is inconsistent")
    return result, issues, missing


def candidate_consequences(panel, original, schedules, schedule_error=None):
    original_components = original[["G00", "tau2", "cov01", "sigma2"]].to_numpy(float)
    output = []
    for row in panel.to_dict("records"):
        delta = row["delta_loglik"]
        if capacity.is_true(row["selected"]) or not np.isfinite(delta) or delta < -LL_TOL:
            continue
        params = np.array([row[f"matched_{k}"] for k in FIT], float)
        has_params = np.isfinite(params).all()
        distinct = has_params and not agrees(params, original_components)
        if not distinct and has_params and delta <= LL_TOL:
            continue
        if not row["admissible"] and delta <= LL_TOL:
            continue
        details = {
            "outcome": row["outcome"], "optimizer": row["optimizer"], "delta_loglik": delta,
            "comparison": "higher saved likelihood" if delta > LL_TOL else "similar likelihood",
            "admissible": row["admissible"],
            "quantities_result": "incomplete",
            "quantities_status": "covariance unavailable",
            "reported_d_0_5": anchored_d(*original_components[:3]),
            "candidate_d_0_5": anchored_d(*params[:3]) if has_params else np.nan,
        }
        for k, before, after in zip(FIT, original_components, params):
            details.update({f"reported_{k}": before, f"candidate_{k}": after})
        if has_params and schedules is None:
            details["quantities_status"] = schedule_error or "original schedules needed"
        elif has_params:
            try:
                status, d, metrics = capacity.evaluate(schedules, *params)
                values = capacity.aggregate(metrics)
                if status == "undefined_zero_target_variance":
                    details["quantities_result"] = "undefined"
                elif np.isfinite(list(values.values())).all():
                    details["quantities_result"] = "evaluated"
                else:
                    raise ValueError("nonfinite decomposition quantities")
                details["quantities_status"] = status
                if np.isfinite(d):
                    details["candidate_d_0_5"] = d
                for k, value in values.items():
                    details.update({f"reported_{k}": original[k], f"candidate_{k}": value,
                                    f"delta_{k}": value - original[k]})
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                details["quantities_status"] = f"evaluation failed: {exc}"
        output.append(details)
    return output


def comparison_summary(details, summary, missing):
    by_status = {}
    for row in details:
        if row["quantities_result"] == "incomplete":
            by_status.setdefault(row["quantities_status"], []).append(row["optimizer"])
    for reason, optimizers in by_status.items():
        missing.append(f"Candidate decomposition comparisons incomplete: {reason} ({', '.join(optimizers)})")
    undefined = sum(row["quantities_result"] == "undefined" for row in details)
    summary.update(
        candidate_comparisons=len(details),
        candidate_quantities_evaluated=sum(row["quantities_result"] == "evaluated" for row in details),
        candidate_quantities_undefined_zero_target_variance=undefined,
        candidate_comparisons_incomplete=sum(map(len, by_status.values())),
    )


def assessment(issues, missing):
    return FOLLOW_UP if issues else INSUFFICIENT if missing else CORROBORATED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--stability-results", type=Path, help="Saved s7_aabc2_kappa_stability run.")
    parser.add_argument("--input-config", type=Path, help="AABC2 data configuration for candidate comparisons.")
    parser.add_argument("--exact-reliability", type=Path, help="Restricted Figure 4 reliability CSV.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.input_config) != bool(args.exact_reliability):
        parser.error("--input-config and --exact-reliability are required together")

    repo = args.repo.expanduser()
    root = repo / "results/frozen_summaries/aabc2"
    figure4 = repo / capacity.FIGURE4_DIR
    summaries, panels, changes = [], [], []

    def optional(path, columns=(), **kwargs):
        try:
            return read(path, columns, **kwargs)
        except (FileNotFoundError, KeyError, pd.errors.EmptyDataError) as exc:
            print(exc, file=sys.stderr)
            return None

    reliable = optional(root / "reliability_model_fit_audit.csv", [
        "outcome", "model_specification", "optimizer", "selected", "llf", "optimizer_converged",
        "fixed_design_rank", "fixed_design_columns", "fixed_design_full_rank", "fixed_parameters_finite",
        "random_effect_covariance_psd", "residual_variance", "slope_variance", "participants",
        "observations", "optimizer_agreement",
    ])
    rlrt = optional(root / "observed_rlrt_optimizer_audit.csv", [
        "outcome", "model", "optimizer", "estimation", "log_likelihood", "fixed_design_rank",
        "fixed_design_columns", "fixed_effects_json", *FIT,
    ])
    profile = optional(repo / capacity.PROFILE_PATH, ["outcome", "dataset_role", "dataset_id",
                       "path_direction", "exact_PSD_boundary", "optimizer_agreement", "d_0_5",
                       "relative_deviance"])
    dec = optional(figure4 / "s7_aabc2_decomposition.csv",
                   ["N", "G00", "tau2", "cov01", "sigma2", "mean_Rdirect", "mean_Delta_fit", "mean_D2"],
                   index_col="outcome")
    kappa = optional(figure4 / "s7_aabc2_kappa_quantiles.csv",
                     ["sqrt_kappa_q50", "mean_kappa"], index_col="outcome")
    evaluated, comparison = None, None
    if args.stability_results:
        path = args.stability_results.expanduser()
        evaluated = optional(path / "aabc2_kappa_cognition_profile.csv",
                             ["path_direction", "exact_PSD_boundary", "d_0_5", "relative_deviance",
                              "median_sqrt_kappa", *FIT])
        comparison = optional(path / "aabc2_kappa_cognition_profile_comparison.csv",
                              ["quantity", "original_fit", "selected_profile", "profile_minus_original"])

    schedules, schedule_errors = {}, {}
    if args.input_config and dec is not None and kappa is not None:
        try:
            sys.path.insert(0, str(repo / "reproduction/s7_s8_aabc2_restricted"))
            import aabc2_common as common
            exact = read(args.exact_reliability.expanduser(), dtype={"participant_id": str})
            specs = common.outcome_specs(common.build_raw_cohorts(args.input_config.expanduser()))
        except SCHEDULE_ERRORS as exc:
            schedule_errors = dict.fromkeys(OUTCOMES, f"schedule loading failed: {exc}")
        else:
            for outcome, (_, n_visits) in OUTCOMES.items():
                try:
                    data = common.prepare_outcome_data(outcome, specs)
                    if len(data) != n_visits:
                        raise ValueError("original visit count differs from the optimizer audit")
                    moments = capacity.schedule_moments(data)
                    capacity.check_observed(outcome, moments, exact, dec, kappa)
                    schedules[outcome] = moments
                except SCHEDULE_ERRORS as exc:
                    schedule_errors[outcome] = f"schedule validation failed: {exc}"

    for outcome, counts in OUTCOMES.items():
        summary, issues, missing = {"outcome": outcome}, [], []
        comparison_missing = []
        if reliable is None or dec is None or kappa is None:
            missing.append("Reference optimizer panel or Figure 4 summaries unavailable")
        else:
            try:
                original = dec.loc[outcome].copy()
                original["mean_R_direct"] = original["mean_Rdirect"]
                original["median_sqrt_kappa"] = kappa.loc[outcome, "sqrt_kappa_q50"]
                original["mean_kappa"] = kappa.loc[outcome, "mean_kappa"]
                if original.N != counts[0]:
                    raise ValueError("Figure 4 participant count differs from the reference cohort")
                summary, issues, missing, panel = optimizer_audit(outcome, reliable, rlrt, original)
                panels.append(panel)
                detail = candidate_consequences(panel, original, schedules.get(outcome),
                                                schedule_errors.get(outcome))
                comparison_summary(detail, summary, comparison_missing)
                changes.extend(detail)
                if outcome == capacity.COGNITION:
                    if profile is None:
                        missing.append("Observed cognition likelihood profile unavailable")
                    else:
                        checked, profile_issues, profile_missing = profile_audit(
                            profile, evaluated, comparison, original, summary["reported_loglik"])
                        summary.update(checked)
                        issues.extend(profile_issues)
                        missing.extend(profile_missing)
            except (KeyError, ValueError) as exc:
                (missing if isinstance(exc, KeyError) else issues).append(str(exc))
        summary["saved_fit_checks_assessment"] = assessment(issues, missing)
        missing.extend(comparison_missing)
        summary["assessment"] = assessment(issues, missing)
        summary["notes"] = " | ".join(issues + missing)
        summaries.append(summary)

    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "observed_fit_summary.csv", index=False)
    if panels:
        pd.concat(panels, ignore_index=True).to_csv(out / "observed_optimizer_candidates.csv", index=False)
    if changes:
        pd.DataFrame(changes).to_csv(out / "observed_candidate_consequences.csv", index=False)
    print(summary[["outcome", "saved_fit_checks_assessment", "assessment", "notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
