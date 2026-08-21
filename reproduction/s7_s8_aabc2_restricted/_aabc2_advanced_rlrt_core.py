#!/usr/bin/env python3
"""Validated computational core for the advanced AABC2 RLRT diagnostics.

Function bodies in this module are extracted verbatim from the notebook cells
that were independently smoke-validated during release preparation.  Runtime
data and fitted-state objects are supplied explicitly by public driver scripts.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import patsy
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from statsmodels.tools.numdiff import approx_hess

import aabc2_common as common

SEED = 20260718
ALPHA = 0.05
OPTIMIZERS = ("lbfgs", "powell", "bfgs")
MAXITER = 1200
PSD_REL_TOL = 1e-8
SLOPE_BOUNDARY_REL_TOL = 1e-8
LLF_REL_TOL = 1e-5
FIXED_AGREEMENT_SE_TOL = 0.25
ROUNDING_TOL = 1e-7
PB_INITIAL_B = 999
PB_MAX_B = 4999
PB_OUTCOMES = (
    "lateral_ventricles", "hippocampus", "global_mean_thickness",
    "global_cortical_cbf", "global_cortical_att", "fluid_cognition",
)
RLRT_USABLE_RATE_TARGET = 0.95
PROFILE1D_BOUNDARY_PLOT_VALUE = 13.0
PROFILE1D_LLF_ABS_TOL = 1e-3
PROFILE1D_FIXED_REL_TOL = 1e-5
PROFILE1D_MAXITER = 900
SIM_ANCHORED_NEAR_THRESHOLD = 0.01

OUTCOMES = None
ANALYSIS_DATA = None
OBSERVED_PANELS = {}
SPLINE_STATES = {}
SCHEDULE_TEMPLATES = {}
SIMULATION_CELL_LOOKUP = {}
simulation_manifest = pd.DataFrame()
observed_rlrt_tests = pd.DataFrame()
rlrt_bootstrap_replicates = pd.DataFrame()

def configure_analysis(outcomes, analysis_data):
    global OUTCOMES, ANALYSIS_DATA, OBSERVED_PANELS, SPLINE_STATES
    OUTCOMES = outcomes
    ANALYSIS_DATA = analysis_data
    OBSERVED_PANELS = {}
    SPLINE_STATES = {}

def fixed_formula(key):
    if OUTCOMES is None:
        raise RuntimeError("configure_analysis() must be called first")
    return common.fixed_formula(key, OUTCOMES)


# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 9
def coerce_boolean(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
    }).fillna(False).astype(bool)

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 9
def midpoint_to_anchor_covariance(random_covariance_midpoint):
    covariance_midpoint = np.asarray(random_covariance_midpoint, float)
    transform = np.array([[1.0, -0.5], [0.0, 1.0]])
    covariance_anchor = transform @ covariance_midpoint @ transform.T
    return (covariance_anchor + covariance_anchor.T) / 2

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 9
def anchored_endpoint_geometry(random_covariance_midpoint):
    covariance_anchor = midpoint_to_anchor_covariance(random_covariance_midpoint)
    variance_0 = float(covariance_anchor[0, 0])
    covariance_0s = float(covariance_anchor[0, 1])
    variance_slope = float(covariance_anchor[1, 1])
    variance_5 = variance_0 + 2 * covariance_0s + variance_slope
    covariance_05 = variance_0 + covariance_0s
    gamma_0_5 = (
        covariance_05 / np.sqrt(variance_0 * variance_5)
        if variance_0 > 0 and variance_5 > 0 else np.nan
    )
    endpoint_covariance = np.array([
        [variance_0, covariance_05], [covariance_05, variance_5]
    ])
    endpoint_eigenvalues = np.linalg.eigvalsh(
        (endpoint_covariance + endpoint_covariance.T) / 2
    )
    endpoint_trace = float(np.trace(endpoint_covariance))
    return {
        "anchor_random_level_variance": variance_0,
        "anchor_level_slope_covariance": covariance_0s,
        "five_year_random_level_variance": variance_5,
        "anchor_to_five_year_covariance": covariance_05,
        "g_0_5": covariance_05,
        "gamma_0_5": gamma_0_5,
        "absolute_gamma_0_5": abs(gamma_0_5) if np.isfinite(gamma_0_5) else np.nan,
        "one_minus_gamma_0_5_squared": 1 - gamma_0_5**2 if np.isfinite(gamma_0_5) else np.nan,
        "anchored_endpoint_minimum_eigenvalue": float(endpoint_eigenvalues.min()),
        "anchored_endpoint_minimum_eigenvalue_fraction": (
            float(endpoint_eigenvalues.min() / endpoint_trace)
            if endpoint_trace > 0 else np.nan
        ),
    }

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 11
def fixed_parameters_finite(result):
    try:
        estimates = np.asarray(result.fe_params, float)
        errors = np.asarray(result.bse_fe, float)
        return bool(
            np.isfinite(estimates).all()
            and np.isfinite(errors).all()
            and (errors >= 0).all()
        )
    except Exception:
        return False

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 11
def hessian_diagnostic(result):
    try:
        hessian = result.model.hessian(result.params_object)
        if isinstance(hessian, tuple):
            hessian = hessian[0]
        information = -(np.asarray(hessian, float) + np.asarray(hessian, float).T) / 2
        eigenvalues = np.linalg.eigvalsh(information)
        tolerance = 1e-8 * max(1.0, float(np.max(np.abs(eigenvalues))))
        return (
            bool(np.min(eigenvalues) > tolerance),
            float(np.min(eigenvalues)),
            float(np.max(eigenvalues)),
        )
    except Exception:
        return False, np.nan, np.nan

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 11
def covariance_diagnostic(result):
    random_covariance = np.asarray(result.cov_re, float)
    random_covariance = (random_covariance + random_covariance.T) / 2
    residual_variance = float(result.scale)
    eigenvalues = np.linalg.eigvalsh(random_covariance)
    tolerance = PSD_REL_TOL * max(
        1.0, float(np.max(np.abs(eigenvalues))), abs(residual_variance)
    )
    covariance_psd = bool(
        np.isfinite(random_covariance).all()
        and residual_variance > 0
        and float(eigenvalues.min()) >= -tolerance
    )
    record = {
        "random_intercept_variance": float(random_covariance[0, 0]),
        "random_covariance_minimum_eigenvalue": float(eigenvalues.min()),
        "random_covariance_psd": covariance_psd,
        "residual_variance": residual_variance,
    }
    if random_covariance.shape == (2, 2):
        variance_intercept = float(random_covariance[0, 0])
        variance_slope = float(random_covariance[1, 1])
        covariance = float(random_covariance[0, 1])
        correlation = (
            covariance / np.sqrt(variance_intercept * variance_slope)
            if variance_intercept > 0 and variance_slope > 0 else np.nan
        )
        innovation_variance = (
            variance_slope - covariance**2 / variance_intercept
            if variance_intercept > 0 else np.nan
        )
        record.update({
            "random_slope_variance": variance_slope,
            "intercept_slope_covariance": covariance,
            "intercept_slope_correlation": correlation,
            "absolute_intercept_slope_correlation": (
                abs(correlation) if np.isfinite(correlation) else np.nan
            ),
            "one_minus_rho_squared": (
                1 - correlation**2 if np.isfinite(correlation) else np.nan
            ),
            "slope_innovation_variance": innovation_variance,
            "innovation_variance_fraction_of_slope": (
                innovation_variance / variance_slope
                if variance_slope > 0 and np.isfinite(innovation_variance) else np.nan
            ),
            "random_slope_boundary": bool(
                not np.isfinite(variance_slope)
                or variance_slope <= SLOPE_BOUNDARY_REL_TOL * max(1.0, residual_variance)
            ),
            **anchored_endpoint_geometry(random_covariance),
        })
    return random_covariance, record

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 11
def results_compatible(first_result, second_result):
    if first_result is None or second_result is None:
        return False
    common = first_result.fe_params.index.intersection(second_result.fe_params.index)
    if len(common) == 0:
        return False
    first_estimate = first_result.fe_params.loc[common].to_numpy(float)
    second_estimate = second_result.fe_params.loc[common].to_numpy(float)
    standard_error_scale = np.maximum(
        first_result.bse_fe.loc[common].to_numpy(float),
        second_result.bse_fe.loc[common].to_numpy(float),
    )
    fixed_z_difference = float(np.max(
        np.abs(first_estimate - second_estimate)
        / np.maximum(standard_error_scale, 1e-8)
    ))
    likelihood_scale = max(
        1.0, abs(float(first_result.llf)), abs(float(second_result.llf))
    )
    likelihood_agrees = (
        abs(float(first_result.llf) - float(second_result.llf))
        <= LLF_REL_TOL * likelihood_scale
    )
    return bool(likelihood_agrees and fixed_z_difference <= FIXED_AGREEMENT_SE_TOL)

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 11
def fit_one_optimizer(data, formula, re_formula, optimizer, reml=True):
    caught_warnings = []
    try:
        response_matrix, fixed_matrix = patsy.dmatrices(
            formula, data, return_type="dataframe"
        )
        fixed_rank = int(np.linalg.matrix_rank(fixed_matrix.to_numpy(float)))
        fixed_columns = int(fixed_matrix.shape[1])
        model = smf.mixedlm(
            formula, data=data, groups=data["pid"], re_formula=re_formula
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings(
                "ignore", category=FutureWarning, module=r"statsmodels\..*"
            )
            result = model.fit(
                reml=reml, method=optimizer, maxiter=MAXITER, disp=False
            )
            caught_warnings = [str(item.message) for item in caught]
        random_covariance, covariance_record = covariance_diagnostic(result)
        hessian_pd, hessian_minimum, hessian_maximum = hessian_diagnostic(result)
        basic_valid = bool(
            result.converged
            and np.isfinite(result.llf)
            and fixed_rank == fixed_columns
            and fixed_parameters_finite(result)
            and covariance_record["random_covariance_psd"]
        )
        strict_valid = bool(basic_valid and hessian_pd)
        record = {
            "optimizer": optimizer,
            "fit_succeeded": True,
            "optimizer_converged": bool(result.converged),
            "log_likelihood": float(result.llf),
            "estimation": "REML" if reml else "ML",
            "fixed_design_rank": fixed_rank,
            "fixed_design_columns": fixed_columns,
            "fixed_design_full_rank": fixed_rank == fixed_columns,
            "fixed_parameters_finite": fixed_parameters_finite(result),
            "fixed_effects_json": json.dumps({
                str(name): float(value) for name, value in result.fe_params.items()
            }, sort_keys=True),
            "fixed_effect_SE_json": json.dumps({
                str(name): float(value) for name, value in result.bse_fe.items()
            }, sort_keys=True),
            "hessian_positive_definite": hessian_pd,
            "hessian_minimum_information_eigenvalue": hessian_minimum,
            "hessian_maximum_information_eigenvalue": hessian_maximum,
            "basic_numerical_validity": basic_valid,
            "strict_numerical_validity": strict_valid,
            "warning_count": len(caught_warnings),
            "warnings": " | ".join(sorted(set(caught_warnings))),
            "error": "",
            **covariance_record,
        }
        return result, record
    except Exception as exc:
        return None, {
            "optimizer": optimizer,
            "fit_succeeded": False,
            "optimizer_converged": False,
            "log_likelihood": np.nan,
            "estimation": "REML" if reml else "ML",
            "fixed_design_rank": np.nan,
            "fixed_design_columns": np.nan,
            "fixed_design_full_rank": False,
            "fixed_parameters_finite": False,
            "fixed_effects_json": "{}",
            "fixed_effect_SE_json": "{}",
            "hessian_positive_definite": False,
            "hessian_minimum_information_eigenvalue": np.nan,
            "hessian_maximum_information_eigenvalue": np.nan,
            "basic_numerical_validity": False,
            "strict_numerical_validity": False,
            "random_covariance_psd": False,
            "warning_count": len(caught_warnings),
            "warnings": " | ".join(sorted(set(caught_warnings))),
            "error": repr(exc),
        }

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 11
def fit_optimizer_panel(
    data, formula, re_formula, reml=True, stop_after_strict_pair=False,
):
    fitted_results = {}
    records = []
    for optimizer in OPTIMIZERS:
        result, record = fit_one_optimizer(
            data, formula, re_formula, optimizer, reml=reml
        )
        fitted_results[optimizer] = result
        records.append(record)
        strict_optimizers = [
            item["optimizer"] for item in records
            if item["strict_numerical_validity"]
        ]
        strict_pair_found = any(
            results_compatible(fitted_results[first], fitted_results[second])
            for position, first in enumerate(strict_optimizers)
            for second in strict_optimizers[position + 1:]
        )
        if stop_after_strict_pair and strict_pair_found:
            break

    basic_eligible = {
        row["optimizer"]: bool(row["basic_numerical_validity"])
        for row in records
    }
    strict_eligible = {
        row["optimizer"]: bool(row["strict_numerical_validity"])
        for row in records
    }
    basic_counts, strict_counts = {}, {}
    for optimizer in fitted_results:
        basic_counts[optimizer] = (
            sum(
                basic_eligible.get(other, False)
                and results_compatible(fitted_results[optimizer], fitted_results[other])
                for other in fitted_results
            ) if basic_eligible.get(optimizer, False) else 0
        )
        strict_counts[optimizer] = (
            sum(
                strict_eligible.get(other, False)
                and results_compatible(fitted_results[optimizer], fitted_results[other])
                for other in fitted_results
            ) if strict_eligible.get(optimizer, False) else 0
        )
    for row in records:
        optimizer = row["optimizer"]
        row["basic_common_solution_count"] = int(basic_counts[optimizer])
        row["strict_common_solution_count"] = int(strict_counts[optimizer])
        row["selected_panel_solution"] = False

    strict_candidates = [
        row for row in records if strict_counts[row["optimizer"]] >= 2
    ]
    basic_candidates = [
        row for row in records if basic_counts[row["optimizer"]] >= 2
    ]
    fallback_candidates = [
        row for row in records if row["basic_numerical_validity"]
    ]
    candidate_pool = strict_candidates or basic_candidates or fallback_candidates
    selected_record = None
    selected_result = None
    if candidate_pool:
        selected_record = max(
            candidate_pool,
            key=lambda row: (
                strict_counts[row["optimizer"]],
                basic_counts[row["optimizer"]],
                row["log_likelihood"],
            ),
        )
        selected_record["selected_panel_solution"] = True
        selected_result = fitted_results[selected_record["optimizer"]]
    return {
        "result": selected_result,
        "selected_record": selected_record,
        "audit": pd.DataFrame(records),
        "attempted_optimizers": len(records),
        "basic_panel_valid": bool(basic_candidates),
        "strict_panel_valid": bool(strict_candidates),
        "maximum_basic_agreement": max(basic_counts.values(), default=0),
        "maximum_strict_agreement": max(strict_counts.values(), default=0),
    }

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 11
def clamp_probability(value):
    if not np.isfinite(value):
        return np.nan
    if -ROUNDING_TOL <= value <= 1 + ROUNDING_TOL:
        return float(np.clip(value, 0, 1))
    return np.nan

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 11
def participant_predictability_metrics(
    random_covariance_midpoint, residual_variance, data, residual_variances=None,
):
    random_covariance_midpoint = np.asarray(random_covariance_midpoint, float)
    eigenvalues = np.linalg.eigvalsh(
        (random_covariance_midpoint + random_covariance_midpoint.T) / 2
    )
    if eigenvalues.min() <= 0 or random_covariance_midpoint[1, 1] <= 0:
        return pd.DataFrame()
    inverse_random = np.linalg.inv(random_covariance_midpoint)
    intercept_variance = float(random_covariance_midpoint[0, 0])
    slope_variance = float(random_covariance_midpoint[1, 1])
    covariance = float(random_covariance_midpoint[0, 1])
    innovation_loading = covariance / intercept_variance
    innovation_variance = slope_variance - covariance**2 / intercept_variance
    rows = []
    for participant_id, group in data.groupby("pid", sort=False):
        times = group.t5_mid.to_numpy(float)
        design = np.column_stack([np.ones(len(group)), times])
        if residual_variances is None:
            observation_variances = np.repeat(float(residual_variance), len(group))
        else:
            observation_variances = np.asarray(
                residual_variances.loc[group.index], float
            )
        weights = 1 / observation_variances
        weighted_mean_time = float(np.sum(weights * times) / np.sum(weights))
        weighted_information = float(
            np.sum(weights * (times - weighted_mean_time)**2)
        )
        direct = (
            slope_variance / (slope_variance + 1 / weighted_information)
            if weighted_information > 0 else np.nan
        )
        posterior_covariance = np.linalg.inv(
            inverse_random + design.T @ (weights[:, None] * design)
        )
        posterior_slope_variance = float(posterior_covariance[1, 1])
        full = clamp_probability(1 - posterior_slope_variance / slope_variance)
        innovation = np.nan
        if innovation_variance > 0:
            loading = np.array([-innovation_loading, 1.0])
            posterior_innovation_variance = float(
                loading @ posterior_covariance @ loading
            )
            innovation = clamp_probability(
                1 - posterior_innovation_variance / innovation_variance
            )
        rows.append({
            "pid": str(participant_id),
            "direct_reliability": direct,
            "full_predictability": full,
            "innovation_reliability": innovation,
            "gap": full - direct if np.isfinite(full) and np.isfinite(direct) else np.nan,
        })
    return pd.DataFrame(rows)

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 11
def summarize_panel_metrics(panel, data):
    if panel["result"] is None:
        return {
            "random_slope_variance": np.nan, "median_direct": np.nan,
            "median_full": np.nan, "median_innovation": np.nan,
            "median_gap": np.nan, "g_0_5": np.nan, "gamma_0_5": np.nan,
        }
    result = panel["result"]
    random_covariance = np.asarray(result.cov_re, float)
    covariance_record = covariance_diagnostic(result)[1]
    participant_table = participant_predictability_metrics(
        random_covariance, float(result.scale), data
    )
    return {
        "random_slope_variance": covariance_record.get("random_slope_variance", np.nan),
        "median_direct": float(participant_table.direct_reliability.median()) if len(participant_table) else np.nan,
        "median_full": float(participant_table.full_predictability.median()) if len(participant_table) else np.nan,
        "median_innovation": float(participant_table.innovation_reliability.median()) if len(participant_table) else np.nan,
        "median_gap": float(participant_table.gap.median()) if len(participant_table) else np.nan,
        "g_0_5": covariance_record.get("g_0_5", np.nan),
        "gamma_0_5": covariance_record.get("gamma_0_5", np.nan),
    }

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 11
def reconstruct_bootstrap_dataset(outcome, seed):
    rng = np.random.default_rng(int(seed))
    source = ANALYSIS_DATA[outcome]
    participant_ids = source.pid.unique()
    sampled_ids = rng.choice(participant_ids, size=len(participant_ids), replace=True)
    pieces = []
    for draw_index, original_id in enumerate(sampled_ids):
        participant = source.loc[source.pid.eq(original_id)].copy()
        participant["original_participant_id"] = str(original_id)
        participant["bootstrap_draw_index"] = int(draw_index)
        participant["pid"] = f"boot_{draw_index:05d}"
        pieces.append(participant)
    reconstructed = pd.concat(pieces, ignore_index=True)
    assert reconstructed.pid.nunique() == len(participant_ids)
    return reconstructed

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 14
def simulate_one_null_rlrt(outcome, replicate):
    seed = SEED + 31_000_000 + 100_000 * PB_OUTCOMES.index(outcome) + int(replicate)
    rng = np.random.default_rng(seed)
    source = ANALYSIS_DATA[outcome]
    formula = fixed_formula(outcome)
    fitted_null = OBSERVED_PANELS[(outcome, "null")]["result"]
    if fitted_null is None:
        return {
            "outcome": outcome, "replicate": int(replicate), "seed": seed,
            "basic_pair_valid": False, "strict_pair_valid": False,
            "restricted_LR": np.nan, "error": "observed null fit unavailable",
        }
    data = source.copy()
    fixed_mean = np.asarray(
        fitted_null.model.exog @ np.asarray(fitted_null.fe_params), float
    )
    random_intercept_variance = max(float(np.asarray(fitted_null.cov_re)[0, 0]), 0.0)
    residual_variance = float(fitted_null.scale)
    participant_ids = data.pid.unique()
    random_intercepts = dict(zip(
        participant_ids,
        rng.normal(0, np.sqrt(random_intercept_variance), len(participant_ids)),
    ))
    data["y"] = (
        fixed_mean
        + data.pid.map(random_intercepts).to_numpy(float)
        + rng.normal(0, np.sqrt(residual_variance), len(data))
    )
    null_panel = fit_optimizer_panel(
        data, formula, "1", reml=True, stop_after_strict_pair=True
    )
    alternative_panel = fit_optimizer_panel(
        data, formula, "1+t5_mid", reml=True, stop_after_strict_pair=True
    )
    basic_pair_valid = bool(
        null_panel["basic_panel_valid"] and alternative_panel["basic_panel_valid"]
    )
    strict_pair_valid = bool(
        null_panel["strict_panel_valid"] and alternative_panel["strict_panel_valid"]
    )
    restricted_lr = (
        max(0.0, 2 * (
            float(alternative_panel["result"].llf) - float(null_panel["result"].llf)
        )) if basic_pair_valid else np.nan
    )
    return {
        "outcome": outcome,
        "replicate": int(replicate),
        "seed": seed,
        "basic_pair_valid": basic_pair_valid,
        "strict_pair_valid": strict_pair_valid,
        "restricted_LR": restricted_lr,
        "null_optimizers_attempted": null_panel["attempted_optimizers"],
        "alternative_optimizers_attempted": alternative_panel["attempted_optimizers"],
        "null_maximum_basic_agreement": null_panel["maximum_basic_agreement"],
        "alternative_maximum_basic_agreement": alternative_panel["maximum_basic_agreement"],
        "null_maximum_strict_agreement": null_panel["maximum_strict_agreement"],
        "alternative_maximum_strict_agreement": alternative_panel["maximum_strict_agreement"],
        "alternative_slope_variance": (
            alternative_panel["selected_record"].get("random_slope_variance", np.nan)
            if alternative_panel["selected_record"] is not None else np.nan
        ),
        "alternative_hessian_minimum": (
            alternative_panel["selected_record"].get(
                "hessian_minimum_information_eigenvalue", np.nan
            ) if alternative_panel["selected_record"] is not None else np.nan
        ),
        "error": "",
    }

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 14
def exact_binomial_interval(successes, trials, confidence=0.95):
    if trials <= 0:
        return np.nan, np.nan
    tail = (1 - confidence) / 2
    lower = 0.0 if successes == 0 else stats.beta.ppf(
        tail, successes, trials - successes + 1
    )
    upper = 1.0 if successes == trials else stats.beta.ppf(
        1 - tail, successes + 1, trials - successes
    )
    return float(lower), float(upper)

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 17
def prepare_reml_problem(data, formula):
    response, fixed = patsy.dmatrices(formula, data, return_type="dataframe")
    aligned = data.loc[response.index].copy()
    y_values = response.iloc[:, 0].to_numpy(float)
    x_values = fixed.to_numpy(float)
    clusters = []
    for _, group in aligned.groupby("pid", sort=False):
        positions = aligned.index.get_indexer(group.index)
        if np.any(positions < 0):
            raise RuntimeError("Cluster index alignment failed.")
        clusters.append({
            "positions": positions,
            "time_anchor": group.t5.to_numpy(float),
            "time_midpoint": group.t5_mid.to_numpy(float),
        })
    return {
        "data": aligned,
        "y": y_values,
        "X": x_values,
        "fixed_names": list(fixed.columns),
        "clusters": clusters,
        "n": len(y_values),
        "p": x_values.shape[1],
    }

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 17
def direct_reml_loglike(problem, random_covariance_anchor, residual_variances):
    y_values = problem["y"]
    x_values = problem["X"]
    residual_variances = np.asarray(residual_variances, float)
    if (
        np.any(~np.isfinite(random_covariance_anchor))
        or np.linalg.eigvalsh(random_covariance_anchor).min() <= 0
        or np.any(~np.isfinite(residual_variances))
        or np.any(residual_variances <= 0)
    ):
        return -np.inf, None
    xt_vinv_x = np.zeros((problem["p"], problem["p"]))
    xt_vinv_y = np.zeros(problem["p"])
    y_vinv_y = 0.0
    logdet_v = 0.0
    try:
        for cluster in problem["clusters"]:
            positions = cluster["positions"]
            z_matrix = np.column_stack([
                np.ones(len(positions)), cluster["time_anchor"]
            ])
            covariance = (
                z_matrix @ random_covariance_anchor @ z_matrix.T
                + np.diag(residual_variances[positions])
            )
            factor = cho_factor(covariance, lower=True, check_finite=False)
            logdet_v += 2 * np.log(np.diag(factor[0])).sum()
            x_cluster = x_values[positions]
            y_cluster = y_values[positions]
            vinv_x = cho_solve(factor, x_cluster, check_finite=False)
            vinv_y = cho_solve(factor, y_cluster, check_finite=False)
            xt_vinv_x += x_cluster.T @ vinv_x
            xt_vinv_y += x_cluster.T @ vinv_y
            y_vinv_y += float(y_cluster @ vinv_y)
        fixed_factor = cho_factor(xt_vinv_x, lower=True, check_finite=False)
        beta_gls = cho_solve(fixed_factor, xt_vinv_y, check_finite=False)
        logdet_fixed = 2 * np.log(np.diag(fixed_factor[0])).sum()
        residual_quadratic = y_vinv_y - float(xt_vinv_y @ beta_gls)
        restricted_loglike = -0.5 * (
            (problem["n"] - problem["p"]) * np.log(2 * np.pi)
            + logdet_v + logdet_fixed + residual_quadratic
        )
        return float(restricted_loglike), beta_gls
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return -np.inf, None

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 17
def random_covariance_midpoint_to_anchor(random_covariance_midpoint):
    return midpoint_to_anchor_covariance(random_covariance_midpoint)

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 17
def endpoint_ratio_from_anchor_covariance(random_covariance_anchor):
    variance_0 = float(random_covariance_anchor[0, 0])
    variance_5 = float(
        random_covariance_anchor[0, 0]
        + 2 * random_covariance_anchor[0, 1]
        + random_covariance_anchor[1, 1]
    )
    return 0.5 * np.log(variance_5 / variance_0)

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 20
def theta_from_midpoint_covariance(random_covariance, residual_variance):
    variance_intercept = float(random_covariance[0, 0])
    variance_slope = float(random_covariance[1, 1])
    correlation = float(
        random_covariance[0, 1] / np.sqrt(variance_intercept * variance_slope)
    )
    return np.array([
        0.5 * np.log(variance_intercept),
        0.5 * np.log(variance_slope),
        np.arctanh(np.clip(correlation, -0.999999, 0.999999)),
        0.5 * np.log(residual_variance),
    ])

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 20
def midpoint_covariance_from_theta(theta):
    sd_intercept = np.exp(theta[0])
    sd_slope = np.exp(theta[1])
    correlation = np.tanh(theta[2])
    covariance = correlation * sd_intercept * sd_slope
    return np.array([
        [sd_intercept**2, covariance],
        [covariance, sd_slope**2],
    ]), np.exp(2 * theta[3])

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 20
def fixed_beta_cluster_loglike(theta, group, beta, fixed_matrix):
    random_covariance, residual_variance = midpoint_covariance_from_theta(theta)
    design = np.column_stack([
        np.ones(len(group)), group.t5_mid.to_numpy(float)
    ])
    covariance = design @ random_covariance @ design.T + residual_variance * np.eye(len(group))
    residual = group.y.to_numpy(float) - fixed_matrix @ beta
    try:
        factor = cho_factor(covariance, lower=True, check_finite=False)
        return float(-0.5 * (
            2 * np.log(np.diag(factor[0])).sum()
            + residual @ cho_solve(factor, residual, check_finite=False)
            + len(group) * np.log(2 * np.pi)
        ))
    except np.linalg.LinAlgError:
        return -1e100

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 20
def numerical_score(function, theta, step=1e-4):
    score = np.zeros(len(theta))
    for index in range(len(theta)):
        increment = np.zeros(len(theta))
        increment[index] = step
        score[index] = (function(theta + increment) - function(theta - increment)) / (2 * step)
    return score

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 20
def cluster_covariance_influence(data, formula, fitted_result, outcome):
    response, fixed = patsy.dmatrices(formula, data, return_type="dataframe")
    aligned = data.loc[response.index].copy()
    fixed_values = fixed.to_numpy(float)
    beta = fitted_result.fe_params.loc[fixed.columns].to_numpy(float)
    theta = theta_from_midpoint_covariance(
        np.asarray(fitted_result.cov_re, float), float(fitted_result.scale)
    )
    group_records = []
    group_functions = []
    for participant_id, group in aligned.groupby("pid", sort=False):
        positions = aligned.index.get_indexer(group.index)
        fixed_group = fixed_values[positions]
        function = lambda parameters, group=group, fixed_group=fixed_group: fixed_beta_cluster_loglike(
            parameters, group, beta, fixed_group
        )
        score = numerical_score(function, theta)
        group_functions.append(function)
        group_records.append({
            "outcome": outcome, "pid": str(participant_id),
            "score_log_intercept_sd": score[0],
            "score_log_slope_sd": score[1],
            "score_atanh_rho": score[2],
            "score_log_residual_sd": score[3],
        })
    scores = np.array([[
        row["score_log_intercept_sd"], row["score_log_slope_sd"],
        row["score_atanh_rho"], row["score_log_residual_sd"],
    ] for row in group_records])
    centered_scores = scores - scores.mean(axis=0, keepdims=True)

    def total_loglike(parameters):
        return sum(function(parameters) for function in group_functions)

    information = -np.asarray(approx_hess(theta, total_loglike), float)
    information = (information + information.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    floor = max(1e-6, 1e-6 * np.max(np.abs(eigenvalues)))
    information_regularized = eigenvectors @ np.diag(
        np.maximum(eigenvalues, floor)
    ) @ eigenvectors.T
    for position, row in enumerate(group_records):
        deletion_delta = -np.linalg.solve(
            information_regularized, centered_scores[position]
        )
        predicted_theta = theta + deletion_delta
        predicted_covariance, _ = midpoint_covariance_from_theta(predicted_theta)
        predicted_geometry = anchored_endpoint_geometry(predicted_covariance)
        row.update({
            "approx_delete_delta_log_intercept_sd": deletion_delta[0],
            "approx_delete_delta_log_slope_sd": deletion_delta[1],
            "approx_delete_delta_atanh_rho": deletion_delta[2],
            "approx_delete_delta_log_residual_sd": deletion_delta[3],
            "influence_norm": float(np.sqrt(
                deletion_delta @ information_regularized @ deletion_delta
            )),
            "approx_deleted_slope_variance": float(predicted_covariance[1, 1]),
            "approx_deleted_g_0_5": predicted_geometry["g_0_5"],
            "approx_deleted_gamma_0_5": predicted_geometry["gamma_0_5"],
            "information_minimum_eigenvalue_before_regularization": float(eigenvalues.min()),
        })
    return pd.DataFrame(group_records).sort_values(
        "influence_norm", ascending=False
    ).reset_index(drop=True)

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 22
def attach_frozen_spline_basis(data, outcome):
    state = SPLINE_STATES[outcome]
    raw_basis = np.asarray(patsy.build_design_matrices(
        [state["design_info"]], data, return_type="dataframe"
    )[0], float)
    linear_design = np.column_stack([
        np.ones(len(data)), data.t5_mid.to_numpy(float)
    ])
    residual_basis = (
        raw_basis - linear_design @ state["projection_coefficients"]
    )
    nonlinear = residual_basis @ state["nonlinear_loadings"]
    result = data.copy()
    result["time_spline_nonlinear_1"] = nonlinear[:, 0]
    result["time_spline_nonlinear_2"] = nonlinear[:, 1]
    return result

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 22
def spline_time_formula(outcome):
    return (
        fixed_formula(outcome)
        + " + time_spline_nonlinear_1 + time_spline_nonlinear_2"
    )

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 22
def anchor_covariance_from_cholesky_parameters(parameters):
    lower = np.array([
        [np.exp(parameters[0]), 0.0],
        [parameters[1], np.exp(parameters[2])],
    ])
    covariance = lower @ lower.T
    return (covariance + covariance.T) / 2

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 22
def fit_first_later_residual_model(data, formula, initial_panel):
    ordered = data.sort_values(["pid", "t5"]).copy()
    ordered["visit_number"] = ordered.groupby("pid").cumcount()
    problem = prepare_reml_problem(ordered, formula)
    aligned = problem["data"]
    first_visit = aligned.groupby("pid").cumcount().eq(0).to_numpy()
    if initial_panel["result"] is None:
        return {
            "fit_succeeded": False, "optimizer_success": False,
            "error": "homoscedastic initial panel unavailable",
        }, pd.DataFrame()
    initial_result = initial_panel["result"]
    initial_anchor = midpoint_to_anchor_covariance(
        np.asarray(initial_result.cov_re, float)
    )
    try:
        initial_lower = np.linalg.cholesky(initial_anchor)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(initial_anchor)
        initial_anchor = eigenvectors @ np.diag(
            np.maximum(eigenvalues, 1e-8)
        ) @ eigenvectors.T
        initial_lower = np.linalg.cholesky(initial_anchor)
    initial = np.array([
        np.log(initial_lower[0, 0]), initial_lower[1, 0],
        np.log(initial_lower[1, 1]),
        0.5 * np.log(float(initial_result.scale)),
        0.5 * np.log(float(initial_result.scale)),
    ])

    def objective(parameters):
        random_covariance_anchor = anchor_covariance_from_cholesky_parameters(
            parameters
        )
        residual_variances = np.where(
            first_visit, np.exp(2 * parameters[3]), np.exp(2 * parameters[4])
        )
        value, _ = direct_reml_loglike(
            problem, random_covariance_anchor, residual_variances
        )
        return -value if np.isfinite(value) else 1e100

    optimizer_rows = []
    fitted_objects = []
    for method in ("L-BFGS-B", "Powell"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = minimize(
                objective, initial, method=method,
                options={"maxiter": 1200, "ftol": 1e-10},
            )
        optimizer_rows.append({
            "optimizer": method,
            "success": bool(fitted.success and np.isfinite(fitted.fun)),
            "negative_restricted_log_likelihood": float(fitted.fun),
            "restricted_log_likelihood": -float(fitted.fun),
            "message": str(fitted.message),
            "iterations": int(getattr(fitted, "nit", -1)),
            "parameters_json": json.dumps([float(value) for value in fitted.x]),
        })
        fitted_objects.append(fitted)
    successful = [
        fitted for fitted in fitted_objects
        if fitted.success and np.isfinite(fitted.fun)
    ]
    if not successful:
        return {
            "fit_succeeded": False, "optimizer_success": False,
            "error": "both direct REML optimizers failed",
        }, pd.DataFrame(optimizer_rows)
    best = min(successful, key=lambda fitted: fitted.fun)
    parameters = np.asarray(best.x, float)
    random_covariance_anchor = anchor_covariance_from_cholesky_parameters(parameters)
    transform_to_midpoint = np.array([[1.0, 0.5], [0.0, 1.0]])
    random_covariance_midpoint = (
        transform_to_midpoint @ random_covariance_anchor @ transform_to_midpoint.T
    )
    residual_first = float(np.exp(2 * parameters[3]))
    residual_later = float(np.exp(2 * parameters[4]))
    residual_series = pd.Series(
        np.where(first_visit, residual_first, residual_later),
        index=aligned.index,
    )
    participant_table = participant_predictability_metrics(
        random_covariance_midpoint, np.nan, aligned,
        residual_variances=residual_series,
    )
    hessian = np.asarray(approx_hess(parameters, objective), float)
    hessian = (hessian + hessian.T) / 2
    hessian_eigenvalues = np.linalg.eigvalsh(hessian)
    anchored_geometry = anchored_endpoint_geometry(random_covariance_midpoint)
    summary = {
        "fit_succeeded": True,
        "optimizer_success": True,
        "optimizer": optimizer_rows[int(np.argmin([row["negative_restricted_log_likelihood"] for row in optimizer_rows]))]["optimizer"],
        "restricted_log_likelihood": -float(best.fun),
        "hessian_positive_definite": bool(hessian_eigenvalues.min() > 1e-8),
        "hessian_minimum_information_eigenvalue": float(hessian_eigenvalues.min()),
        "random_intercept_variance_midpoint": float(random_covariance_midpoint[0, 0]),
        "random_slope_variance": float(random_covariance_midpoint[1, 1]),
        "intercept_slope_covariance_midpoint": float(random_covariance_midpoint[0, 1]),
        "residual_variance_first_visit": residual_first,
        "residual_variance_later_visit": residual_later,
        "later_to_first_residual_variance_ratio": residual_later / residual_first,
        "median_direct": float(participant_table.direct_reliability.median()),
        "median_full": float(participant_table.full_predictability.median()),
        "median_innovation": float(participant_table.innovation_reliability.median()),
        "median_gap": float(participant_table.gap.median()),
        "g_0_5": anchored_geometry["g_0_5"],
        "gamma_0_5": anchored_geometry["gamma_0_5"],
        "error": "",
    }
    return summary, pd.DataFrame(optimizer_rows)

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 22
def fit_adequacy_dataset(task):
    outcome = str(task["outcome"])
    data = (
        ANALYSIS_DATA[outcome].copy()
        if task["dataset_role"] == "observed"
        else reconstruct_bootstrap_dataset(outcome, int(task["seed"]))
    )
    linear_panel = fit_optimizer_panel(
        data, fixed_formula(outcome), "1+t5_mid", reml=True,
        stop_after_strict_pair=False,
    )
    spline_data = attach_frozen_spline_basis(data, outcome)
    spline_panel = fit_optimizer_panel(
        spline_data, spline_time_formula(outcome), "1+t5_mid", reml=True,
        stop_after_strict_pair=False,
    )
    heteroscedastic_summary, heteroscedastic_optimizer_audit = (
        fit_first_later_residual_model(data, fixed_formula(outcome), linear_panel)
    )
    summary_rows = []
    optimizer_rows = []
    for model_name, panel in (
        ("linear_homoscedastic_REML", linear_panel),
        ("natural_cubic_spline_3df_homoscedastic_REML", spline_panel),
    ):
        model_data = spline_data if model_name.startswith("natural_cubic") else data
        metrics = summarize_panel_metrics(panel, model_data)
        summary_rows.append({
            **task, "model": model_name,
            "participants": data.pid.nunique(), "observations": len(data),
            "basic_panel_valid": panel["basic_panel_valid"],
            "strict_panel_valid": panel["strict_panel_valid"],
            "restricted_log_likelihood": (
                float(panel["result"].llf) if panel["result"] is not None else np.nan
            ),
            **metrics,
        })
        audit = panel["audit"].copy()
        for key, value in task.items():
            audit[key] = value
        audit["model"] = model_name
        optimizer_rows.extend(audit.to_dict("records"))
    summary_rows.append({
        **task, "model": "linear_first_vs_later_residual_REML",
        "participants": data.pid.nunique(), "observations": len(data),
        **heteroscedastic_summary,
    })
    if len(heteroscedastic_optimizer_audit):
        audit = heteroscedastic_optimizer_audit.copy()
        for key, value in task.items():
            audit[key] = value
        audit["model"] = "linear_first_vs_later_residual_REML"
        optimizer_rows.extend(audit.to_dict("records"))
    return summary_rows, optimizer_rows

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 25
def build_schedule_template(name, participant_times):
    participant_times = [np.asarray(times, float) for times in participant_times]
    flat_times = np.concatenate(participant_times)
    group_index = np.concatenate([
        np.repeat(index, len(times))
        for index, times in enumerate(participant_times)
    ])
    positions_by_length = {}
    offset = 0
    for index, times in enumerate(participant_times):
        positions = np.arange(offset, offset + len(times))
        positions_by_length.setdefault(len(times), []).append((index, positions, times))
        offset += len(times)
    by_length = {}
    for length, records in positions_by_length.items():
        positions = np.vstack([record[1] for record in records])
        times = np.vstack([record[2] for record in records])
        design = np.stack([np.ones_like(times), times], axis=2)
        by_length[length] = {
            "positions": positions, "times": times, "design": design,
        }
    information = np.array([
        np.sum((times - times.mean())**2) for times in participant_times
    ])
    return {
        "name": name,
        "participant_times": participant_times,
        "flat_times": flat_times,
        "group_index": group_index,
        "participants": len(participant_times),
        "observations": len(flat_times),
        "by_length": by_length,
        "schedule_information": information,
        "reference_information": float(np.median(information)),
    }

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 25
def schedule_predictability_summary(random_covariance, residual_variance, template):
    if random_covariance[1, 1] <= 0:
        return {
            "mean_Rdirect": 0.0, "mean_Pfull": 0.0,
            "mean_Rinnovation": 0.0, "mean_Delta": 0.0,
        }
    eigenvalues = np.linalg.eigvalsh(random_covariance)
    if eigenvalues.min() <= 0:
        return {
            "mean_Rdirect": np.nan, "mean_Pfull": np.nan,
            "mean_Rinnovation": np.nan, "mean_Delta": np.nan,
        }
    inverse_random = np.linalg.inv(random_covariance)
    variance_intercept = random_covariance[0, 0]
    variance_slope = random_covariance[1, 1]
    covariance = random_covariance[0, 1]
    innovation_variance = variance_slope - covariance**2 / variance_intercept
    direct_values, full_values, innovation_values = [], [], []
    for length, block in template["by_length"].items():
        times = block["times"]
        information = np.sum((times - times.mean(axis=1, keepdims=True))**2, axis=1)
        direct = variance_slope / (
            variance_slope + residual_variance / information
        )
        n = np.repeat(length, len(times))
        sum_time = times.sum(axis=1)
        sum_time_squared = (times**2).sum(axis=1)
        precision_00 = inverse_random[0, 0] + n / residual_variance
        precision_01 = inverse_random[0, 1] + sum_time / residual_variance
        precision_11 = inverse_random[1, 1] + sum_time_squared / residual_variance
        determinant = precision_00 * precision_11 - precision_01**2
        posterior_00 = precision_11 / determinant
        posterior_01 = -precision_01 / determinant
        posterior_11 = precision_00 / determinant
        full = 1 - posterior_11 / variance_slope
        if innovation_variance > 0:
            loading0 = -covariance / variance_intercept
            posterior_innovation = (
                loading0**2 * posterior_00
                + 2 * loading0 * posterior_01
                + posterior_11
            )
            innovation = 1 - posterior_innovation / innovation_variance
        else:
            innovation = np.repeat(np.nan, len(times))
        direct_values.append(direct)
        full_values.append(np.clip(full, 0, 1))
        innovation_values.append(np.clip(innovation, 0, 1))
    direct = np.concatenate(direct_values)
    full = np.concatenate(full_values)
    innovation = np.concatenate(innovation_values)
    return {
        "mean_Rdirect": float(np.nanmean(direct)),
        "mean_Pfull": float(np.nanmean(full)),
        "mean_Rinnovation": float(np.nanmean(innovation)),
        "mean_Delta": float(np.nanmean(full - direct)),
    }

# Source: AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb, cell 26
def generate_simulation_outcome(cell_id, replicate):
    cell = SIMULATION_CELL_LOOKUP[cell_id]
    template = SCHEDULE_TEMPLATES[cell["schedule"]]
    seed = SEED + 47_000_000 + 100_000 * simulation_manifest.index[
        simulation_manifest.cell_id.eq(cell_id)
    ][0] + int(replicate)
    rng = np.random.default_rng(seed)
    residual_variance = 1.0
    slope_variance = (
        cell["q1"] / template["reference_information"]
        if cell["q1"] > 0 else 0.0
    )
    intercept_variance = cell["q0"]
    covariance = (
        cell["rho"] * np.sqrt(intercept_variance * slope_variance)
        if slope_variance > 0 else 0.0
    )
    random_covariance = np.array([
        [intercept_variance, covariance], [covariance, slope_variance]
    ])
    random_effects = rng.multivariate_normal(
        np.zeros(2), random_covariance, size=template["participants"],
        check_valid="ignore",
    )
    times = template["flat_times"]
    group_index = template["group_index"]
    fixed_mean = 0.10 * times
    if cell["misspecification"] == "mild_curvature_max_0p25_residual_SD":
        curvature = times**2
        curvature = curvature - np.mean(curvature)
        curvature = 0.25 * curvature / np.max(np.abs(curvature))
        fixed_mean = fixed_mean + curvature
    random_mean = (
        random_effects[group_index, 0]
        + random_effects[group_index, 1] * times
    )
    errors = np.zeros(template["observations"])
    if cell["misspecification"] == "residual_AR1_rho_0p4":
        offset = 0
        for participant_times in template["participant_times"]:
            count = len(participant_times)
            orders = np.arange(count)
            correlation = 0.4 ** np.abs(orders[:, None] - orders[None, :])
            errors[offset:offset + count] = rng.multivariate_normal(
                np.zeros(count), correlation
            )
            offset += count
    else:
        error_sd = np.ones(template["observations"])
        if cell["misspecification"] == "late_visit_residual_variance_doubled":
            offset = 0
            for participant_times in template["participant_times"]:
                count = len(participant_times)
                latest = np.argmax(participant_times)
                error_sd[offset + latest] = np.sqrt(2.0)
                offset += count
        errors = rng.normal(0, error_sd)
    y_values = fixed_mean + random_mean + errors
    return y_values, seed

# Source: advanced_rlrt_repair_pass_1, cell 5
def prepare_vectorized_reml_problem(data, formula):
    response, fixed = patsy.dmatrices(formula, data, return_type="dataframe")
    aligned = data.loc[response.index].copy().reset_index(drop=True)
    y_values = response.iloc[:, 0].to_numpy(float)
    x_values = fixed.to_numpy(float)
    blocks = {}
    records_by_length = {}
    for _, group in aligned.groupby("pid", sort=False):
        positions = group.index.to_numpy(int)
        records_by_length.setdefault(len(group), []).append(positions)
    for length, records in records_by_length.items():
        positions = np.vstack(records)
        midpoint_times = aligned.loc[
            positions.ravel(), "t5_mid"
        ].to_numpy(float).reshape(positions.shape)
        blocks[length] = {
            "positions": positions,
            "random_design": np.stack(
                [np.ones_like(midpoint_times), midpoint_times], axis=2
            ),
        }
    return {
        "data": aligned,
        "y": y_values,
        "X": x_values,
        "fixed_names": list(fixed.columns),
        "blocks": blocks,
        "n": len(y_values),
        "p": x_values.shape[1],
    }

# Source: advanced_rlrt_repair_pass_1, cell 5
def vectorized_direct_reml_loglike(problem, random_covariance, residual_variance):
    random_covariance = np.asarray(random_covariance, float)
    residual_variance = float(residual_variance)
    eigenvalues = np.linalg.eigvalsh(
        (random_covariance + random_covariance.T) / 2
    )
    covariance_tolerance = 1e-10 * max(
        1.0, float(np.max(np.abs(eigenvalues))), abs(residual_variance)
    )
    if (
        residual_variance <= 0
        or not np.isfinite(residual_variance)
        or np.any(~np.isfinite(random_covariance))
        or eigenvalues.min() < -covariance_tolerance
    ):
        return -np.inf, None

    xt_vinv_x = np.zeros((problem["p"], problem["p"]))
    xt_vinv_y = np.zeros(problem["p"])
    y_vinv_y = 0.0
    logdet_v = 0.0
    try:
        for length, block in problem["blocks"].items():
            positions = block["positions"]
            z_values = block["random_design"]
            x_values = problem["X"][positions]
            y_values = problem["y"][positions]
            covariance = (
                np.einsum(
                    "mki,ij,mlj->mkl",
                    z_values, random_covariance, z_values,
                    optimize=True,
                )
                + residual_variance * np.eye(length)[None, :, :]
            )
            signs, logdets = np.linalg.slogdet(covariance)
            if np.any(signs <= 0):
                return -np.inf, None
            vinv_x = np.linalg.solve(covariance, x_values)
            vinv_y = np.linalg.solve(
                covariance, y_values[..., None]
            )[..., 0]
            xt_vinv_x += np.einsum(
                "mkp,mkq->pq", x_values, vinv_x, optimize=True
            )
            xt_vinv_y += np.einsum(
                "mkp,mk->p", x_values, vinv_y, optimize=True
            )
            y_vinv_y += float(np.einsum(
                "mk,mk->", y_values, vinv_y, optimize=True
            ))
            logdet_v += float(logdets.sum())
        fixed_factor = cho_factor(
            xt_vinv_x, lower=True, check_finite=False
        )
        beta = cho_solve(
            fixed_factor, xt_vinv_y, check_finite=False
        )
        logdet_fixed = 2 * np.log(np.diag(fixed_factor[0])).sum()
        residual_quadratic = y_vinv_y - float(xt_vinv_y @ beta)
        loglike = -0.5 * (
            (problem["n"] - problem["p"]) * np.log(2 * np.pi)
            + logdet_v + logdet_fixed + residual_quadratic
        )
        return float(loglike), beta
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return -np.inf, None

# Source: advanced_rlrt_repair_pass_1, cell 5
def unpack_direct_parameters(parameters, alternative):
    parameters = np.asarray(parameters, float)
    if alternative:
        lower = np.array([
            [np.exp(parameters[0]), 0.0],
            [parameters[1], np.exp(parameters[2])],
        ])
        return lower @ lower.T, np.exp(2 * parameters[3])
    return (
        np.array([[np.exp(2 * parameters[0]), 0.0], [0.0, 0.0]]),
        np.exp(2 * parameters[1]),
    )

# Source: advanced_rlrt_repair_pass_1, cell 5
def direct_multistarts(y_values, alternative):
    outcome_variance = max(float(np.var(y_values, ddof=1)), 1e-4)
    if not alternative:
        return [
            np.array([
                0.5 * np.log(intercept_fraction * outcome_variance),
                0.5 * np.log(residual_fraction * outcome_variance),
            ])
            for intercept_fraction, residual_fraction in (
                (0.10, 0.90), (0.50, 0.50), (0.90, 0.10)
            )
        ]

    configurations = [
        (0.50, 0.05, 0.00, 0.50),
        (0.80, 0.005, 0.00, 0.20),
        (0.50, 0.50, 0.70, 0.50),
        (0.50, 0.50, -0.70, 0.50),
        (0.25, 2.00, 0.00, 0.75),
        (0.50, 1.00, 0.95, 0.50),
        (0.50, 1.00, -0.95, 0.50),
    ]
    starts = []
    for int_fraction, slope_fraction, rho, residual_fraction in configurations:
        intercept_variance = max(int_fraction * outcome_variance, 1e-8)
        slope_variance = max(slope_fraction * outcome_variance, 1e-8)
        residual_variance = max(residual_fraction * outcome_variance, 1e-8)
        lower00 = np.sqrt(intercept_variance)
        lower10 = rho * np.sqrt(slope_variance)
        lower11 = np.sqrt(max(slope_variance * (1 - rho**2), 1e-10))
        starts.append(np.array([
            np.log(lower00), lower10, np.log(lower11),
            0.5 * np.log(residual_variance),
        ]))
    return starts

# Source: advanced_rlrt_repair_pass_1, cell 5
def direct_solution_compatible(first, second):
    if first is None or second is None:
        return False
    likelihood_scale = max(
        1.0, abs(first["log_likelihood"]), abs(second["log_likelihood"])
    )
    likelihood_agrees = abs(
        first["log_likelihood"] - second["log_likelihood"]
    ) <= 1e-6 * likelihood_scale
    beta_scale = max(
        1.0,
        float(np.max(np.abs(first["fixed_effects"]))),
        float(np.max(np.abs(second["fixed_effects"]))),
    )
    fixed_agrees = np.max(np.abs(
        first["fixed_effects"] - second["fixed_effects"]
    )) <= 1e-5 * beta_scale
    return bool(likelihood_agrees and fixed_agrees)

# Source: advanced_rlrt_repair_pass_1, cell 5
def parameter_hessian_diagnostic(parameters, objective):
    try:
        information = np.asarray(approx_hess(parameters, objective), float)
        information = (information + information.T) / 2
        eigenvalues = np.linalg.eigvalsh(information)
        tolerance = 1e-8 * max(
            1.0, float(np.max(np.abs(eigenvalues)))
        )
        return (
            bool(eigenvalues.min() > tolerance),
            float(eigenvalues.min()),
            float(eigenvalues.max()),
        )
    except Exception:
        return False, np.nan, np.nan

# Source: advanced_rlrt_repair_pass_1, cell 5
def fit_direct_multistart(problem, alternative, maxiter=700):
    starts = direct_multistarts(problem["y"], alternative)
    outcome_scale = max(float(np.std(problem["y"], ddof=1)), 1e-3)
    if alternative:
        bounds = [
            (-12, 6), (-10 * outcome_scale, 10 * outcome_scale),
            (-12, 6), (-12, 6),
        ]
    else:
        bounds = [(-12, 6), (-12, 6)]

    def objective(parameters):
        random_covariance, residual_variance = unpack_direct_parameters(
            parameters, alternative
        )
        loglike, _ = vectorized_direct_reml_loglike(
            problem, random_covariance, residual_variance
        )
        return -loglike if np.isfinite(loglike) else 1e100

    candidates = []
    for method in ("L-BFGS-B", "Powell"):
        for start_index, start in enumerate(starts):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = minimize(
                        objective, start, method=method, bounds=bounds,
                        options={"maxiter": maxiter, "ftol": 1e-10},
                    )
                random_covariance, residual_variance = unpack_direct_parameters(
                    fitted.x, alternative
                )
                loglike, fixed_effects = vectorized_direct_reml_loglike(
                    problem, random_covariance, residual_variance
                )
                succeeded = bool(
                    fitted.success and np.isfinite(loglike)
                    and fixed_effects is not None
                )
                candidates.append({
                    "method": method,
                    "start_index": int(start_index),
                    "optimizer_success": bool(fitted.success),
                    "fit_succeeded": succeeded,
                    "optimizer_message": str(fitted.message),
                    "parameters": np.asarray(fitted.x, float),
                    "log_likelihood": float(loglike),
                    "fixed_effects": (
                        np.asarray(fixed_effects, float)
                        if fixed_effects is not None
                        else np.full(problem["p"], np.nan)
                    ),
                    "random_covariance": random_covariance,
                    "residual_variance": float(residual_variance),
                })
            except Exception as exc:
                candidates.append({
                    "method": method, "start_index": int(start_index),
                    "optimizer_success": False, "fit_succeeded": False,
                    "optimizer_message": repr(exc),
                    "parameters": np.asarray(start, float),
                    "log_likelihood": -np.inf,
                    "fixed_effects": np.full(problem["p"], np.nan),
                    "random_covariance": np.full((2, 2), np.nan),
                    "residual_variance": np.nan,
                })

    method_best = {}
    for method in ("L-BFGS-B", "Powell"):
        successful = [
            candidate for candidate in candidates
            if candidate["method"] == method and candidate["fit_succeeded"]
        ]
        if successful:
            method_best[method] = max(
                successful, key=lambda candidate: candidate["log_likelihood"]
            )

    for candidate in method_best.values():
        hessian_pd, hessian_minimum, hessian_maximum = (
            parameter_hessian_diagnostic(candidate["parameters"], objective)
        )
        candidate["hessian_positive_definite"] = hessian_pd
        candidate["hessian_minimum_information_eigenvalue"] = hessian_minimum
        candidate["hessian_maximum_information_eigenvalue"] = hessian_maximum

    first = method_best.get("L-BFGS-B")
    second = method_best.get("Powell")
    optimizer_agreement = direct_solution_compatible(first, second)
    basic_panel_valid = bool(optimizer_agreement)
    strict_panel_valid = bool(
        optimizer_agreement
        and first.get("hessian_positive_definite", False)
        and second.get("hessian_positive_definite", False)
    ) if first is not None and second is not None else False
    selected = None
    if first is not None and second is not None and optimizer_agreement:
        selected = max(
            (first, second), key=lambda candidate: candidate["log_likelihood"]
        )
    elif method_best:
        selected = max(
            method_best.values(), key=lambda candidate: candidate["log_likelihood"]
        )

    audit_rows = []
    for method in ("L-BFGS-B", "Powell"):
        candidate = method_best.get(method)
        audit_rows.append({
            "method": method,
            "successful_starts": int(sum(
                item["fit_succeeded"] and item["method"] == method
                for item in candidates
            )),
            "attempted_starts": len(starts),
            "best_log_likelihood": (
                candidate["log_likelihood"] if candidate is not None else np.nan
            ),
            "best_start_index": (
                candidate["start_index"] if candidate is not None else np.nan
            ),
            "hessian_positive_definite": (
                candidate.get("hessian_positive_definite", False)
                if candidate is not None else False
            ),
            "hessian_minimum_information_eigenvalue": (
                candidate.get("hessian_minimum_information_eigenvalue", np.nan)
                if candidate is not None else np.nan
            ),
        })
    return {
        "selected": selected,
        "basic_panel_valid": basic_panel_valid,
        "strict_panel_valid": strict_panel_valid,
        "optimizer_agreement": optimizer_agreement,
        "audit": audit_rows,
    }

# Source: advanced_rlrt_repair_pass_1, cell 5
def reconstruct_null_rlrt_dataset(outcome, replicate):
    seed = (
        SEED + 31_000_000
        + 100_000 * PB_OUTCOMES.index(outcome)
        + int(replicate)
    )
    rng = np.random.default_rng(seed)
    source = ANALYSIS_DATA[outcome]
    fitted_null = OBSERVED_PANELS[(outcome, "null")]["result"]
    if fitted_null is None:
        raise RuntimeError(f"Observed null fit unavailable for {outcome}.")
    data = source.copy()
    fixed_mean = np.asarray(
        fitted_null.model.exog @ np.asarray(fitted_null.fe_params), float
    )
    intercept_variance = max(
        float(np.asarray(fitted_null.cov_re)[0, 0]), 0.0
    )
    residual_variance = float(fitted_null.scale)
    participant_ids = data.pid.unique()
    random_intercepts = dict(zip(
        participant_ids,
        rng.normal(0, np.sqrt(intercept_variance), len(participant_ids)),
    ))
    data["y"] = (
        fixed_mean
        + data.pid.map(random_intercepts).to_numpy(float)
        + rng.normal(0, np.sqrt(residual_variance), len(data))
    )
    return data, seed

# Source: advanced_rlrt_repair_pass_1, cell 5
def repair_one_ventricular_rlrt(replicate):
    outcome = "lateral_ventricles"
    try:
        data, seed = reconstruct_null_rlrt_dataset(outcome, replicate)
        problem = prepare_vectorized_reml_problem(
            data, fixed_formula(outcome)
        )
        null_fit = fit_direct_multistart(problem, alternative=False)
        alternative_fit = fit_direct_multistart(problem, alternative=True)
        null_selected = null_fit["selected"]
        alternative_selected = alternative_fit["selected"]
        basic_pair_valid = bool(
            null_fit["basic_panel_valid"]
            and alternative_fit["basic_panel_valid"]
            and null_selected is not None
            and alternative_selected is not None
        )
        strict_pair_valid = bool(
            null_fit["strict_panel_valid"]
            and alternative_fit["strict_panel_valid"]
        )
        restricted_lr = (
            max(0.0, 2 * (
                alternative_selected["log_likelihood"]
                - null_selected["log_likelihood"]
            )) if basic_pair_valid else np.nan
        )
        selected_covariance = (
            alternative_selected["random_covariance"]
            if alternative_selected is not None
            else np.full((2, 2), np.nan)
        )
        alternative_hessian_minimum = np.nan
        if alternative_fit["audit"]:
            alternative_hessian_minimum = float(np.nanmin([
                row["hessian_minimum_information_eigenvalue"]
                for row in alternative_fit["audit"]
            ]))
        return {
            "outcome": outcome,
            "replicate": int(replicate),
            "seed": int(seed),
            "basic_pair_valid": basic_pair_valid,
            "strict_pair_valid": strict_pair_valid,
            "restricted_LR": restricted_lr,
            "null_optimizers_attempted": 2,
            "alternative_optimizers_attempted": 2,
            "null_maximum_basic_agreement": (
                2 if null_fit["basic_panel_valid"] else 0
            ),
            "alternative_maximum_basic_agreement": (
                2 if alternative_fit["basic_panel_valid"] else 0
            ),
            "null_maximum_strict_agreement": (
                2 if null_fit["strict_panel_valid"] else 0
            ),
            "alternative_maximum_strict_agreement": (
                2 if alternative_fit["strict_panel_valid"] else 0
            ),
            "alternative_slope_variance": float(selected_covariance[1, 1]),
            "alternative_hessian_minimum": alternative_hessian_minimum,
            "repair_method": "multistart_vectorized_direct_REML_LBFGSB_and_Powell",
            "null_optimizer_agreement": null_fit["optimizer_agreement"],
            "alternative_optimizer_agreement": alternative_fit["optimizer_agreement"],
            "repair_error": "",
            "error": "",
        }
    except Exception as exc:
        return {
            "outcome": outcome,
            "replicate": int(replicate),
            "seed": int(
                SEED + 31_000_000
                + 100_000 * PB_OUTCOMES.index(outcome)
                + int(replicate)
            ),
            "basic_pair_valid": False,
            "strict_pair_valid": False,
            "restricted_LR": np.nan,
            "repair_method": "multistart_vectorized_direct_REML_LBFGSB_and_Powell",
            "repair_error": repr(exc),
            "error": repr(exc),
        }

# Source: advanced_rlrt_repair_pass_1, cell 9
def direct_reml_loglike_allow_psd(
    problem, random_covariance_anchor, residual_variances
):
    random_covariance_anchor = np.asarray(
        random_covariance_anchor, float
    )
    residual_variances = np.asarray(residual_variances, float)
    eigenvalues = np.linalg.eigvalsh(
        (random_covariance_anchor + random_covariance_anchor.T) / 2
    )
    tolerance = 1e-10 * max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
        float(np.max(np.abs(residual_variances))),
    )
    if (
        np.any(~np.isfinite(random_covariance_anchor))
        or eigenvalues.min() < -tolerance
        or np.any(~np.isfinite(residual_variances))
        or np.any(residual_variances <= 0)
    ):
        return -np.inf, None

    xt_vinv_x = np.zeros((problem["p"], problem["p"]))
    xt_vinv_y = np.zeros(problem["p"])
    y_vinv_y = 0.0
    logdet_v = 0.0
    try:
        for cluster in problem["clusters"]:
            positions = cluster["positions"]
            random_design = np.column_stack([
                np.ones(len(positions)), cluster["time_anchor"]
            ])
            marginal_covariance = (
                random_design
                @ random_covariance_anchor
                @ random_design.T
                + np.diag(residual_variances[positions])
            )
            # G may be PSD/singular; the marginal observation covariance
            # must remain positive definite.
            factor = cho_factor(
                marginal_covariance, lower=True, check_finite=False
            )
            logdet_v += 2 * np.log(np.diag(factor[0])).sum()
            x_cluster = problem["X"][positions]
            y_cluster = problem["y"][positions]
            vinv_x = cho_solve(factor, x_cluster, check_finite=False)
            vinv_y = cho_solve(factor, y_cluster, check_finite=False)
            xt_vinv_x += x_cluster.T @ vinv_x
            xt_vinv_y += x_cluster.T @ vinv_y
            y_vinv_y += float(y_cluster @ vinv_y)
        fixed_factor = cho_factor(
            xt_vinv_x, lower=True, check_finite=False
        )
        beta_gls = cho_solve(
            fixed_factor, xt_vinv_y, check_finite=False
        )
        logdet_fixed = 2 * np.log(np.diag(fixed_factor[0])).sum()
        residual_quadratic = y_vinv_y - float(xt_vinv_y @ beta_gls)
        restricted_loglike = -0.5 * (
            (problem["n"] - problem["p"]) * np.log(2 * np.pi)
            + logdet_v + logdet_fixed + residual_quadratic
        )
        return float(restricted_loglike), beta_gls
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return -np.inf, None

# Source: advanced_rlrt_repair_pass_1, cell 9
def endpoint_covariance_from_d(
    log_slope_sd, d_0_5, log_sd5_over_sd0, gamma_sign
):
    d_0_5 = float(np.clip(d_0_5, 0.0, 1.0))
    gamma = float(gamma_sign) * np.sqrt(max(0.0, 1.0 - d_0_5))
    slope_sd = np.exp(float(log_slope_sd))
    ratio = np.exp(float(log_sd5_over_sd0))
    slope_scale = 1 + ratio**2 - 2 * gamma * ratio
    if not np.isfinite(slope_scale) or slope_scale <= 1e-14:
        return None, gamma
    sd0 = slope_sd / np.sqrt(slope_scale)
    sd5 = ratio * sd0
    endpoint_covariance = np.array([
        [sd0**2, gamma * sd0 * sd5],
        [gamma * sd0 * sd5, sd5**2],
    ])
    endpoint_to_anchor = np.array([[1.0, 0.0], [-1.0, 1.0]])
    random_covariance_anchor = (
        endpoint_to_anchor
        @ endpoint_covariance
        @ endpoint_to_anchor.T
    )
    return (
        (random_covariance_anchor + random_covariance_anchor.T) / 2,
        gamma,
    )

# Source: advanced_rlrt_repair_pass_1, cell 14
def simulation_problem_from_template(template, y_values):
    return {
        "y": np.asarray(y_values, float),
        "X": np.column_stack([
            np.ones(template["observations"]),
            template["flat_times"],
        ]),
        "fixed_names": ["Intercept", "t5_mid"],
        "blocks": {
            length: {
                "positions": block["positions"],
                "random_design": block["design"],
            }
            for length, block in template["by_length"].items()
        },
        "n": int(template["observations"]),
        "p": 2,
    }

# Source: advanced_rlrt_repair_pass_1, cell 14
def covariance_metrics_from_matrix(random_covariance):
    random_covariance = np.asarray(random_covariance, float)
    eigenvalues = np.linalg.eigvalsh(
        (random_covariance + random_covariance.T) / 2
    )
    variance_intercept = float(random_covariance[0, 0])
    variance_slope = float(random_covariance[1, 1])
    covariance = float(random_covariance[0, 1])
    rho = (
        covariance / np.sqrt(variance_intercept * variance_slope)
        if variance_intercept > 0 and variance_slope > 0 else np.nan
    )
    innovation_variance = (
        variance_slope - covariance**2 / variance_intercept
        if variance_intercept > 0 else np.nan
    )
    geometry = anchored_endpoint_geometry(random_covariance)
    return {
        "random_intercept_variance": variance_intercept,
        "random_slope_variance": variance_slope,
        "intercept_slope_covariance": covariance,
        "intercept_slope_correlation": rho,
        "absolute_intercept_slope_correlation": (
            abs(rho) if np.isfinite(rho) else np.nan
        ),
        "covariance_minimum_eigenvalue": float(eigenvalues.min()),
        "one_minus_rho_squared": (
            1 - rho**2 if np.isfinite(rho) else np.nan
        ),
        "normalized_determinant": (
            np.linalg.det(random_covariance)
            / np.trace(random_covariance)**2
            if np.trace(random_covariance) > 0 else np.nan
        ),
        "innovation_variance": innovation_variance,
        "innovation_fraction": (
            innovation_variance / variance_slope
            if variance_slope > 0 else np.nan
        ),
        **geometry,
    }

# Source: advanced_rlrt_repair_pass_1, cell 14
def repair_one_operating_replicate(cell_id, replicate):
    try:
        cell = SIMULATION_CELL_LOOKUP[cell_id]
        template = SCHEDULE_TEMPLATES[cell["schedule"]]
        y_values, seed = generate_simulation_outcome(
            cell_id, int(replicate)
        )
        problem = simulation_problem_from_template(template, y_values)
        alternative_fit = fit_direct_multistart(
            problem, alternative=True, maxiter=700
        )
        null_fit = fit_direct_multistart(
            problem, alternative=False, maxiter=700
        )
        alternative_selected = alternative_fit["selected"]
        null_selected = null_fit["selected"]
        basic_fit_valid = bool(
            alternative_fit["basic_panel_valid"]
            and null_fit["basic_panel_valid"]
            and alternative_selected is not None
            and null_selected is not None
        )
        strict_fit_valid = bool(
            alternative_fit["strict_panel_valid"]
            and null_fit["strict_panel_valid"]
        )
        row = {
            "cell_id": cell_id,
            "replicate": int(replicate),
            "seed": int(seed),
            "fit_valid": basic_fit_valid,
            "strict_fit_valid": strict_fit_valid,
            "restricted_LR": (
                max(0.0, 2 * (
                    alternative_selected["log_likelihood"]
                    - null_selected["log_likelihood"]
                )) if basic_fit_valid else np.nan
            ),
            "null_converged": bool(null_fit["basic_panel_valid"]),
            "null_optimizer": "L-BFGS-B+Powell multistart",
            "alternative_converged": bool(
                alternative_fit["basic_panel_valid"]
            ),
            "alternative_optimizer": "L-BFGS-B+Powell multistart",
            "alternative_optimizers_attempted": 2,
            "optimizer_agreement": bool(
                alternative_fit["optimizer_agreement"]
            ),
            "null_optimizer_agreement": bool(
                null_fit["optimizer_agreement"]
            ),
            "repair_attempted": True,
            "repair_succeeded": strict_fit_valid,
            "repair_method": "multistart_vectorized_direct_REML_LBFGSB_and_Powell",
            "error": "",
        }
        if alternative_selected is None:
            row.update({
                "hessian_positive_definite": False,
                "hessian_minimum_information_eigenvalue": np.nan,
                "boundary_flag": True,
            })
            return row

        method_hessians = [
            item["hessian_minimum_information_eigenvalue"]
            for item in alternative_fit["audit"]
        ]
        random_covariance = alternative_selected["random_covariance"]
        residual_variance = float(
            alternative_selected["residual_variance"]
        )
        covariance_metrics = covariance_metrics_from_matrix(
            random_covariance
        )
        row.update(covariance_metrics)
        row["residual_variance"] = residual_variance
        row["hessian_positive_definite"] = bool(
            alternative_fit["strict_panel_valid"]
        )
        row["hessian_minimum_information_eigenvalue"] = float(
            np.nanmin(method_hessians)
        )
        row["boundary_flag"] = bool(
            random_covariance[1, 1]
            <= SLOPE_BOUNDARY_REL_TOL * max(1.0, residual_variance)
            or covariance_metrics["covariance_minimum_eigenvalue"]
            <= SLOPE_BOUNDARY_REL_TOL * max(
                1.0, float(np.trace(random_covariance))
            )
        )
        row.update(schedule_predictability_summary(
            random_covariance, residual_variance, template
        ))
        row["estimated_anchored_near"] = bool(
            np.isfinite(row[
                "anchored_endpoint_minimum_eigenvalue_fraction"
            ])
            and row[
                "anchored_endpoint_minimum_eigenvalue_fraction"
            ] <= SIM_ANCHORED_NEAR_THRESHOLD
        )
        return row
    except Exception as exc:
        return {
            "cell_id": cell_id,
            "replicate": int(replicate),
            "fit_valid": False,
            "strict_fit_valid": False,
            "hessian_positive_definite": False,
            "optimizer_agreement": False,
            "repair_attempted": True,
            "repair_succeeded": False,
            "repair_method": "multistart_vectorized_direct_REML_LBFGSB_and_Powell",
            "error": repr(exc),
        }

# Source: advanced_rlrt_repair_pass_2, cell 3
def summarize_one_rlrt_final(outcome, requested_B, frame=None):
    source = rlrt_bootstrap_replicates if frame is None else frame
    observed = observed_rlrt_tests.set_index("outcome").loc[outcome]
    rows = source.loc[
        source.outcome.eq(outcome)
        & pd.to_numeric(
            source.replicate, errors="coerce"
        ).le(int(requested_B))
    ].drop_duplicates("replicate", keep="last").copy()
    rows["basic_pair_valid"] = coerce_boolean(rows["basic_pair_valid"])
    rows["strict_pair_valid"] = coerce_boolean(rows["strict_pair_valid"])

    completed_B = int(rows.replicate.nunique())
    finite_lr = np.isfinite(
        pd.to_numeric(rows.restricted_LR, errors="coerce")
    )
    usable = rows.loc[rows.basic_pair_valid & finite_lr].copy()
    hessian_positive = rows.loc[
        rows.strict_pair_valid & finite_lr
    ].copy()
    usable_B = int(usable.replicate.nunique())
    hessian_positive_B = int(
        hessian_positive.replicate.nunique()
    )
    exceedances = int(
        (
            usable.restricted_LR
            >= float(observed.observed_restricted_LR)
        ).sum()
    )
    p_value = (
        (exceedances + 1) / (usable_B + 1)
        if usable_B else np.nan
    )
    mc_low, mc_high = exact_binomial_interval(
        exceedances, usable_B
    )
    extension_requested = bool(
        int(requested_B) > int(PB_INITIAL_B)
    )
    extension_completed = bool(completed_B >= int(requested_B))
    usable_rate = (
        usable_B / completed_B if completed_B else np.nan
    )
    hessian_rate = (
        hessian_positive_B / completed_B
        if completed_B else np.nan
    )
    observed_basic_valid = bool(
        observed.null_basic_panel_valid
        and observed.alternative_basic_panel_valid
    )
    return {
        "outcome": outcome,
        "test_specification": (
            "REML parametric bootstrap; RI null versus unstructured "
            "RI+t5_mid slope; observed schedules/covariates; a bootstrap "
            "replicate is usable when both restricted likelihoods are finite "
            "and multistart optimizers agree. Positive-Hessian status is "
            "reported diagnostically and is not an exclusion rule."
        ),
        "requested_B": int(requested_B),
        "completed_B": completed_B,
        "usable_optimizer_agreeing_denominator": usable_B,
        "unusable_denominator": completed_B - usable_B,
        "usable_optimizer_agreeing_rate": usable_rate,
        "Hessian_positive_denominator": hessian_positive_B,
        "Hessian_positive_rate": hessian_rate,
        "observed_restricted_LR": float(
            observed.observed_restricted_LR
        ),
        "tail_exceedances": exceedances,
        "bootstrap_p_value_plus_one": p_value,
        "MC_interval_low": mc_low,
        "MC_interval_high": mc_high,
        "MC_interval_overlaps_0p05": bool(
            np.isfinite(mc_low)
            and mc_low <= ALPHA <= mc_high
        ),
        "extension_requested": extension_requested,
        "extension_completed": extension_completed,
        "observed_basic_optimizer_panel_valid": observed_basic_valid,
        "p_value_licensed": bool(
            extension_completed
            and observed_basic_valid
            and np.isfinite(usable_rate)
            and usable_rate >= RLRT_USABLE_RATE_TARGET
        ),
    }

# Source: advanced_rlrt_repair_pass_2, cell 5
def profile1d_objective(
    parameters, problem, d_0_5, gamma_sign
):
    random_covariance_anchor, _ = endpoint_covariance_from_d(
        parameters[0], d_0_5, parameters[1], gamma_sign
    )
    if random_covariance_anchor is None:
        return 1e100
    residual_variance = float(np.exp(2 * parameters[2]))
    log_likelihood, _ = direct_reml_loglike_allow_psd(
        problem,
        random_covariance_anchor,
        np.repeat(residual_variance, problem["n"]),
    )
    return (
        -float(log_likelihood)
        if np.isfinite(log_likelihood) else 1e100
    )

# Source: advanced_rlrt_repair_pass_2, cell 5
def unique_clipped_starts(starts, bounds):
    unique = []
    for start in starts:
        if start is None:
            continue
        values = np.asarray(start, float)
        if values.shape != (3,) or not np.isfinite(values).all():
            continue
        values = np.array([
            np.clip(value, lower, upper)
            for value, (lower, upper) in zip(values, bounds)
        ])
        if not any(np.max(np.abs(values - prior)) < 1e-10 for prior in unique):
            unique.append(values)
    return unique

# Source: advanced_rlrt_repair_pass_2, cell 5
def run_profile1d_optimizer(
    method, starts, bounds, problem, d_0_5, gamma_sign
):
    candidates = []
    for start_index, start in enumerate(starts):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                options = (
                    {
                        "maxiter": PROFILE1D_MAXITER,
                        "ftol": 1e-11,
                        "maxls": 50,
                    }
                    if method == "L-BFGS-B"
                    else {
                        "maxiter": PROFILE1D_MAXITER,
                        "ftol": 1e-10,
                        "xtol": 1e-7,
                    }
                )
                fitted = minimize(
                    profile1d_objective,
                    start,
                    args=(problem, d_0_5, gamma_sign),
                    method=method,
                    bounds=bounds,
                    options=options,
                )
            parameters = np.asarray(fitted.x, float)
            random_covariance_anchor, gamma = (
                endpoint_covariance_from_d(
                    parameters[0], d_0_5,
                    parameters[1], gamma_sign
                )
            )
            residual_variance = float(
                np.exp(2 * parameters[2])
            )
            log_likelihood, fixed_effects = (
                direct_reml_loglike_allow_psd(
                    problem,
                    random_covariance_anchor,
                    np.repeat(
                        residual_variance, problem["n"]
                    ),
                )
                if random_covariance_anchor is not None
                else (-np.inf, None)
            )
            finite_solution = bool(
                np.isfinite(log_likelihood)
                and fixed_effects is not None
                and np.isfinite(fixed_effects).all()
            )
            candidates.append({
                "optimizer": method,
                "start_index": int(start_index),
                "optimizer_success": bool(
                    fitted.success and finite_solution
                ),
                "finite_solution": finite_solution,
                "log_likelihood": float(log_likelihood),
                "parameters": parameters,
                "fixed_effects": (
                    np.asarray(fixed_effects, float)
                    if fixed_effects is not None
                    else np.full(problem["p"], np.nan)
                ),
                "gamma_0_5": float(gamma),
                "message": str(fitted.message),
                "iterations": int(
                    getattr(fitted, "nit", -1)
                ),
            })
        except Exception as exc:
            candidates.append({
                "optimizer": method,
                "start_index": int(start_index),
                "optimizer_success": False,
                "finite_solution": False,
                "log_likelihood": -np.inf,
                "parameters": np.asarray(start, float),
                "fixed_effects": np.full(problem["p"], np.nan),
                "gamma_0_5": (
                    float(gamma_sign)
                    * np.sqrt(max(0.0, 1.0 - d_0_5))
                ),
                "message": repr(exc),
                "iterations": -1,
            })
    finite = [
        row for row in candidates if row["finite_solution"]
    ]
    successful = [
        row for row in finite if row["optimizer_success"]
    ]
    best = (
        max(
            successful or finite,
            key=lambda row: row["log_likelihood"],
        )
        if finite else None
    )
    return best, candidates

# Source: advanced_rlrt_repair_pass_2, cell 5
def profile1d_solutions_agree(first, second):
    if first is None or second is None:
        return False, np.nan, np.nan
    if not (
        first["optimizer_success"]
        and second["optimizer_success"]
    ):
        return False, np.nan, np.nan
    likelihood_difference = abs(
        first["log_likelihood"]
        - second["log_likelihood"]
    )
    fixed_scale = max(
        1.0,
        float(np.max(np.abs(first["fixed_effects"]))),
        float(np.max(np.abs(second["fixed_effects"]))),
    )
    fixed_difference = float(np.max(np.abs(
        first["fixed_effects"]
        - second["fixed_effects"]
    )))
    agrees = bool(
        likelihood_difference <= PROFILE1D_LLF_ABS_TOL
        and fixed_difference
        <= PROFILE1D_FIXED_REL_TOL * fixed_scale
    )
    return agrees, likelihood_difference, fixed_difference

# Source: advanced_rlrt_repair_pass_2, cell 5
def fit_profile1d_d_value(
    problem, d_0_5, frozen_start, warm_starts_by_sign,
    legacy_start, bounds
):
    sign_summaries = []
    audit_rows = []
    next_warm = {}
    for gamma_sign in (1.0, -1.0):
        starts = unique_clipped_starts(
            [
                warm_starts_by_sign.get(gamma_sign),
                legacy_start,
                frozen_start,
                np.array([
                    frozen_start[0], 0.0, frozen_start[2]
                ]),
            ],
            bounds,
        )
        lbfgsb_best, lbfgsb_candidates = (
            run_profile1d_optimizer(
                "L-BFGS-B", starts, bounds,
                problem, d_0_5, gamma_sign
            )
        )
        powell_starts = list(starts)
        if lbfgsb_best is not None:
            powell_starts.insert(
                0, lbfgsb_best["parameters"]
            )
        powell_starts = unique_clipped_starts(
            powell_starts, bounds
        )
        powell_best, powell_candidates = (
            run_profile1d_optimizer(
                "Powell", powell_starts, bounds,
                problem, d_0_5, gamma_sign
            )
        )
        agrees, ll_difference, fixed_difference = (
            profile1d_solutions_agree(
                lbfgsb_best, powell_best
            )
        )

        # One cross-polish pass. This is deliberately bounded: it resolves
        # start dependence without turning one profile point into an
        # unbounded multistart search.
        if (
            not agrees
            and lbfgsb_best is not None
            and powell_best is not None
        ):
            polished_lbfgsb, polished_lbfgsb_candidates = (
                run_profile1d_optimizer(
                    "L-BFGS-B",
                    [powell_best["parameters"]],
                    bounds, problem, d_0_5, gamma_sign,
                )
            )
            polished_powell, polished_powell_candidates = (
                run_profile1d_optimizer(
                    "Powell",
                    [lbfgsb_best["parameters"]],
                    bounds, problem, d_0_5, gamma_sign,
                )
            )
            lbfgsb_candidates.extend(
                polished_lbfgsb_candidates
            )
            powell_candidates.extend(
                polished_powell_candidates
            )
            if (
                polished_lbfgsb is not None
                and (
                    lbfgsb_best is None
                    or (
                        polished_lbfgsb["optimizer_success"]
                        and not lbfgsb_best["optimizer_success"]
                    )
                    or (
                        polished_lbfgsb["optimizer_success"]
                        == lbfgsb_best["optimizer_success"]
                        and polished_lbfgsb["log_likelihood"]
                        > lbfgsb_best["log_likelihood"]
                    )
                )
            ):
                lbfgsb_best = polished_lbfgsb
            if (
                polished_powell is not None
                and (
                    powell_best is None
                    or (
                        polished_powell["optimizer_success"]
                        and not powell_best["optimizer_success"]
                    )
                    or (
                        polished_powell["optimizer_success"]
                        == powell_best["optimizer_success"]
                        and polished_powell["log_likelihood"]
                        > powell_best["log_likelihood"]
                    )
                )
            ):
                powell_best = polished_powell
            agrees, ll_difference, fixed_difference = (
                profile1d_solutions_agree(
                    lbfgsb_best, powell_best
                )
            )

        finite_best = [
            candidate
            for candidate in (lbfgsb_best, powell_best)
            if candidate is not None
            and candidate["finite_solution"]
        ]
        representative = (
            max(
                finite_best,
                key=lambda row: row["log_likelihood"],
            )
            if finite_best else None
        )
        if representative is not None:
            next_warm[gamma_sign] = (
                representative["parameters"].copy()
            )
        sign_summaries.append({
            "gamma_sign": float(gamma_sign),
            "optimizer_agreement": bool(agrees),
            "likelihood_agreement_absolute_difference": (
                ll_difference
            ),
            "fixed_effect_agreement_maximum_absolute_difference": (
                fixed_difference
            ),
            "profile_log_likelihood": (
                float(representative["log_likelihood"])
                if representative is not None else -np.inf
            ),
            "parameters": (
                representative["parameters"].copy()
                if representative is not None
                else np.full(3, np.nan)
            ),
            "gamma_0_5": (
                float(representative["gamma_0_5"])
                if representative is not None
                else float(gamma_sign)
                * np.sqrt(max(0.0, 1.0 - d_0_5))
            ),
        })
        for optimizer, best, candidates in (
            ("L-BFGS-B", lbfgsb_best, lbfgsb_candidates),
            ("Powell", powell_best, powell_candidates),
        ):
            audit_rows.append({
                "d_0_5": float(d_0_5),
                "gamma_sign": float(gamma_sign),
                "optimizer": optimizer,
                "optimizer_success": bool(
                    best is not None
                    and best["optimizer_success"]
                ),
                "finite_solution": bool(
                    best is not None
                    and best["finite_solution"]
                ),
                "log_likelihood": (
                    float(best["log_likelihood"])
                    if best is not None else -np.inf
                ),
                "log_slope_sd": (
                    float(best["parameters"][0])
                    if best is not None else np.nan
                ),
                "log_sd5_over_sd0": (
                    float(best["parameters"][1])
                    if best is not None else np.nan
                ),
                "log_residual_sd": (
                    float(best["parameters"][2])
                    if best is not None else np.nan
                ),
                "fixed_effects_json": (
                    json.dumps([
                        float(value)
                        for value in best["fixed_effects"]
                    ])
                    if best is not None else "[]"
                ),
                "optimizer_message": (
                    best["message"]
                    if best is not None
                    else "no finite solution"
                ),
                "attempts": int(len(candidates)),
                "optimizer_agreement_for_sign": bool(agrees),
                "likelihood_agreement_absolute_difference": (
                    ll_difference
                ),
                "fixed_effect_agreement_maximum_absolute_difference": (
                    fixed_difference
                ),
            })

    agreed_signs = [
        row for row in sign_summaries
        if row["optimizer_agreement"]
        and np.isfinite(row["profile_log_likelihood"])
    ]
    finite_signs = [
        row for row in sign_summaries
        if np.isfinite(row["profile_log_likelihood"])
    ]
    chosen = (
        max(
            agreed_signs,
            key=lambda row: row["profile_log_likelihood"],
        )
        if agreed_signs
        else max(
            finite_signs,
            key=lambda row: row["profile_log_likelihood"],
        )
        if finite_signs else None
    )
    summary = {
        "d_0_5": float(d_0_5),
        "minus_log10_d_0_5": (
            -np.log10(d_0_5)
            if d_0_5 > 0
            else PROFILE1D_BOUNDARY_PLOT_VALUE
        ),
        "exact_PSD_boundary": bool(d_0_5 == 0),
        "profile_log_likelihood": (
            float(chosen["profile_log_likelihood"])
            if chosen is not None else -np.inf
        ),
        "optimizer_agreement": bool(
            chosen is not None
            and chosen["optimizer_agreement"]
        ),
        "selected_gamma_sign": (
            float(chosen["gamma_sign"])
            if chosen is not None else np.nan
        ),
        "selected_gamma_0_5": (
            float(chosen["gamma_0_5"])
            if chosen is not None else np.nan
        ),
        "optimized_log_slope_sd": (
            float(chosen["parameters"][0])
            if chosen is not None else np.nan
        ),
        "optimized_log_sd5_over_sd0": (
            float(chosen["parameters"][1])
            if chosen is not None else np.nan
        ),
        "optimized_log_residual_sd": (
            float(chosen["parameters"][2])
            if chosen is not None else np.nan
        ),
        "likelihood_agreement_absolute_difference": (
            float(
                chosen[
                    "likelihood_agreement_absolute_difference"
                ]
            )
            if chosen is not None
            and np.isfinite(
                chosen[
                    "likelihood_agreement_absolute_difference"
                ]
            )
            else np.nan
        ),
        "fixed_effect_agreement_maximum_absolute_difference": (
            float(
                chosen[
                    "fixed_effect_agreement_maximum_absolute_difference"
                ]
            )
            if chosen is not None
            and np.isfinite(
                chosen[
                    "fixed_effect_agreement_maximum_absolute_difference"
                ]
            )
            else np.nan
        ),
    }
    return summary, audit_rows, next_warm

# Source: advanced_rlrt_repair_pass_2, cell 8
def replace_raw_endpoint_covariance_names(frame):
    result = frame.copy()
    rename_map = {
        column: column.replace("g_0_5", "cov_L0_L5")
        for column in result.columns
        if "g_0_5" in column
    }
    if rename_map:
        result = result.rename(columns=rename_map)
    if "gamma_0_5" in result.columns:
        result["d_0_5"] = (
            1 - pd.to_numeric(
                result.gamma_0_5, errors="coerce"
            ) ** 2
        ).clip(lower=0)
    if "approx_deleted_gamma_0_5" in result.columns:
        result["approx_deleted_d_0_5"] = (
            1 - pd.to_numeric(
                result.approx_deleted_gamma_0_5,
                errors="coerce",
            ) ** 2
        ).clip(lower=0)
    return result
