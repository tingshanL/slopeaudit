#!/usr/bin/env python3
"""Pure numerical helpers for the S4 operating-characteristic simulation.

This module contains numerical geometry, REML fitting, predictability,
and robust-refit helpers only.

It deliberately contains no:
- AABC2 data reads;
- empirical schedule values;
- schedule or cell manifests;
- filesystem discovery;
- output/checkpoint paths;
- simulation execution loops.

Schedule construction, data generation, execution, repair orchestration,
and deterministic summarization live in separate entry points.
"""

from __future__ import annotations

import hashlib
import math
import warnings

import numpy as np
import pandas as pd
import patsy
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess


# ============================================================================
# SLOPE_BOUNDARY_REL_TOL
# ============================================================================
SLOPE_BOUNDARY_REL_TOL = 1e-8


# ============================================================================
# anchored_endpoint_geometry
# ============================================================================
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


# ============================================================================
# midpoint_to_anchor_covariance
# ============================================================================
def midpoint_to_anchor_covariance(random_covariance_midpoint):
    covariance_midpoint = np.asarray(random_covariance_midpoint, float)
    transform = np.array([[1.0, -0.5], [0.0, 1.0]])
    covariance_anchor = transform @ covariance_midpoint @ transform.T
    return (covariance_anchor + covariance_anchor.T) / 2


# ============================================================================
# alternative_covariance_from_parameters
# ============================================================================
def alternative_covariance_from_parameters(parameters):
    lower = np.array([
        [np.exp(parameters[0]), 0.0],
        [parameters[1], np.exp(parameters[2])],
    ])
    return lower @ lower.T, np.exp(2 * parameters[3])


# ============================================================================
# fit_specialized_reml
# ============================================================================
def fit_specialized_reml(template, y_values, alternative):
    outcome_variance = max(float(np.var(y_values, ddof=1)), 1e-4)
    if alternative:
        initial = np.array([
            0.5 * np.log(0.5 * outcome_variance), 0.0,
            0.5 * np.log(0.05 * outcome_variance),
            0.5 * np.log(0.5 * outcome_variance),
        ])
        bounds = [(-12, 4), (-5, 5), (-12, 4), (-12, 4)]
        unpack = alternative_covariance_from_parameters
    else:
        initial = np.array([
            0.5 * np.log(0.5 * outcome_variance),
            0.5 * np.log(0.5 * outcome_variance),
        ])
        bounds = [(-12, 4), (-12, 4)]
        unpack = null_covariance_from_parameters

    def objective(parameters):
        random_covariance, residual_variance = unpack(parameters)
        loglike, _ = vectorized_reml_loglike(
            template, y_values, random_covariance, residual_variance
        )
        return -loglike if np.isfinite(loglike) else 1e100

    attempts = []
    for method in ("L-BFGS-B", "Powell"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = minimize(
                objective, initial, method=method,
                bounds=bounds if method == "L-BFGS-B" else None,
                options={"maxiter": 600, "ftol": 1e-9},
            )
        attempts.append((method, fitted))
        if fitted.success and np.isfinite(fitted.fun):
            break
    successful = [
        (method, fitted) for method, fitted in attempts
        if fitted.success and np.isfinite(fitted.fun)
    ]
    if not successful:
        return {
            "fit_valid": False, "converged": False,
            "log_likelihood": np.nan, "error": "all optimizers failed",
        }
    method, best = min(successful, key=lambda item: item[1].fun)
    random_covariance, residual_variance = unpack(best.x)
    eigenvalues = np.linalg.eigvalsh(random_covariance)
    record = {
        "fit_valid": True,
        "converged": bool(best.success),
        "optimizer": method,
        "optimizers_attempted": len(attempts),
        "log_likelihood": -float(best.fun),
        "random_intercept_variance": float(random_covariance[0, 0]),
        "random_slope_variance": float(random_covariance[1, 1]),
        "intercept_slope_covariance": float(random_covariance[0, 1]),
        "residual_variance": float(residual_variance),
        "covariance_minimum_eigenvalue": float(eigenvalues.min()),
        "boundary_flag": bool(
            alternative and (
                random_covariance[1, 1]
                <= SLOPE_BOUNDARY_REL_TOL * max(1.0, residual_variance)
                or eigenvalues.min()
                <= SLOPE_BOUNDARY_REL_TOL * max(1.0, np.trace(random_covariance))
            )
        ),
        "error": "",
    }
    if alternative:
        hessian = np.asarray(approx_hess(best.x, objective), float)
        hessian = (hessian + hessian.T) / 2
        hessian_eigenvalues = np.linalg.eigvalsh(hessian)
        correlation = (
            random_covariance[0, 1]
            / np.sqrt(random_covariance[0, 0] * random_covariance[1, 1])
            if random_covariance[0, 0] > 0 and random_covariance[1, 1] > 0
            else np.nan
        )
        innovation_variance = (
            random_covariance[1, 1]
            - random_covariance[0, 1]**2 / random_covariance[0, 0]
            if random_covariance[0, 0] > 0 else np.nan
        )
        record.update({
            "hessian_positive_definite": bool(hessian_eigenvalues.min() > 1e-8),
            "hessian_minimum_information_eigenvalue": float(hessian_eigenvalues.min()),
            "intercept_slope_correlation": correlation,
            "absolute_intercept_slope_correlation": abs(correlation) if np.isfinite(correlation) else np.nan,
            "one_minus_rho_squared": 1 - correlation**2 if np.isfinite(correlation) else np.nan,
            "normalized_determinant": (
                np.linalg.det(random_covariance) / np.trace(random_covariance)**2
                if np.trace(random_covariance) > 0 else np.nan
            ),
            "innovation_variance": innovation_variance,
            "innovation_fraction": (
                innovation_variance / random_covariance[1, 1]
                if random_covariance[1, 1] > 0 else np.nan
            ),
            **anchored_endpoint_geometry(random_covariance),
        })
    return record


# ============================================================================
# null_covariance_from_parameters
# ============================================================================
def null_covariance_from_parameters(parameters):
    return np.array([[np.exp(2 * parameters[0]), 0.0], [0.0, 0.0]]), np.exp(2 * parameters[1])


# ============================================================================
# schedule_predictability_summary
# ============================================================================
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


# ============================================================================
# vectorized_reml_loglike
# ============================================================================
def vectorized_reml_loglike(template, y_values, random_covariance_midpoint, residual_variance):
    if (
        residual_variance <= 0
        or np.any(~np.isfinite(random_covariance_midpoint))
        or np.linalg.eigvalsh(random_covariance_midpoint).min() < -1e-10
    ):
        return -np.inf, None
    xt_vinv_x = np.zeros((2, 2))
    xt_vinv_y = np.zeros(2)
    y_vinv_y = 0.0
    logdet_v = 0.0
    try:
        for length, block in template["by_length"].items():
            positions = block["positions"]
            design = block["design"]
            outcomes = y_values[positions]
            covariance = (
                np.einsum(
                    "mki,ij,mlj->mkl",
                    design, random_covariance_midpoint, design,
                    optimize=True,
                )
                + residual_variance * np.eye(length)[None, :, :]
            )
            signs, logdets = np.linalg.slogdet(covariance)
            if np.any(signs <= 0):
                return -np.inf, None
            vinv_design = np.linalg.solve(covariance, design)
            vinv_outcome = np.linalg.solve(covariance, outcomes[..., None])[..., 0]
            xt_vinv_x += np.einsum(
                "mki,mkj->ij", design, vinv_design, optimize=True
            )
            xt_vinv_y += np.einsum(
                "mki,mk->i", design, vinv_outcome, optimize=True
            )
            y_vinv_y += float(np.einsum(
                "mk,mk->", outcomes, vinv_outcome, optimize=True
            ))
            logdet_v += float(logdets.sum())
        beta = np.linalg.solve(xt_vinv_x, xt_vinv_y)
        sign_fixed, logdet_fixed = np.linalg.slogdet(xt_vinv_x)
        if sign_fixed <= 0:
            return -np.inf, None
        residual_quadratic = y_vinv_y - float(xt_vinv_y @ beta)
        loglike = -0.5 * (
            (template["observations"] - 2) * np.log(2 * np.pi)
            + logdet_v + logdet_fixed + residual_quadratic
        )
        return float(loglike), beta
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return -np.inf, None


# ============================================================================
# direct_multistarts
# ============================================================================
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


# ============================================================================
# direct_solution_compatible
# ============================================================================
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


# ============================================================================
# fit_direct_multistart
# ============================================================================
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


# ============================================================================
# parameter_hessian_diagnostic
# ============================================================================
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


# ============================================================================
# prepare_vectorized_reml_problem
# ============================================================================
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


# ============================================================================
# unpack_direct_parameters
# ============================================================================
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


# ============================================================================
# vectorized_direct_reml_loglike
# ============================================================================
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


# ============================================================================
# covariance_metrics_from_matrix
# ============================================================================
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


# ============================================================================
# simulation_problem_from_template
# ============================================================================
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
