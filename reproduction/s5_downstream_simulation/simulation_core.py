#!/usr/bin/env python3
"""Pure numerical core for the S5 downstream simulation.

This module contains the numerical routines from the completed
`AABC_final500_covariance_singularity_simulation` analysis. Source parameters
are passed explicitly rather than read from notebook-global state, so the
module requires no AABC file discovery, participant identifiers, source or
result paths, task loop, checkpoint I/O, or output I/O.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import warnings

import numpy as np
import pandas as pd
import patsy
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable


OPTIMIZERS = ("lbfgs", "powell", "bfgs")
MAXITER = 1200

SLOPE_BOUNDARY_REL_TOL = 1e-8
PSD_REL_TOL = 1e-8

PRIMARY_NEAR_SINGULAR_ABS_RHO = 0.95
PRIMARY_SMALL_NORMALIZED_DETERMINANT = 0.10
PRIMARY_SMALL_EIGEN_FRACTION = 1e-4


__all__ = [
    "OPTIMIZERS",
    "MAXITER",
    "SLOPE_BOUNDARY_REL_TOL",
    "PSD_REL_TOL",
    "PRIMARY_NEAR_SINGULAR_ABS_RHO",
    "PRIMARY_SMALL_NORMALIZED_DETERMINANT",
    "PRIMARY_SMALL_EIGEN_FRACTION",
    "hessian_diagnostic",
    "fixed_finite",
    "covariance_geometry",
    "clamp_rounding_only",
    "reliability_geometry",
    "fit_optimizer_sequence",
    "reml_meta_regression",
    "eb_slope_table",
    "uncertainty_aware_slopes",
    "simulated_dataset",
    "attach_task_metadata",
    "run_lr_task",
    "run_evaluation_task",
]


def hessian_diagnostic(result):
    try:
        hessian = result.model.hessian(result.params_object)
        if isinstance(hessian, tuple):
            hessian = hessian[0]
        information = -(np.asarray(hessian, float)+np.asarray(hessian, float).T)/2
        eigenvalues = np.linalg.eigvalsh(information)
        tolerance = 1e-8*max(1.0, float(np.max(np.abs(eigenvalues))))
        return bool(np.min(eigenvalues)>tolerance), float(np.min(eigenvalues)), float(np.max(eigenvalues))
    except Exception:
        return False, np.nan, np.nan


def fixed_finite(result):
    try:
        estimates = np.asarray(result.fe_params, float)
        errors = np.asarray(result.bse_fe, float)
        return bool(np.isfinite(estimates).all() and np.isfinite(errors).all() and (errors>=0).all())
    except Exception:
        return False


def covariance_geometry(result, has_random_slope):
    residual = float(result.scale)
    random_covariance = np.asarray(result.cov_re, float)
    covariance_eigenvalues = np.linalg.eigvalsh((random_covariance+random_covariance.T)/2)
    covariance_minimum = float(np.min(covariance_eigenvalues))
    covariance_trace = float(np.trace(random_covariance))
    covariance_scale = max(1.0, float(np.max(np.abs(covariance_eigenvalues))), abs(residual))
    covariance_psd = bool(
        np.isfinite(random_covariance).all() and residual>0
        and covariance_minimum >= -PSD_REL_TOL*covariance_scale
    )
    hessian_pd, hessian_minimum, hessian_maximum = hessian_diagnostic(result)

    output = {
        "converged": bool(result.converged),
        "reml_log_likelihood": float(result.llf),
        "fixed_parameters_finite": fixed_finite(result),
        "hessian_positive_definite": hessian_pd,
        "hessian_min_information_eigenvalue": hessian_minimum,
        "hessian_max_information_eigenvalue": hessian_maximum,
        "residual_variance": residual,
        "random_intercept_variance": float(random_covariance[0,0]),
        "random_slope_variance": np.nan,
        "intercept_slope_covariance": np.nan,
        "intercept_slope_correlation": np.nan,
        "absolute_intercept_slope_correlation": np.nan,
        "covariance_minimum_eigenvalue": covariance_minimum,
        "covariance_trace": covariance_trace,
        "minimum_eigenvalue_fraction_of_trace": (
            covariance_minimum/covariance_trace if covariance_trace>0 else np.nan
        ),
        "covariance_psd": covariance_psd,
        "one_minus_rho_squared": np.nan,
        "normalized_determinant": np.nan,
        "innovation_variance": np.nan,
        "innovation_fraction": np.nan,
        "residual_variance_boundary": bool(not np.isfinite(residual) or residual<=PSD_REL_TOL),
        "intercept_variance_boundary": bool(
            not np.isfinite(random_covariance[0,0])
            or random_covariance[0,0] <= PSD_REL_TOL*covariance_scale
        ),
        "slope_variance_boundary": False,
        "covariance_boundary": bool(covariance_minimum <= PSD_REL_TOL*covariance_scale),
        "near_singular_abs_rho_095": False,
        "near_singular_normalized_determinant_010": False,
        "near_singular_eigen_fraction_1e4": bool(
            covariance_trace>0 and covariance_minimum/covariance_trace <= PRIMARY_SMALL_EIGEN_FRACTION
        ),
    }
    if has_random_slope and random_covariance.shape == (2,2):
        intercept_variance = float(random_covariance[0,0])
        slope_variance = float(random_covariance[1,1])
        covariance = float(random_covariance[0,1])
        correlation = (
            covariance/np.sqrt(intercept_variance*slope_variance)
            if intercept_variance>0 and slope_variance>0 else np.nan
        )
        normalized_determinant = (
            float(np.linalg.det(random_covariance)/(intercept_variance*slope_variance))
            if intercept_variance>0 and slope_variance>0 else np.nan
        )
        one_minus_rho_squared = 1-correlation**2 if np.isfinite(correlation) else np.nan
        innovation_variance = (
            slope_variance-covariance**2/intercept_variance
            if intercept_variance>0 else np.nan
        )
        innovation_fraction = (
            innovation_variance/slope_variance
            if slope_variance>0 and np.isfinite(innovation_variance) else np.nan
        )
        slope_boundary_tolerance = SLOPE_BOUNDARY_REL_TOL*max(1.0, residual)
        output.update({
            "random_slope_variance": slope_variance,
            "intercept_slope_covariance": covariance,
            "intercept_slope_correlation": correlation,
            "absolute_intercept_slope_correlation": abs(correlation) if np.isfinite(correlation) else np.nan,
            "one_minus_rho_squared": one_minus_rho_squared,
            "normalized_determinant": normalized_determinant,
            "innovation_variance": innovation_variance,
            "innovation_fraction": innovation_fraction,
            "slope_variance_boundary": bool(not np.isfinite(slope_variance) or slope_variance<=slope_boundary_tolerance),
            "near_singular_abs_rho_095": bool(np.isfinite(correlation) and abs(correlation)>=PRIMARY_NEAR_SINGULAR_ABS_RHO),
            "near_singular_normalized_determinant_010": bool(
                np.isfinite(normalized_determinant) and normalized_determinant<=PRIMARY_SMALL_NORMALIZED_DETERMINANT
            ),
        })
    output["any_boundary_flag"] = bool(
        output["residual_variance_boundary"]
        or output["intercept_variance_boundary"]
        or output["slope_variance_boundary"]
        or output["covariance_boundary"]
    )
    output["basic_numerical_validity"] = bool(
        output["converged"] and output["fixed_parameters_finite"] and output["covariance_psd"]
        and np.isfinite(output["reml_log_likelihood"])
    )
    output["strict_numerical_validity"] = bool(
        output["basic_numerical_validity"] and output["hessian_positive_definite"]
        and not output["covariance_boundary"]
        and (not has_random_slope or not output["slope_variance_boundary"])
    )
    return output


def clamp_rounding_only(value, tolerance=1e-7):
    if not np.isfinite(value):
        return np.nan
    if -tolerance <= value <= 1+tolerance:
        return float(np.clip(value,0,1))
    return np.nan


def reliability_geometry(result, data):
    random_covariance = np.asarray(result.cov_re, float)
    residual = float(result.scale)
    output = {
        "mean_direct_predictability": np.nan, "median_direct_predictability": np.nan,
        "mean_full_posterior_predictability": np.nan, "median_full_posterior_predictability": np.nan,
        "mean_innovation_predictability": np.nan, "median_innovation_predictability": np.nan,
    }
    if random_covariance.shape!=(2,2):
        return output
    eigenvalues = np.linalg.eigvalsh((random_covariance+random_covariance.T)/2)
    intercept_variance, slope_variance = float(random_covariance[0,0]), float(random_covariance[1,1])
    covariance = float(random_covariance[0,1])
    if not (np.min(eigenvalues)>0 and intercept_variance>0 and slope_variance>0 and residual>0):
        return output
    inverse_random = np.linalg.inv(random_covariance)
    innovation_loading = covariance/intercept_variance
    innovation_variance = slope_variance-covariance**2/intercept_variance
    direct_values, full_values, innovation_values = [], [], []
    for _, group in data.groupby("pid", sort=False):
        information = float(np.sum((group.t-group.t.mean())**2))
        direct = slope_variance/(slope_variance+residual/information) if information>0 else np.nan
        random_design = np.column_stack([np.ones(len(group)),group.t.to_numpy(float)])
        posterior_covariance = np.linalg.inv(inverse_random+random_design.T@random_design/residual)
        full = clamp_rounding_only(1-posterior_covariance[1,1]/slope_variance)
        innovation = np.nan
        if innovation_variance>0:
            transform = np.array([-innovation_loading,1.0])
            posterior_innovation = float(transform@posterior_covariance@transform)
            innovation = clamp_rounding_only(1-posterior_innovation/innovation_variance)
        direct_values.append(direct); full_values.append(full); innovation_values.append(innovation)
    for values, name in [
        (direct_values,"direct_predictability"),
        (full_values,"full_posterior_predictability"),
        (innovation_values,"innovation_predictability"),
    ]:
        output[f"mean_{name}"] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
        output[f"median_{name}"] = float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
    return output


def fit_optimizer_sequence(data, formula, re_formula, model_role):
    has_random_slope = "t" in re_formula
    attempts, candidates, fitted_candidates = [], [], []
    selected_result = None
    for optimizer in OPTIMIZERS:
        caught = []
        try:
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always")
                result = smf.mixedlm(
                    formula, data=data, groups=data["pid"], re_formula=re_formula
                ).fit(reml=True, method=optimizer, maxiter=MAXITER, disp=False)
                caught = [str(item.message) for item in warning_records]
            geometry = covariance_geometry(result, has_random_slope)
            if has_random_slope:
                geometry.update(reliability_geometry(result,data))
            else:
                geometry.update({
                    "mean_direct_predictability":np.nan,"median_direct_predictability":np.nan,
                    "mean_full_posterior_predictability":np.nan,"median_full_posterior_predictability":np.nan,
                    "mean_innovation_predictability":np.nan,"median_innovation_predictability":np.nan,
                })
            attempt = {
                "model_role":model_role,"optimizer":optimizer,"fit_succeeded":True,
                "warning_count":len(caught),"warnings":" | ".join(sorted(set(caught))),"error":"",
                **geometry,
            }
            attempts.append(attempt)
            fitted_candidates.append((result,attempt))
            if geometry["basic_numerical_validity"]:
                candidates.append((result,attempt))
            if geometry["strict_numerical_validity"]:
                selected_result = result
                selected_attempt = attempt
                break
        except Exception as exc:
            attempts.append({
                "model_role":model_role,"optimizer":optimizer,"fit_succeeded":False,
                "converged":False,"basic_numerical_validity":False,"strict_numerical_validity":False,
                "warning_count":len(caught),"warnings":" | ".join(sorted(set(caught))),"error":repr(exc),
            })
    if selected_result is None and candidates:
        selected_result, selected_attempt = candidates[0]  # frozen optimizer order; no selective best-fit retention
    if selected_result is None and fitted_candidates:
        # Preserve a fitted but numerically invalid replicate in the diagnostic denominator.
        # It is never treated as a valid inferential result.
        selected_result, selected_attempt = fitted_candidates[0]
    if selected_result is None:
        return None, None, attempts
    for attempt in attempts:
        attempt["selected"] = attempt is selected_attempt
        attempt["selection_status"] = (
            "strictly_valid" if selected_attempt.get("strict_numerical_validity",False)
            else (
                "basic_valid_fallback" if selected_attempt.get("basic_numerical_validity",False)
                else "invalid_fitted_fallback"
            )
        ) if attempt is selected_attempt else "not_selected"
    return selected_result, dict(selected_attempt), attempts


def reml_meta_regression(y, sampling_variance, design, column_names):
    y=np.asarray(y,float); variance=np.asarray(sampling_variance,float); design=np.asarray(design,float)
    good=np.isfinite(y)&np.isfinite(variance)&(variance>0)&np.isfinite(design).all(axis=1)
    y,variance,design=y[good],variance[good],design[good]
    if np.linalg.matrix_rank(design)<design.shape[1]:
        raise np.linalg.LinAlgError("Rank-deficient meta-regression design")
    def evaluate(tau2):
        total=variance+max(0.0,float(tau2)); weights=1/total
        xtwx=design.T@(weights[:,None]*design); inverse=np.linalg.inv(xtwx)
        beta=inverse@(design.T@(weights*y)); residual=y-design@beta
        sign,logdet=np.linalg.slogdet(xtwx)
        if sign<=0:return np.inf,None,None
        nll=.5*(np.log(total).sum()+logdet+np.sum(weights*residual**2)+(len(y)-design.shape[1])*np.log(2*np.pi))
        return float(nll),beta,inverse
    upper=max(float(np.var(y,ddof=1))*10,float(np.median(variance))*100,1e-6)
    optimization=minimize_scalar(lambda tau:evaluate(tau)[0],bounds=(0,upper),method="bounded",options={"xatol":1e-10})
    candidates=[0.0,float(optimization.x)]; tau2=min(candidates,key=lambda value:evaluate(value)[0])
    _,beta,covariance=evaluate(tau2)
    return pd.Series(beta,index=column_names),pd.Series(np.sqrt(np.diag(covariance)),index=column_names),tau2


def eb_slope_table(result,data,predictor_included):
    representatives=data.groupby("pid",as_index=False).head(1)[["pid","x"]].copy()
    deviations=[]
    for participant_id in representatives.pid:
        random_effect=result.random_effects[participant_id]
        deviations.append(float(random_effect.get("t",np.asarray(random_effect)[-1])))
    representatives["EB_random_deviation"]=np.asarray(deviations)
    representatives["EB_total_slope"]=float(result.fe_params["t"])+np.asarray(deviations)
    if predictor_included:
        representatives["EB_total_slope"] += float(result.fe_params["x:t"])*representatives["x"]
    truth=data.groupby("pid",as_index=False).head(1)[["pid","true_b1"]]
    representatives=representatives.merge(truth,on="pid",validate="one_to_one")
    assert "true_b1" in representatives and not any(column.startswith("true_b1_") for column in representatives)
    return representatives


def uncertainty_aware_slopes(data,residual_variance):
    rows=[]
    for participant_id,group in data.groupby("pid",sort=False):
        centered=group.t.to_numpy(float)-float(group.t.mean())
        information=float(np.sum(centered**2))
        rows.append({
            "pid":participant_id,"x":float(group.x.iloc[0]),
            "unshrunk_slope":float(np.sum(centered*group.y.to_numpy(float))/information),
            "slope_sampling_variance":float(residual_variance)/information,
        })
    return pd.DataFrame(rows)


def simulated_dataset(source_parameters, regime, effect_condition, seed, force_zero_slope=False):
    parameters=source_parameters[regime]
    rng=np.random.default_rng(int(seed))
    data=parameters["data"].copy()
    participant_ids=data.pid.unique()
    predictor_map=dict(zip(participant_ids,rng.normal(size=len(participant_ids))))
    data["x"]=data.pid.map(predictor_map)
    beta_x=.10*parameters["outcome_SD"]
    beta_xt=0.0 if effect_condition=="beta_xt_zero" else .10*parameters["outcome_SD"]
    if force_zero_slope or regime=="true_zero_slope_variance":
        random_covariance=np.array([[max(parameters["RI_intercept_variance"],1e-8),0.0],[0.0,0.0]])
        residual_variance=parameters["RI_residual_variance"]
    else:
        random_covariance=parameters["reference_random_covariance"]
        residual_variance=parameters["reference_residual_variance"]
    random_effects=rng.multivariate_normal(np.zeros(2),random_covariance,size=len(participant_ids),check_valid="ignore")
    random_map=dict(zip(participant_ids,random_effects))
    data["true_b1"]=[random_map[participant_id][1] for participant_id in data.pid]
    random_intercept=np.array([random_map[participant_id][0] for participant_id in data.pid])
    data["y"]=(
        parameters["beta0"]+parameters["beta_t"]*data.t+beta_x*data.x+beta_xt*data.x*data.t
        +random_intercept+data["true_b1"]*data.t
        +rng.normal(0,np.sqrt(residual_variance),len(data))
    )
    return data,beta_xt,float(random_covariance[1,1]),parameters["outcome_SD"]


def attach_task_metadata(rows,metadata):
    return [{**metadata,**row} for row in rows]


def run_lr_task(source_parameters, row_type, regime, replicate, seed):
    data,_,true_slope_variance,outcome_sd=simulated_dataset(
        source_parameters, regime,"beta_xt_zero",seed,force_zero_slope=True
    )
    alternative,alternative_diagnostic,alternative_attempts=fit_optimizer_sequence(
        data,"y ~ t + x + x:t","1+t","random_slope_primary"
    )
    null,null_diagnostic,null_attempts=fit_optimizer_sequence(
        data,"y ~ t + x + x:t","1","random_intercept_null"
    )
    metadata={
        "row_type":row_type,"regime":regime,"source_outcome":source_parameters[regime]["outcome"],
        "effect_condition":"beta_xt_zero","replicate":int(replicate),"seed":int(seed),
        "true_slope_variance":true_slope_variance,"true_beta_xt":0.0,"outcome_SD":outcome_sd,
    }
    diagnostics=[]
    if alternative_diagnostic is not None:diagnostics.append({**metadata,**alternative_diagnostic})
    if null_diagnostic is not None:diagnostics.append({**metadata,**null_diagnostic})
    attempts=attach_task_metadata(alternative_attempts+null_attempts,metadata)
    valid=alternative is not None and null is not None
    replicate_row={
        **metadata,"task_valid":valid,
        "restricted_LR":max(0.0,2*(float(alternative.llf)-float(null.llf))) if valid else np.nan,
        "error":"" if valid else "one or both required mixed models unavailable",
    }
    return replicate_row,diagnostics,attempts


def run_evaluation_task(source_parameters, regime, effect_condition, replicate, seed):
    data,true_beta_xt,true_slope_variance,outcome_sd=simulated_dataset(source_parameters,regime,effect_condition,seed)
    included,included_diagnostic,included_attempts=fit_optimizer_sequence(
        data,"y ~ t + x + x:t","1+t","random_slope_primary"
    )
    intercept_only,RI_diagnostic,RI_attempts=fit_optimizer_sequence(
        data,"y ~ t + x + x:t","1","random_intercept_null"
    )
    omitted,omitted_diagnostic,omitted_attempts=fit_optimizer_sequence(
        data,"y ~ t + x","1+t","random_slope_predictor_omitted"
    )
    metadata={
        "row_type":"evaluation","regime":regime,"source_outcome":source_parameters[regime]["outcome"],
        "effect_condition":effect_condition,"replicate":int(replicate),"seed":int(seed),
        "true_slope_variance":true_slope_variance,"true_beta_xt":true_beta_xt,"outcome_SD":outcome_sd,
    }
    diagnostics=[]
    for item in [included_diagnostic,RI_diagnostic,omitted_diagnostic]:
        if item is not None:diagnostics.append({**metadata,**item})
    attempts=attach_task_metadata(included_attempts+RI_attempts+omitted_attempts,metadata)
    row={**metadata,"task_valid":False,"error":""}
    method_names=[
        "one_stage_LMM","one_stage_GEE","uncertainty_aware_two_stage",
        "BLUP_as_outcome_attenuation","predictor_double_use_overconfidence",
    ]
    for method in method_names:
        row.update({f"{method}_beta":np.nan,f"{method}_SE":np.nan,f"{method}_p":np.nan,f"{method}_valid":False})
    if included is None or intercept_only is None or omitted is None:
        row["error"]="one or more required mixed models unavailable"
        return row,diagnostics,attempts
    try:
        beta,se=float(included.fe_params["x:t"]),float(included.bse_fe["x:t"])
        row.update({"one_stage_LMM_beta":beta,"one_stage_LMM_SE":se,
                    "one_stage_LMM_p":2*norm.sf(abs(beta/se)),"one_stage_LMM_valid":np.isfinite(beta) and se>0})
        gee=smf.gee(
            "y ~ t + x + x:t",groups="pid",data=data,cov_struct=Exchangeable(),family=sm.families.Gaussian()
        ).fit(cov_type="robust")
        beta,se=float(gee.params["x:t"]),float(gee.bse["x:t"])
        row.update({"one_stage_GEE_beta":beta,"one_stage_GEE_SE":se,
                    "one_stage_GEE_p":2*norm.sf(abs(beta/se)),"one_stage_GEE_valid":np.isfinite(beta) and se>0})

        included_people=eb_slope_table(included,data,True)
        omitted_people=eb_slope_table(omitted,data,False)
        double=smf.ols("EB_total_slope ~ x",data=included_people).fit(cov_type="HC3")
        blup=smf.ols("EB_total_slope ~ x",data=omitted_people).fit(cov_type="HC3")
        for method,fit in [
            ("predictor_double_use_overconfidence",double),
            ("BLUP_as_outcome_attenuation",blup),
        ]:
            beta,se=float(fit.params["x"]),float(fit.bse["x"])
            row.update({f"{method}_beta":beta,f"{method}_SE":se,
                        f"{method}_p":float(fit.pvalues["x"]),f"{method}_valid":np.isfinite(beta) and se>0})

        raw=uncertainty_aware_slopes(data,float(included.scale))
        response,design=patsy.dmatrices("unshrunk_slope ~ x",raw,return_type="dataframe")
        meta_beta,meta_se,tau2=reml_meta_regression(
            response.iloc[:,0],raw.loc[response.index,"slope_sampling_variance"],design,design.columns
        )
        beta,se=float(meta_beta["x"]),float(meta_se["x"])
        row.update({"uncertainty_aware_two_stage_beta":beta,"uncertainty_aware_two_stage_SE":se,
                    "uncertainty_aware_two_stage_p":2*norm.sf(abs(beta/se)),
                    "uncertainty_aware_two_stage_valid":np.isfinite(beta) and se>0,
                    "uncertainty_aware_tau2":tau2})
        row["restricted_LR"]=max(0.0,2*(float(included.llf)-float(intercept_only.llf)))
        primary_geometry=included_diagnostic or {}
        for column in [
            "mean_direct_predictability","median_direct_predictability",
            "mean_full_posterior_predictability","median_full_posterior_predictability",
            "mean_innovation_predictability","median_innovation_predictability",
        ]:
            row[f"primary_{column}"]=primary_geometry.get(column,np.nan)
        row["task_valid"]=bool(all(row[f"{method}_valid"] for method in method_names))
    except Exception as exc:
        row["error"]=repr(exc)
    return row,diagnostics,attempts
