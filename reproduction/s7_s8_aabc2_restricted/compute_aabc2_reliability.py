#!/usr/bin/env python3
"""Compute exact AABC2 slope-reliability summaries.

This restricted-data entry point fits only the reference and original-frozen
mixed models needed for the six-outcome reliability decomposition. It writes
participant-level exact reliability only to ``restricted/`` and writes only
aggregate summaries to ``public/``.

The public schedule table also appends percentile intervals from the frozen
499-replicate thickness/cognition reliability-gap bootstrap supplied through
``--bootstrap-metrics``. Boundary-aware RLRT inference, covariance-geometry
diagnostics, and simulation studies are handled by separate reproduction
steps.
"""

from __future__ import annotations

import os
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from tqdm.auto import tqdm

import aabc2_common as common

OPTIMIZERS = ("lbfgs", "powell", "bfgs")
MAXITER = 1200
SLOPE_BOUNDARY_REL_TOL = 1e-8
PSD_REL_TOL = 1e-8
LLF_REL_TOL = 1e-5
FIXED_AGREEMENT_SE_TOL = 0.25
RELIABILITY_GAP_BOOT_B = 499

parser = argparse.ArgumentParser()
parser.add_argument("--input-config", required=True)
parser.add_argument(
    "--bootstrap-metrics",
    required=True,
    help="Full 09_reliability_gap_bootstrap_replicates.csv (499 per outcome).",
)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()

ROOT_DIR = Path(args.output_dir).expanduser().resolve()
if ROOT_DIR.exists() and any(ROOT_DIR.iterdir()):
    raise FileExistsError(f"Output directory is non-empty: {ROOT_DIR}")
PUBLIC_DIR = ROOT_DIR / "public"
RESTRICTED_DIR = ROOT_DIR / "restricted"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_METRICS = Path(args.bootstrap_metrics).expanduser().resolve()
if not BOOTSTRAP_METRICS.is_file():
    raise FileNotFoundError(BOOTSTRAP_METRICS)

raw = common.build_raw_cohorts(args.input_config)
OUTCOMES, ANALYSIS_DATA, cohort_check = common.build_analysis_data(raw)

def fixed_formula(key):
    return common.fixed_formula(key, OUTCOMES)

def original_frozen_formula(key):
    return common.original_frozen_formula(key, OUTCOMES)

MODEL_SPECS = {
    "reference_unstructured_mid": {
        "re_formula": "1 + t5_mid", "vc_formula": None, "slope_time": "t5_mid",
    },
    "original_frozen_structural": {
        "re_formula": "1",
        "vc_formula": {"slope": "0 + t5_person_centered"},
        "slope_time": "t5_person_centered",
    },
    "original_frozen_anchor": {
        "re_formula": "1", "vc_formula": {"slope": "0 + t5"}, "slope_time": "t5",
    },
}

ORIGINAL_SPEC = {
    key: (
        "original_frozen_structural"
        if OUTCOMES[key]["modality"] == "structural MRI"
        else "original_frozen_anchor"
    )
    for key in OUTCOMES
}

def hessian_diagnostic(result):
    try:
        h = result.model.hessian(result.params_object)
        if isinstance(h, tuple):
            h = h[0]
        info = -(np.asarray(h, float) + np.asarray(h, float).T) / 2
        eig = np.linalg.eigvalsh(info)
        tol = 1e-8 * max(1.0, np.max(np.abs(eig)))
        return bool(np.min(eig) > tol), float(np.min(eig)), float(np.max(eig))
    except Exception:
        return False, np.nan, np.nan

def random_components(result, model_spec):
    residual = float(result.scale)
    if model_spec in ("reference_unstructured_mid", "predictor_omitted_reference"):
        G = np.asarray(result.cov_re, float)
    elif model_spec in ("legacy_independent_baseline", "origin_independent_mid", "original_frozen_structural", "original_frozen_anchor"):
        G = np.diag([float(np.asarray(result.cov_re)[0, 0]), float(np.asarray(result.vcomp)[0])])
    elif model_spec == "random_intercept_only":
        G = np.array([[float(np.asarray(result.cov_re)[0, 0])]])
    else:
        raise KeyError(model_spec)
    eig = np.linalg.eigvalsh((G + G.T) / 2)
    tol = PSD_REL_TOL * max(1.0, np.max(np.abs(eig)), residual)
    psd = bool(np.isfinite(G).all() and residual > 0 and np.min(eig) >= -tol)
    slope = float(G[1, 1]) if G.shape == (2, 2) else np.nan
    boundary_tol = SLOPE_BOUNDARY_REL_TOL * max(1.0, residual)
    boundary = bool(G.shape == (2, 2) and (not np.isfinite(slope) or slope <= boundary_tol))
    return {"G": G, "residual_variance": residual, "random_effect_covariance_psd": psd,
            "slope_variance": slope, "random_slope_boundary": boundary,
            "random_slope_interior": bool(G.shape == (2, 2) and psd and not boundary),
            "random_covariance_min_eigenvalue": float(np.min(eig))}

def build_mixed(d, formula, model_spec):
    s = MODEL_SPECS[model_spec]
    return smf.mixedlm(formula, d, groups=d["pid"], re_formula=s["re_formula"], vc_formula=s["vc_formula"])

def fixed_finite(result):
    try:
        fe = np.asarray(result.fe_params, float)
        se = np.asarray(result.bse_fe, float)
        return bool(np.isfinite(fe).all() and np.isfinite(se).all() and (se >= 0).all())
    except Exception:
        return False

def fits_compatible(a, b):
    fa, fb = a["result"].fe_params, b["result"].fe_params
    common = fa.index.intersection(fb.index)
    if len(common) == 0:
        return False
    scale = np.maximum(a["result"].bse_fe.loc[common].to_numpy(float), b["result"].bse_fe.loc[common].to_numpy(float))
    z = np.max(np.abs(fa.loc[common].to_numpy(float)-fb.loc[common].to_numpy(float))/np.maximum(scale, 1e-8))
    ll_scale = max(1.0, abs(a["llf"]), abs(b["llf"]))
    return bool(abs(a["llf"]-b["llf"]) <= LLF_REL_TOL*ll_scale and z <= FIXED_AGREEMENT_SE_TOL)

def fit_optimizer_panel(key, model_spec, formula_override=None):
    d = ANALYSIS_DATA[key]
    formula = fixed_formula(key) if formula_override is None else formula_override
    ymat, xmat = patsy.dmatrices(formula, d, return_type="dataframe")
    rank, columns = int(np.linalg.matrix_rank(xmat.to_numpy())), int(xmat.shape[1])
    attempts, successful = [], []
    for optimizer in OPTIMIZERS:
        caught = []
        try:
            model = build_mixed(d, formula, model_spec)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                result = model.fit(reml=True, method=optimizer, maxiter=MAXITER, disp=False)
                caught = [str(w.message) for w in ws]
            comp = random_components(result, model_spec)
            hpd, hmin, hmax = hessian_diagnostic(result)
            rec = {
                "outcome": key, "label": OUTCOMES[key]["label"], "model_specification": model_spec,
                "optimizer": optimizer, "optimizer_converged": bool(result.converged), "llf": float(result.llf),
                "fixed_design_rank": rank, "fixed_design_columns": columns, "fixed_design_full_rank": rank == columns,
                "fixed_parameters_finite": fixed_finite(result), "full_hessian_positive_definite": hpd,
                "hessian_min_information_eigenvalue": hmin, "hessian_max_information_eigenvalue": hmax,
                "participants": d.pid.nunique(), "observations": len(d), "warning_count": len(caught),
                "warnings": " | ".join(sorted(set(caught))), "error": "",
                **{k: v for k, v in comp.items() if k != "G"},
            }
            attempts.append(rec)
            if rec["optimizer_converged"] and rec["fixed_design_full_rank"] and rec["fixed_parameters_finite"] and np.isfinite(rec["llf"]):
                successful.append({"result": result, "record": rec, "llf": rec["llf"], "optimizer": optimizer})
        except Exception as exc:
            attempts.append({
                "outcome": key, "label": OUTCOMES[key]["label"], "model_specification": model_spec,
                "optimizer": optimizer, "optimizer_converged": False, "llf": np.nan,
                "fixed_design_rank": rank, "fixed_design_columns": columns, "fixed_design_full_rank": rank == columns,
                "fixed_parameters_finite": False, "full_hessian_positive_definite": False,
                "hessian_min_information_eigenvalue": np.nan, "hessian_max_information_eigenvalue": np.nan,
                "participants": d.pid.nunique(), "observations": len(d), "warning_count": len(caught),
                "warnings": " | ".join(sorted(set(caught))), "error": repr(exc),
                "random_effect_covariance_psd": False, "slope_variance": np.nan,
                "random_slope_boundary": np.nan, "random_slope_interior": np.nan,
                "random_covariance_min_eigenvalue": np.nan, "residual_variance": np.nan,
            })
    if not successful:
        raise RuntimeError(f"No usable optimizer for {key}, {model_spec}")
    for a in successful:
        a["agreement_count"] = sum(fits_compatible(a, b) for b in successful)
    best_count = max(a["agreement_count"] for a in successful)
    pool = [a for a in successful if a["agreement_count"] == best_count]
    psd_pool = [a for a in pool if a["record"]["random_effect_covariance_psd"]]
    selected = max(psd_pool or pool, key=lambda a: a["llf"])
    optimizer_agreement = best_count >= 2
    fixed_effect_usable = bool(selected["record"]["optimizer_converged"] and optimizer_agreement and selected["record"]["fixed_design_full_rank"] and selected["record"]["fixed_parameters_finite"])
    audit = pd.DataFrame(attempts)
    audit["selected"] = audit["optimizer"].eq(selected["optimizer"])
    audit["optimizer_common_solution_count"] = best_count
    audit["optimizer_agreement"] = optimizer_agreement
    audit["fixed_effect_usable"] = audit["selected"] & fixed_effect_usable
    return {"result": selected["result"], "data": d, "formula": formula, "model_spec": model_spec,
            "selected_optimizer": selected["optimizer"], "optimizer_agreement": optimizer_agreement,
            "fixed_effect_usable": fixed_effect_usable, "audit": audit,
            "components": random_components(selected["result"], model_spec)}

def clamp_rounding_only(x, tol=1e-7):
    if not np.isfinite(x):
        return np.nan
    if -tol <= x <= 1 + tol:
        return float(np.clip(x, 0, 1))
    return np.nan

def direct_grr(slope_var, residual_var, I_i):
    if not (np.isfinite(slope_var) and slope_var > 0 and np.isfinite(residual_var) and residual_var > 0 and I_i > 0):
        return np.nan
    return float(slope_var / (slope_var + residual_var / I_i))

# Fit only the 12 model panels required by the reliability decomposition.
FITS = {}
fit_audits = []
for key in tqdm(OUTCOMES, desc="Reliability model panels", unit="outcome"):
    for model_spec, formula in (
        ("reference_unstructured_mid", fixed_formula(key)),
        (ORIGINAL_SPEC[key], original_frozen_formula(key)),
    ):
        fit = fit_optimizer_panel(key, model_spec, formula_override=formula)
        FITS[(key, model_spec)] = fit
        fit_audits.append(fit["audit"])

reliability_model_fit_audit = pd.concat(fit_audits, ignore_index=True)

reliability_rows = []
for key in tqdm(OUTCOMES, desc="Exact slope reliability", unit="outcome"):
    ref = FITS[(key, "reference_unstructured_mid")]
    original = FITS[(key, ORIGINAL_SPEC[key])]
    d = ref["data"]
    G = np.asarray(ref["components"]["G"], float)
    residual = float(ref["components"]["residual_variance"])
    G00, G01, G11 = float(G[0, 0]), float(G[0, 1]), float(G[1, 1])
    corr = G01 / np.sqrt(G00 * G11) if G00 > 0 and G11 > 0 else np.nan
    innovation_a = G01 / G00 if G00 > 0 else np.nan
    innovation_prior_var = G11 - G01**2 / G00 if G00 > 0 else np.nan
    ref_interior = (
        ref["components"]["random_slope_interior"]
        and ref["components"]["random_effect_covariance_psd"]
    )
    Ginv = (
        np.linalg.inv(G)
        if ref_interior and np.min(np.linalg.eigvalsh(G)) > 0
        else None
    )

    orig_comp = original["components"]
    orig_time = MODEL_SPECS[ORIGINAL_SPEC[key]]["slope_time"]

    for pid, g in d.groupby("pid", sort=False):
        information = float(np.sum((g.t5_mid - g.t5_mid.mean()) ** 2))
        direct = direct_grr(G11, residual, information)
        full = innovation = post_slope_var = post_innovation_var = np.nan

        if Ginv is not None:
            Z = np.column_stack([np.ones(len(g)), g.t5_mid.to_numpy(float)])
            posterior_cov = np.linalg.inv(Ginv + Z.T @ Z / residual)
            post_slope_var = float(posterior_cov[1, 1])
            full = clamp_rounding_only(1 - post_slope_var / G11)
            L = np.array([-innovation_a, 1.0])
            post_innovation_var = float(L @ posterior_cov @ L)
            innovation = (
                clamp_rounding_only(1 - post_innovation_var / innovation_prior_var)
                if innovation_prior_var > 0
                else np.nan
            )

        go = original["data"].loc[original["data"].pid.eq(pid)]
        orig_information = float(np.sum((go[orig_time] - go[orig_time].mean()) ** 2))
        original_R = direct_grr(
            orig_comp["slope_variance"],
            orig_comp["residual_variance"],
            orig_information,
        )

        reliability_rows.append({
            "outcome": key,
            "label": OUTCOMES[key]["label"],
            "modality": OUTCOMES[key]["modality"],
            "participant_id": pid,
            "scan_count": len(g),
            "followup_years": float((g.t5.max() - g.t5.min()) * 5),
            "random_intercept_variance": G00,
            "random_slope_variance": G11,
            "intercept_slope_covariance": G01,
            "intercept_slope_correlation": corr,
            "residual_variance": residual,
            "direct_schedule_information_I_i": information,
            "original_frozen_model_specification": ORIGINAL_SPEC[key],
            "original_frozen_R_i": original_R,
            "same_reference_fit_direct_GRR": direct,
            "full_posterior_slope_predictability": full,
            "conditional_slope_innovation_prior_variance": innovation_prior_var,
            "posterior_slope_variance": post_slope_var,
            "posterior_innovation_variance": post_innovation_var,
            "conditional_slope_innovation_reliability": innovation,
            "level_assisted_reliability_gain": (
                full - direct
                if np.isfinite(full) and np.isfinite(direct)
                else np.nan
            ),
        })

exact_posterior_slope_reliability = pd.DataFrame(reliability_rows)
exact_posterior_slope_reliability["followup_band"] = pd.cut(
    exact_posterior_slope_reliability.followup_years,
    [-np.inf, 2, 4, np.inf],
    labels=["<=2y", "2-4y", ">4y"],
)

METRICS = [
    "original_frozen_R_i",
    "same_reference_fit_direct_GRR",
    "full_posterior_slope_predictability",
    "conditional_slope_innovation_reliability",
    "level_assisted_reliability_gain",
]
agg = {}
for metric in METRICS:
    agg[f"median_{metric}"] = (metric, "median")
    agg[f"q25_{metric}"] = (metric, lambda x: x.quantile(.25))
    agg[f"q75_{metric}"] = (metric, lambda x: x.quantile(.75))
    agg[f"mean_{metric}"] = (metric, "mean")
agg.update({
    "participants": ("participant_id", "nunique"),
    "median_followup_years": ("followup_years", "median"),
    "n_direct_ge_025": ("same_reference_fit_direct_GRR", lambda x: int((x >= .25).sum())),
    "n_direct_ge_040": ("same_reference_fit_direct_GRR", lambda x: int((x >= .40).sum())),
    "sum_direct_schedule_GRR": ("same_reference_fit_direct_GRR", "sum"),
    "sum_full_posterior_predictability": ("full_posterior_slope_predictability", "sum"),
    "sum_conditional_innovation_reliability": ("conditional_slope_innovation_reliability", "sum"),
})

reliability_by_schedule = (
    exact_posterior_slope_reliability
    .groupby(
        ["outcome", "label", "modality", "scan_count", "followup_band"],
        observed=True,
        dropna=False,
    )
    .agg(**agg)
    .reset_index()
)

reliability_summary = (
    exact_posterior_slope_reliability
    .groupby(["outcome", "label", "modality"])
    .agg(
        participants=("participant_id", "nunique"),
        original_frozen_R_i=("original_frozen_R_i", "median"),
        direct_schedule_GRR=("same_reference_fit_direct_GRR", "median"),
        full_posterior_slope_predictability=("full_posterior_slope_predictability", "median"),
        conditional_slope_innovation_reliability=("conditional_slope_innovation_reliability", "median"),
        level_assisted_reliability_gain=("level_assisted_reliability_gain", "median"),
        intercept_slope_correlation=("intercept_slope_correlation", "first"),
    )
    .reset_index()
)

# Frozen full reliability-gap bootstrap contract.
bootstrap = pd.read_csv(BOOTSTRAP_METRICS, low_memory=False)
required_bootstrap = {
    "outcome", "replicate", "valid", "median_direct_GRR",
    "median_full_predictability", "median_innovation_reliability",
    "median_full_minus_direct_gap", "median_full_minus_innovation_gap",
}
missing = required_bootstrap - set(bootstrap.columns)
if missing:
    raise ValueError(f"Bootstrap metrics missing columns: {sorted(missing)}")

if bootstrap["valid"].dtype != bool:
    bootstrap["valid"] = (
        bootstrap["valid"].astype(str).str.lower().map({"true": True, "false": False})
    )
if bootstrap["valid"].isna().any():
    raise ValueError("Bootstrap valid column contains unrecognized values")

expected = pd.Series({"global_mean_thickness": 499, "fluid_cognition": 499})
counts = bootstrap.groupby("outcome").replicate.nunique().reindex(expected.index)
if not counts.equals(expected):
    raise RuntimeError("Unexpected reliability-gap replicate counts:\n" + counts.to_string())
if not bootstrap["valid"].all():
    raise RuntimeError("Final reliability summary requires the frozen all-valid 998-row bootstrap state")

interval_rows = []
for key, group in bootstrap.loc[bootstrap.valid].groupby("outcome"):
    for metric in [
        "median_direct_GRR",
        "median_full_predictability",
        "median_innovation_reliability",
        "median_full_minus_direct_gap",
        "median_full_minus_innovation_gap",
    ]:
        low, median, high = group[metric].quantile([.025, .5, .975])
        interval_rows.append({
            "status": "completed",
            "outcome": key,
            "metric": metric,
            "requested_B": RELIABILITY_GAP_BOOT_B,
            "valid_B": len(group),
            "validity_rate": len(group) / RELIABILITY_GAP_BOOT_B,
            "percentile_2.5": low,
            "bootstrap_median": median,
            "percentile_97.5": high,
        })
reliability_gap_intervals = pd.DataFrame(interval_rows)

reliability_export_by_schedule = pd.concat(
    [
        reliability_by_schedule.assign(row_type="schedule_summary"),
        reliability_gap_intervals.assign(row_type="outcome_bootstrap_interval"),
    ],
    ignore_index=True,
    sort=False,
)

# Public/restricted routing.
reliability_model_fit_audit.to_csv(
    PUBLIC_DIR / "04a_reliability_model_fit_audit.csv", index=False
)
reliability_summary.to_csv(
    PUBLIC_DIR / "04b_reliability_summary.csv", index=False
)
reliability_export_by_schedule.to_csv(
    PUBLIC_DIR / "05_reliability_by_schedule.csv", index=False
)
exact_posterior_slope_reliability.to_csv(
    RESTRICTED_DIR / "04_exact_posterior_slope_reliability.csv", index=False
)

# Privacy and shape invariants.
assert exact_posterior_slope_reliability.participant_id.nunique() >= 1
for frame in (reliability_model_fit_audit, reliability_summary, reliability_export_by_schedule):
    forbidden = {"participant_id", "pid", "id_event", "original_participant_id"} & set(frame.columns)
    if forbidden:
        raise RuntimeError(f"Public reliability output contains participant columns: {sorted(forbidden)}")
assert not (PUBLIC_DIR / "04_exact_posterior_slope_reliability.csv").exists()
assert (RESTRICTED_DIR / "04_exact_posterior_slope_reliability.csv").exists()

print("AABC2 reliability summaries complete.")
print("participant-level rows (restricted):", len(exact_posterior_slope_reliability))
print("schedule/bootstrap rows (public):", len(reliability_export_by_schedule))
print("PARTICIPANT-LEVEL TABLES EXPORTED PUBLICLY: 0")
