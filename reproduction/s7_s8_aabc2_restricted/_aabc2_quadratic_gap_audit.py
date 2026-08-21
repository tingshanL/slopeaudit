#!/usr/bin/env python3
"""Restricted-data reproduction of the quadratic within-person gap audit.

This refactors the frozen
AABC_quadratic_within_person_gap_audit.ipynb
workflow into an explicit non-notebook release entry point.

The 60-row selection manifest from the preceding gap-optimizer workflow
must be supplied explicitly. No recursive file discovery or fallback to
the 499-row bootstrap table is used.

Public aggregate outputs and restricted participant-level outputs are
separated. The resumable bootstrap checkpoint is private.
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

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
from scipy import stats
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import aabc2_common as common


# ---------------------------------------------------------------------------
# Frozen controls
# ---------------------------------------------------------------------------

SEED = 20260718

TARGET_OUTCOMES = (
    "global_mean_thickness",
    "fluid_cognition",
)

TRAJECTORIES = (
    "linear",
    "quadratic",
)

OPTIMIZERS = (
    "lbfgs",
    "powell",
    "bfgs",
)

MAXITER = 1200

GAP_METRIC = "median_full_minus_direct_gap"
N_PER_REGIME = 10
QUADRATIC_TERM = "I(t5_mid ** 2)"

LLF_REL_TOL = 1e-5
FIXED_AGREEMENT_SE_TOL = 0.25
PSD_REL_TOL = 1e-8
BOUNDARY_REL_TOL = 1e-8

# Descriptive mechanism thresholds inherited from the prior gap audit.
NEAR_ABS_RHO = 0.95
SMALL_INNOVATION_FRACTION = 0.10
HIGH_FULL_PREDICTABILITY = 0.40
LOW_DIRECT_RELIABILITY = 0.25
LOW_INNOVATION_RELIABILITY = 0.25


REQUIRED_SELECTION_COLUMNS = {
    "outcome",
    "replicate",
    "gap_regime",
    "bootstrap_seed",
    GAP_METRIC,
}


# ---------------------------------------------------------------------------
# CLI / output policy
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input-config",
    required=True,
    help="Private JSON containing explicit restricted AABC input paths.",
)

parser.add_argument(
    "--selected-manifest",
    required=True,
    help=(
        "Explicit upstream 01_selected_gap_replicates.csv "
        "from gap_optimizer_diagnostic.py."
    ),
)

parser.add_argument(
    "--output-dir",
    required=True,
)

parser.add_argument(
    "--n-jobs",
    type=int,
    default=2,
    help="Parallel workers; capped at 3 to match the validated workflow.",
)

parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from the private quadratic bootstrap checkpoint.",
)

args = parser.parse_args()

N_JOBS = max(
    1,
    min(
        int(args.n_jobs),
        3,
    ),
)

ROOT_DIR = (
    Path(args.output_dir)
    .expanduser()
    .resolve()
)

if ROOT_DIR.exists():
    nonempty = any(ROOT_DIR.iterdir())

    if nonempty and not args.resume:
        raise FileExistsError(
            f"Output directory is non-empty: {ROOT_DIR}. "
            "Use a fresh directory or pass --resume."
        )

PUBLIC_DIR = ROOT_DIR / "public"
RESTRICTED_DIR = ROOT_DIR / "restricted"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

for _dir in (
    PUBLIC_DIR,
    RESTRICTED_DIR,
    CHECKPOINT_DIR,
):
    _dir.mkdir(
        parents=True,
        exist_ok=True,
    )

CHECKPOINT_FILE = (
    CHECKPOINT_DIR
    / "_quadratic_selected_resamples_checkpoint.pkl"
)


def display(*args, **kwargs):
    """Suppress notebook-only display calls."""
    return None


# ---------------------------------------------------------------------------
# Validated common restricted-data loader
# ---------------------------------------------------------------------------

raw = common.build_raw_cohorts(
    args.input_config
)

OUTCOMES, ANALYSIS_DATA, cohort_check = (
    common.build_analysis_data(raw)
)


def fixed_formula(key):
    return common.fixed_formula(
        key,
        OUTCOMES,
    )


# ==============================================================================
# trajectory_formula — frozen notebook cell 8
# ==============================================================================
def trajectory_formula(outcome, trajectory):
    if trajectory not in TRAJECTORIES:
        raise ValueError(f"Unknown trajectory: {trajectory}")
    base = fixed_formula(outcome)
    return base if trajectory == "linear" else f"{base} + {QUADRATIC_TERM}"

# ==============================================================================
# clamp_rounding_only — frozen notebook cell 12
# ==============================================================================
def clamp_rounding_only(value, tolerance=1e-7):
    if not np.isfinite(value):
        return np.nan
    if -tolerance <= value <= 1+tolerance:
        return float(np.clip(value, 0, 1))
    return np.nan

# ==============================================================================
# direct_reliability — frozen notebook cell 12
# ==============================================================================
def direct_reliability(slope_variance, residual_variance, information):
    if not (
        np.isfinite(slope_variance) and slope_variance > 0
        and np.isfinite(residual_variance) and residual_variance > 0
        and np.isfinite(information) and information > 0
    ):
        return np.nan
    return float(slope_variance/(slope_variance + residual_variance/information))

# ==============================================================================
# hessian_diagnostic — frozen notebook cell 12
# ==============================================================================
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

# ==============================================================================
# fixed_parameters_finite — frozen notebook cell 12
# ==============================================================================
def fixed_parameters_finite(result):
    try:
        estimates = np.asarray(result.fe_params, float)
        errors = np.asarray(result.bse_fe, float)
        return bool(np.isfinite(estimates).all() and np.isfinite(errors).all() and (errors>=0).all())
    except Exception:
        return False

# ==============================================================================
# reconstruct_bootstrap_dataset — frozen notebook cell 12
# ==============================================================================
def reconstruct_bootstrap_dataset(outcome, replicate, seed):
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
    data = pd.concat(pieces, ignore_index=True)
    assert data.pid.nunique() == len(participant_ids)
    return data

# ==============================================================================
# covariance_and_predictability — frozen notebook cell 12
# ==============================================================================
def covariance_and_predictability(result, data):
    random_covariance = np.asarray(result.cov_re, float)
    residual_variance = float(result.scale)
    symmetric_covariance = (random_covariance+random_covariance.T)/2
    eigenvalues = np.linalg.eigvalsh(symmetric_covariance)
    minimum_eigenvalue = float(np.min(eigenvalues))
    covariance_trace = float(np.trace(random_covariance))
    scale = max(1.0, float(np.max(np.abs(eigenvalues))), abs(residual_variance))
    covariance_psd = bool(
        np.isfinite(random_covariance).all() and residual_variance>0
        and minimum_eigenvalue >= -PSD_REL_TOL*scale
    )

    intercept_variance = float(random_covariance[0,0])
    slope_variance = float(random_covariance[1,1])
    intercept_slope_covariance = float(random_covariance[0,1])
    rho = (
        intercept_slope_covariance/np.sqrt(intercept_variance*slope_variance)
        if intercept_variance>0 and slope_variance>0 else np.nan
    )
    one_minus_rho_squared = 1-rho**2 if np.isfinite(rho) else np.nan
    normalized_determinant = (
        float(np.linalg.det(random_covariance)/(intercept_variance*slope_variance))
        if intercept_variance>0 and slope_variance>0 else np.nan
    )
    innovation_variance = (
        slope_variance-intercept_slope_covariance**2/intercept_variance
        if intercept_variance>0 else np.nan
    )
    innovation_fraction = (
        innovation_variance/slope_variance
        if slope_variance>0 and np.isfinite(innovation_variance) else np.nan
    )

    participant_rows = []
    inverse_random = None
    if minimum_eigenvalue>0 and residual_variance>0 and slope_variance>0:
        inverse_random = np.linalg.inv(random_covariance)
    innovation_loading = (
        intercept_slope_covariance/intercept_variance if intercept_variance>0 else np.nan
    )
    for participant_id, group in data.groupby("pid", sort=False):
        information = float(np.sum((group.t5_mid-group.t5_mid.mean())**2))
        direct = direct_reliability(slope_variance, residual_variance, information)
        full = np.nan
        innovation = np.nan
        if inverse_random is not None:
            random_design = np.column_stack([
                np.ones(len(group)), group.t5_mid.to_numpy(float)
            ])
            posterior_covariance = np.linalg.inv(
                inverse_random + random_design.T@random_design/residual_variance
            )
            full = clamp_rounding_only(1-posterior_covariance[1,1]/slope_variance)
            if innovation_variance>0:
                transform = np.array([-innovation_loading, 1.0])
                posterior_innovation = float(transform@posterior_covariance@transform)
                innovation = clamp_rounding_only(1-posterior_innovation/innovation_variance)
        first = group.iloc[0]
        participant_rows.append({
            "participant_boot_id": participant_id,
            "original_participant_id": first.get("original_participant_id", str(participant_id)),
            "bootstrap_draw_index": first.get("bootstrap_draw_index", np.nan),
            "scan_count": len(group),
            "followup_years": float((group.t5.max()-group.t5.min())*5),
            "schedule_information": information,
            "direct_reliability": direct,
            "full_posterior_predictability": full,
            "innovation_reliability": innovation,
        })
    participant_metrics = pd.DataFrame(participant_rows)

    summary = {
        "random_intercept_variance": intercept_variance,
        "random_slope_variance": slope_variance,
        "intercept_slope_covariance": intercept_slope_covariance,
        "intercept_slope_correlation": rho,
        "absolute_intercept_slope_correlation": abs(rho) if np.isfinite(rho) else np.nan,
        "one_minus_rho_squared": one_minus_rho_squared,
        "normalized_determinant": normalized_determinant,
        "random_covariance_min_eigenvalue": minimum_eigenvalue,
        "random_covariance_trace": covariance_trace,
        "minimum_eigenvalue_fraction_of_trace": (
            minimum_eigenvalue/covariance_trace if covariance_trace>0 else np.nan
        ),
        "random_covariance_psd": covariance_psd,
        "residual_variance": residual_variance,
        "slope_innovation_variance": innovation_variance,
        "innovation_variance_fraction_of_slope": innovation_fraction,
        "intercept_variance_boundary": bool(
            not np.isfinite(intercept_variance) or intercept_variance<=BOUNDARY_REL_TOL*scale
        ),
        "slope_variance_boundary": bool(
            not np.isfinite(slope_variance) or slope_variance<=BOUNDARY_REL_TOL*scale
        ),
        "covariance_boundary": bool(minimum_eigenvalue<=BOUNDARY_REL_TOL*scale),
    }
    summary["any_boundary_flag"] = bool(
        summary["intercept_variance_boundary"]
        or summary["slope_variance_boundary"]
        or summary["covariance_boundary"]
    )
    for source_column, label in [
        ("direct_reliability", "direct_reliability"),
        ("full_posterior_predictability", "full_posterior_predictability"),
        ("innovation_reliability", "innovation_reliability"),
    ]:
        values = participant_metrics[source_column]
        summary[f"median_{label}"] = float(values.median())
        summary[f"q25_{label}"] = float(values.quantile(.25))
        summary[f"q75_{label}"] = float(values.quantile(.75))
    summary["median_full_minus_direct_gap"] = (
        summary["median_full_posterior_predictability"]
        - summary["median_direct_reliability"]
    )
    summary["median_full_minus_innovation_gap"] = (
        summary["median_full_posterior_predictability"]
        - summary["median_innovation_reliability"]
    )
    return summary, participant_metrics

# ==============================================================================
# results_compatible — frozen notebook cell 12
# ==============================================================================
def results_compatible(first, second):
    if first is None or second is None:
        return False
    common = first.fe_params.index.intersection(second.fe_params.index)
    if not len(common):
        return False
    scale = np.maximum(
        first.bse_fe.loc[common].to_numpy(float),
        second.bse_fe.loc[common].to_numpy(float),
    )
    fixed_distance = float(np.max(
        np.abs(
            first.fe_params.loc[common].to_numpy(float)
            - second.fe_params.loc[common].to_numpy(float)
        )/np.maximum(scale, 1e-8)
    ))
    likelihood_scale = max(1.0, abs(float(first.llf)), abs(float(second.llf)))
    return bool(
        abs(float(first.llf)-float(second.llf)) <= LLF_REL_TOL*likelihood_scale
        and fixed_distance <= FIXED_AGREEMENT_SE_TOL
    )

# ==============================================================================
# fit_one_optimizer — frozen notebook cell 13
# ==============================================================================
def fit_one_optimizer(data, outcome, trajectory, optimizer, reml):
    formula = trajectory_formula(outcome, trajectory)
    caught_warnings = []
    result = None
    try:
        y_matrix, x_matrix = patsy.dmatrices(formula, data, return_type="dataframe")
        design_rank = int(np.linalg.matrix_rank(x_matrix.to_numpy(float)))
        design_columns = int(x_matrix.shape[1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = smf.mixedlm(
                formula, data=data, groups=data["pid"], re_formula="1+t5_mid"
            ).fit(reml=reml, method=optimizer, maxiter=MAXITER, disp=False)
            caught_warnings = [str(item.message) for item in caught]

        geometry, participant_metrics = covariance_and_predictability(result, data)
        hessian_pd, hessian_minimum, hessian_maximum = hessian_diagnostic(result)
        quadratic_coefficient = float(result.fe_params.get(QUADRATIC_TERM, np.nan))
        quadratic_se = float(result.bse_fe.get(QUADRATIC_TERM, np.nan))
        quadratic_p = (
            float(2*stats.norm.sf(abs(quadratic_coefficient/quadratic_se)))
            if np.isfinite(quadratic_coefficient) and quadratic_se>0 else np.nan
        )
        record = {
            "outcome": outcome, "trajectory": trajectory,
            "estimation": "REML" if reml else "ML", "optimizer": optimizer,
            "formula": formula, "fit_succeeded": True,
            "optimizer_converged": bool(result.converged),
            "log_likelihood": float(result.llf),
            "AIC": float(result.aic) if np.isfinite(result.aic) else np.nan,
            "BIC": float(result.bic) if np.isfinite(result.bic) else np.nan,
            "fixed_design_rank": design_rank, "fixed_design_columns": design_columns,
            "fixed_design_full_rank": design_rank==design_columns,
            "fixed_parameters_finite": fixed_parameters_finite(result),
            "hessian_positive_definite": hessian_pd,
            "hessian_min_information_eigenvalue": hessian_minimum,
            "hessian_max_information_eigenvalue": hessian_maximum,
            "quadratic_coefficient": quadratic_coefficient,
            "quadratic_coefficient_SE": quadratic_se,
            "quadratic_coefficient_p": quadratic_p,
            "fixed_effects_json": json.dumps({str(k): float(v) for k,v in result.fe_params.items()}),
            "fixed_effect_SE_json": json.dumps({str(k): float(v) for k,v in result.bse_fe.items()}),
            "warning_count": len(caught_warnings),
            "warnings": " | ".join(sorted(set(caught_warnings))), "error": "",
            **geometry,
        }
        record["basic_numerical_validity"] = bool(
            record["optimizer_converged"] and record["fixed_design_full_rank"]
            and record["fixed_parameters_finite"] and record["random_covariance_psd"]
            and np.isfinite(record["log_likelihood"])
        )
        record["strict_numerical_validity"] = bool(
            record["basic_numerical_validity"] and record["hessian_positive_definite"]
        )
        return result, record, participant_metrics
    except Exception as exc:
        empty = {
            "outcome": outcome, "trajectory": trajectory,
            "estimation": "REML" if reml else "ML", "optimizer": optimizer,
            "formula": formula, "fit_succeeded": False, "optimizer_converged": False,
            "log_likelihood": np.nan, "AIC": np.nan, "BIC": np.nan,
            "fixed_design_rank": np.nan, "fixed_design_columns": np.nan,
            "fixed_design_full_rank": False, "fixed_parameters_finite": False,
            "hessian_positive_definite": False,
            "hessian_min_information_eigenvalue": np.nan,
            "hessian_max_information_eigenvalue": np.nan,
            "quadratic_coefficient": np.nan, "quadratic_coefficient_SE": np.nan,
            "quadratic_coefficient_p": np.nan,
            "random_intercept_variance": np.nan, "random_slope_variance": np.nan,
            "intercept_slope_covariance": np.nan,
            "intercept_slope_correlation": np.nan,
            "absolute_intercept_slope_correlation": np.nan,
            "one_minus_rho_squared": np.nan, "normalized_determinant": np.nan,
            "random_covariance_min_eigenvalue": np.nan,
            "random_covariance_trace": np.nan,
            "minimum_eigenvalue_fraction_of_trace": np.nan,
            "random_covariance_psd": False, "residual_variance": np.nan,
            "slope_innovation_variance": np.nan,
            "innovation_variance_fraction_of_slope": np.nan,
            "intercept_variance_boundary": False, "slope_variance_boundary": False,
            "covariance_boundary": False, "any_boundary_flag": False,
            "median_direct_reliability": np.nan,
            "median_full_posterior_predictability": np.nan,
            "median_innovation_reliability": np.nan,
            "median_full_minus_direct_gap": np.nan,
            "median_full_minus_innovation_gap": np.nan,
            "fixed_effects_json": "{}", "fixed_effect_SE_json": "{}",
            "warning_count": len(caught_warnings),
            "warnings": " | ".join(sorted(set(caught_warnings))),
            "error": repr(exc), "basic_numerical_validity": False,
            "strict_numerical_validity": False,
        }
        return result, empty, pd.DataFrame()

# ==============================================================================
# fit_optimizer_panel — frozen notebook cell 13
# ==============================================================================
def fit_optimizer_panel(data, outcome, trajectory, reml):
    results, records, participant_tables = {}, [], {}
    for optimizer in OPTIMIZERS:
        result, record, participant_metrics = fit_one_optimizer(
            data, outcome, trajectory, optimizer, reml
        )
        results[optimizer] = result
        records.append(record)
        participant_tables[optimizer] = participant_metrics

    basic_eligible = {
        row["optimizer"]: bool(row["basic_numerical_validity"]) for row in records
    }
    strict_eligible = {
        row["optimizer"]: bool(row["strict_numerical_validity"]) for row in records
    }
    for row in records:
        optimizer = row["optimizer"]
        row["optimizer_common_solution_count"] = (
            sum(
                basic_eligible[other] and results_compatible(results[optimizer], results[other])
                for other in OPTIMIZERS
            ) if basic_eligible[optimizer] else 0
        )
        row["hessian_positive_common_solution_count"] = (
            sum(
                strict_eligible[other] and results_compatible(results[optimizer], results[other])
                for other in OPTIMIZERS
            ) if strict_eligible[optimizer] else 0
        )
        row["at_least_two_optimizers_agree"] = row["optimizer_common_solution_count"]>=2
        row["hessian_positive_optimizer_agreement"] = (
            row["hessian_positive_common_solution_count"]>=2
        )
        row["selected_basic_representative"] = False
        row["selected_strict_representative"] = False

    basic_candidates = [row for row in records if row["at_least_two_optimizers_agree"]]
    if basic_candidates:
        selected = max(
            basic_candidates,
            key=lambda row: (row["optimizer_common_solution_count"], row["log_likelihood"]),
        )
        selected["selected_basic_representative"] = True

    strict_candidates = [row for row in records if row["hessian_positive_optimizer_agreement"]]
    selected_participants = pd.DataFrame()
    if strict_candidates:
        selected = max(
            strict_candidates,
            key=lambda row: (
                row["hessian_positive_common_solution_count"],
                row["optimizer_common_solution_count"], row["log_likelihood"],
            ),
        )
        selected["selected_strict_representative"] = True
        selected_participants = participant_tables[selected["optimizer"]].copy()
        if not selected_participants.empty:
            selected_participants["representative_optimizer"] = selected["optimizer"]
    return records, selected_participants

# ==============================================================================
# fit_selected_resample — frozen notebook cell 18
# ==============================================================================
def fit_selected_resample(selection_record):
    selection_record = dict(selection_record)
    outcome = str(selection_record["outcome"])
    replicate = int(selection_record["replicate"])
    data = reconstruct_bootstrap_dataset(
        outcome, replicate, int(selection_record["bootstrap_seed"])
    )
    optimizer_rows = []
    representative_participant_tables = []
    for trajectory in TRAJECTORIES:
        records, participant_metrics = fit_optimizer_panel(
            data, outcome, trajectory, reml=True
        )
        metadata = {
            "replicate": replicate,
            "gap_regime": str(selection_record["gap_regime"]),
            "selection_gap": float(selection_record[GAP_METRIC]),
            "bootstrap_seed": int(selection_record["bootstrap_seed"]),
            "participants": int(data.pid.nunique()), "observations": int(len(data)),
        }
        for record in records:
            record.update(metadata)
        optimizer_rows.extend(records)
        if not participant_metrics.empty:
            representative_participant_tables.append(participant_metrics.assign(
                outcome=outcome, replicate=replicate, trajectory=trajectory,
                gap_regime=str(selection_record["gap_regime"]),
                selection_gap=float(selection_record[GAP_METRIC]),
            ))
    return optimizer_rows, representative_participant_tables

# ---------------------------------------------------------------------------
# Exact target-cohort and trajectory-formula preflight
# ---------------------------------------------------------------------------

target_cohort_preflight = (
    cohort_check.loc[
        cohort_check.outcome.isin(
            TARGET_OUTCOMES
        )
    ]
    .copy()
)

assert set(
    target_cohort_preflight.outcome
) == set(TARGET_OUTCOMES)

assert (
    target_cohort_preflight
    .set_index("outcome")
    .loc[
        "global_mean_thickness",
        "participants",
    ]
    == 622
)

assert (
    target_cohort_preflight
    .set_index("outcome")
    .loc[
        "fluid_cognition",
        "participants",
    ]
    == 691
)

assert (
    "C" not in globals()
    or callable(globals()["C"])
)

trajectory_preflight_rows = []

for _outcome in TARGET_OUTCOMES:
    _source_index = (
        ANALYSIS_DATA[_outcome].index
    )

    for _trajectory in TRAJECTORIES:
        _formula = trajectory_formula(
            _outcome,
            _trajectory,
        )

        _y, _x = patsy.dmatrices(
            _formula,
            ANALYSIS_DATA[_outcome],
            return_type="dataframe",
        )

        _rank = int(
            np.linalg.matrix_rank(
                _x.to_numpy(float)
            )
        )

        trajectory_preflight_rows.append({
            "outcome": _outcome,
            "trajectory": _trajectory,
            "formula": _formula,
            "rows": len(_y),
            "columns": _x.shape[1],
            "rank": _rank,
            "full_rank":
                _rank == _x.shape[1],
            "same_complete_case_rows_as_source":
                _y.index.equals(
                    _source_index
                ),
        })

trajectory_formula_preflight = (
    pd.DataFrame(
        trajectory_preflight_rows
    )
)

assert len(
    trajectory_formula_preflight
) == 4

assert (
    trajectory_formula_preflight
    .full_rank
    .all()
)

assert (
    trajectory_formula_preflight
    .same_complete_case_rows_as_source
    .all()
)


# ---------------------------------------------------------------------------
# Explicit upstream 60-row manifest
# ---------------------------------------------------------------------------

SELECTED_MANIFEST_PATH = (
    Path(args.selected_manifest)
    .expanduser()
    .resolve()
)

if not SELECTED_MANIFEST_PATH.is_file():
    raise FileNotFoundError(
        SELECTED_MANIFEST_PATH
    )

selected_gap_replicates = pd.read_csv(
    SELECTED_MANIFEST_PATH
)

_missing = (
    REQUIRED_SELECTION_COLUMNS
    - set(
        selected_gap_replicates.columns
    )
)

if _missing:
    raise ValueError(
        "Selected manifest missing required columns: "
        f"{sorted(_missing)}"
    )

selected_gap_replicates[
    "replicate"
] = (
    pd.to_numeric(
        selected_gap_replicates[
            "replicate"
        ],
        errors="raise",
    )
    .astype(int)
)

selected_gap_replicates[
    "bootstrap_seed"
] = (
    pd.to_numeric(
        selected_gap_replicates[
            "bootstrap_seed"
        ],
        errors="raise",
    )
    .astype(int)
)

selected_gap_replicates = (
    selected_gap_replicates.loc[
        selected_gap_replicates
        .outcome
        .isin(TARGET_OUTCOMES)
    ]
    .copy()
)

selected_gap_replicates[
    "selection_id"
] = (
    selected_gap_replicates
    .outcome.astype(str)
    + "__"
    + selected_gap_replicates
      .replicate.astype(str)
)

selection_counts = (
    selected_gap_replicates
    .groupby(
        [
            "outcome",
            "gap_regime",
        ]
    )
    .size()
    .unstack(fill_value=0)
)

assert len(
    selected_gap_replicates
) == 60

assert (
    selected_gap_replicates
    .selection_id
    .nunique()
    == 60
)

assert not (
    selected_gap_replicates
    .duplicated(
        ["outcome", "replicate"]
    )
    .any()
)

assert (
    selection_counts
    .reindex(
        columns=[
            "lowest",
            "median",
            "highest",
        ]
    )
    .eq(N_PER_REGIME)
    .all()
    .all()
)


# Deliberate release-only provenance change:
# explicit upstream file and portable basename.
selection_audit = pd.DataFrame([
    {
        "selection_source":
            "explicit prior selected manifest",
        "source_path":
            SELECTED_MANIFEST_PATH.name,
        "selected_rows":
            len(
                selected_gap_replicates
            ),
        "unique_selection_ids":
            selected_gap_replicates
            .selection_id
            .nunique(),
    }
])


# ===== Frozen notebook cell 15: observed ML/REML fits =====
# 10 — Fit 2 outcomes × 2 trajectories × ML/REML × 3 optimizers
observed_optimizer_rows = []
observed_participant_tables = []
observed_tasks = [
    (outcome, trajectory, reml)
    for outcome in TARGET_OUTCOMES
    for reml in (False, True)
    for trajectory in TRAJECTORIES
]
for outcome, trajectory, reml in tqdm(
    observed_tasks, desc="Observed ML/REML optimizer panels", unit="panel"
):
    records, participant_metrics = fit_optimizer_panel(
        ANALYSIS_DATA[outcome], outcome, trajectory, reml
    )
    observed_optimizer_rows.extend(records)
    if not participant_metrics.empty:
        observed_participant_tables.append(participant_metrics.assign(
            outcome=outcome, trajectory=trajectory,
            estimation="REML" if reml else "ML",
        ))

observed_optimizer_fits = pd.DataFrame(observed_optimizer_rows)
observed_participant_reliability = (
    pd.concat(observed_participant_tables, ignore_index=True)
    if observed_participant_tables else pd.DataFrame()
)
assert len(observed_optimizer_fits)==2*2*2*3
display(observed_optimizer_fits[[
    "outcome", "trajectory", "estimation", "optimizer", "optimizer_converged",
    "hessian_positive_definite", "basic_numerical_validity",
    "strict_numerical_validity", "optimizer_common_solution_count",
    "hessian_positive_common_solution_count", "selected_strict_representative",
    "log_likelihood", "quadratic_coefficient", "quadratic_coefficient_p",
]])


# ===== Frozen notebook cell 16: observed comparisons =====
# 11 — Observed ML likelihood-ratio comparison and REML covariance contrast
observed_strict_representatives = observed_optimizer_fits.loc[
    observed_optimizer_fits.selected_strict_representative
].copy()

ml_rows = []
for outcome in TARGET_OUTCOMES:
    subset = observed_strict_representatives.loc[
        observed_strict_representatives.outcome.eq(outcome)
        & observed_strict_representatives.estimation.eq("ML")
    ].set_index("trajectory")
    row = {"outcome": outcome, "comparison_valid": False}
    if {"linear", "quadratic"}.issubset(subset.index):
        linear = subset.loc["linear"]
        quadratic = subset.loc["quadratic"]
        lr = max(0.0, 2*(float(quadratic.log_likelihood)-float(linear.log_likelihood)))
        row.update({
            "comparison_valid": True,
            "linear_optimizer": linear.optimizer,
            "quadratic_optimizer": quadratic.optimizer,
            "linear_ML_log_likelihood": float(linear.log_likelihood),
            "quadratic_ML_log_likelihood": float(quadratic.log_likelihood),
            "LR_statistic_df1": lr,
            "LR_p_value": float(stats.chi2.sf(lr,1)),
            "quadratic_coefficient": float(quadratic.quadratic_coefficient),
            "quadratic_coefficient_SE": float(quadratic.quadratic_coefficient_SE),
            "quadratic_coefficient_p_Wald": float(quadratic.quadratic_coefficient_p),
            "delta_AIC_quadratic_minus_linear": float(quadratic.AIC-linear.AIC),
            "delta_BIC_quadratic_minus_linear": float(quadratic.BIC-linear.BIC),
        })
    else:
        row["reason"] = "one or both ML trajectories lacked a Hessian-positive optimizer-agreeing fit"
    ml_rows.append(row)
observed_ML_likelihood_comparison = pd.DataFrame(ml_rows)

reml_representatives = observed_strict_representatives.loc[
    observed_strict_representatives.estimation.eq("REML")
].copy()
reml_columns = [
    "outcome", "trajectory", "optimizer", "random_intercept_variance",
    "random_slope_variance", "intercept_slope_covariance",
    "intercept_slope_correlation", "absolute_intercept_slope_correlation",
    "one_minus_rho_squared", "minimum_eigenvalue_fraction_of_trace",
    "slope_innovation_variance", "innovation_variance_fraction_of_slope",
    "median_direct_reliability", "median_full_posterior_predictability",
    "median_innovation_reliability", "median_full_minus_direct_gap",
    "quadratic_coefficient", "quadratic_coefficient_SE", "quadratic_coefficient_p",
]
observed_REML_covariance_reliability = reml_representatives[reml_columns].copy()

display(observed_ML_likelihood_comparison.round(6))
display(observed_REML_covariance_reliability.round(6))


# ===== Frozen notebook cell 19: checkpointed bootstrap fits =====
# 13 — Resumable 60-dataset × 2-trajectory × 3-optimizer execution
selection_signature = json.dumps({
    "selection": json.loads(selected_gap_replicates.sort_values(
        ["outcome", "replicate"]
    )[["outcome", "replicate", "gap_regime", "bootstrap_seed", GAP_METRIC]].to_json(
        orient="records", double_precision=15
    )),
    "trajectories": TRAJECTORIES, "optimizers": OPTIMIZERS,
    "quadratic_term": QUADRATIC_TERM,
}, sort_keys=True)

if CHECKPOINT_FILE.exists():
    checkpoint = pd.read_pickle(CHECKPOINT_FILE)
    if checkpoint.get("selection_signature") != selection_signature:
        raise RuntimeError(
            "The checkpoint belongs to a different selection/formula configuration. "
            "Move it before intentionally starting a different audit."
        )
    bootstrap_optimizer_fits = checkpoint.get("bootstrap_optimizer_fits", pd.DataFrame())
    bootstrap_representative_participants = checkpoint.get(
        "bootstrap_representative_participants", pd.DataFrame()
    )
else:
    bootstrap_optimizer_fits = pd.DataFrame()
    bootstrap_representative_participants = pd.DataFrame()

completed = set()
if not bootstrap_optimizer_fits.empty:
    complete_counts = bootstrap_optimizer_fits.groupby(
        ["outcome", "replicate"]
    ).size()
    completed = set(complete_counts.loc[complete_counts.eq(len(TRAJECTORIES)*len(OPTIMIZERS))].index)

tasks = [
    row._asdict() for row in selected_gap_replicates.itertuples(index=False)
    if (row.outcome, int(row.replicate)) not in completed
]

progress = tqdm(total=len(tasks), desc="Quadratic audit selected resamples", unit="dataset")
batch_size = max(2, 2*N_JOBS)
for start in range(0, len(tasks), batch_size):
    batch = tasks[start:start+batch_size]
    batch_results = Parallel(
        n_jobs=N_JOBS, backend="loky", batch_size=1, pre_dispatch=N_JOBS
    )(delayed(fit_selected_resample)(record) for record in batch)
    new_optimizer_rows, new_participant_tables = [], []
    for optimizer_rows, participant_tables in batch_results:
        new_optimizer_rows.extend(optimizer_rows)
        new_participant_tables.extend(participant_tables)
    bootstrap_optimizer_fits = pd.concat([
        bootstrap_optimizer_fits, pd.DataFrame(new_optimizer_rows)
    ], ignore_index=True)
    if new_participant_tables:
        bootstrap_representative_participants = pd.concat([
            bootstrap_representative_participants, *new_participant_tables
        ], ignore_index=True)
    pd.to_pickle({
        "selection_signature": selection_signature,
        "bootstrap_optimizer_fits": bootstrap_optimizer_fits,
        "bootstrap_representative_participants": bootstrap_representative_participants,
    }, CHECKPOINT_FILE)
    progress.update(len(batch))
progress.close()

bootstrap_optimizer_fits = bootstrap_optimizer_fits.drop_duplicates(
    ["outcome", "replicate", "trajectory", "optimizer"], keep="last"
).sort_values(["outcome", "gap_regime", "replicate", "trajectory", "optimizer"]).reset_index(drop=True)
if not bootstrap_representative_participants.empty:
    bootstrap_representative_participants = bootstrap_representative_participants.drop_duplicates(
        ["outcome", "replicate", "trajectory", "participant_boot_id"], keep="last"
    ).reset_index(drop=True)

assert len(bootstrap_optimizer_fits)==60*2*3
assert bootstrap_optimizer_fits.groupby(
    ["outcome", "replicate", "trajectory"]
).optimizer.nunique().eq(3).all()
display(bootstrap_optimizer_fits.head())


# ===== Frozen notebook cell 21: strict representatives/pairs =====
# 14 — One strict optimizer-agreeing representative per resample and trajectory
bootstrap_strict_representatives = bootstrap_optimizer_fits.loc[
    bootstrap_optimizer_fits.selected_strict_representative
].copy()

panel_denominators = bootstrap_optimizer_fits.groupby(
    ["outcome", "trajectory", "gap_regime"]
).agg(
    selected_replicates=("replicate", "nunique"),
    converged_optimizer_fits=("optimizer_converged", "sum"),
    hessian_positive_optimizer_fits=("hessian_positive_definite", "sum"),
    basic_valid_optimizer_fits=("basic_numerical_validity", "sum"),
    strict_valid_optimizer_fits=("strict_numerical_validity", "sum"),
    panels_with_two_basic_agreeing=(
        "at_least_two_optimizers_agree",
        lambda values: int(bootstrap_optimizer_fits.loc[values.index].groupby("replicate")[
            "at_least_two_optimizers_agree"
        ].any().sum()),
    ),
    panels_with_strict_representative=("selected_strict_representative", "sum"),
).reset_index()

representative_columns = [
    "outcome", "replicate", "gap_regime", "selection_gap", "trajectory", "optimizer",
    "random_intercept_variance", "random_slope_variance", "intercept_slope_covariance",
    "intercept_slope_correlation", "absolute_intercept_slope_correlation",
    "one_minus_rho_squared", "normalized_determinant",
    "random_covariance_min_eigenvalue", "minimum_eigenvalue_fraction_of_trace",
    "residual_variance", "slope_innovation_variance",
    "innovation_variance_fraction_of_slope", "median_direct_reliability",
    "median_full_posterior_predictability", "median_innovation_reliability",
    "median_full_minus_direct_gap", "median_full_minus_innovation_gap",
    "quadratic_coefficient", "quadratic_coefficient_SE", "quadratic_coefficient_p",
    "hessian_min_information_eigenvalue",
]
bootstrap_representative_table = bootstrap_strict_representatives[representative_columns].copy()

linear = bootstrap_representative_table.loc[
    bootstrap_representative_table.trajectory.eq("linear")
].drop(columns="trajectory").add_prefix("linear_")
quadratic = bootstrap_representative_table.loc[
    bootstrap_representative_table.trajectory.eq("quadratic")
].drop(columns="trajectory").add_prefix("quadratic_")
bootstrap_linear_quadratic_pairs = linear.merge(
    quadratic,
    left_on=["linear_outcome", "linear_replicate"],
    right_on=["quadratic_outcome", "quadratic_replicate"],
    how="outer", validate="one_to_one", indicator=True,
)
bootstrap_linear_quadratic_pairs["outcome"] = bootstrap_linear_quadratic_pairs[
    "linear_outcome"
].fillna(bootstrap_linear_quadratic_pairs["quadratic_outcome"])
bootstrap_linear_quadratic_pairs["replicate"] = bootstrap_linear_quadratic_pairs[
    "linear_replicate"
].fillna(bootstrap_linear_quadratic_pairs["quadratic_replicate"])
bootstrap_linear_quadratic_pairs["gap_regime"] = bootstrap_linear_quadratic_pairs[
    "linear_gap_regime"
].fillna(bootstrap_linear_quadratic_pairs["quadratic_gap_regime"])
bootstrap_linear_quadratic_pairs["paired_strict_representatives"] = (
    bootstrap_linear_quadratic_pairs._merge.eq("both")
)
bootstrap_linear_quadratic_pairs["quadratic_minus_linear_gap"] = (
    bootstrap_linear_quadratic_pairs.quadratic_median_full_minus_direct_gap
    - bootstrap_linear_quadratic_pairs.linear_median_full_minus_direct_gap
)

display(panel_denominators)
display(bootstrap_linear_quadratic_pairs[[
    "outcome", "replicate", "gap_regime", "paired_strict_representatives",
    "linear_median_full_minus_direct_gap", "quadratic_median_full_minus_direct_gap",
    "quadratic_minus_linear_gap", "quadratic_quadratic_coefficient",
]])


# ===== Frozen notebook cell 22: regime summaries =====
# 15 — Regime summaries under linear and quadratic fixed trajectories
bootstrap_gap_regime_summary = bootstrap_representative_table.groupby(
    ["outcome", "trajectory", "gap_regime"]
).agg(
    representative_replicates=("replicate", "size"),
    median_gap=("median_full_minus_direct_gap", "median"),
    median_direct=("median_direct_reliability", "median"),
    median_full=("median_full_posterior_predictability", "median"),
    median_innovation=("median_innovation_reliability", "median"),
    median_abs_rho=("absolute_intercept_slope_correlation", "median"),
    median_one_minus_rho_squared=("one_minus_rho_squared", "median"),
    median_min_eigen_fraction=("minimum_eigenvalue_fraction_of_trace", "median"),
    median_innovation_variance=("slope_innovation_variance", "median"),
    median_innovation_fraction=("innovation_variance_fraction_of_slope", "median"),
    median_quadratic_coefficient=("quadratic_coefficient", "median"),
).reset_index()
display(bootstrap_gap_regime_summary.round(6))


# ===== Frozen notebook cell 23: branch audit =====
# 16 — Determine whether original high-gap membership survives quadratic adjustment
branch_rows = []
membership_rows = []
for outcome in TARGET_OUTCOMES:
    quadratic_all = bootstrap_representative_table.loc[
        bootstrap_representative_table.outcome.eq(outcome)
        & bootstrap_representative_table.trajectory.eq("quadratic")
    ].copy()
    quadratic_all = quadratic_all.sort_values(
        ["median_full_minus_direct_gap", "replicate"], ascending=[False, True]
    )
    quadratic_top10_ids = set(quadratic_all.head(N_PER_REGIME).replicate.astype(int))

    for row in quadratic_all.itertuples(index=False):
        membership_rows.append({
            "outcome": outcome, "replicate": int(row.replicate),
            "original_gap_regime": row.gap_regime,
            "quadratic_gap": float(row.median_full_minus_direct_gap),
            "quadratic_top10_gap_member": int(row.replicate) in quadratic_top10_ids,
        })

    high_quadratic = quadratic_all.loc[quadratic_all.gap_regime.eq("highest")]
    low_quadratic = quadratic_all.loc[quadratic_all.gap_regime.eq("lowest")]
    paired_high = bootstrap_linear_quadratic_pairs.loc[
        bootstrap_linear_quadratic_pairs.outcome.eq(outcome)
        & bootstrap_linear_quadratic_pairs.gap_regime.eq("highest")
        & bootstrap_linear_quadratic_pairs.paired_strict_representatives
    ].dropna(subset=[
        "linear_median_full_minus_direct_gap",
        "quadratic_median_full_minus_direct_gap",
    ])

    retained_top10 = int(high_quadratic.replicate.astype(int).isin(quadratic_top10_ids).sum())
    valid_high = len(high_quadratic)
    valid_low = len(low_quadratic)
    paired_high_n = len(paired_high)
    median_linear_high_gap = float(
        paired_high.linear_median_full_minus_direct_gap.median()
    ) if paired_high_n else np.nan
    median_quadratic_high_gap = float(
        high_quadratic.median_full_minus_direct_gap.median()
    ) if valid_high else np.nan
    median_quadratic_low_gap = float(
        low_quadratic.median_full_minus_direct_gap.median()
    ) if valid_low else np.nan
    median_paired_change = float(
        paired_high.quadratic_minus_linear_gap.median()
    ) if paired_high_n else np.nan

    wilcoxon_p = np.nan
    if paired_high_n>=5 and np.any(np.abs(paired_high.quadratic_minus_linear_gap)>0):
        wilcoxon_p = float(stats.wilcoxon(
            paired_high.quadratic_median_full_minus_direct_gap,
            paired_high.linear_median_full_minus_direct_gap,
            alternative="two-sided", zero_method="wilcox",
        ).pvalue)
    rank_p = np.nan
    if valid_high and valid_low:
        rank_p = float(stats.mannwhitneyu(
            high_quadratic.median_full_minus_direct_gap,
            low_quadratic.median_full_minus_direct_gap,
            alternative="greater",
        ).pvalue)

    mechanism = {
        "median_abs_rho": float(high_quadratic.absolute_intercept_slope_correlation.median()) if valid_high else np.nan,
        "median_innovation_fraction": float(high_quadratic.innovation_variance_fraction_of_slope.median()) if valid_high else np.nan,
        "median_direct": float(high_quadratic.median_direct_reliability.median()) if valid_high else np.nan,
        "median_full": float(high_quadratic.median_full_posterior_predictability.median()) if valid_high else np.nan,
        "median_innovation": float(high_quadratic.median_innovation_reliability.median()) if valid_high else np.nan,
    }
    mechanism_persists = bool(
        mechanism["median_abs_rho"]>=NEAR_ABS_RHO
        and mechanism["median_innovation_fraction"]<=SMALL_INNOVATION_FRACTION
        and mechanism["median_full"]>=HIGH_FULL_PREDICTABILITY
        and mechanism["median_direct"]<LOW_DIRECT_RELIABILITY
        and mechanism["median_innovation"]<LOW_INNOVATION_RELIABILITY
    )

    if valid_high<8 or valid_low<8 or paired_high_n<8:
        classification = "insufficient numerical support"
    elif retained_top10>=8 and median_quadratic_high_gap>median_quadratic_low_gap:
        classification = (
            "high-gap branch survives quadratic adjustment"
            if mechanism_persists
            else "rank-defined high-gap branch survives, but the full covariance mechanism is qualified"
        )
    elif retained_top10<=2 or median_quadratic_high_gap<=median_quadratic_low_gap:
        classification = "high-gap branch does not survive quadratic adjustment"
    else:
        classification = "mixed persistence after quadratic adjustment"

    branch_rows.append({
        "outcome": outcome,
        "quadratic_strict_high_replicates": valid_high,
        "quadratic_strict_low_replicates": valid_low,
        "paired_strict_high_replicates": paired_high_n,
        "original_high_replicates_retained_in_quadratic_top10": retained_top10,
        "median_linear_high_gap_in_pairs": median_linear_high_gap,
        "median_quadratic_high_gap": median_quadratic_high_gap,
        "median_quadratic_low_gap": median_quadratic_low_gap,
        "median_paired_quadratic_minus_linear_gap": median_paired_change,
        "paired_Wilcoxon_p": wilcoxon_p,
        "quadratic_high_vs_low_Mann_Whitney_p": rank_p,
        **mechanism,
        "full_covariance_mechanism_persists": mechanism_persists,
        "classification": classification,
        "classification_is_automatic_paper_gate": False,
    })

quadratic_rank_membership = pd.DataFrame(membership_rows)
high_gap_branch_audit = pd.DataFrame(branch_rows)
display(high_gap_branch_audit.round(6))


# ===== Frozen notebook cell 24: diagnostic figure =====
# 17 — Diagnostic figure: matched gap changes and quadratic regime separation
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
regime_order = ["lowest", "median", "highest"]
regime_colors = {"lowest": "#2b6cb0", "median": "#718096", "highest": "#c53030"}

for row_index, outcome in enumerate(TARGET_OUTCOMES):
    ax = axes[row_index, 0]
    pairs = bootstrap_linear_quadratic_pairs.loc[
        bootstrap_linear_quadratic_pairs.outcome.eq(outcome)
        & bootstrap_linear_quadratic_pairs.paired_strict_representatives
    ]
    for regime in regime_order:
        subset = pairs.loc[pairs.gap_regime.eq(regime)]
        ax.scatter(
            subset.linear_median_full_minus_direct_gap,
            subset.quadratic_median_full_minus_direct_gap,
            color=regime_colors[regime], label=regime, s=42, alpha=.8,
        )
    finite = pairs[[
        "linear_median_full_minus_direct_gap",
        "quadratic_median_full_minus_direct_gap",
    ]].to_numpy(float)
    finite = finite[np.isfinite(finite).all(axis=1)]
    if len(finite):
        limits = [float(np.min(finite)), float(np.max(finite))]
        padding = .05*max(limits[1]-limits[0], .01)
        limits = [limits[0]-padding, limits[1]+padding]
        ax.plot(limits, limits, color="black", lw=.8, ls="--")
        ax.set_xlim(limits); ax.set_ylim(limits)
    ax.set_xlabel("Linear REML full-minus-direct gap")
    ax.set_ylabel("Quadratic REML full-minus-direct gap")
    ax.set_title(OUTCOMES[outcome]["label"])
    ax.legend(frameon=False, title="Original selection")

    ax = axes[row_index, 1]
    data = bootstrap_representative_table.loc[
        bootstrap_representative_table.outcome.eq(outcome)
        & bootstrap_representative_table.trajectory.eq("quadratic")
    ]
    groups = [
        data.loc[data.gap_regime.eq(regime), "median_full_minus_direct_gap"].dropna()
        for regime in regime_order
    ]
    ax.boxplot(groups, labels=regime_order, showfliers=True)
    for x_position, regime in enumerate(regime_order, 1):
        values = groups[x_position-1]
        jitter = np.linspace(-.06, .06, len(values)) if len(values) else []
        ax.scatter(
            np.full(len(values), x_position)+jitter, values,
            color=regime_colors[regime], s=28, alpha=.75,
        )
    ax.set_xlabel("Original linear-gap selection regime")
    ax.set_ylabel("Quadratic REML full-minus-direct gap")
    ax.set_title("Quadratic-adjusted separation")

fig.suptitle("Does the high-gap branch survive within-person quadratic adjustment?", y=1.01)
fig.tight_layout()
figure_path = PUBLIC_DIR / "quadratic_high_gap_branch_audit.png"
fig.savefig(figure_path, dpi=220, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Release output routing
# ---------------------------------------------------------------------------

PUBLIC_EXPORTS = {
    "00_selection_audit.csv":
        selection_audit,

    "01_selected_gap_replicates.csv":
        selected_gap_replicates,

    "02_trajectory_formula_preflight.csv":
        trajectory_formula_preflight,

    "03_observed_optimizer_fits.csv":
        observed_optimizer_fits,

    "04_observed_ML_likelihood_comparison.csv":
        observed_ML_likelihood_comparison,

    "05_observed_REML_covariance_reliability.csv":
        observed_REML_covariance_reliability,

    "06_bootstrap_optimizer_fits.csv":
        bootstrap_optimizer_fits,

    "08_bootstrap_panel_denominators.csv":
        panel_denominators,

    "09_bootstrap_representative_table.csv":
        bootstrap_representative_table,

    "10_bootstrap_linear_quadratic_pairs.csv":
        bootstrap_linear_quadratic_pairs,

    "11_bootstrap_gap_regime_summary.csv":
        bootstrap_gap_regime_summary,

    "12_quadratic_rank_membership.csv":
        quadratic_rank_membership,

    "13_high_gap_branch_audit.csv":
        high_gap_branch_audit,
}

for _filename, _frame in (
    PUBLIC_EXPORTS.items()
):
    _frame.to_csv(
        PUBLIC_DIR / _filename,
        index=False,
    )


# Contains original participant IDs from selected bootstrap draws.
bootstrap_representative_participants.to_csv(
    RESTRICTED_DIR
    / "07_bootstrap_representative_participant_reliability.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# Final scientific/release invariants
# ---------------------------------------------------------------------------

assert len(
    trajectory_formula_preflight
) == 4

assert (
    trajectory_formula_preflight
    .full_rank
    .all()
)

assert len(
    selected_gap_replicates
) == 60

assert (
    selected_gap_replicates
    .selection_id
    .nunique()
    == 60
)

assert len(
    observed_optimizer_fits
) == 24

assert len(
    observed_ML_likelihood_comparison
) == 2

assert len(
    observed_REML_covariance_reliability
) == 4

assert len(
    bootstrap_optimizer_fits
) == 360

assert (
    bootstrap_optimizer_fits
    .groupby(
        [
            "outcome",
            "replicate",
            "trajectory",
        ]
    )
    .optimizer
    .nunique()
    .eq(3)
    .all()
)

assert len(
    panel_denominators
) == 12

assert len(
    bootstrap_representative_table
) == 120

assert len(
    bootstrap_linear_quadratic_pairs
) == 60

assert len(
    bootstrap_gap_regime_summary
) == 12

assert len(
    quadratic_rank_membership
) == 60

assert len(
    high_gap_branch_audit
) == 2

assert not (
    high_gap_branch_audit
    .classification_is_automatic_paper_gate
    .any()
)

assert (
    figure_path.exists()
    and figure_path.stat().st_size > 0
)


# Privacy invariant.
assert not (
    PUBLIC_DIR
    / "07_bootstrap_representative_participant_reliability.csv"
).exists()

assert (
    RESTRICTED_DIR
    / "07_bootstrap_representative_participant_reliability.csv"
).exists()


# Portable provenance invariant.
assert (
    selection_audit[
        "source_path"
    ]
    .eq(
        SELECTED_MANIFEST_PATH.name
    )
    .all()
)

assert not (
    selection_audit[
        "source_path"
    ]
    .astype(str)
    .str.contains(
        r"[/\\]",
        regex=True,
    )
    .any()
)


print()
print("=" * 80)
print("QUADRATIC GAP AUDIT COMPLETE")
print("=" * 80)

print(
    "observed optimizer rows:",
    len(observed_optimizer_fits),
)

print(
    "bootstrap optimizer rows:",
    len(bootstrap_optimizer_fits),
)

print(
    "restricted participant rows:",
    len(
        bootstrap_representative_participants
    ),
)

print(
    "public CSV files:",
    len(
        list(
            PUBLIC_DIR.glob("*.csv")
        )
    ),
)

print(
    "restricted CSV files:",
    len(
        list(
            RESTRICTED_DIR.glob("*.csv")
        )
    ),
)

print(
    "checkpoint exists:",
    CHECKPOINT_FILE.exists(),
)

print(
    "figure exists:",
    figure_path.exists(),
)

print(
    "participant-level table routed only to restricted/:",
    not (
        PUBLIC_DIR
        / "07_bootstrap_representative_participant_reliability.csv"
    ).exists()
    and (
        RESTRICTED_DIR
        / "07_bootstrap_representative_participant_reliability.csv"
    ).exists(),
)
