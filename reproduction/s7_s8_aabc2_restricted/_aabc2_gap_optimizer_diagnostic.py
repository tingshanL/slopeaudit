#!/usr/bin/env python3
"""Restricted-data reproduction of the thickness/cognition gap diagnostic.

This refactors the frozen
AABC_thickness_cognition_gap_regime_optimizer_diagnostic.ipynb
workflow into an explicit, non-notebook release entry point.

The upstream reliability-gap bootstrap table must be supplied explicitly.
No filesystem discovery or "newest matching file" logic is used.

Public aggregate products and restricted participant-level products are
written to separate directories. The optimizer checkpoint is private.
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

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm.auto import tqdm

import aabc2_common as common


# ---------------------------------------------------------------------------
# Frozen numerical/scientific controls
# ---------------------------------------------------------------------------

SEED = 20260718

OPTIMIZERS = (
    "lbfgs",
    "powell",
    "bfgs",
)

MAXITER = 1200

GAP_METRIC = "median_full_minus_direct_gap"
N_PER_REGIME = 10

TARGET_OUTCOMES = (
    "global_mean_thickness",
    "fluid_cognition",
)

LLF_REL_TOL = 1e-5
FIXED_AGREEMENT_SE_TOL = 0.25
PSD_REL_TOL = 1e-8

# Diagnostic conventions—not new reliability licensing gates.
NEAR_ABS_RHO = 0.95
SMALL_INNOVATION_FRACTION = 0.10
HIGH_FULL_PREDICTABILITY = 0.40
LOW_DIRECT_RELIABILITY = 0.25
LOW_INNOVATION_RELIABILITY = 0.25


REQUIRED_GAP_COLUMNS = {
    "outcome",
    "replicate",
    "valid",
    "median_direct_GRR",
    "median_full_predictability",
    "median_innovation_reliability",
    "median_full_minus_direct_gap",
    "median_full_minus_innovation_gap",
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
    "--bootstrap-metrics",
    required=True,
    help=(
        "Explicit upstream "
        "09_reliability_gap_bootstrap_replicates.csv"
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
    help="Resume from the private optimizer checkpoint if present.",
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
    / "_selected_optimizer_fits_checkpoint.pkl"
)


def display(*args, **kwargs):
    """Suppress notebook-only rich display calls."""
    return None


# ---------------------------------------------------------------------------
# Validated common AABC loader
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


# Target-cohort and formula invariants from the frozen notebook.
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

formula_preflight_rows = []

for _outcome in TARGET_OUTCOMES:
    _formula = fixed_formula(
        _outcome
    )

    _y, _x = patsy.dmatrices(
        _formula,
        ANALYSIS_DATA[_outcome],
        return_type="dataframe",
    )

    _rank = int(
        np.linalg.matrix_rank(
            _x.to_numpy()
        )
    )

    formula_preflight_rows.append({
        "outcome": _outcome,
        "formula_role": "common_reference",
        "rows": len(_y),
        "columns": _x.shape[1],
        "rank": _rank,
        "full_rank":
            _rank == _x.shape[1],
    })

target_formula_preflight = pd.DataFrame(
    formula_preflight_rows
)

assert (
    len(target_formula_preflight) == 2
    and target_formula_preflight
        .full_rank
        .all()
)


# ---------------------------------------------------------------------------
# Explicit upstream bootstrap input
# ---------------------------------------------------------------------------

BOOTSTRAP_METRICS_PATH = (
    Path(args.bootstrap_metrics)
    .expanduser()
    .resolve()
)

if not BOOTSTRAP_METRICS_PATH.is_file():
    raise FileNotFoundError(
        BOOTSTRAP_METRICS_PATH
    )

bootstrap_gap_source = pd.read_csv(
    BOOTSTRAP_METRICS_PATH
)

missing = (
    REQUIRED_GAP_COLUMNS
    - set(bootstrap_gap_source.columns)
)

if missing:
    raise ValueError(
        "Bootstrap table is missing required columns: "
        f"{sorted(missing)}"
    )

bootstrap_gap_source[
    "replicate"
] = (
    pd.to_numeric(
        bootstrap_gap_source["replicate"],
        errors="raise",
    )
    .astype(int)
)

if bootstrap_gap_source[
    "valid"
].dtype != bool:
    bootstrap_gap_source[
        "valid"
    ] = (
        bootstrap_gap_source["valid"]
        .astype(str)
        .str.lower()
        .map({
            "true": True,
            "false": False,
        })
    )

if bootstrap_gap_source[
    "valid"
].isna().any():
    raise ValueError(
        "Could not parse every value in the bootstrap valid column."
    )

bootstrap_gap_source = (
    bootstrap_gap_source.loc[
        bootstrap_gap_source.outcome.isin(
            TARGET_OUTCOMES
        )
    ]
    .copy()
)

assert not bootstrap_gap_source.duplicated(
    ["outcome", "replicate"]
).any()

assert set(
    bootstrap_gap_source.outcome
) == set(TARGET_OUTCOMES)

assert (
    bootstrap_gap_source
    .groupby("outcome")
    .replicate
    .nunique()
    .eq(499)
    .all()
)

gap_source_audit = (
    bootstrap_gap_source
    .groupby("outcome")
    .agg(
        requested_replicates=(
            "replicate",
            "nunique",
        ),
        valid_replicates=(
            "valid",
            "sum",
        ),
        finite_gap_replicates=(
            GAP_METRIC,
            lambda x:
                np.isfinite(x).sum(),
        ),
    )
    .reset_index()
)

# Deliberate release change: portable provenance rather than an absolute path.
gap_source_audit[
    "source_file"
] = BOOTSTRAP_METRICS_PATH.name


# ==============================================================================
# select_gap_regimes — frozen notebook cell 11
# ==============================================================================
def select_gap_regimes(group, n=N_PER_REGIME):
    group = group.loc[group.valid & np.isfinite(group[GAP_METRIC])].copy()
    group = group.sort_values([GAP_METRIC, "replicate"], kind="mergesort")
    if len(group) < 3*n:
        raise ValueError(f"{group.outcome.iloc[0]} has only {len(group)} valid finite-gap replicates; need {3*n}.")

    low = group.head(n).copy()
    high = group.tail(n).copy()
    excluded = set(low.replicate) | set(high.replicate)
    middle_pool = group.loc[~group.replicate.isin(excluded)].copy()
    median_gap = float(group[GAP_METRIC].median())
    middle_pool["distance_from_outcome_median_gap"] = (middle_pool[GAP_METRIC] - median_gap).abs()
    middle = middle_pool.sort_values(
        ["distance_from_outcome_median_gap", "replicate"], kind="mergesort"
    ).head(n).copy()

    low["gap_regime"] = "lowest"
    middle["gap_regime"] = "median"
    high["gap_regime"] = "highest"
    selected = pd.concat([low, middle, high], ignore_index=True)
    selected["outcome_median_gap"] = median_gap
    selected["distance_from_outcome_median_gap"] = (selected[GAP_METRIC] - median_gap).abs()
    rank_map = dict(zip(group.replicate, np.arange(1, len(group)+1)))
    selected["gap_rank_among_valid"] = selected.replicate.map(rank_map).astype(int)
    selected["gap_empirical_quantile"] = selected.gap_rank_among_valid / (len(group)+1)
    return selected

# ==============================================================================
# clamp_rounding_only — frozen notebook cell 13
# ==============================================================================
def clamp_rounding_only(value, tolerance=1e-7):
    if not np.isfinite(value):
        return np.nan
    if -tolerance <= value <= 1+tolerance:
        return float(np.clip(value, 0, 1))
    return np.nan

# ==============================================================================
# direct_grr — frozen notebook cell 13
# ==============================================================================
def direct_grr(slope_variance, residual_variance, information):
    if not (
        np.isfinite(slope_variance) and slope_variance > 0
        and np.isfinite(residual_variance) and residual_variance > 0
        and np.isfinite(information) and information > 0
    ):
        return np.nan
    return float(slope_variance / (slope_variance + residual_variance/information))

# ==============================================================================
# hessian_diagnostic — frozen notebook cell 13
# ==============================================================================
def hessian_diagnostic(result):
    try:
        hessian = result.model.hessian(result.params_object)
        if isinstance(hessian, tuple):
            hessian = hessian[0]
        information_matrix = -(np.asarray(hessian, float) + np.asarray(hessian, float).T)/2
        eigenvalues = np.linalg.eigvalsh(information_matrix)
        tolerance = 1e-8 * max(1.0, float(np.max(np.abs(eigenvalues))))
        return bool(np.min(eigenvalues) > tolerance), float(np.min(eigenvalues)), float(np.max(eigenvalues))
    except Exception:
        return False, np.nan, np.nan

# ==============================================================================
# fixed_parameters_finite — frozen notebook cell 13
# ==============================================================================
def fixed_parameters_finite(result):
    try:
        estimates = np.asarray(result.fe_params, float)
        errors = np.asarray(result.bse_fe, float)
        return bool(np.isfinite(estimates).all() and np.isfinite(errors).all() and (errors >= 0).all())
    except Exception:
        return False

# ==============================================================================
# reconstruct_bootstrap_dataset — frozen notebook cell 13
# ==============================================================================
def reconstruct_bootstrap_dataset(outcome, replicate, seed):
    rng = np.random.default_rng(int(seed))
    source_data = ANALYSIS_DATA[outcome]
    participant_ids = source_data.pid.unique()
    sampled_ids = rng.choice(participant_ids, size=len(participant_ids), replace=True)
    pieces = []
    for draw_index, original_id in enumerate(sampled_ids):
        participant = source_data.loc[source_data.pid.eq(original_id)].copy()
        participant["original_participant_id"] = str(original_id)
        participant["bootstrap_draw_index"] = int(draw_index)
        participant["pid"] = f"boot_{draw_index:05d}"
        pieces.append(participant)
    data = pd.concat(pieces, ignore_index=True)
    assert data.pid.nunique() == len(participant_ids)
    assert len(data) == sum(len(source_data.loc[source_data.pid.eq(pid)]) for pid in sampled_ids)
    return data

# ==============================================================================
# fit_one_optimizer — frozen notebook cell 13
# ==============================================================================
def fit_one_optimizer(data, outcome, optimizer):
    formula = fixed_formula(outcome)
    caught_warnings = []
    try:
        y_matrix, x_matrix = patsy.dmatrices(formula, data, return_type="dataframe")
        design_rank = int(np.linalg.matrix_rank(x_matrix.to_numpy(float)))
        design_columns = int(x_matrix.shape[1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = smf.mixedlm(
                formula, data=data, groups=data["pid"], re_formula="1+t5_mid"
            ).fit(reml=True, method=optimizer, maxiter=MAXITER, disp=False)
            caught_warnings = [str(w.message) for w in caught]

        random_covariance = np.asarray(result.cov_re, float)
        random_eigenvalues = np.linalg.eigvalsh((random_covariance + random_covariance.T)/2)
        residual_variance = float(result.scale)
        covariance_tolerance = PSD_REL_TOL * max(
            1.0, float(np.max(np.abs(random_eigenvalues))), abs(residual_variance)
        )
        covariance_psd = bool(
            np.isfinite(random_covariance).all() and residual_variance > 0
            and np.min(random_eigenvalues) >= -covariance_tolerance
        )
        hessian_pd, hessian_min, hessian_max = hessian_diagnostic(result)

        intercept_variance = float(random_covariance[0, 0])
        slope_variance = float(random_covariance[1, 1])
        intercept_slope_covariance = float(random_covariance[0, 1])
        correlation = (
            intercept_slope_covariance / np.sqrt(intercept_variance*slope_variance)
            if intercept_variance > 0 and slope_variance > 0 else np.nan
        )
        innovation_variance = (
            slope_variance - intercept_slope_covariance**2/intercept_variance
            if intercept_variance > 0 else np.nan
        )
        innovation_fraction = (
            innovation_variance/slope_variance
            if slope_variance > 0 and np.isfinite(innovation_variance) else np.nan
        )

        participant_rows = []
        inverse_random = None
        if np.min(random_eigenvalues) > 0 and slope_variance > 0:
            inverse_random = np.linalg.inv(random_covariance)
        innovation_loading = (
            intercept_slope_covariance/intercept_variance if intercept_variance > 0 else np.nan
        )
        for participant_id, group in data.groupby("pid", sort=False):
            information = float(np.sum((group.t5_mid-group.t5_mid.mean())**2))
            direct = direct_grr(slope_variance, residual_variance, information)
            full = np.nan
            innovation = np.nan
            posterior_slope_variance = np.nan
            posterior_innovation_variance = np.nan
            if inverse_random is not None:
                random_design = np.column_stack([np.ones(len(group)), group.t5_mid.to_numpy(float)])
                posterior_covariance = np.linalg.inv(
                    inverse_random + random_design.T@random_design/residual_variance
                )
                posterior_slope_variance = float(posterior_covariance[1, 1])
                full = clamp_rounding_only(1-posterior_slope_variance/slope_variance)
                if innovation_variance > 0:
                    transform = np.array([-innovation_loading, 1.0])
                    posterior_innovation_variance = float(transform@posterior_covariance@transform)
                    innovation = clamp_rounding_only(
                        1-posterior_innovation_variance/innovation_variance
                    )
            first = group.iloc[0]
            participant_rows.append({
                "participant_boot_id": participant_id,
                "original_participant_id": first.original_participant_id,
                "bootstrap_draw_index": int(first.bootstrap_draw_index),
                "scan_count": len(group),
                "followup_years": float((group.t5.max()-group.t5.min())*5),
                "schedule_information": information,
                "direct_reliability": direct,
                "full_posterior_predictability": full,
                "innovation_reliability": innovation,
                "posterior_slope_variance": posterior_slope_variance,
                "posterior_innovation_variance": posterior_innovation_variance,
            })
        participant_metrics = pd.DataFrame(participant_rows)

        reliability_summary = {}
        for source_column, prefix in [
            ("direct_reliability", "direct_reliability"),
            ("full_posterior_predictability", "full_posterior_predictability"),
            ("innovation_reliability", "innovation_reliability"),
        ]:
            reliability_summary[f"median_{prefix}"] = float(participant_metrics[source_column].median())
            reliability_summary[f"q25_{prefix}"] = float(participant_metrics[source_column].quantile(.25))
            reliability_summary[f"q75_{prefix}"] = float(participant_metrics[source_column].quantile(.75))

        record = {
            "optimizer": optimizer,
            "fit_succeeded": True,
            "optimizer_converged": bool(result.converged),
            "reml_log_likelihood": float(result.llf),
            "fixed_design_rank": design_rank,
            "fixed_design_columns": design_columns,
            "fixed_design_full_rank": design_rank == design_columns,
            "fixed_parameters_finite": fixed_parameters_finite(result),
            "hessian_positive_definite": hessian_pd,
            "hessian_min_information_eigenvalue": hessian_min,
            "hessian_max_information_eigenvalue": hessian_max,
            "random_intercept_variance": intercept_variance,
            "random_slope_variance": slope_variance,
            "intercept_slope_covariance": intercept_slope_covariance,
            "intercept_slope_correlation": correlation,
            "absolute_intercept_slope_correlation": abs(correlation) if np.isfinite(correlation) else np.nan,
            "random_covariance_min_eigenvalue": float(np.min(random_eigenvalues)),
            "random_covariance_psd": covariance_psd,
            "residual_variance": residual_variance,
            "slope_innovation_variance": innovation_variance,
            "innovation_variance_fraction_of_slope": innovation_fraction,
            "fixed_effects_json": json.dumps({str(k): float(v) for k, v in result.fe_params.items()}),
            "fixed_effect_SE_json": json.dumps({str(k): float(v) for k, v in result.bse_fe.items()}),
            "warning_count": len(caught_warnings),
            "warnings": " | ".join(sorted(set(caught_warnings))),
            "error": "",
            **reliability_summary,
        }
        record["strict_numerical_validity"] = bool(
            record["optimizer_converged"]
            and record["fixed_design_full_rank"]
            and record["fixed_parameters_finite"]
            and record["random_covariance_psd"]
            and record["hessian_positive_definite"]
        )
        return result, record, participant_metrics
    except Exception as exc:
        record = {
            "optimizer": optimizer, "fit_succeeded": False,
            "optimizer_converged": False, "reml_log_likelihood": np.nan,
            "fixed_design_rank": np.nan, "fixed_design_columns": np.nan,
            "fixed_design_full_rank": False, "fixed_parameters_finite": False,
            "hessian_positive_definite": False,
            "hessian_min_information_eigenvalue": np.nan,
            "hessian_max_information_eigenvalue": np.nan,
            "random_intercept_variance": np.nan, "random_slope_variance": np.nan,
            "intercept_slope_covariance": np.nan, "intercept_slope_correlation": np.nan,
            "absolute_intercept_slope_correlation": np.nan,
            "random_covariance_min_eigenvalue": np.nan, "random_covariance_psd": False,
            "residual_variance": np.nan, "slope_innovation_variance": np.nan,
            "innovation_variance_fraction_of_slope": np.nan,
            "fixed_effects_json": "{}", "fixed_effect_SE_json": "{}",
            "warning_count": len(caught_warnings),
            "warnings": " | ".join(sorted(set(caught_warnings))),
            "strict_numerical_validity": False, "error": repr(exc),
        }
        for prefix in ["direct_reliability", "full_posterior_predictability", "innovation_reliability"]:
            for statistic in ["median", "q25", "q75"]:
                record[f"{statistic}_{prefix}"] = np.nan
        return None, record, pd.DataFrame()

# ==============================================================================
# results_compatible — frozen notebook cell 13
# ==============================================================================
def results_compatible(first, second):
    if first is None or second is None:
        return False
    common = first.fe_params.index.intersection(second.fe_params.index)
    if len(common) == 0:
        return False
    first_estimate = first.fe_params.loc[common].to_numpy(float)
    second_estimate = second.fe_params.loc[common].to_numpy(float)
    scale = np.maximum(
        first.bse_fe.loc[common].to_numpy(float),
        second.bse_fe.loc[common].to_numpy(float),
    )
    maximum_standardized_difference = float(
        np.max(np.abs(first_estimate-second_estimate)/np.maximum(scale, 1e-8))
    )
    likelihood_scale = max(1.0, abs(float(first.llf)), abs(float(second.llf)))
    likelihood_agrees = abs(float(first.llf)-float(second.llf)) <= LLF_REL_TOL*likelihood_scale
    return bool(likelihood_agrees and maximum_standardized_difference <= FIXED_AGREEMENT_SE_TOL)

# ==============================================================================
# fit_selected_resample — frozen notebook cell 14
# ==============================================================================
def fit_selected_resample(selection_record):
    selection_record = dict(selection_record)
    outcome = str(selection_record["outcome"])
    replicate = int(selection_record["replicate"])
    data = reconstruct_bootstrap_dataset(outcome, replicate, int(selection_record["bootstrap_seed"]))

    fitted_results = {}
    optimizer_records = []
    participant_tables = []
    for optimizer in OPTIMIZERS:
        result, record, participant_metrics = fit_one_optimizer(data, outcome, optimizer)
        fitted_results[optimizer] = result
        record.update({
            "outcome": outcome, "replicate": replicate,
            "gap_regime": selection_record["gap_regime"],
            "selection_gap": float(selection_record[GAP_METRIC]),
            "bootstrap_seed": int(selection_record["bootstrap_seed"]),
            "participants": int(data.pid.nunique()), "observations": int(len(data)),
        })
        optimizer_records.append(record)
        if not participant_metrics.empty:
            participant_metrics = participant_metrics.assign(
                outcome=outcome, replicate=replicate,
                gap_regime=selection_record["gap_regime"],
                selection_gap=float(selection_record[GAP_METRIC]),
                optimizer=optimizer,
            )
            participant_tables.append(participant_metrics)

    # Count agreement only among fits usable for likelihood/fixed-effect comparison.
    comparison_eligible = {
        row["optimizer"]: bool(
            row["fit_succeeded"] and row["optimizer_converged"]
            and row["fixed_design_full_rank"] and row["fixed_parameters_finite"]
            and row["random_covariance_psd"]
        )
        for row in optimizer_records
    }
    agreement_counts = {}
    for optimizer in OPTIMIZERS:
        if not comparison_eligible[optimizer]:
            agreement_counts[optimizer] = 0
            continue
        agreement_counts[optimizer] = sum(
            comparison_eligible[other]
            and results_compatible(fitted_results[optimizer], fitted_results[other])
            for other in OPTIMIZERS
        )

    # A separate, stricter count requires every member of the agreeing pair to
    # pass the full Hessian-inclusive numerical-validity definition.
    strict_eligible = {
        row["optimizer"]: bool(row["strict_numerical_validity"])
        for row in optimizer_records
    }
    hessian_positive_agreement_counts = {}
    for optimizer in OPTIMIZERS:
        if not strict_eligible[optimizer]:
            hessian_positive_agreement_counts[optimizer] = 0
            continue
        hessian_positive_agreement_counts[optimizer] = sum(
            strict_eligible[other]
            and results_compatible(fitted_results[optimizer], fitted_results[other])
            for other in OPTIMIZERS
        )

    for row in optimizer_records:
        row["optimizer_common_solution_count"] = int(agreement_counts[row["optimizer"]])
        row["agrees_with_at_least_one_other_optimizer"] = bool(
            agreement_counts[row["optimizer"]] >= 2
        )
        row["hessian_positive_common_solution_count"] = int(
            hessian_positive_agreement_counts[row["optimizer"]]
        )
        row["hessian_positive_optimizer_agreement"] = bool(
            hessian_positive_agreement_counts[row["optimizer"]] >= 2
        )
        row["selected_representative"] = False

    representative_candidates = [
        row for row in optimizer_records if row["hessian_positive_optimizer_agreement"]
    ]
    if representative_candidates:
        representative = max(
            representative_candidates,
            key=lambda row: (
                row["hessian_positive_common_solution_count"],
                row["optimizer_common_solution_count"],
                row["reml_log_likelihood"],
            ),
        )
        representative["selected_representative"] = True
        representative_optimizer = representative["optimizer"]
        for table in participant_tables:
            table["selected_representative"] = table.optimizer.eq(representative_optimizer)
    else:
        for table in participant_tables:
            table["selected_representative"] = False

    return optimizer_records, participant_tables

# ---------------------------------------------------------------------------
# Frozen deterministic low / median / high selection
# ---------------------------------------------------------------------------

selected_gap_replicates = pd.concat(
    [
        select_gap_regimes(group)
        for _, group
        in bootstrap_gap_source.groupby(
            "outcome",
            sort=False,
        )
    ],
    ignore_index=True,
)

outcome_seed_index = {
    "global_mean_thickness": 0,
    "fluid_cognition": 1,
}

selected_gap_replicates[
    "bootstrap_seed"
] = selected_gap_replicates.apply(
    lambda row:
        SEED
        + 8_000_000
        + 100_000
          * outcome_seed_index[
              row.outcome
          ]
        + int(row.replicate),
    axis=1,
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
        ["outcome", "gap_regime"]
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

assert not selected_gap_replicates.duplicated(
    ["outcome", "replicate"]
).any()


# ===== Frozen notebook cell 15: full checkpointed fit run =====
# 11 — Checkpointed 60-dataset × 3-optimizer run
selection_signature = selected_gap_replicates.sort_values(["outcome", "replicate"])[
    ["outcome", "replicate", "gap_regime", "bootstrap_seed", GAP_METRIC]
].to_json(orient="records", double_precision=15)

if CHECKPOINT_FILE.exists():
    checkpoint = pd.read_pickle(CHECKPOINT_FILE)
    if checkpoint.get("selection_signature") != selection_signature:
        raise RuntimeError(
            "The existing checkpoint belongs to a different selection manifest. "
            "Move or delete the checkpoint before intentionally starting the new manifest."
        )
    optimizer_fit_audit = checkpoint.get("optimizer_fit_audit", pd.DataFrame())
    selected_participant_reliability = checkpoint.get("selected_participant_reliability", pd.DataFrame())
else:
    optimizer_fit_audit = pd.DataFrame()
    selected_participant_reliability = pd.DataFrame()

completed = set()
if not optimizer_fit_audit.empty:
    completed_counts = optimizer_fit_audit.groupby(["outcome", "replicate"]).optimizer.nunique()
    completed = set(completed_counts.loc[completed_counts.eq(len(OPTIMIZERS))].index)

tasks = [
    row._asdict() for row in selected_gap_replicates.itertuples(index=False)
    if (row.outcome, int(row.replicate)) not in completed
]

progress = tqdm(total=len(tasks), desc="Selected resampled datasets", unit="dataset")
batch_size = max(2, 2*N_JOBS)
for start in range(0, len(tasks), batch_size):
    batch = tasks[start:start+batch_size]
    batch_results = Parallel(
        n_jobs=N_JOBS, backend="loky", batch_size=1, pre_dispatch=N_JOBS
    )(delayed(fit_selected_resample)(record) for record in batch)

    new_optimizer_rows = []
    new_participant_tables = []
    for optimizer_rows, participant_tables in batch_results:
        new_optimizer_rows.extend(optimizer_rows)
        new_participant_tables.extend(participant_tables)
    optimizer_fit_audit = pd.concat(
        [optimizer_fit_audit, pd.DataFrame(new_optimizer_rows)], ignore_index=True
    )
    if new_participant_tables:
        selected_participant_reliability = pd.concat(
            [selected_participant_reliability, *new_participant_tables], ignore_index=True
        )
    pd.to_pickle({
        "selection_signature": selection_signature,
        "optimizer_fit_audit": optimizer_fit_audit,
        "selected_participant_reliability": selected_participant_reliability,
    }, CHECKPOINT_FILE)
    progress.update(len(batch))
progress.close()

optimizer_fit_audit = optimizer_fit_audit.drop_duplicates(
    ["outcome", "replicate", "optimizer"], keep="last"
).sort_values(["outcome", "gap_regime", "replicate", "optimizer"]).reset_index(drop=True)
if not selected_participant_reliability.empty:
    selected_participant_reliability = selected_participant_reliability.drop_duplicates(
        ["outcome", "replicate", "optimizer", "participant_boot_id"], keep="last"
    ).reset_index(drop=True)

assert len(optimizer_fit_audit) == 60*len(OPTIMIZERS)
assert optimizer_fit_audit.groupby(["outcome", "replicate"]).optimizer.nunique().eq(3).all()
display(optimizer_fit_audit.head())


# ===== Frozen notebook cell 17: replicate agreement =====
# 12 — One row per selected replicate and one representative strict fit where available
replicate_rows = []
for (outcome, replicate), group in optimizer_fit_audit.groupby(["outcome", "replicate"], sort=False):
    representative = group.loc[group.selected_representative]
    base = selected_gap_replicates.loc[
        selected_gap_replicates.outcome.eq(outcome)
        & selected_gap_replicates.replicate.eq(replicate)
    ].iloc[0]
    row = {
        "outcome": outcome, "replicate": int(replicate),
        "gap_regime": base.gap_regime, "selection_gap": float(base[GAP_METRIC]),
        "optimizers_attempted": group.optimizer.nunique(),
        "optimizers_converged": int(group.optimizer_converged.sum()),
        "hessian_positive_optimizers": int(group.hessian_positive_definite.sum()),
        "strictly_valid_optimizers": int(group.strict_numerical_validity.sum()),
        "maximum_common_solution_count": int(group.optimizer_common_solution_count.max()),
        "maximum_hessian_positive_common_solution_count": int(
            group.hessian_positive_common_solution_count.max()
        ),
        "at_least_two_optimizers_agree": bool(group.optimizer_common_solution_count.max() >= 2),
        "has_hessian_positive_optimizer_agreement": bool(group.hessian_positive_optimizer_agreement.any()),
        "representative_available": len(representative) == 1,
    }
    if len(representative) == 1:
        selected = representative.iloc[0]
        for column in [
            "optimizer", "reml_log_likelihood", "hessian_min_information_eigenvalue",
            "random_intercept_variance", "random_slope_variance",
            "intercept_slope_covariance", "intercept_slope_correlation",
            "absolute_intercept_slope_correlation", "random_covariance_min_eigenvalue",
            "residual_variance", "slope_innovation_variance",
            "innovation_variance_fraction_of_slope", "median_direct_reliability",
            "median_full_posterior_predictability", "median_innovation_reliability",
        ]:
            row[f"representative_{column}"] = selected[column]
    replicate_rows.append(row)

replicate_optimizer_agreement = pd.DataFrame(replicate_rows)
assert len(replicate_optimizer_agreement) == 60
display(replicate_optimizer_agreement.head())


# ===== Frozen notebook cell 18: regime summaries =====
# 13 — Regime-level summaries
representatives = replicate_optimizer_agreement.loc[
    replicate_optimizer_agreement.representative_available
].copy()

regime_count_table = replicate_optimizer_agreement.groupby(["outcome", "gap_regime"]).agg(
    selected_replicates=("replicate", "size"),
    at_least_two_agree_n=("at_least_two_optimizers_agree", "sum"),
    hessian_positive_agreeing_n=("has_hessian_positive_optimizer_agreement", "sum"),
    representative_n=("representative_available", "sum"),
).reset_index()

regime_metric_table = representatives.groupby(["outcome", "gap_regime"]).agg(
    median_gap=("selection_gap", "median"),
    median_abs_rho=("representative_absolute_intercept_slope_correlation", "median"),
    median_innovation_variance=("representative_slope_innovation_variance", "median"),
    median_innovation_fraction=("representative_innovation_variance_fraction_of_slope", "median"),
    median_direct_reliability=("representative_median_direct_reliability", "median"),
    median_full_predictability=("representative_median_full_posterior_predictability", "median"),
    median_innovation_reliability=("representative_median_innovation_reliability", "median"),
    proportion_near_abs_rho_1=(
        "representative_absolute_intercept_slope_correlation",
        lambda x: float(np.mean(x >= NEAR_ABS_RHO)),
    ),
    proportion_small_innovation_fraction=(
        "representative_innovation_variance_fraction_of_slope",
        lambda x: float(np.mean(x <= SMALL_INNOVATION_FRACTION)),
    ),
).reset_index()

gap_regime_summary = regime_count_table.merge(
    regime_metric_table, on=["outcome", "gap_regime"], how="left", validate="one_to_one"
)
display(gap_regime_summary.round(5))


# ===== Frozen notebook cell 19: diagnostic checks =====
# 14 — Requested checks, reported per outcome and overall without a combined GO decision
check_rows = []

for outcome, group in replicate_optimizer_agreement.groupby("outcome"):
    agreement_rate = float(group.at_least_two_optimizers_agree.mean())
    check_rows.append({
        "outcome": outcome,
        "check": "at least two optimizers reach the same likelihood/fixed-effect solution for most selected replicates",
        "observed": f"{int(group.at_least_two_optimizers_agree.sum())}/{len(group)} = {agreement_rate:.3f}",
        "criterion": "literal majority (>0.50); >=0.80 is additionally labeled strong",
        "criterion_met": bool(agreement_rate > .50),
        "strong_support": bool(agreement_rate >= .80),
    })

    regime_index = gap_regime_summary.loc[gap_regime_summary.outcome.eq(outcome)].set_index("gap_regime")
    low_count = int(regime_index.loc["lowest", "hessian_positive_agreeing_n"])
    high_count = int(regime_index.loc["highest", "hessian_positive_agreeing_n"])
    check_rows.append({
        "outcome": outcome,
        "check": "both low- and high-gap regimes remain among Hessian-positive optimizer-agreeing fits",
        "observed": f"lowest {low_count}/10; highest {high_count}/10",
        "criterion": "at least one retained fit in each tail; counts and rates remain primary",
        "criterion_met": bool(low_count > 0 and high_count > 0),
        "strong_support": bool(low_count >= 8 and high_count >= 8),
    })

    low = regime_index.loc["lowest"]
    high = regime_index.loc["highest"]
    mechanism_checks = {
        "near-|rho|=1": bool(high.median_abs_rho >= NEAR_ABS_RHO),
        "very small innovation fraction": bool(high.median_innovation_fraction <= SMALL_INNOVATION_FRACTION),
        "high full predictability": bool(high.median_full_predictability >= HIGH_FULL_PREDICTABILITY),
        "higher |rho| than low-gap tail": bool(high.median_abs_rho > low.median_abs_rho),
        "smaller innovation fraction than low-gap tail": bool(high.median_innovation_fraction < low.median_innovation_fraction),
        "higher full predictability than low-gap tail": bool(high.median_full_predictability > low.median_full_predictability),
    }
    check_rows.append({
        "outcome": outcome,
        "check": "high-gap regime corresponds to near-|rho|=1, small innovation variance, and high full predictability",
        "observed": (
            f"median |rho|={high.median_abs_rho:.3f}; innovation/slope={high.median_innovation_fraction:.3f}; "
            f"full={high.median_full_predictability:.3f}; component checks="
            + json.dumps(mechanism_checks)
        ),
        "criterion": (
            f"median |rho|>={NEAR_ABS_RHO}, innovation/slope<={SMALL_INNOVATION_FRACTION}, "
            f"full>={HIGH_FULL_PREDICTABILITY}, plus expected high-vs-low ordering"
        ),
        "criterion_met": bool(all(mechanism_checks.values())),
        "strong_support": bool(all(mechanism_checks.values())),
    })

    reliability_checks = {
        "direct remains low": bool(high.median_direct_reliability < LOW_DIRECT_RELIABILITY),
        "innovation remains low": bool(high.median_innovation_reliability < LOW_INNOVATION_RELIABILITY),
    }
    check_rows.append({
        "outcome": outcome,
        "check": "direct and innovation reliability remain low in the high-gap regime",
        "observed": (
            f"median direct={high.median_direct_reliability:.3f}; "
            f"median innovation={high.median_innovation_reliability:.3f}"
        ),
        "criterion": f"direct <{LOW_DIRECT_RELIABILITY} and innovation <{LOW_INNOVATION_RELIABILITY}",
        "criterion_met": bool(all(reliability_checks.values())),
        "strong_support": bool(all(reliability_checks.values())),
    })

prespecified_diagnostic_checks = pd.DataFrame(check_rows)
prespecified_diagnostic_checks["automatic_scientific_gate"] = False
display(prespecified_diagnostic_checks)


# ===== Frozen notebook cell 20: diagnostic figure =====
# 15 — Compact diagnostic figure
colors = {"lowest": "#2b6cb0", "median": "#718096", "highest": "#c53030"}
regime_order = ["lowest", "median", "highest"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for row_index, outcome in enumerate(TARGET_OUTCOMES):
    data = representatives.loc[representatives.outcome.eq(outcome)]
    for regime in regime_order:
        subset = data.loc[data.gap_regime.eq(regime)]
        axes[row_index, 0].scatter(
            subset.selection_gap, subset.representative_absolute_intercept_slope_correlation,
            label=regime, color=colors[regime], alpha=.8
        )
        axes[row_index, 1].scatter(
            subset.selection_gap, subset.representative_innovation_variance_fraction_of_slope,
            label=regime, color=colors[regime], alpha=.8
        )
        for metric_index, (column, marker) in enumerate([
            ("representative_median_direct_reliability", "o"),
            ("representative_median_full_posterior_predictability", "s"),
            ("representative_median_innovation_reliability", "^"),
        ]):
            x = regime_order.index(regime) + np.linspace(-.08, .08, max(1, len(subset)))
            axes[row_index, 2].scatter(
                x, subset[column], color=colors[regime], marker=marker,
                alpha=.75, label=(column.replace("representative_median_", "").replace("_reliability", "") if regime=="lowest" else None)
            )

    axes[row_index, 0].axhline(NEAR_ABS_RHO, color="black", ls="--", lw=.8)
    axes[row_index, 0].set_ylabel(f"{OUTCOMES[outcome]['label']}\n|intercept-slope correlation|")
    axes[row_index, 0].set_xlabel("Selected full-minus-direct gap")
    axes[row_index, 1].axhline(SMALL_INNOVATION_FRACTION, color="black", ls="--", lw=.8)
    axes[row_index, 1].set_ylabel("Innovation variance / slope variance")
    axes[row_index, 1].set_xlabel("Selected full-minus-direct gap")
    axes[row_index, 2].set_xticks(range(3), regime_order)
    axes[row_index, 2].axhline(.25, color="grey", ls="--", lw=.8)
    axes[row_index, 2].axhline(.40, color="black", ls=":", lw=.8)
    axes[row_index, 2].set_ylabel("Median reliability/predictability")
    axes[row_index, 2].legend(frameon=False, fontsize=8)

axes[0, 0].legend(frameon=False)
fig.suptitle("Selected gap regimes: covariance geometry and slope-information diagnostics", y=1.01)
fig.tight_layout()
figure_path = PUBLIC_DIR / "gap_regime_covariance_reliability_diagnostic.png"
fig.savefig(figure_path, dpi=220, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# Release output routing
# ---------------------------------------------------------------------------

PUBLIC_EXPORTS = {
    "00_gap_source_audit.csv":
        gap_source_audit,

    "01_selected_gap_replicates.csv":
        selected_gap_replicates,

    "02_optimizer_fit_audit.csv":
        optimizer_fit_audit,

    "04_replicate_optimizer_agreement.csv":
        replicate_optimizer_agreement,

    "05_gap_regime_summary.csv":
        gap_regime_summary,

    "06_prespecified_diagnostic_checks.csv":
        prespecified_diagnostic_checks,
}

for _filename, _frame in (
    PUBLIC_EXPORTS.items()
):
    _frame.to_csv(
        PUBLIC_DIR / _filename,
        index=False,
    )


# Contains original real participant identifiers from the bootstrap draws.
selected_participant_reliability.to_csv(
    RESTRICTED_DIR
    / "03_selected_participant_reliability.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# Final invariants
# ---------------------------------------------------------------------------

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
    optimizer_fit_audit
) == 180

assert (
    optimizer_fit_audit
    .groupby(
        ["outcome", "replicate"]
    )
    .optimizer
    .nunique()
    .eq(3)
    .all()
)

assert len(
    replicate_optimizer_agreement
) == 60

assert (
    replicate_optimizer_agreement
    .groupby(
        ["outcome", "gap_regime"]
    )
    .size()
    .eq(10)
    .all()
)

assert len(
    gap_regime_summary
) == 6

assert len(
    prespecified_diagnostic_checks
) == 8

assert not (
    prespecified_diagnostic_checks
    .automatic_scientific_gate
    .any()
)

assert (
    figure_path.exists()
    and figure_path.stat().st_size > 0
)

# Privacy invariant.
assert not (
    PUBLIC_DIR
    / "03_selected_participant_reliability.csv"
).exists()

assert (
    RESTRICTED_DIR
    / "03_selected_participant_reliability.csv"
).exists()

# Public source audit must contain portable provenance only.
assert (
    gap_source_audit[
        "source_file"
    ]
    .eq(
        BOOTSTRAP_METRICS_PATH.name
    )
    .all()
)

assert not (
    gap_source_audit[
        "source_file"
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
print("GAP-OPTIMIZER DIAGNOSTIC COMPLETE")
print("=" * 80)

print(
    "selected datasets:",
    len(selected_gap_replicates),
)

print(
    "optimizer rows:",
    len(optimizer_fit_audit),
)

print(
    "participant-level rows:",
    len(
        selected_participant_reliability
    ),
)

print(
    "replicate agreement rows:",
    len(
        replicate_optimizer_agreement
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
        / "03_selected_participant_reliability.csv"
    ).exists()
    and (
        RESTRICTED_DIR
        / "03_selected_participant_reliability.csv"
    ).exists(),
)
