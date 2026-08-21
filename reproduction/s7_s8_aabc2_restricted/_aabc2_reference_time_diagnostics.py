#!/usr/bin/env python3
"""Restricted-data reference-time and covariance-geometry diagnostics.

Refactors the frozen AABC_section10_1_items2to4.ipynb workflow.

All upstream inputs are explicit. No recursive discovery, newest-file
selection or absolute-path audit is part of the release workflow.

The full 499 × two-outcome bootstrap geometry enrichment is resumable.
Participant-level reference-time results and the bootstrap checkpoint
are private. Aggregate scientific tables and figures are public.
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
    os.environ.setdefault(
        _name,
        "1",
    )

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

from joblib import Parallel, delayed
from tqdm.auto import tqdm

import aabc2_common as common


# ---------------------------------------------------------------------------
# Frozen controls
# ---------------------------------------------------------------------------

SEED = 20260718

MODEL_ORIGIN_YEARS = 2.5

REFERENCE_TIMES_YEARS = {
    "anchor_0y": 0.0,
    "five_year_5y": 5.0,
}

SINGULARITY_THRESHOLDS = (
    1e-2,
    1e-3,
    1e-4,
    1e-5,
)

TARGET_BOOTSTRAP_OUTCOMES = (
    "global_mean_thickness",
    "fluid_cognition",
)

EXPECTED_BOOTSTRAP_REPS = 499

OPTIMIZERS = (
    "lbfgs",
    "powell",
    "bfgs",
)

MAXITER = 1200

PSD_REL_TOL = 1e-8
ROUNDING_TOL = 1e-7


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input-config",
    required=True,
    help=(
        "Private JSON containing explicit "
        "restricted AABC raw-input paths."
    ),
)

parser.add_argument(
    "--exact-reliability",
    required=True,
    help=(
        "Restricted six-outcome "
        "04_exact_posterior_slope_reliability.csv."
    ),
)

parser.add_argument(
    "--gap-bootstrap",
    required=True,
    help=(
        "Six-outcome "
        "09_reliability_gap_bootstrap_replicates.csv."
    ),
)

parser.add_argument(
    "--selected-participants",
    required=True,
    help=(
        "Restricted gap-optimizer "
        "03_selected_participant_reliability.csv."
    ),
)

parser.add_argument(
    "--representative-fits",
    required=True,
    help=(
        "Gap-optimizer "
        "04_replicate_optimizer_agreement.csv."
    ),
)

parser.add_argument(
    "--simulation-geometry",
    required=True,
    help=(
        "Final simulation "
        "04_model_fit_covariance_diagnostics.csv."
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
    help=(
        "Parallel bootstrap workers; "
        "capped at 3."
    ),
)

parser.add_argument(
    "--resume",
    action="store_true",
    help=(
        "Resume the private full-499 "
        "geometry checkpoint."
    ),
)

args = parser.parse_args()


N_JOBS = max(
    1,
    min(
        int(args.n_jobs),
        3,
    ),
)

BATCH_SIZE = max(
    2,
    2 * N_JOBS,
)

# A release run always completes missing enrichment tasks.
RUN_FULL_499_GEOMETRY_ENRICHMENT = True


ROOT_DIR = (
    Path(args.output_dir)
    .expanduser()
    .resolve()
)

if ROOT_DIR.exists():
    if (
        any(ROOT_DIR.iterdir())
        and
        not args.resume
    ):
        raise FileExistsError(
            f"Output directory is non-empty: "
            f"{ROOT_DIR}. "
            "Use a fresh directory or pass --resume."
        )


PUBLIC_DIR = (
    ROOT_DIR
    / "public"
)

RESTRICTED_DIR = (
    ROOT_DIR
    / "restricted"
)

CHECKPOINT_DIR = (
    ROOT_DIR
    / "checkpoints"
)

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
    / "_full499_geometry_checkpoint.pkl"
)


# ---------------------------------------------------------------------------
# Explicit upstream files
# ---------------------------------------------------------------------------

EXACT_RELIABILITY_PATH = (
    Path(args.exact_reliability)
    .expanduser()
    .resolve()
)

GAP_BOOTSTRAP_PATH = (
    Path(args.gap_bootstrap)
    .expanduser()
    .resolve()
)

SELECTED_PARTICIPANT_PATH = (
    Path(args.selected_participants)
    .expanduser()
    .resolve()
)

REPRESENTATIVE_FIT_PATH = (
    Path(args.representative_fits)
    .expanduser()
    .resolve()
)

SIMULATION_GEOMETRY_PATH = (
    Path(args.simulation_geometry)
    .expanduser()
    .resolve()
)

for _path in (
    EXACT_RELIABILITY_PATH,
    GAP_BOOTSTRAP_PATH,
    SELECTED_PARTICIPANT_PATH,
    REPRESENTATIVE_FIT_PATH,
    SIMULATION_GEOMETRY_PATH,
):
    if not _path.is_file():
        raise FileNotFoundError(
            _path
        )


def display(*args, **kwargs):
    """Suppress notebook-only display calls."""
    return None


# ---------------------------------------------------------------------------
# Validated common AABC data loader
# ---------------------------------------------------------------------------

raw = common.build_raw_cohorts(
    args.input_config
)

OUTCOMES, ANALYSIS_DATA, cohort_check = (
    common.build_analysis_data(
        raw
    )
)


def fixed_formula(key):
    return common.fixed_formula(
        key,
        OUTCOMES,
    )


assert len(
    OUTCOMES
) == 6

assert set(
    TARGET_BOOTSTRAP_OUTCOMES
).issubset(
    OUTCOMES
)


# ---------------------------------------------------------------------------
# Explicit upstream table loading / schema checks
# ---------------------------------------------------------------------------

exact_source = pd.read_csv(
    EXACT_RELIABILITY_PATH,
    dtype={
        "participant_id": str,
    },
)

_exact_required = {
    "outcome",
    "participant_id",
    "random_intercept_variance",
    "random_slope_variance",
    "intercept_slope_covariance",
    "residual_variance",
    "conditional_slope_innovation_reliability",
}

_missing = (
    _exact_required
    - set(exact_source.columns)
)

if _missing:
    raise ValueError(
        "Exact reliability input missing: "
        f"{sorted(_missing)}"
    )


gap_source = pd.read_csv(
    GAP_BOOTSTRAP_PATH,
    low_memory=False,
)

_gap_required = {
    "outcome",
    "replicate",
    "valid",
    "median_direct_GRR",
    "median_full_predictability",
    "median_innovation_reliability",
    "median_full_minus_direct_gap",
}

_missing = (
    _gap_required
    - set(gap_source.columns)
)

if _missing:
    raise ValueError(
        "Gap-bootstrap input missing: "
        f"{sorted(_missing)}"
    )


selected_participant_source = (
    pd.read_csv(
        SELECTED_PARTICIPANT_PATH,
        dtype={
            "participant_boot_id": str,
            "original_participant_id": str,
        },
        low_memory=False,
    )
)

_selected_required = {
    "outcome",
    "replicate",
    "gap_regime",
    "optimizer",
    "participant_boot_id",
    "original_participant_id",
    "scan_count",
    "followup_years",
    "schedule_information",
    "selected_representative",
}

_missing = (
    _selected_required
    - set(
        selected_participant_source.columns
    )
)

if _missing:
    raise ValueError(
        "Selected-participant input missing: "
        f"{sorted(_missing)}"
    )


representative_fit_source = (
    pd.read_csv(
        REPRESENTATIVE_FIT_PATH,
        low_memory=False,
    )
)

_representative_required = {
    "outcome",
    "replicate",
    "gap_regime",
    "representative_available",
    "representative_random_intercept_variance",
    "representative_random_slope_variance",
    "representative_intercept_slope_correlation",
    "representative_random_covariance_min_eigenvalue",
}

_missing = (
    _representative_required
    - set(
        representative_fit_source.columns
    )
)

if _missing:
    raise ValueError(
        "Representative-fit input missing: "
        f"{sorted(_missing)}"
    )


# Simulation geometry is required for the frozen 56-row output 11.
_simulation_header = pd.read_csv(
    SIMULATION_GEOMETRY_PATH,
    nrows=2,
)

_simulation_required = {
    "regime",
    "replicate",
    "model_role",
    "one_minus_rho_squared",
    "minimum_eigenvalue_fraction_of_trace",
    "mean_full_posterior_predictability",
}

_missing = (
    _simulation_required
    - set(
        _simulation_header.columns
    )
)

if _missing:
    raise ValueError(
        "Simulation geometry input missing: "
        f"{sorted(_missing)}"
    )


# ==============================================================================
# coerce_bool — frozen notebook cell 4
# ==============================================================================
def coerce_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapped = series.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False,
    })
    return mapped.fillna(False).astype(bool)

# ==============================================================================
# clamp_rounding_only — frozen notebook cell 11
# ==============================================================================
def clamp_rounding_only(value, tolerance=ROUNDING_TOL):
    if not np.isfinite(value):
        return np.nan
    if -tolerance <= value <= 1 + tolerance:
        return float(np.clip(value, 0, 1))
    return np.nan

# ==============================================================================
# covariance_at_reference — frozen notebook cell 11
# ==============================================================================
def covariance_at_reference(G_mid, reference_years):
    G_mid = np.asarray(G_mid, float)
    delta = (float(reference_years) - MODEL_ORIGIN_YEARS) / 5.0
    transform = np.array([[1.0, delta], [0.0, 1.0]])
    G_ref = transform @ G_mid @ transform.T
    G_ref = (G_ref + G_ref.T) / 2
    return G_ref, delta

# ==============================================================================
# covariance_geometry — frozen notebook cell 11
# ==============================================================================
def covariance_geometry(G):
    G = np.asarray(G, float)
    eig = np.linalg.eigvalsh((G + G.T) / 2)
    trace = float(np.trace(G))
    G00, G01, G11 = float(G[0, 0]), float(G[0, 1]), float(G[1, 1])
    rho = G01 / np.sqrt(G00 * G11) if G00 > 0 and G11 > 0 else np.nan
    one_minus = 1.0 - rho**2 if np.isfinite(rho) else np.nan
    if np.isfinite(one_minus) and -ROUNDING_TOL <= one_minus < 0:
        one_minus = 0.0
    eigen_fraction = float(eig.min() / trace) if trace > 0 else np.nan
    innovation = G11 - G01**2 / G00 if G00 > 0 else np.nan
    return {
        "random_intercept_variance": G00,
        "random_slope_variance": G11,
        "intercept_slope_covariance": G01,
        "intercept_slope_correlation": rho,
        "one_minus_rho_squared": one_minus,
        "covariance_minimum_eigenvalue": float(eig.min()),
        "covariance_trace": trace,
        "minimum_eigenvalue_fraction_of_trace": eigen_fraction,
        "innovation_variance_denominator": innovation,
    }

# ==============================================================================
# participant_reference_metrics — frozen notebook cell 11
# ==============================================================================
def participant_reference_metrics(G_mid, residual_variance, data, reference_years):
    G_ref, delta = covariance_at_reference(G_mid, reference_years)
    geometry = covariance_geometry(G_ref)
    innovation_prior = geometry["innovation_variance_denominator"]
    eig = np.linalg.eigvalsh(G_ref)
    inverse_random = np.linalg.inv(G_ref) if eig.min() > 0 else None
    innovation_loading = (
        G_ref[0, 1] / G_ref[0, 0] if G_ref[0, 0] > 0 else np.nan
    )
    rows = []
    for participant_id, group in data.groupby("pid", sort=False):
        x_ref = group["t5"].to_numpy(float) - float(reference_years) / 5.0
        full = innovation_reliability = np.nan
        posterior_slope_variance = posterior_innovation_variance = np.nan
        if inverse_random is not None and residual_variance > 0:
            design = np.column_stack([np.ones(len(group)), x_ref])
            posterior_covariance = np.linalg.inv(
                inverse_random + design.T @ design / residual_variance
            )
            posterior_slope_variance = float(posterior_covariance[1, 1])
            full = clamp_rounding_only(
                1 - posterior_slope_variance / geometry["random_slope_variance"]
            )
            if np.isfinite(innovation_prior) and innovation_prior > 0:
                loading = np.array([-innovation_loading, 1.0])
                posterior_innovation_variance = float(
                    loading @ posterior_covariance @ loading
                )
                innovation_reliability = clamp_rounding_only(
                    1 - posterior_innovation_variance / innovation_prior
                )
        rows.append({
            "participant_id": str(participant_id),
            "scan_count": int(len(group)),
            "followup_years": float((group.t5.max() - group.t5.min()) * 5),
            "mean_observation_time_years": float(group.t5.mean() * 5),
            "reference_years_after_anchor": float(reference_years),
            "reference_shift_in_five_year_units": delta,
            "innovation_variance_denominator": innovation_prior,
            "posterior_innovation_variance": posterior_innovation_variance,
            "innovation_reliability": innovation_reliability,
            "posterior_slope_variance": posterior_slope_variance,
            "full_posterior_slope_predictability": full,
            "intercept_slope_correlation_at_reference": geometry["intercept_slope_correlation"],
        })
    return pd.DataFrame(rows), geometry

# ==============================================================================
# fixed_parameters_finite — frozen notebook cell 14
# ==============================================================================
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

# ==============================================================================
# hessian_diagnostic — frozen notebook cell 14
# ==============================================================================
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

# ==============================================================================
# reconstruct_bootstrap_dataset — frozen notebook cell 14
# ==============================================================================
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
    data = pd.concat(pieces, ignore_index=True)
    assert data.pid.nunique() == len(participant_ids)
    return data

# ==============================================================================
# fit_bootstrap_model — frozen notebook cell 14
# ==============================================================================
def fit_bootstrap_model(data, outcome, optimizer_order):
    last_error = ""
    for optimizer in optimizer_order:
        caught = []
        try:
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always")
                result = smf.mixedlm(
                    fixed_formula(outcome), data=data, groups=data["pid"],
                    re_formula="1+t5_mid",
                ).fit(
                    reml=True, method=optimizer, maxiter=MAXITER, disp=False,
                )
                caught = [str(item.message) for item in warning_records]
            G = np.asarray(result.cov_re, float)
            residual = float(result.scale)
            eigenvalues = np.linalg.eigvalsh((G + G.T) / 2)
            tolerance = PSD_REL_TOL * max(
                1.0, float(np.max(np.abs(eigenvalues))), abs(residual)
            )
            covariance_psd = bool(
                np.isfinite(G).all() and residual > 0
                and float(eigenvalues.min()) >= -tolerance
            )
            basic_valid = bool(
                result.converged and np.isfinite(result.llf)
                and fixed_parameters_finite(result) and covariance_psd
            )
            if basic_valid:
                return result, optimizer, caught, ""
            last_error = "nonconverged, nonfinite, or non-PSD"
        except Exception as exc:
            last_error = repr(exc)
    return None, "", caught, last_error or "all optimizers failed"

# ==============================================================================
# median_reference_metrics — frozen notebook cell 14
# ==============================================================================
def median_reference_metrics(G_mid, residual, data, reference_years):
    participant_table, geometry = participant_reference_metrics(
        G_mid, residual, data, reference_years
    )
    return {
        "innovation_variance_denominator": geometry["innovation_variance_denominator"],
        "intercept_slope_correlation": geometry["intercept_slope_correlation"],
        "median_innovation_reliability": float(participant_table.innovation_reliability.median()),
        "median_full_predictability": float(participant_table.full_posterior_slope_predictability.median()),
        "finite_innovation_n": int(np.isfinite(participant_table.innovation_reliability).sum()),
    }

# ==============================================================================
# direct_grr — frozen notebook cell 14
# ==============================================================================
def direct_grr(slope_variance, residual_variance, information):
    if not (
        np.isfinite(slope_variance) and slope_variance > 0
        and np.isfinite(residual_variance) and residual_variance > 0
        and np.isfinite(information) and information > 0
    ):
        return np.nan
    return float(slope_variance / (slope_variance + residual_variance / information))

# ==============================================================================
# fit_one_bootstrap_geometry — frozen notebook cell 14
# ==============================================================================
def fit_one_bootstrap_geometry(task):
    outcome = str(task["outcome"])
    replicate = int(task["replicate"])
    seed = int(task["seed"])
    expected_optimizer = str(task.get("expected_optimizer", "")).strip()
    fallback_optimizer = str(task.get("fallback_optimizer", "")).strip()
    optimizer_order = tuple(dict.fromkeys([
        optimizer for optimizer in (expected_optimizer, fallback_optimizer, *OPTIMIZERS)
        if optimizer in OPTIMIZERS
    ]))
    data = reconstruct_bootstrap_dataset(outcome, seed)
    result, optimizer, caught, error = fit_bootstrap_model(data, outcome, optimizer_order)
    base = {
        "outcome": outcome,
        "replicate": replicate,
        "seed": seed,
        "expected_optimizer_from_v3": expected_optimizer,
        "optimizer": optimizer,
        "fit_succeeded": result is not None,
        "error": error,
        "warning_count": len(caught),
        "warnings": " | ".join(sorted(set(caught))),
    }
    if result is None:
        return base

    G = np.asarray(result.cov_re, float)
    residual = float(result.scale)
    geometry = covariance_geometry(G)
    eigenvalues = np.linalg.eigvalsh((G + G.T) / 2)
    interior = bool(eigenvalues.min() > 0 and geometry["random_slope_variance"] > 0)
    hessian_pd, hessian_minimum, hessian_maximum = hessian_diagnostic(result)
    base.update({
        "converged": bool(result.converged),
        "reml_log_likelihood": float(result.llf),
        "residual_variance": residual,
        "random_covariance_interior": interior,
        "hessian_positive_definite": hessian_pd,
        "hessian_min_information_eigenvalue": hessian_minimum,
        "hessian_max_information_eigenvalue": hessian_maximum,
        **geometry,
    })

    if interior:
        for reference_label, reference_years in {
            **REFERENCE_TIMES_YEARS, "model_midpoint_2p5y": MODEL_ORIGIN_YEARS,
        }.items():
            metrics = median_reference_metrics(G, residual, data, reference_years)
            for name, value in metrics.items():
                base[f"{reference_label}__{name}"] = value

        information = data.groupby("pid")["t5_mid"].apply(
            lambda x: float(np.sum((x - x.mean()) ** 2))
        )
        direct_values = [
            direct_grr(geometry["random_slope_variance"], residual, value)
            for value in information
        ]
        base["recomputed_median_direct_GRR"] = float(np.nanmedian(direct_values))
        base["recomputed_median_full_predictability"] = base[
            "model_midpoint_2p5y__median_full_predictability"
        ]
        base["recomputed_median_innovation_reliability"] = base[
            "model_midpoint_2p5y__median_innovation_reliability"
        ]
        base["recomputed_median_full_minus_direct_gap"] = (
            base["recomputed_median_full_predictability"]
            - base["recomputed_median_direct_GRR"]
        )
    return base

# ==============================================================================
# threshold_sweep — frozen notebook cell 18
# ==============================================================================
def threshold_sweep(
    frame,
    metric_columns,
    group_columns,
    scope,
    eligible_column=None,
    high_gap_column=None,
    high_gap_threshold=0.10,
    high_full_column=None,
    high_full_threshold=0.40,
):
    rows = []
    if group_columns:
        grouped = frame.groupby(group_columns, dropna=False, sort=False)
    else:
        grouped = [((), frame)]
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        labels = dict(zip(group_columns, group_key))
        if eligible_column is not None:
            group = group.loc[coerce_bool(group[eligible_column])].copy()
        for metric_label, metric_column in metric_columns.items():
            eligible = group.loc[np.isfinite(pd.to_numeric(group[metric_column], errors="coerce"))].copy()
            values = pd.to_numeric(eligible[metric_column], errors="coerce")
            for threshold in SINGULARITY_THRESHOLDS:
                near = values <= threshold
                row = {
                    "scope": scope,
                    **labels,
                    "geometry_metric": metric_label,
                    "threshold": threshold,
                    "eligible_n": len(eligible),
                    "near_singular_n": int(near.sum()),
                    "near_singular_proportion": float(near.mean()) if len(near) else np.nan,
                }
                if high_gap_column is not None and high_gap_column in eligible:
                    high_gap = pd.to_numeric(eligible[high_gap_column], errors="coerce") >= high_gap_threshold
                    row.update({
                        "high_gap_threshold": high_gap_threshold,
                        "high_gap_n": int(high_gap.sum()),
                        "P_near_given_high_gap": float(near.loc[high_gap].mean()) if high_gap.any() else np.nan,
                        "P_high_gap_given_near": float(high_gap.loc[near].mean()) if near.any() else np.nan,
                    })
                if high_full_column is not None and high_full_column in eligible:
                    high_full = pd.to_numeric(eligible[high_full_column], errors="coerce") >= high_full_threshold
                    row.update({
                        "high_full_threshold": high_full_threshold,
                        "high_full_n": int(high_full.sum()),
                        "P_near_given_high_full": float(near.loc[high_full].mean()) if high_full.any() else np.nan,
                        "P_high_full_given_near": float(high_full.loc[near].mean()) if near.any() else np.nan,
                    })
                rows.append(row)
    result = pd.DataFrame(rows)
    return result

# ==============================================================================
# assert_nested_threshold_counts — frozen notebook cell 18
# ==============================================================================
def assert_nested_threshold_counts(table, grouping_columns):
    for _, group in table.groupby(grouping_columns, dropna=False):
        ordered = group.sort_values("threshold")
        counts = ordered.near_singular_n.to_numpy(float)
        assert np.all(np.diff(counts) >= 0), ordered

# ---------------------------------------------------------------------------
# Input normalization after frozen helper definitions are available
# ---------------------------------------------------------------------------

gap_source[
    "valid"
] = coerce_bool(
    gap_source[
        "valid"
    ]
)

gap_source[
    "replicate"
] = (
    pd.to_numeric(
        gap_source[
            "replicate"
        ],
        errors="raise",
    )
    .astype(int)
)

gap_source = (
    gap_source.loc[
        gap_source
        .outcome
        .isin(
            TARGET_BOOTSTRAP_OUTCOMES
        )
    ]
    .copy()
)

assert not (
    gap_source
    .duplicated(
        [
            "outcome",
            "replicate",
        ]
    )
    .any()
)

assert set(
    gap_source.outcome
) == set(
    TARGET_BOOTSTRAP_OUTCOMES
)

assert (
    gap_source
    .groupby("outcome")
    .replicate
    .nunique()
    .eq(
        EXPECTED_BOOTSTRAP_REPS
    )
    .all()
)


selected_participant_source[
    "selected_representative"
] = coerce_bool(
    selected_participant_source[
        "selected_representative"
    ]
)

representative_fit_source[
    "representative_available"
] = coerce_bool(
    representative_fit_source[
        "representative_available"
    ]
)


# ===== Frozen cell 12: observed reference-time calculations =====
# 8 — Observed-fit innovation reliability at anchor and five-year reference
observed_reference_rows = []
observed_geometry_rows = []

for outcome in tqdm(OUTCOMES, desc="Observed reference-time innovation reliability", unit="outcome"):
    source = exact_source.loc[exact_source.outcome.eq(outcome)].copy()
    if source.empty:
        raise ValueError(f"No rows for {outcome} in {EXACT_RELIABILITY_PATH}")
    first = source.iloc[0]
    G_mid = np.array([
        [first.random_intercept_variance, first.intercept_slope_covariance],
        [first.intercept_slope_covariance, first.random_slope_variance],
    ], dtype=float)
    residual = float(first.residual_variance)
    data = ANALYSIS_DATA[outcome]
    assert set(data.pid.astype(str)) == set(source.participant_id.astype(str))
    for reference_label, reference_years in REFERENCE_TIMES_YEARS.items():
        participant_table, geometry = participant_reference_metrics(
            G_mid, residual, data, reference_years
        )
        participant_table.insert(0, "outcome", outcome)
        participant_table.insert(1, "label", OUTCOMES[outcome]["label"])
        participant_table.insert(2, "reference_label", reference_label)
        observed_reference_rows.append(participant_table)
        observed_geometry_rows.append({
            "outcome": outcome,
            "label": OUTCOMES[outcome]["label"],
            "reference_label": reference_label,
            "reference_years_after_anchor": reference_years,
            **geometry,
        })

observed_reference_participant = pd.concat(observed_reference_rows, ignore_index=True)
observed_reference_geometry = pd.DataFrame(observed_geometry_rows)

observed_reference_summary = observed_reference_participant.groupby(
    ["outcome", "label", "reference_label", "reference_years_after_anchor"],
    as_index=False,
).agg(
    participants=("participant_id", "nunique"),
    innovation_variance_denominator=("innovation_variance_denominator", "first"),
    intercept_slope_correlation_at_reference=("intercept_slope_correlation_at_reference", "first"),
    median_innovation_reliability=("innovation_reliability", "median"),
    q25_innovation_reliability=("innovation_reliability", lambda x: x.quantile(.25)),
    q75_innovation_reliability=("innovation_reliability", lambda x: x.quantile(.75)),
    mean_innovation_reliability=("innovation_reliability", "mean"),
    finite_innovation_reliability_n=("innovation_reliability", lambda x: int(np.isfinite(x).sum())),
    median_full_posterior_predictability=("full_posterior_slope_predictability", "median"),
)

denominator_wide = observed_reference_summary.pivot(
    index=["outcome", "label"],
    columns="reference_label",
    values=["innovation_variance_denominator", "median_innovation_reliability"],
)
denominator_wide.columns = ["__".join(column) for column in denominator_wide.columns]
observed_denominator_change = denominator_wide.reset_index()
anchor_denom = "innovation_variance_denominator__anchor_0y"
five_denom = "innovation_variance_denominator__five_year_5y"
observed_denominator_change["denominator_absolute_change_5y_minus_anchor"] = (
    observed_denominator_change[five_denom] - observed_denominator_change[anchor_denom]
)
observed_denominator_change["denominator_ratio_5y_over_anchor"] = (
    observed_denominator_change[five_denom] / observed_denominator_change[anchor_denom]
)

# Numerical validation against the exact V3 midpoint calculation.
midpoint_validation_rows = []
for outcome in OUTCOMES:
    source = exact_source.loc[exact_source.outcome.eq(outcome)].copy()
    first = source.iloc[0]
    G_mid = np.array([
        [first.random_intercept_variance, first.intercept_slope_covariance],
        [first.intercept_slope_covariance, first.random_slope_variance],
    ], dtype=float)
    recomputed, _ = participant_reference_metrics(
        G_mid, float(first.residual_variance), ANALYSIS_DATA[outcome], MODEL_ORIGIN_YEARS
    )
    comparison = source[["participant_id", "conditional_slope_innovation_reliability"]].merge(
        recomputed[["participant_id", "innovation_reliability"]],
        on="participant_id", validate="one_to_one",
    )
    difference = (
        comparison.conditional_slope_innovation_reliability
        - comparison.innovation_reliability
    ).abs()
    midpoint_validation_rows.append({
        "outcome": outcome,
        "participants": len(comparison),
        "finite_pairs": int(np.isfinite(difference).sum()),
        "maximum_absolute_difference": float(difference.max()),
    })
midpoint_validation = pd.DataFrame(midpoint_validation_rows)
assert midpoint_validation.maximum_absolute_difference.fillna(0).max() < 1e-6

display(observed_reference_summary.round(6))
display(observed_denominator_change.round(6))
display(midpoint_validation)



# ===== Frozen cell 15: resumable full-499 geometry enrichment =====
# 10 — Resumable full-499 × two-outcome enrichment with tqdm
optimizer_modes = (
    gap_source.loc[gap_source.valid & gap_source.get("optimizer", pd.Series(index=gap_source.index, dtype=object)).isin(OPTIMIZERS)]
    .groupby("outcome")["optimizer"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else OPTIMIZERS[0])
    .to_dict()
    if "optimizer" in gap_source.columns else {}
)

outcome_seed_index = {"global_mean_thickness": 0, "fluid_cognition": 1}
task_manifest = gap_source[["outcome", "replicate"]].copy()
task_manifest["seed"] = task_manifest.apply(
    lambda row: SEED + 8_000_000 + 100_000 * outcome_seed_index[row.outcome] + int(row.replicate),
    axis=1,
)
if "optimizer" in gap_source.columns:
    task_manifest = task_manifest.merge(
        gap_source[["outcome", "replicate", "optimizer"]].rename(
            columns={"optimizer": "expected_optimizer"}
        ),
        on=["outcome", "replicate"], how="left", validate="one_to_one",
    )
else:
    task_manifest["expected_optimizer"] = ""
task_manifest["expected_optimizer"] = task_manifest["expected_optimizer"].fillna("")
task_manifest["fallback_optimizer"] = task_manifest.outcome.map(optimizer_modes).fillna(OPTIMIZERS[0])
task_manifest["task_id"] = task_manifest.outcome + "__" + task_manifest.replicate.astype(str)
assert len(task_manifest) == len(TARGET_BOOTSTRAP_OUTCOMES) * EXPECTED_BOOTSTRAP_REPS
assert task_manifest.task_id.nunique() == len(task_manifest)

checkpoint_signature = json.dumps({
    "analysis_version": "section10.1-items2to4-v1",
    "seed": SEED,
    "outcomes": TARGET_BOOTSTRAP_OUTCOMES,
    "expected_B": EXPECTED_BOOTSTRAP_REPS,
    "reference_times": REFERENCE_TIMES_YEARS,
    "model_origin_years": MODEL_ORIGIN_YEARS,
    "optimizers": OPTIMIZERS,
}, sort_keys=True)

if CHECKPOINT_FILE.exists():
    checkpoint = pd.read_pickle(CHECKPOINT_FILE)
    if checkpoint.get("signature") != checkpoint_signature:
        raise RuntimeError(
            "Existing checkpoint has a different frozen configuration. Move it aside before intentionally restarting."
        )
    bootstrap_geometry = checkpoint.get("bootstrap_geometry", pd.DataFrame())
else:
    bootstrap_geometry = pd.DataFrame()

completed = set()
if not bootstrap_geometry.empty:
    completed = set(
        bootstrap_geometry.outcome.astype(str)
        + "__" + bootstrap_geometry.replicate.astype(int).astype(str)
    )
pending = task_manifest.loc[~task_manifest.task_id.isin(completed)].to_dict("records")

if pending and not RUN_FULL_499_GEOMETRY_ENRICHMENT:
    raise RuntimeError(
        f"{len(pending)} full-bootstrap geometry tasks are missing. "
        "Set RUN_FULL_499_GEOMETRY_ENRICHMENT=True to complete them."
    )

progress = tqdm(total=len(pending), desc="Full-499 bootstrap covariance enrichment", unit="replicate")
for start in range(0, len(pending), BATCH_SIZE):
    batch = pending[start:start + BATCH_SIZE]
    rows = Parallel(
        n_jobs=N_JOBS, backend="loky", batch_size=1, pre_dispatch=N_JOBS,
    )(delayed(fit_one_bootstrap_geometry)(task) for task in batch)
    bootstrap_geometry = pd.concat(
        [bootstrap_geometry, pd.DataFrame(rows)], ignore_index=True,
    )
    bootstrap_geometry = bootstrap_geometry.drop_duplicates(
        ["outcome", "replicate"], keep="last"
    )
    pd.to_pickle({
        "signature": checkpoint_signature,
        "bootstrap_geometry": bootstrap_geometry,
    }, CHECKPOINT_FILE)
    progress.update(len(batch))
progress.close()

bootstrap_geometry["replicate"] = pd.to_numeric(
    bootstrap_geometry["replicate"], errors="raise"
).astype(int)
bootstrap_geometry = bootstrap_geometry.sort_values(
    ["outcome", "replicate"]
).reset_index(drop=True)
assert len(bootstrap_geometry) == len(task_manifest)
assert not bootstrap_geometry.duplicated(["outcome", "replicate"]).any()

full_bootstrap_geometry = gap_source.merge(
    bootstrap_geometry,
    on=["outcome", "replicate"], how="left", validate="one_to_one",
    suffixes=("_v3", "_refit"),
)
full_bootstrap_geometry["geometry_eligible"] = (
    full_bootstrap_geometry["valid"]
    & full_bootstrap_geometry["fit_succeeded"].fillna(False)
    & full_bootstrap_geometry["random_covariance_interior"].fillna(False)
    & np.isfinite(full_bootstrap_geometry["one_minus_rho_squared"])
    & np.isfinite(full_bootstrap_geometry["minimum_eigenvalue_fraction_of_trace"])
)

metric_pairs = {
    "median_direct_GRR": "recomputed_median_direct_GRR",
    "median_full_predictability": "recomputed_median_full_predictability",
    "median_innovation_reliability": "recomputed_median_innovation_reliability",
    "median_full_minus_direct_gap": "recomputed_median_full_minus_direct_gap",
}
for original, recomputed in metric_pairs.items():
    full_bootstrap_geometry[f"absolute_difference__{original}"] = (
        full_bootstrap_geometry[original] - full_bootstrap_geometry[recomputed]
    ).abs()

bootstrap_reproduction_audit = full_bootstrap_geometry.groupby("outcome").agg(
    requested_B=("replicate", "nunique"),
    v3_valid_B=("valid", "sum"),
    geometry_eligible_B=("geometry_eligible", "sum"),
    optimizer_reproduced_rate=(
        "expected_optimizer_from_v3",
        lambda x: float(np.mean(x.astype(str) == full_bootstrap_geometry.loc[x.index, "optimizer_refit"].astype(str))),
    ),
    maximum_direct_difference=("absolute_difference__median_direct_GRR", "max"),
    maximum_full_difference=("absolute_difference__median_full_predictability", "max"),
    maximum_innovation_difference=("absolute_difference__median_innovation_reliability", "max"),
    maximum_gap_difference=("absolute_difference__median_full_minus_direct_gap", "max"),
).reset_index()
display(bootstrap_reproduction_audit)



# ===== Frozen cell 16: bootstrap reference-time summaries =====
# 11 — Bootstrap distributions of reference-specific denominators and reliability
bootstrap_reference_long_rows = []
for reference_label in REFERENCE_TIMES_YEARS:
    denominator_column = f"{reference_label}__innovation_variance_denominator"
    reliability_column = f"{reference_label}__median_innovation_reliability"
    correlation_column = f"{reference_label}__intercept_slope_correlation"
    temp = full_bootstrap_geometry[[
        "outcome", "replicate", "geometry_eligible",
        denominator_column, reliability_column, correlation_column,
    ]].copy()
    temp = temp.rename(columns={
        denominator_column: "innovation_variance_denominator",
        reliability_column: "median_innovation_reliability",
        correlation_column: "intercept_slope_correlation_at_reference",
    })
    temp["reference_label"] = reference_label
    temp["reference_years_after_anchor"] = REFERENCE_TIMES_YEARS[reference_label]
    bootstrap_reference_long_rows.append(temp)

bootstrap_reference_long = pd.concat(bootstrap_reference_long_rows, ignore_index=True)
bootstrap_reference_summary = (
    bootstrap_reference_long.loc[bootstrap_reference_long.geometry_eligible]
    .groupby(["outcome", "reference_label", "reference_years_after_anchor"], as_index=False)
    .agg(
        valid_B=("replicate", "nunique"),
        denominator_q025=("innovation_variance_denominator", lambda x: x.quantile(.025)),
        denominator_median=("innovation_variance_denominator", "median"),
        denominator_q975=("innovation_variance_denominator", lambda x: x.quantile(.975)),
        median_innovation_reliability_q025=("median_innovation_reliability", lambda x: x.quantile(.025)),
        median_innovation_reliability_median=("median_innovation_reliability", "median"),
        median_innovation_reliability_q975=("median_innovation_reliability", lambda x: x.quantile(.975)),
        rho_reference_q025=("intercept_slope_correlation_at_reference", lambda x: x.quantile(.025)),
        rho_reference_median=("intercept_slope_correlation_at_reference", "median"),
        rho_reference_q975=("intercept_slope_correlation_at_reference", lambda x: x.quantile(.975)),
    )
)

bootstrap_change = full_bootstrap_geometry.loc[
    full_bootstrap_geometry.geometry_eligible,
    [
        "outcome", "replicate",
        "anchor_0y__innovation_variance_denominator",
        "five_year_5y__innovation_variance_denominator",
        "anchor_0y__median_innovation_reliability",
        "five_year_5y__median_innovation_reliability",
    ],
].copy()
bootstrap_change["denominator_ratio_5y_over_anchor"] = (
    bootstrap_change["five_year_5y__innovation_variance_denominator"]
    / bootstrap_change["anchor_0y__innovation_variance_denominator"]
)
bootstrap_change["median_innovation_reliability_change_5y_minus_anchor"] = (
    bootstrap_change["five_year_5y__median_innovation_reliability"]
    - bootstrap_change["anchor_0y__median_innovation_reliability"]
)
bootstrap_reference_change_summary = bootstrap_change.groupby("outcome", as_index=False).agg(
    valid_B=("replicate", "nunique"),
    denominator_ratio_q025=("denominator_ratio_5y_over_anchor", lambda x: x.quantile(.025)),
    denominator_ratio_median=("denominator_ratio_5y_over_anchor", "median"),
    denominator_ratio_q975=("denominator_ratio_5y_over_anchor", lambda x: x.quantile(.975)),
    reliability_change_q025=("median_innovation_reliability_change_5y_minus_anchor", lambda x: x.quantile(.025)),
    reliability_change_median=("median_innovation_reliability_change_5y_minus_anchor", "median"),
    reliability_change_q975=("median_innovation_reliability_change_5y_minus_anchor", lambda x: x.quantile(.975)),
)

display(bootstrap_reference_summary.round(6))
display(bootstrap_reference_change_summary.round(6))



# ===== Frozen cell 19: empirical threshold sweep =====
# 13 — Full-499 empirical bootstrap threshold sweep
EMPIRICAL_METRICS = {
    "1-rho^2": "one_minus_rho_squared",
    "lambda_min(G)/trace(G)": "minimum_eigenvalue_fraction_of_trace",
}

empirical_sweep_by_outcome = threshold_sweep(
    full_bootstrap_geometry,
    metric_columns=EMPIRICAL_METRICS,
    group_columns=["outcome"],
    scope="empirical_full499_bootstrap",
    eligible_column="geometry_eligible",
    high_gap_column="median_full_minus_direct_gap",
    high_gap_threshold=0.10,
    high_full_column="median_full_predictability",
    high_full_threshold=0.40,
)
empirical_pooled = full_bootstrap_geometry.copy()
empirical_pooled["outcome"] = "pooled_two_outcomes"
empirical_sweep_pooled = threshold_sweep(
    empirical_pooled,
    metric_columns=EMPIRICAL_METRICS,
    group_columns=["outcome"],
    scope="empirical_full499_bootstrap",
    eligible_column="geometry_eligible",
    high_gap_column="median_full_minus_direct_gap",
    high_gap_threshold=0.10,
    high_full_column="median_full_predictability",
    high_full_threshold=0.40,
)
empirical_singularity_sweep = pd.concat(
    [empirical_sweep_by_outcome, empirical_sweep_pooled], ignore_index=True
)
assert set(empirical_singularity_sweep.threshold) == set(SINGULARITY_THRESHOLDS)
assert_nested_threshold_counts(
    empirical_singularity_sweep,
    ["scope", "outcome", "geometry_metric"],
)
display(empirical_singularity_sweep.round(5))



# ===== Frozen cell 20: selected-60 diagnostic sweep =====
# 14 — Selected-60 optimizer-panel sweep (diagnostic, not prevalence)
selected60 = representative_fit_source.loc[
    representative_fit_source.representative_available
].copy()
selected60["one_minus_rho_squared"] = 1 - (
    pd.to_numeric(selected60["representative_intercept_slope_correlation"], errors="coerce") ** 2
)
selected60.loc[
    selected60.one_minus_rho_squared.between(-ROUNDING_TOL, 0, inclusive="left"),
    "one_minus_rho_squared",
] = 0.0
selected60["covariance_trace"] = (
    pd.to_numeric(selected60["representative_random_intercept_variance"], errors="coerce")
    + pd.to_numeric(selected60["representative_random_slope_variance"], errors="coerce")
)
selected60["minimum_eigenvalue_fraction_of_trace"] = (
    pd.to_numeric(selected60["representative_random_covariance_min_eigenvalue"], errors="coerce")
    / selected60["covariance_trace"]
)
selected60["geometry_eligible"] = (
    selected60.representative_available
    & np.isfinite(selected60.one_minus_rho_squared)
    & np.isfinite(selected60.minimum_eigenvalue_fraction_of_trace)
)

selected60_singularity_sweep = threshold_sweep(
    selected60,
    metric_columns=EMPIRICAL_METRICS,
    group_columns=["outcome", "gap_regime"],
    scope="selected60_optimizer_diagnostic_not_prevalence",
    eligible_column="geometry_eligible",
    high_gap_column="selection_gap",
    high_gap_threshold=0.10,
    high_full_column="representative_median_full_posterior_predictability",
    high_full_threshold=0.40,
)
assert_nested_threshold_counts(
    selected60_singularity_sweep,
    ["scope", "outcome", "gap_regime", "geometry_metric"],
)
display(selected60_singularity_sweep.round(5))



# ===== Frozen cell 21: final simulation sweep =====
# 15 — Completed final-500 simulation sweep, if its saved diagnostics are present
if SIMULATION_GEOMETRY_PATH is not None:
    simulation_geometry = pd.read_csv(SIMULATION_GEOMETRY_PATH)
    simulation_geometry = simulation_geometry.loc[
        simulation_geometry.model_role.eq("random_slope_primary")
    ].copy()
    if "true_slope_variance" in simulation_geometry:
        simulation_geometry["truth_slope_status"] = np.where(
            pd.to_numeric(simulation_geometry.true_slope_variance, errors="coerce").eq(0),
            "true_zero", "positive_variance",
        )
    else:
        simulation_geometry["truth_slope_status"] = np.where(
            simulation_geometry.regime.astype(str).str.contains("zero", case=False),
            "true_zero", "positive_variance",
        )
    simulation_geometry["geometry_eligible"] = (
        np.isfinite(pd.to_numeric(simulation_geometry.one_minus_rho_squared, errors="coerce"))
        & np.isfinite(pd.to_numeric(simulation_geometry.minimum_eigenvalue_fraction_of_trace, errors="coerce"))
    )
    simulation_singularity_sweep = threshold_sweep(
        simulation_geometry,
        metric_columns=EMPIRICAL_METRICS,
        group_columns=["truth_slope_status", "regime"],
        scope="final500_simulation_fitted_random_slope_model",
        eligible_column="geometry_eligible",
        high_full_column="mean_full_posterior_predictability",
        high_full_threshold=0.40,
    )
    assert_nested_threshold_counts(
        simulation_singularity_sweep,
        ["scope", "truth_slope_status", "regime", "geometry_metric"],
    )
else:
    simulation_geometry = pd.DataFrame()
    simulation_singularity_sweep = pd.DataFrame(columns=[
        "scope", "truth_slope_status", "regime", "geometry_metric", "threshold",
        "eligible_n", "near_singular_n", "near_singular_proportion",
    ])
    warnings.warn(
        "Final-500 simulation diagnostics were not found. The empirical and selected-60 sweeps are complete; the optional simulation sweep is empty."
    )

display(simulation_singularity_sweep.round(5))



# ===== Frozen cell 23: selected schedule summaries =====
# 16 — Selected-representative participant schedules and branch summaries
selected_participants = selected_participant_source.loc[
    selected_participant_source.selected_representative
].copy()
selected_participants["replicate"] = pd.to_numeric(
    selected_participants.replicate, errors="raise"
).astype(int)
selected_participants["scan_count"] = pd.to_numeric(
    selected_participants.scan_count, errors="coerce"
)
selected_participants["followup_years"] = pd.to_numeric(
    selected_participants.followup_years, errors="coerce"
)
selected_participants["schedule_information"] = pd.to_numeric(
    selected_participants.schedule_information, errors="coerce"
)
selected_participants["two_scan"] = selected_participants.scan_count.eq(2)
selected_participants["three_or_more_scans"] = selected_participants.scan_count.ge(3)
selected_participants["followup_le_2y"] = selected_participants.followup_years.le(2)
selected_participants["followup_gt_4y"] = selected_participants.followup_years.gt(4)

selected_counts = selected_participants.groupby(
    ["outcome", "gap_regime"]
).replicate.nunique().unstack(fill_value=0)
assert selected_counts.reindex(columns=["lowest", "median", "highest"]).eq(10).all().all()

replicate_schedule = selected_participants.groupby(
    ["outcome", "gap_regime", "replicate"], as_index=False
).agg(
    participant_draws=("participant_boot_id", "nunique"),
    unique_original_participants=("original_participant_id", "nunique"),
    median_scan_count=("scan_count", "median"),
    proportion_two_scan=("two_scan", "mean"),
    proportion_three_or_more_scans=("three_or_more_scans", "mean"),
    median_followup_years=("followup_years", "median"),
    q25_followup_years=("followup_years", lambda x: x.quantile(.25)),
    q75_followup_years=("followup_years", lambda x: x.quantile(.75)),
    proportion_followup_le_2y=("followup_le_2y", "mean"),
    proportion_followup_gt_4y=("followup_gt_4y", "mean"),
    median_schedule_information=("schedule_information", "median"),
    median_direct_reliability=("direct_reliability", "median"),
    median_full_posterior_predictability=("full_posterior_predictability", "median"),
    median_innovation_reliability=("innovation_reliability", "median"),
)

def q25(x):
    return x.quantile(.25)

def q75(x):
    return x.quantile(.75)

schedule_by_branch_replicate_weighted = replicate_schedule.groupby(
    ["outcome", "gap_regime"], as_index=False
).agg(
    selected_replicates=("replicate", "nunique"),
    participant_draws_per_replicate_median=("participant_draws", "median"),
    unique_original_participants_per_replicate_median=("unique_original_participants", "median"),
    scan_count_median_across_replicates=("median_scan_count", "median"),
    proportion_two_scan_q25=("proportion_two_scan", q25),
    proportion_two_scan_median=("proportion_two_scan", "median"),
    proportion_two_scan_q75=("proportion_two_scan", q75),
    proportion_three_or_more_scans_median=("proportion_three_or_more_scans", "median"),
    followup_years_q25_of_replicate_medians=("median_followup_years", q25),
    followup_years_median_of_replicate_medians=("median_followup_years", "median"),
    followup_years_q75_of_replicate_medians=("median_followup_years", q75),
    proportion_followup_le_2y_median=("proportion_followup_le_2y", "median"),
    proportion_followup_gt_4y_median=("proportion_followup_gt_4y", "median"),
    schedule_information_q25=("median_schedule_information", q25),
    schedule_information_median=("median_schedule_information", "median"),
    schedule_information_q75=("median_schedule_information", q75),
    direct_reliability_median=("median_direct_reliability", "median"),
    full_predictability_median=("median_full_posterior_predictability", "median"),
    innovation_reliability_median=("median_innovation_reliability", "median"),
)

schedule_by_branch_draw_level = selected_participants.groupby(
    ["outcome", "gap_regime"], as_index=False
).agg(
    selected_replicates=("replicate", "nunique"),
    participant_draws=("participant_boot_id", "size"),
    median_scan_count=("scan_count", "median"),
    proportion_two_scan=("two_scan", "mean"),
    proportion_three_or_more_scans=("three_or_more_scans", "mean"),
    followup_years_q25=("followup_years", q25),
    followup_years_median=("followup_years", "median"),
    followup_years_q75=("followup_years", q75),
    schedule_information_q25=("schedule_information", q25),
    schedule_information_median=("schedule_information", "median"),
    schedule_information_q75=("schedule_information", q75),
)

display(schedule_by_branch_replicate_weighted.round(4))
display(schedule_by_branch_draw_level.round(4))



# ===== Frozen cell 25: figures =====
# 17 — Reference-time and empirical-threshold figures
outcome_order = list(OUTCOMES)
label_map = {key: value["label"] for key, value in OUTCOMES.items()}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for reference_label, color, marker in [
    ("anchor_0y", "#2b6cb0", "o"),
    ("five_year_5y", "#c53030", "s"),
]:
    subset = (
        observed_reference_summary.loc[
            observed_reference_summary.reference_label.eq(reference_label)
        ]
        .set_index("outcome").reindex(outcome_order).reset_index()
    )
    x = np.arange(len(subset))
    axes[0].plot(
        x, np.log10(subset.innovation_variance_denominator),
        marker=marker, color=color, label=reference_label,
    )
    axes[1].plot(
        x, subset.median_innovation_reliability,
        marker=marker, color=color, label=reference_label,
    )
axes[0].set_ylabel(r"$\log_{10}\{\mathrm{Var}(u_{\mathrm{ref}})\}$")
axes[1].set_ylabel("Median innovation reliability")
for axis in axes:
    axis.set_xticks(np.arange(len(outcome_order)))
    axis.set_xticklabels([label_map[key] for key in outcome_order], rotation=35, ha="right")
    axis.grid(alpha=.2)
    axis.legend(frameon=False)
axes[0].set_title("Changing innovation denominator")
axes[1].set_title("Reference-specific innovation reliability")
fig.tight_layout()
reference_figure_path = PUBLIC_DIR / "reference_time_innovation_denominator_and_reliability.png"
fig.savefig(reference_figure_path, dpi=220, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for axis, metric in zip(axes, EMPIRICAL_METRICS):
    subset = empirical_singularity_sweep.loc[
        empirical_singularity_sweep.geometry_metric.eq(metric)
        & ~empirical_singularity_sweep.outcome.eq("pooled_two_outcomes")
    ]
    for outcome, color in zip(TARGET_BOOTSTRAP_OUTCOMES, ["#805ad5", "#dd6b20"]):
        line = subset.loc[subset.outcome.eq(outcome)].sort_values("threshold")
        axis.plot(
            line.threshold, line.near_singular_proportion,
            marker="o", label=label_map[outcome], color=color,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Near-singularity threshold")
    axis.set_title(metric)
    axis.grid(alpha=.2)
    axis.legend(frameon=False)
axes[0].set_ylabel("Proportion among eligible bootstrap fits")
fig.tight_layout()
threshold_figure_path = PUBLIC_DIR / "empirical_full499_singularity_threshold_sweep.png"
fig.savefig(threshold_figure_path, dpi=220, bbox_inches="tight")
plt.close(fig)



# ===== Frozen reporting-notes assignment from cell 26 =====
reporting_notes = pd.DataFrame([
    {
        "topic": "reference times",
        "required wording": "Anchor (0 years) and five-year follow-up are scientific reference times obtained by exact covariance transformation from the 2.5-year fitted origin; no model was refitted.",
    },
    {
        "topic": "innovation denominator",
        "required wording": "Every innovation-reliability estimate is accompanied by Var(u_ref); values are boundary-sensitive when this denominator approaches zero.",
    },
    {
        "topic": "full-499 sweep denominator",
        "required wording": "Near-singularity proportions use V3-valid, successfully reproduced, interior fits with finite geometry; eligible n is reported for every threshold.",
    },
    {
        "topic": "selected-60 sweep",
        "required wording": "The low/central/high panel was deliberately selected and is a numerical diagnostic, not a random sample or prevalence estimate.",
    },
    {
        "topic": "schedule table",
        "required wording": "The primary schedule-by-branch table gives each of the 10 selected bootstrap datasets equal weight within branch; draw-level summaries are supplemental.",
    },
    {
        "topic": "geometry reference",
        "required wording": "Empirical rho and eigenvalue-fraction thresholds are evaluated at the frozen 2.5-year model origin; these geometry diagnostics are not invariant to a time-origin shift.",
    },
])

# ===== Frozen completion-checklist assignment from cell 26 =====
completion = pd.DataFrame([
    {
        "requirement": "two scientific reference times for all six outcomes",
        "complete": observed_reference_summary.groupby("outcome").reference_label.nunique().eq(2).all()
        and observed_reference_summary.outcome.nunique() == 6,
    },
    {
        "requirement": "innovation denominator reported with every reference-time summary",
        "complete": observed_reference_summary.innovation_variance_denominator.notna().all(),
    },
    {
        "requirement": "midpoint calculation reproduces V3",
        "complete": midpoint_validation.maximum_absolute_difference.fillna(0).max() < 1e-6,
    },
    {
        "requirement": "all 499 empirical tasks retained per target outcome",
        "complete": full_bootstrap_geometry.groupby("outcome").replicate.nunique().eq(EXPECTED_BOOTSTRAP_REPS).all(),
    },
    {
        "requirement": "both geometry metrics swept at exactly four frozen thresholds",
        "complete": set(empirical_singularity_sweep.geometry_metric) == set(EMPIRICAL_METRICS)
        and set(empirical_singularity_sweep.threshold) == set(SINGULARITY_THRESHOLDS),
    },
    {
        "requirement": "selected-60 diagnostic labeled separately",
        "complete": selected60_singularity_sweep.scope.eq(
            "selected60_optimizer_diagnostic_not_prevalence"
        ).all(),
    },
    {
        "requirement": "10 selected datasets per outcome and branch in schedule table",
        "complete": selected_counts.reindex(columns=["lowest", "median", "highest"]).eq(10).all().all(),
    },
    {
        "requirement": "two figures saved",
        "complete": reference_figure_path.exists() and threshold_figure_path.exists(),
    },
])

# ---------------------------------------------------------------------------
# Release output routing
# ---------------------------------------------------------------------------

PUBLIC_EXPORTS = {
    "02_observed_reference_time_summary.csv":
        observed_reference_summary,

    "03_observed_reference_time_denominator_change.csv":
        observed_denominator_change,

    "04_midpoint_reproduction_validation.csv":
        midpoint_validation,

    "05_full499_bootstrap_covariance_geometry.csv":
        full_bootstrap_geometry,

    "06_full499_bootstrap_reproduction_audit.csv":
        bootstrap_reproduction_audit,

    "07_full499_reference_time_bootstrap_summary.csv":
        bootstrap_reference_summary,

    "08_full499_reference_time_change_summary.csv":
        bootstrap_reference_change_summary,

    "09_empirical_full499_singularity_threshold_sweep.csv":
        empirical_singularity_sweep,

    "10_selected60_singularity_threshold_sweep.csv":
        selected60_singularity_sweep,

    "11_simulation_singularity_threshold_sweep.csv":
        simulation_singularity_sweep,

    "12_selected_schedule_per_replicate.csv":
        replicate_schedule,

    "13_schedule_by_branch_replicate_weighted.csv":
        schedule_by_branch_replicate_weighted,

    "14_schedule_by_branch_draw_level.csv":
        schedule_by_branch_draw_level,

    "15_reporting_notes.csv":
        reporting_notes,

    "16_completion_checklist.csv":
        completion,
}


for _filename, _frame in (
    PUBLIC_EXPORTS.items()
):
    _frame.to_csv(
        PUBLIC_DIR
        / _filename,
        index=False,
    )


# Participant IDs occur in this table;
# it is intentionally restricted.
observed_reference_participant.to_csv(
    RESTRICTED_DIR
    / "01_observed_reference_time_participant.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# Final scientific / privacy invariants
# ---------------------------------------------------------------------------

assert len(
    observed_reference_participant
) == 8322

assert len(
    observed_reference_summary
) == 12

assert len(
    observed_denominator_change
) == 6

assert len(
    midpoint_validation
) == 6

assert (
    midpoint_validation
    .maximum_absolute_difference
    .fillna(0)
    .max()
    < 1e-6
)

assert len(
    full_bootstrap_geometry
) == 998

assert (
    full_bootstrap_geometry
    .groupby("outcome")
    .replicate
    .nunique()
    .eq(499)
    .all()
)

assert len(
    bootstrap_reproduction_audit
) == 2

assert len(
    bootstrap_reference_summary
) == 4

assert len(
    bootstrap_reference_change_summary
) == 2

assert len(
    empirical_singularity_sweep
) == 24

assert len(
    selected60_singularity_sweep
) == 48

assert len(
    simulation_singularity_sweep
) == 56

assert len(
    replicate_schedule
) == 60

assert len(
    schedule_by_branch_replicate_weighted
) == 6

assert len(
    schedule_by_branch_draw_level
) == 6

assert len(
    reporting_notes
) == 6

assert len(
    completion
) == 8

assert (
    completion[
        "complete"
    ]
    .all()
)


# No participant identifiers are allowed in any public CSV schema.
_FORBIDDEN_PUBLIC_COLUMNS = {
    "participant_id",
    "participant_boot_id",
    "original_participant_id",
    "bootstrap_draw_index",
    "id_event",
    "pid",
}

for _filename, _frame in (
    PUBLIC_EXPORTS.items()
):
    _leaked = (
        set(_frame.columns)
        & _FORBIDDEN_PUBLIC_COLUMNS
    )

    if _leaked:
        raise RuntimeError(
            f"Public output {_filename} "
            "contains participant-level columns: "
            f"{sorted(_leaked)}"
        )


assert not (
    PUBLIC_DIR
    / "01_observed_reference_time_participant.csv"
).exists()

assert (
    RESTRICTED_DIR
    / "01_observed_reference_time_participant.csv"
).exists()


assert (
    CHECKPOINT_FILE.parent
    == CHECKPOINT_DIR
)


assert (
    reference_figure_path.parent
    == PUBLIC_DIR
)

assert (
    threshold_figure_path.parent
    == PUBLIC_DIR
)

assert (
    reference_figure_path.exists()
    and
    reference_figure_path
    .stat()
    .st_size > 0
)

assert (
    threshold_figure_path.exists()
    and
    threshold_figure_path
    .stat()
    .st_size > 0
)


print()
print("=" * 80)
print(
    "REFERENCE-TIME DIAGNOSTICS COMPLETE"
)
print("=" * 80)

print(
    "observed participant rows:",
    len(
        observed_reference_participant
    ),
)

print(
    "full-499 geometry rows:",
    len(
        full_bootstrap_geometry
    ),
)

print(
    "simulation sweep rows:",
    len(
        simulation_singularity_sweep
    ),
)

print(
    "public CSV files:",
    len(
        list(
            PUBLIC_DIR.glob(
                "*.csv"
            )
        )
    ),
)

print(
    "restricted CSV files:",
    len(
        list(
            RESTRICTED_DIR.glob(
                "*.csv"
            )
        )
    ),
)

print(
    "checkpoint exists:",
    CHECKPOINT_FILE.exists(),
)

print(
    "01 participant table restricted only:",
    (
        not (
            PUBLIC_DIR
            / "01_observed_reference_time_participant.csv"
        ).exists()
        and
        (
            RESTRICTED_DIR
            / "01_observed_reference_time_participant.csv"
        ).exists()
    ),
)

print(
    "two public figures:",
    (
        reference_figure_path.exists()
        and
        threshold_figure_path.exists()
    ),
)
