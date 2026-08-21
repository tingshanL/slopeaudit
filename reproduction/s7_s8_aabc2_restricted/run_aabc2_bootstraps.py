#!/usr/bin/env python3
"""Run the AABC2 reliability-gap bootstrap.

This public restricted-data entry point reproduces the frozen
participant-resampling bootstrap for global mean cortical thickness and
fluid cognition.

The bootstrap estimates the sampling distribution of median direct
schedule reliability, full posterior slope predictability, conditional
slope-innovation reliability, and their gaps.

This script deliberately does NOT run the historical boundary-RLRT
bootstrap. Final boundary-aware slope-variance inference is handled by
the separately validated advanced RLRT workflow.

No participant identifiers or participant-level real-data tables are
exported. Bootstrap replicate summaries contain aggregate metrics only.

Restricted source data are not distributed with this repository.
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
import warnings
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import patsy
import statsmodels.formula.api as smf
from tqdm.auto import tqdm

import aabc2_common as common


# ---------------------------------------------------------------------------
# Frozen numerical controls
# ---------------------------------------------------------------------------

SEED = 20260718

OPTIMIZERS = (
    "lbfgs",
    "powell",
    "bfgs",
)

MAXITER = 1200

SLOPE_BOUNDARY_REL_TOL = 1e-8
PSD_REL_TOL = 1e-8
LLF_REL_TOL = 1e-5
FIXED_AGREEMENT_SE_TOL = 0.25

RELIABILITY_GAP_BOOT_B = 499

BOOTSTRAP_OUTCOMES = (
    "global_mean_thickness",
    "fluid_cognition",
)

MODEL_SPECS = {
    "reference_unstructured_mid": {
        "re_formula": "1 + t5_mid",
        "vc_formula": None,
        "slope_time": "t5_mid",
    },
    "random_intercept_only": {
        "re_formula": "1",
        "vc_formula": None,
        "slope_time": None,
    },
}


# ---------------------------------------------------------------------------
# CLI / output policy
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input-config",
    required=True,
    help=(
        "Private JSON containing explicit paths to the restricted "
        "AABC2 source inputs."
    ),
)

parser.add_argument(
    "--output-dir",
    required=True,
    help="Fresh directory for aggregate bootstrap outputs.",
)

parser.add_argument(
    "--mode",
    choices=("smoke", "full"),
    default="smoke",
    help=(
        "smoke runs only the requested fixed replicates; "
        "full runs all 499 replicates for each of the two outcomes."
    ),
)

parser.add_argument(
    "--replicates",
    nargs="+",
    type=int,
    default=(1, 2, 3),
    help=(
        "Replicate numbers used in smoke mode. "
        "Ignored in full mode."
    ),
)

parser.add_argument(
    "--n-jobs",
    type=int,
    default=2,
    help="Parallel workers for full mode; capped at 3.",
)

args = parser.parse_args()

N_JOBS = max(
    1,
    min(
        int(args.n_jobs),
        3,
    ),
)

OUTPUT_DIR = (
    Path(args.output_dir)
    .expanduser()
    .resolve()
)

if OUTPUT_DIR.exists():
    if any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is non-empty: {OUTPUT_DIR}. "
            "Use a fresh directory."
        )
else:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


# ---------------------------------------------------------------------------
# Restricted-data preparation
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


@contextmanager
def tqdm_joblib(bar):
    """Route joblib batch completion through one tqdm progress bar."""

    old = joblib.parallel.BatchCompletionCallBack

    class Callback(old):
        def __call__(self, *args, **kwargs):
            bar.update(
                n=self.batch_size
            )
            return super().__call__(
                *args,
                **kwargs,
            )

    joblib.parallel.BatchCompletionCallBack = Callback

    try:
        yield bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        bar.close()




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

    if model_spec == "reference_unstructured_mid":
        G = np.asarray(result.cov_re, float)
    elif model_spec == "random_intercept_only":
        G = np.array([[float(np.asarray(result.cov_re)[0, 0])]])
    else:
        raise KeyError(model_spec)

    eig = np.linalg.eigvalsh((G + G.T) / 2)
    tol = PSD_REL_TOL * max(1.0, np.max(np.abs(eig)), residual)

    psd = bool(
        np.isfinite(G).all()
        and residual > 0
        and np.min(eig) >= -tol
    )

    slope = float(G[1, 1]) if G.shape == (2, 2) else np.nan

    boundary_tol = SLOPE_BOUNDARY_REL_TOL * max(1.0, residual)

    boundary = bool(
        G.shape == (2, 2)
        and (not np.isfinite(slope) or slope <= boundary_tol)
    )

    return {
        "G": G,
        "residual_variance": residual,
        "random_effect_covariance_psd": psd,
        "slope_variance": slope,
        "random_slope_boundary": boundary,
        "random_slope_interior": bool(
            G.shape == (2, 2) and psd and not boundary
        ),
        "random_covariance_min_eigenvalue": float(np.min(eig)),
    }


def build_mixed(d, formula, model_spec):
    s = MODEL_SPECS[model_spec]
    return smf.mixedlm(
        formula,
        d,
        groups=d["pid"],
        re_formula=s["re_formula"],
        vc_formula=s["vc_formula"],
    )


def fixed_finite(result):
    try:
        fe = np.asarray(result.fe_params, float)
        se = np.asarray(result.bse_fe, float)
        return bool(
            np.isfinite(fe).all()
            and np.isfinite(se).all()
            and (se >= 0).all()
        )
    except Exception:
        return False


def fits_compatible(a, b):
    fa = a["result"].fe_params
    fb = b["result"].fe_params
    common_names = fa.index.intersection(fb.index)

    if len(common_names) == 0:
        return False

    scale = np.maximum(
        a["result"].bse_fe.loc[common_names].to_numpy(float),
        b["result"].bse_fe.loc[common_names].to_numpy(float),
    )

    z = np.max(
        np.abs(
            fa.loc[common_names].to_numpy(float)
            - fb.loc[common_names].to_numpy(float)
        )
        / np.maximum(scale, 1e-8)
    )

    ll_scale = max(1.0, abs(a["llf"]), abs(b["llf"]))

    return bool(
        abs(a["llf"] - b["llf"]) <= LLF_REL_TOL * ll_scale
        and z <= FIXED_AGREEMENT_SE_TOL
    )


def fit_optimizer_panel(key, model_spec):
    d = ANALYSIS_DATA[key]
    formula = fixed_formula(key)

    _, xmat = patsy.dmatrices(
        formula,
        d,
        return_type="dataframe",
    )

    rank = int(np.linalg.matrix_rank(xmat.to_numpy()))
    columns = int(xmat.shape[1])

    attempts = []
    successful = []

    for optimizer in OPTIMIZERS:
        caught = []

        try:
            model = build_mixed(d, formula, model_spec)

            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                result = model.fit(
                    reml=True,
                    method=optimizer,
                    maxiter=MAXITER,
                    disp=False,
                )
                caught = [str(w.message) for w in ws]

            comp = random_components(result, model_spec)
            hpd, hmin, hmax = hessian_diagnostic(result)

            rec = {
                "optimizer": optimizer,
                "optimizer_converged": bool(result.converged),
                "llf": float(result.llf),
                "fixed_design_full_rank": rank == columns,
                "fixed_parameters_finite": fixed_finite(result),
                "full_hessian_positive_definite": hpd,
                "hessian_min_information_eigenvalue": hmin,
                "hessian_max_information_eigenvalue": hmax,
                "warnings": " | ".join(sorted(set(caught))),
                **{k: v for k, v in comp.items() if k != "G"},
            }

            attempts.append(rec)

            if (
                rec["optimizer_converged"]
                and rec["fixed_design_full_rank"]
                and rec["fixed_parameters_finite"]
                and np.isfinite(rec["llf"])
            ):
                successful.append({
                    "result": result,
                    "record": rec,
                    "llf": rec["llf"],
                    "optimizer": optimizer,
                })

        except Exception as exc:
            attempts.append({
                "optimizer": optimizer,
                "error": repr(exc),
            })

    if not successful:
        raise RuntimeError(
            f"No usable optimizer for {key}, {model_spec}"
        )

    for a in successful:
        a["agreement_count"] = sum(
            fits_compatible(a, b)
            for b in successful
        )

    best_count = max(
        a["agreement_count"]
        for a in successful
    )

    pool = [
        a for a in successful
        if a["agreement_count"] == best_count
    ]

    psd_pool = [
        a for a in pool
        if a["record"]["random_effect_covariance_psd"]
    ]

    selected = max(
        psd_pool or pool,
        key=lambda a: a["llf"],
    )

    return {
        "result": selected["result"],
        "data": d,
        "formula": formula,
        "selected_optimizer": selected["optimizer"],
        "components": random_components(
            selected["result"],
            model_spec,
        ),
    }


def fit_quick(d, formula, model_spec, optimizer_order):
    last = None

    for optimizer in optimizer_order:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = build_mixed(
                    d,
                    formula,
                    model_spec,
                ).fit(
                    reml=True,
                    method=optimizer,
                    maxiter=MAXITER,
                    disp=False,
                )

            comp = random_components(result, model_spec)

            if (
                result.converged
                and np.isfinite(result.llf)
                and fixed_finite(result)
                and comp["random_effect_covariance_psd"]
            ):
                return result, optimizer, ""

            last = "nonconverged/nonfinite/non-PSD"

        except Exception as exc:
            last = repr(exc)

    return None, "", last or "all optimizers failed"


def clamp_rounding_only(x, tol=1e-7):
    if not np.isfinite(x):
        return np.nan
    if -tol <= x <= 1 + tol:
        return float(np.clip(x, 0, 1))
    return np.nan


def direct_grr(slope_var, residual_var, information):
    if not (
        np.isfinite(slope_var)
        and slope_var > 0
        and np.isfinite(residual_var)
        and residual_var > 0
        and information > 0
    ):
        return np.nan

    return float(
        slope_var
        / (
            slope_var
            + residual_var / information
        )
    )


def bootstrap_reference_metrics(key, replicate, seed):
    rng = np.random.default_rng(seed)

    source = ANALYSIS_DATA[key]
    ids = source.pid.unique()

    sampled = rng.choice(
        ids,
        size=len(ids),
        replace=True,
    )

    pieces = []

    for draw, original_id in enumerate(sampled):
        part = source.loc[
            source.pid.eq(original_id)
        ].copy()

        part["pid"] = f"boot_{draw:05d}"
        pieces.append(part)

    data = pd.concat(
        pieces,
        ignore_index=True,
    )

    observed = FITS[
        (key, "reference_unstructured_mid")
    ]

    order = tuple(dict.fromkeys([
        observed["selected_optimizer"],
        *OPTIMIZERS,
    ]))

    fit, optimizer, error = fit_quick(
        data,
        fixed_formula(key),
        "reference_unstructured_mid",
        order,
    )

    if fit is None:
        return {
            "outcome": key,
            "replicate": replicate,
            "valid": False,
            "error": error,
        }

    random_cov = np.asarray(
        fit.cov_re,
        float,
    )

    residual = float(fit.scale)

    if (
        np.min(np.linalg.eigvalsh(random_cov)) <= 0
        or random_cov[1, 1] <= 0
    ):
        return {
            "outcome": key,
            "replicate": replicate,
            "valid": False,
            "error": "noninterior random covariance",
        }

    inverse_random = np.linalg.inv(random_cov)

    a = (
        random_cov[0, 1]
        / random_cov[0, 0]
    )

    innovation_prior = (
        random_cov[1, 1]
        - random_cov[0, 1] ** 2
        / random_cov[0, 0]
    )

    direct_values = []
    full_values = []
    innovation_values = []

    for _, group in data.groupby("pid"):
        information = float(
            np.sum(
                (
                    group.t5_mid
                    - group.t5_mid.mean()
                ) ** 2
            )
        )

        direct_values.append(
            direct_grr(
                random_cov[1, 1],
                residual,
                information,
            )
        )

        design = np.column_stack([
            np.ones(len(group)),
            group.t5_mid.to_numpy(float),
        ])

        posterior_cov = np.linalg.inv(
            inverse_random
            + design.T @ design / residual
        )

        full_values.append(
            clamp_rounding_only(
                1
                - posterior_cov[1, 1]
                / random_cov[1, 1]
            )
        )

        transform = np.array([-a, 1.0])

        innovation_values.append(
            clamp_rounding_only(
                1
                - float(
                    transform
                    @ posterior_cov
                    @ transform
                )
                / innovation_prior
            )
        )

    direct_median = float(
        np.nanmedian(direct_values)
    )
    full_median = float(
        np.nanmedian(full_values)
    )
    innovation_median = float(
        np.nanmedian(innovation_values)
    )

    return {
        "outcome": key,
        "replicate": replicate,
        "valid": True,
        "optimizer": optimizer,
        "median_direct_GRR": direct_median,
        "median_full_predictability": full_median,
        "median_innovation_reliability": innovation_median,
        "median_full_minus_direct_gap":
            full_median - direct_median,
        "median_full_minus_innovation_gap":
            full_median - innovation_median,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Minimal observed reference fits required by the bootstrap
# ---------------------------------------------------------------------------

FITS = {}

for key in BOOTSTRAP_OUTCOMES:
    fit = fit_optimizer_panel(
        key,
        "reference_unstructured_mid",
    )

    FITS[
        (
            key,
            "reference_unstructured_mid",
        )
    ] = fit

    print(
        "Observed reference fit:",
        key,
        "optimizer=",
        fit["selected_optimizer"],
        "llf=",
        float(
            fit["result"].llf
        ),
    )


# ---------------------------------------------------------------------------
# Deterministic bootstrap task construction
# ---------------------------------------------------------------------------

if args.mode == "full":
    replicates = tuple(
        range(
            1,
            RELIABILITY_GAP_BOOT_B + 1,
        )
    )
else:
    replicates = tuple(
        int(rep)
        for rep in args.replicates
    )

    if not replicates:
        raise ValueError(
            "Smoke mode requires at least one replicate."
        )

    invalid = [
        rep for rep in replicates
        if rep < 1
        or rep > RELIABILITY_GAP_BOOT_B
    ]

    if invalid:
        raise ValueError(
            "Smoke replicate numbers must be between 1 and 499: "
            + ", ".join(
                map(str, invalid)
            )
        )

    if len(set(replicates)) != len(replicates):
        raise ValueError(
            "Smoke replicate numbers must be unique."
        )


tasks = []

for outcome_index, key in enumerate(
    BOOTSTRAP_OUTCOMES
):
    for replicate in replicates:
        seed = (
            SEED
            + 8_000_000
            + 100_000 * outcome_index
            + replicate
        )

        tasks.append(
            (
                key,
                replicate,
                seed,
            )
        )


if args.mode == "full":
    with tqdm_joblib(
        tqdm(
            total=len(tasks),
            desc=(
                "Thickness/cognition "
                "reliability-gap bootstrap"
            ),
            unit="rep",
        )
    ):
        rows = Parallel(
            n_jobs=N_JOBS,
            backend="loky",
            batch_size=1,
            pre_dispatch=N_JOBS,
        )(
            delayed(
                bootstrap_reference_metrics
            )(
                *task
            )
            for task in tasks
        )

else:
    rows = []

    for task in tqdm(
        tasks,
        desc="Reliability-gap smoke",
        unit="rep",
    ):
        rows.append(
            bootstrap_reference_metrics(
                *task
            )
        )


bootstrap = pd.DataFrame(
    rows
)


# ---------------------------------------------------------------------------
# Frozen output contract
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "outcome",
    "replicate",
    "valid",
    "optimizer",
    "median_direct_GRR",
    "median_full_predictability",
    "median_innovation_reliability",
    "median_full_minus_direct_gap",
    "median_full_minus_innovation_gap",
    "error",
]

for column in EXPECTED_COLUMNS:
    if column not in bootstrap.columns:
        bootstrap[column] = np.nan

bootstrap = bootstrap[
    EXPECTED_COLUMNS
]


if args.mode == "full":
    if len(bootstrap) != 998:
        raise RuntimeError(
            "Expected 998 full bootstrap rows; "
            f"found {len(bootstrap)}."
        )

    counts = (
        bootstrap.groupby(
            "outcome"
        )[
            "replicate"
        ]
        .nunique()
    )

    expected_counts = pd.Series(
        {
            "global_mean_thickness": 499,
            "fluid_cognition": 499,
        }
    )

    counts = counts.reindex(
        expected_counts.index
    )

    if not counts.equals(
        expected_counts
    ):
        raise RuntimeError(
            "Unexpected full bootstrap replicate counts:\n"
            + counts.to_string()
        )

    invalid = bootstrap.loc[
        ~bootstrap["valid"].fillna(False),
        [
            "outcome",
            "replicate",
            "error",
        ],
    ]

    if not invalid.empty:
        raise RuntimeError(
            "Full bootstrap reproduction contains invalid replicates; "
            "it therefore does not reproduce the frozen 998-valid-row "
            "analysis state:\n"
            + invalid.to_string(index=False)
        )


if args.mode == "full":
    output_path = (
        OUTPUT_DIR
        / "09_reliability_gap_bootstrap_replicates.csv"
    )
else:
    output_path = (
        OUTPUT_DIR
        / "09_reliability_gap_bootstrap_smoke.csv"
    )


bootstrap.to_csv(
    output_path,
    index=False,
)


print()
print("=== RELIABILITY-GAP BOOTSTRAP SUMMARY ===")
print("mode:", args.mode)
print("rows:", len(bootstrap))
print(
    "valid rows:",
    int(
        bootstrap[
            "valid"
        ].fillna(False).sum()
    ),
)

print()
print(
    bootstrap.groupby(
        "outcome"
    ).agg(
        rows=(
            "replicate",
            "size",
        ),
        valid=(
            "valid",
            "sum",
        ),
        min_replicate=(
            "replicate",
            "min",
        ),
        max_replicate=(
            "replicate",
            "max",
        ),
    ).to_string()
)

print()
print("PARTICIPANT-LEVEL TABLES EXPORTED: 0")
print(f"WROTE: {output_path}")
