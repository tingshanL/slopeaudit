#!/usr/bin/env python3
"""Fit the six restricted-data AABC2 outcome models.

This public entry point reconstructs the frozen AABC2 analysis cohorts,
checks the fixed-effect design matrices, fits the prespecified mixed-model
optimizer panels and participant-clustered GEE comparators, and writes
aggregate model-fit and population-estimand tables.

It does not run boundary-aware bootstrap inference, participant-level
reliability calculations, reliability bootstraps, simulation studies, or
downstream two-stage diagnostics. Those belong to later reproduction steps.

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
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from tqdm.auto import tqdm

import aabc2_common as common


# ---------------------------------------------------------------------------
# Frozen numerical controls
# ---------------------------------------------------------------------------

ALPHA = 0.05

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
    help="Fresh directory for aggregate observed-fit outputs.",
)

args = parser.parse_args()


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


def predictor_omitted_formula(key):
    return common.predictor_omitted_formula(
        key,
        OUTCOMES,
    )


def original_frozen_formula(key):
    return common.original_frozen_formula(
        key,
        OUTCOMES,
    )


# ---------------------------------------------------------------------------
# Frozen model specifications
# ---------------------------------------------------------------------------

MODEL_SPECS = {
    "reference_unstructured_mid": {
        "re_formula": "1 + t5_mid",
        "vc_formula": None,
        "slope_time": "t5_mid",
    },
    "legacy_independent_baseline": {
        "re_formula": "1",
        "vc_formula": {
            "slope": "0 + t5",
        },
        "slope_time": "t5",
    },
    "origin_independent_mid": {
        "re_formula": "1",
        "vc_formula": {
            "slope": "0 + t5_mid",
        },
        "slope_time": "t5_mid",
    },
    "random_intercept_only": {
        "re_formula": "1",
        "vc_formula": None,
        "slope_time": None,
    },
    "original_frozen_structural": {
        "re_formula": "1",
        "vc_formula": {
            "slope": "0 + t5_person_centered",
        },
        "slope_time": "t5_person_centered",
    },
    "original_frozen_anchor": {
        "re_formula": "1",
        "vc_formula": {
            "slope": "0 + t5",
        },
        "slope_time": "t5",
    },
    "predictor_omitted_reference": {
        "re_formula": "1 + t5_mid",
        "vc_formula": None,
        "slope_time": "t5_mid",
    },
}


CORE_MODEL_SPECS = (
    "reference_unstructured_mid",
    "legacy_independent_baseline",
    "origin_independent_mid",
    "random_intercept_only",
)


ORIGINAL_SPEC = {
    key: (
        "original_frozen_structural"
        if OUTCOMES[key]["modality"] == "structural MRI"
        else "original_frozen_anchor"
    )
    for key in OUTCOMES
}


# ---------------------------------------------------------------------------
# Formula preflight
# ---------------------------------------------------------------------------

assert (
    "C" not in globals()
    or callable(globals()["C"])
), (
    "Global name C is non-callable and would shadow Patsy C(...)."
)


formula_preflight_rows = []

for key in OUTCOMES:
    for formula_role, formula in (
        (
            "common_reference",
            fixed_formula(key),
        ),
        (
            "predictor_omitted",
            predictor_omitted_formula(key),
        ),
        (
            "original_frozen",
            original_frozen_formula(key),
        ),
    ):
        ymat, xmat = patsy.dmatrices(
            formula,
            ANALYSIS_DATA[key],
            return_type="dataframe",
        )

        rank = int(
            np.linalg.matrix_rank(
                xmat.to_numpy()
            )
        )

        formula_preflight_rows.append({
            "outcome": key,
            "formula_role": formula_role,
            "rows": len(ymat),
            "columns": xmat.shape[1],
            "rank": rank,
            "full_rank": (
                rank == xmat.shape[1]
            ),
        })


formula_preflight = pd.DataFrame(
    formula_preflight_rows
)

if not formula_preflight["full_rank"].all():
    raise RuntimeError(
        "At least one fixed-effect design matrix is rank deficient:\n"
        + formula_preflight.loc[
            ~formula_preflight["full_rank"]
        ].to_string(index=False)
    )


# ---------------------------------------------------------------------------
# Mixed-model / GEE helpers
# ---------------------------------------------------------------------------

def hessian_diagnostic(result):
    try:
        hessian = result.model.hessian(
            result.params_object
        )

        if isinstance(hessian, tuple):
            hessian = hessian[0]

        information = -(
            np.asarray(hessian, float)
            + np.asarray(hessian, float).T
        ) / 2

        eigenvalues = np.linalg.eigvalsh(
            information
        )

        tolerance = (
            1e-8
            * max(
                1.0,
                np.max(
                    np.abs(eigenvalues)
                ),
            )
        )

        return (
            bool(
                np.min(eigenvalues)
                > tolerance
            ),
            float(
                np.min(eigenvalues)
            ),
            float(
                np.max(eigenvalues)
            ),
        )

    except Exception:
        return (
            False,
            np.nan,
            np.nan,
        )


def random_components(
    result,
    model_spec,
):
    residual = float(
        result.scale
    )

    if model_spec in (
        "reference_unstructured_mid",
        "predictor_omitted_reference",
    ):
        G = np.asarray(
            result.cov_re,
            float,
        )

    elif model_spec in (
        "legacy_independent_baseline",
        "origin_independent_mid",
        "original_frozen_structural",
        "original_frozen_anchor",
    ):
        G = np.diag([
            float(
                np.asarray(
                    result.cov_re
                )[0, 0]
            ),
            float(
                np.asarray(
                    result.vcomp
                )[0]
            ),
        ])

    elif model_spec == "random_intercept_only":
        G = np.array([[
            float(
                np.asarray(
                    result.cov_re
                )[0, 0]
            )
        ]])

    else:
        raise KeyError(
            model_spec
        )

    eigenvalues = np.linalg.eigvalsh(
        (G + G.T) / 2
    )

    tolerance = (
        PSD_REL_TOL
        * max(
            1.0,
            np.max(
                np.abs(eigenvalues)
            ),
            residual,
        )
    )

    psd = bool(
        np.isfinite(G).all()
        and residual > 0
        and np.min(eigenvalues)
        >= -tolerance
    )

    slope = (
        float(G[1, 1])
        if G.shape == (2, 2)
        else np.nan
    )

    boundary_tolerance = (
        SLOPE_BOUNDARY_REL_TOL
        * max(
            1.0,
            residual,
        )
    )

    boundary = bool(
        G.shape == (2, 2)
        and (
            not np.isfinite(slope)
            or slope
            <= boundary_tolerance
        )
    )

    return {
        "G": G,
        "residual_variance":
            residual,
        "random_effect_covariance_psd":
            psd,
        "slope_variance":
            slope,
        "random_slope_boundary":
            boundary,
        "random_slope_interior":
            bool(
                G.shape == (2, 2)
                and psd
                and not boundary
            ),
        "random_covariance_min_eigenvalue":
            float(
                np.min(eigenvalues)
            ),
    }


def build_mixed(
    data,
    formula,
    model_spec,
):
    specification = (
        MODEL_SPECS[
            model_spec
        ]
    )

    return smf.mixedlm(
        formula,
        data,
        groups=data["pid"],
        re_formula=specification[
            "re_formula"
        ],
        vc_formula=specification[
            "vc_formula"
        ],
    )


def fixed_finite(result):
    try:
        fixed_effects = np.asarray(
            result.fe_params,
            float,
        )

        standard_errors = np.asarray(
            result.bse_fe,
            float,
        )

        return bool(
            np.isfinite(
                fixed_effects
            ).all()
            and np.isfinite(
                standard_errors
            ).all()
            and (
                standard_errors >= 0
            ).all()
        )

    except Exception:
        return False


def fits_compatible(
    fit_a,
    fit_b,
):
    fixed_a = (
        fit_a["result"]
        .fe_params
    )

    fixed_b = (
        fit_b["result"]
        .fe_params
    )

    common_names = (
        fixed_a.index
        .intersection(
            fixed_b.index
        )
    )

    if len(common_names) == 0:
        return False

    scale = np.maximum(
        fit_a["result"]
        .bse_fe
        .loc[common_names]
        .to_numpy(float),
        fit_b["result"]
        .bse_fe
        .loc[common_names]
        .to_numpy(float),
    )

    z_max = np.max(
        np.abs(
            fixed_a
            .loc[common_names]
            .to_numpy(float)
            -
            fixed_b
            .loc[common_names]
            .to_numpy(float)
        )
        /
        np.maximum(
            scale,
            1e-8,
        )
    )

    ll_scale = max(
        1.0,
        abs(
            fit_a["llf"]
        ),
        abs(
            fit_b["llf"]
        ),
    )

    return bool(
        abs(
            fit_a["llf"]
            - fit_b["llf"]
        )
        <= (
            LLF_REL_TOL
            * ll_scale
        )
        and z_max
        <= FIXED_AGREEMENT_SE_TOL
    )


def fit_optimizer_panel(
    key,
    model_spec,
    formula_override=None,
):
    data = ANALYSIS_DATA[
        key
    ]

    formula = (
        fixed_formula(key)
        if formula_override is None
        else formula_override
    )

    _, xmat = patsy.dmatrices(
        formula,
        data,
        return_type="dataframe",
    )

    rank = int(
        np.linalg.matrix_rank(
            xmat.to_numpy()
        )
    )

    columns = int(
        xmat.shape[1]
    )

    attempts = []
    successful = []

    for optimizer in OPTIMIZERS:
        caught = []

        try:
            model = build_mixed(
                data,
                formula,
                model_spec,
            )

            with warnings.catch_warnings(
                record=True
            ) as warning_records:
                warnings.simplefilter(
                    "always"
                )

                result = model.fit(
                    reml=True,
                    method=optimizer,
                    maxiter=MAXITER,
                    disp=False,
                )

                caught = [
                    str(
                        warning.message
                    )
                    for warning
                    in warning_records
                ]

            components = (
                random_components(
                    result,
                    model_spec,
                )
            )

            (
                hessian_positive,
                hessian_min,
                hessian_max,
            ) = hessian_diagnostic(
                result
            )

            record = {
                "outcome":
                    key,
                "label":
                    OUTCOMES[key][
                        "label"
                    ],
                "model_specification":
                    model_spec,
                "optimizer":
                    optimizer,
                "optimizer_converged":
                    bool(
                        result.converged
                    ),
                "llf":
                    float(
                        result.llf
                    ),
                "fixed_design_rank":
                    rank,
                "fixed_design_columns":
                    columns,
                "fixed_design_full_rank":
                    rank == columns,
                "fixed_parameters_finite":
                    fixed_finite(
                        result
                    ),
                "full_hessian_positive_definite":
                    hessian_positive,
                "hessian_min_information_eigenvalue":
                    hessian_min,
                "hessian_max_information_eigenvalue":
                    hessian_max,
                "participants":
                    data[
                        "pid"
                    ].nunique(),
                "observations":
                    len(data),
                "warning_count":
                    len(caught),
                "warnings":
                    " | ".join(
                        sorted(
                            set(caught)
                        )
                    ),
                "error":
                    "",
                **{
                    component_key:
                        component_value
                    for (
                        component_key,
                        component_value
                    )
                    in components.items()
                    if component_key
                    != "G"
                },
            }

            attempts.append(
                record
            )

            if (
                record[
                    "optimizer_converged"
                ]
                and record[
                    "fixed_design_full_rank"
                ]
                and record[
                    "fixed_parameters_finite"
                ]
                and np.isfinite(
                    record["llf"]
                )
            ):
                successful.append({
                    "result":
                        result,
                    "record":
                        record,
                    "llf":
                        record["llf"],
                    "optimizer":
                        optimizer,
                })

        except Exception as exc:
            attempts.append({
                "outcome":
                    key,
                "label":
                    OUTCOMES[key][
                        "label"
                    ],
                "model_specification":
                    model_spec,
                "optimizer":
                    optimizer,
                "optimizer_converged":
                    False,
                "llf":
                    np.nan,
                "fixed_design_rank":
                    rank,
                "fixed_design_columns":
                    columns,
                "fixed_design_full_rank":
                    rank == columns,
                "fixed_parameters_finite":
                    False,
                "full_hessian_positive_definite":
                    False,
                "hessian_min_information_eigenvalue":
                    np.nan,
                "hessian_max_information_eigenvalue":
                    np.nan,
                "participants":
                    data[
                        "pid"
                    ].nunique(),
                "observations":
                    len(data),
                "warning_count":
                    len(caught),
                "warnings":
                    " | ".join(
                        sorted(
                            set(caught)
                        )
                    ),
                "error":
                    repr(exc),
                "random_effect_covariance_psd":
                    False,
                "slope_variance":
                    np.nan,
                "random_slope_boundary":
                    np.nan,
                "random_slope_interior":
                    np.nan,
                "random_covariance_min_eigenvalue":
                    np.nan,
                "residual_variance":
                    np.nan,
            })

    if not successful:
        raise RuntimeError(
            "No usable optimizer for "
            f"{key}, {model_spec}"
        )

    for fit in successful:
        fit[
            "agreement_count"
        ] = sum(
            fits_compatible(
                fit,
                other,
            )
            for other
            in successful
        )

    best_count = max(
        fit[
            "agreement_count"
        ]
        for fit
        in successful
    )

    pool = [
        fit
        for fit
        in successful
        if fit[
            "agreement_count"
        ]
        == best_count
    ]

    psd_pool = [
        fit
        for fit
        in pool
        if fit[
            "record"
        ][
            "random_effect_covariance_psd"
        ]
    ]

    selected = max(
        psd_pool
        or pool,
        key=lambda fit:
            fit["llf"],
    )

    optimizer_agreement = (
        best_count >= 2
    )

    fixed_effect_usable = bool(
        selected[
            "record"
        ][
            "optimizer_converged"
        ]
        and optimizer_agreement
        and selected[
            "record"
        ][
            "fixed_design_full_rank"
        ]
        and selected[
            "record"
        ][
            "fixed_parameters_finite"
        ]
    )

    audit = pd.DataFrame(
        attempts
    )

    audit["selected"] = (
        audit[
            "optimizer"
        ].eq(
            selected[
                "optimizer"
            ]
        )
    )

    audit[
        "optimizer_common_solution_count"
    ] = best_count

    audit[
        "optimizer_agreement"
    ] = optimizer_agreement

    audit[
        "fixed_effect_usable"
    ] = (
        audit[
            "selected"
        ]
        & fixed_effect_usable
    )

    return {
        "result":
            selected[
                "result"
            ],
        "data":
            data,
        "formula":
            formula,
        "model_spec":
            model_spec,
        "selected_optimizer":
            selected[
                "optimizer"
            ],
        "optimizer_agreement":
            optimizer_agreement,
        "fixed_effect_usable":
            fixed_effect_usable,
        "audit":
            audit,
        "components":
            random_components(
                selected[
                    "result"
                ],
                model_spec,
            ),
    }


def fit_gee(key):
    data = ANALYSIS_DATA[
        key
    ]

    formula = fixed_formula(
        key
    )

    result = smf.gee(
        formula,
        groups="pid",
        data=data,
        cov_struct=Exchangeable(),
        family=sm.families.Gaussian(),
    ).fit(
        maxiter=500,
        cov_type="robust",
    )

    return {
        "result":
            result,
        "data":
            data,
        "formula":
            formula,
        "fixed_effect_usable":
            bool(
                np.isfinite(
                    result.params
                ).all()
                and np.isfinite(
                    result.bse
                ).all()
            ),
    }


def wald_joint(
    result,
    terms,
):
    names = list(
        result.params.index
    )

    indices = [
        names.index(term)
        for term
        in terms
        if term in names
    ]

    if len(indices) != len(terms):
        return np.nan

    restriction = np.zeros(
        (
            len(indices),
            len(names),
        )
    )

    for row, column in enumerate(
        indices
    ):
        restriction[
            row,
            column,
        ] = 1

    return float(
        np.asarray(
            result.wald_test(
                restriction,
                scalar=True,
            ).pvalue
        ).squeeze()
    )


# ---------------------------------------------------------------------------
# Observed fits
# ---------------------------------------------------------------------------

FITS = {}
GEE_FITS = {}
fit_audits = []


common_tasks = [
    (
        key,
        model_spec,
        fixed_formula(key),
    )
    for key in OUTCOMES
    for model_spec
    in CORE_MODEL_SPECS
]

original_tasks = [
    (
        key,
        ORIGINAL_SPEC[key],
        original_frozen_formula(
            key
        ),
    )
    for key in OUTCOMES
]

omitted_tasks = [
    (
        key,
        "predictor_omitted_reference",
        predictor_omitted_formula(
            key
        ),
    )
    for key in OUTCOMES
]

all_tasks = (
    common_tasks
    + original_tasks
    + omitted_tasks
)


for (
    key,
    model_spec,
    formula,
) in tqdm(
    all_tasks,
    desc="Observed REML optimizer panels",
    unit="model",
):
    fit = fit_optimizer_panel(
        key,
        model_spec,
        formula_override=formula,
    )

    FITS[
        (
            key,
            model_spec,
        )
    ] = fit

    fit_audits.append(
        fit["audit"]
    )


for key in tqdm(
    OUTCOMES,
    desc="Participant-clustered robust GEE",
    unit="outcome",
):
    GEE_FITS[
        key
    ] = fit_gee(
        key
    )


observed_model_fit_audit = pd.concat(
    fit_audits,
    ignore_index=True,
)


# ---------------------------------------------------------------------------
# Population estimands
# ---------------------------------------------------------------------------

def fixed_covariance(
    result,
    fixed_only,
):
    if fixed_only:
        names = list(
            result.fe_params.index
        )

        covariance = (
            result.cov_params()
            .loc[
                names,
                names,
            ]
            .to_numpy(float)
        )

        return (
            covariance,
            names,
        )

    names = list(
        result.params.index
    )

    return (
        result.cov_params()
        .loc[
            names,
            names,
        ]
        .to_numpy(float),
        names,
    )


def effect_row(
    key,
    method,
    result,
    term,
    fixed_only,
):
    params = (
        result.fe_params
        if (
            hasattr(
                result,
                "fe_params",
            )
            and fixed_only
        )
        else result.params
    )

    bse = (
        result.bse_fe
        if (
            hasattr(
                result,
                "bse_fe",
            )
            and fixed_only
        )
        else result.bse
    )

    beta = float(
        params.get(
            term,
            np.nan,
        )
    )

    se = float(
        bse.get(
            term,
            np.nan,
        )
    )

    return {
        "outcome":
            key,
        "label":
            OUTCOMES[key][
                "label"
            ],
        "modality":
            OUTCOMES[key][
                "modality"
            ],
        "method":
            method,
        "term":
            term,
        "estimate":
            beta,
        "SE":
            se,
        "CI95_low":
            beta
            - 1.96 * se,
        "CI95_high":
            beta
            + 1.96 * se,
        "p_value":
            (
                2
                * norm.sf(
                    abs(
                        beta / se
                    )
                )
                if se > 0
                else np.nan
            ),
    }


def marginal_time_contrast(
    key,
    result,
    data,
    fixed_only,
    method,
):
    representatives = (
        data.groupby(
            "pid",
            as_index=False,
        )
        .head(1)
        .copy()
    )

    data_t0 = (
        representatives.copy()
    )

    data_t1 = (
        representatives.copy()
    )

    data_t0[
        "t5_mid"
    ] = 0.0

    data_t1[
        "t5_mid"
    ] = 1.0

    design_info = (
        result.model.data.design_info
    )

    x0 = np.asarray(
        patsy.build_design_matrices(
            [design_info],
            data_t0,
        )[0]
    )

    x1 = np.asarray(
        patsy.build_design_matrices(
            [design_info],
            data_t1,
        )[0]
    )

    contrast = (
        x1 - x0
    ).mean(
        axis=0
    )

    params = np.asarray(
        (
            result.fe_params
            if fixed_only
            else result.params
        ),
        float,
    )

    covariance, _ = (
        fixed_covariance(
            result,
            fixed_only,
        )
    )

    beta = float(
        contrast
        @ params[
            :len(contrast)
        ]
    )

    se = float(
        np.sqrt(
            max(
                0.0,
                contrast
                @ covariance[
                    :len(contrast),
                    :len(contrast),
                ]
                @ contrast,
            )
        )
    )

    return {
        "outcome":
            key,
        "label":
            OUTCOMES[key][
                "label"
            ],
        "modality":
            OUTCOMES[key][
                "modality"
            ],
        "method":
            method,
        "term":
            "participant_marginal_change_per_5y",
        "estimate":
            beta,
        "SE":
            se,
        "CI95_low":
            beta
            - 1.96 * se,
        "CI95_high":
            beta
            + 1.96 * se,
        "p_value":
            (
                2
                * norm.sf(
                    abs(
                        beta / se
                    )
                )
                if se > 0
                else np.nan
            ),
        "standardization":
            (
                "equal weight per participant over observed "
                "baseline covariate distribution"
            ),
    }


AGE_TERMS = (
    "t5_mid:age0_c",
    "t5_mid:age0_sq_c",
)


population_rows = []

for key in OUTCOMES:
    reference_fit = (
        FITS[
            (
                key,
                "reference_unstructured_mid",
            )
        ]
    )

    gee_fit = (
        GEE_FITS[
            key
        ]
    )

    for (
        method,
        result,
        fixed_only,
    ) in (
        (
            "one_stage_reference_LMM",
            reference_fit[
                "result"
            ],
            True,
        ),
        (
            "one_stage_participant_clustered_GEE",
            gee_fit[
                "result"
            ],
            False,
        ),
    ):
        for term in (
            "t5_mid",
            *AGE_TERMS,
        ):
            population_rows.append(
                effect_row(
                    key,
                    method,
                    result,
                    term,
                    fixed_only,
                )
            )

        population_rows.append({
            "outcome":
                key,
            "label":
                OUTCOMES[key][
                    "label"
                ],
            "modality":
                OUTCOMES[key][
                    "modality"
                ],
            "method":
                method,
            "term":
                "joint_age_and_age2_by_time",
            "estimate":
                np.nan,
            "SE":
                np.nan,
            "CI95_low":
                np.nan,
            "CI95_high":
                np.nan,
            "p_value":
                wald_joint(
                    result,
                    AGE_TERMS,
                ),
        })

        population_rows.append(
            marginal_time_contrast(
                key,
                result,
                reference_fit[
                    "data"
                ],
                fixed_only,
                method,
            )
        )


population_estimands_LMM_GEE = (
    pd.DataFrame(
        population_rows
    )
)


# ---------------------------------------------------------------------------
# Public aggregate exports
# ---------------------------------------------------------------------------

if len(
    observed_model_fit_audit
) != 108:
    raise RuntimeError(
        "Expected 108 optimizer-audit rows; "
        f"found {len(observed_model_fit_audit)}."
    )

if len(
    observed_model_fit_audit.columns
) != 28:
    raise RuntimeError(
        "Expected 28 base fit-audit columns; "
        f"found {len(observed_model_fit_audit.columns)}."
    )

if len(
    population_estimands_LMM_GEE
) != 60:
    raise RuntimeError(
        "Expected 60 population-estimand rows; "
        f"found {len(population_estimands_LMM_GEE)}."
    )


fit_path = (
    OUTPUT_DIR
    / "01_observed_model_fit_audit.csv"
)

population_path = (
    OUTPUT_DIR
    / "02_population_estimands_LMM_GEE.csv"
)


observed_model_fit_audit.to_csv(
    fit_path,
    index=False,
)

population_estimands_LMM_GEE.to_csv(
    population_path,
    index=False,
)


print("Restricted AABC2 observed models fitted successfully.")
print()
print(cohort_check.to_string(index=False))
print()
print(
    "FORMULA PREFLIGHT FULL RANK:",
    bool(
        formula_preflight[
            "full_rank"
        ].all()
    ),
)
print(
    "OBSERVED FIT-AUDIT ROWS:",
    len(
        observed_model_fit_audit
    ),
)
print(
    "OBSERVED FIT-AUDIT COLUMNS:",
    len(
        observed_model_fit_audit.columns
    ),
)
print(
    "POPULATION ESTIMAND ROWS:",
    len(
        population_estimands_LMM_GEE
    ),
)
print("PARTICIPANT-LEVEL TABLES EXPORTED: 0")
print(f"WROTE: {fit_path}")
print(f"WROTE: {population_path}")
