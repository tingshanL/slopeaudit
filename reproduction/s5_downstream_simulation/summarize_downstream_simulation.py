#!/usr/bin/env python3
"""Deterministically summarize raw S5 downstream-simulation results.

This script performs no simulation generation and no mixed-model fitting.
It consumes explicitly supplied raw generator outputs and reproduces the
frozen S5 calibration, method-performance, singularity, sensitivity, and
descriptive-summary calculations.

For the empirical AABC-derived study, input/output results remain HOLD pending
written AABC permission. This reviewer-facing script itself contains no
participant-level AABC data or identifiers.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


ALPHA = 0.05

HIGH_FALSE_FULL_THRESHOLD = 0.40
PRIMARY_NEAR_SINGULAR_ABS_RHO = 0.95
PRIMARY_SMALL_NORMALIZED_DETERMINANT = 0.10
PRIMARY_SMALL_EIGEN_FRACTION = 1e-4

FULL_REPS_PER_CELL = 500

REGIME_TO_OUTCOME = {
    "high_slope_information": "lateral_ventricles",
    "intermediate": "hippocampus",
    "near_boundary": "global_cortical_att",
    "true_zero_slope_variance": "fluid_cognition",
}

METHODS = [
    "one_stage_LMM",
    "one_stage_GEE",
    "uncertainty_aware_two_stage",
    "BLUP_as_outcome_attenuation",
    "predictor_double_use_overconfidence",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--anchor-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--task-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--simulation-replicates-raw",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--model-fit-diagnostics",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--optimizer-attempts",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help=(
            "full enforces the frozen 8,000-task / "
            "500-replicate-per-cell design; smoke validates "
            "the explicitly supplied smaller task manifest."
        ),
    )

    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} does not exist or is not a file."
        )

    return path


def load_outcome_map(anchor_manifest_path: Path):
    path = require_file(
        anchor_manifest_path,
        "source-anchor manifest",
    )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    regimes = payload.get("regimes")

    if not isinstance(regimes, dict):
        raise ValueError(
            "Anchor manifest lacks top-level 'regimes'."
        )

    outcome_map = {}

    for regime, expected_outcome in REGIME_TO_OUTCOME.items():
        if regime not in regimes:
            raise ValueError(
                f"Anchor manifest lacks regime {regime!r}."
            )

        outcome = regimes[regime].get("outcome")

        if outcome != expected_outcome:
            raise ValueError(
                f"{regime}: expected outcome "
                f"{expected_outcome!r}, got {outcome!r}."
            )

        outcome_map[regime] = outcome

    return outcome_map


def validate_raw_inputs(
    task_manifest,
    simulation_replicates,
    model_fit_diagnostics,
    optimizer_attempts,
    mode,
):
    task_required = {
        "row_type",
        "regime",
        "effect_condition",
        "replicate",
        "seed",
        "task_id",
    }

    if not task_required.issubset(task_manifest.columns):
        raise ValueError(
            "Task manifest lacks required columns."
        )

    if task_manifest["task_id"].duplicated().any():
        raise ValueError(
            "Task manifest contains duplicate task IDs."
        )

    if simulation_replicates["task_id"].duplicated().any():
        raise ValueError(
            "Raw simulation results contain duplicate task IDs."
        )

    expected_ids = set(
        task_manifest["task_id"].astype(str)
    )
    result_ids = set(
        simulation_replicates["task_id"].astype(str)
    )

    if expected_ids != result_ids:
        raise ValueError(
            "Task IDs in raw simulation results differ "
            "from the task manifest."
        )

    diagnostic_keys = [
        "task_id",
        "model_role",
    ]

    if model_fit_diagnostics.duplicated(
        diagnostic_keys
    ).any():
        raise ValueError(
            "Model-fit diagnostics contain duplicate "
            "(task_id, model_role) rows."
        )

    attempt_keys = [
        "task_id",
        "model_role",
        "optimizer",
    ]

    if optimizer_attempts.duplicated(
        attempt_keys
    ).any():
        raise ValueError(
            "Optimizer audit contains duplicate "
            "(task_id, model_role, optimizer) rows."
        )

    for frame, name in [
        (
            model_fit_diagnostics,
            "model-fit diagnostics",
        ),
        (
            optimizer_attempts,
            "optimizer attempts",
        ),
    ]:
        unknown = (
            set(frame["task_id"].astype(str))
            - expected_ids
        )

        if unknown:
            raise ValueError(
                f"{name} contain task IDs absent from "
                "the task manifest."
            )

    cell_counts = (
        task_manifest.groupby(
            [
                "row_type",
                "regime",
                "effect_condition",
            ],
            sort=False,
        )["replicate"]
        .nunique()
    )

    if mode == "full":
        if len(task_manifest) != 8000:
            raise ValueError(
                "Full mode requires exactly 8,000 tasks."
            )

        if len(cell_counts) != 16:
            raise ValueError(
                "Full mode requires exactly 16 cells."
            )

        if not cell_counts.eq(
            FULL_REPS_PER_CELL
        ).all():
            raise ValueError(
                "Full mode requires exactly 500 "
                "replicates in every cell."
            )

        if set(task_manifest["regime"]) != set(
            REGIME_TO_OUTCOME
        ):
            raise ValueError(
                "Full mode requires all four frozen regimes."
            )

    return cell_counts


def concentration_test(
    data,
    near_column,
    high_column="false_high_full_predictability",
):
    data = data.loc[
        data["finite_full_predictability"]
        & data[near_column].notna()
    ].copy()

    table = (
        pd.crosstab(
            data[high_column],
            data[near_column],
        )
        .reindex(
            index=[False, True],
            columns=[False, True],
            fill_value=0,
        )
    )

    a = float(table.loc[True, True])
    b = float(table.loc[True, False])
    c = float(table.loc[False, True])
    d = float(table.loc[False, False])

    odds_ratio, fisher_p = stats.fisher_exact(
        [
            [a, b],
            [c, d],
        ]
    )

    # Frozen Haldane-Anscombe correction.
    ac = a + 0.5
    bc = b + 0.5
    cc = c + 0.5
    dc = d + 0.5

    risk_near = ac / (ac + cc)
    risk_not = bc / (bc + dc)
    risk_ratio = risk_near / risk_not

    log_rr_se = np.sqrt(
        1 / ac
        - 1 / (ac + cc)
        + 1 / bc
        - 1 / (bc + dc)
    )

    return {
        "near_definition":
            near_column,
        "finite_replicates":
            len(data),
        "high_and_near":
            int(a),
        "high_and_not_near":
            int(b),
        "not_high_and_near":
            int(c),
        "not_high_and_not_near":
            int(d),
        "P_near_given_high":
            (
                a / (a + b)
                if a + b
                else np.nan
            ),
        "P_near_given_not_high":
            (
                c / (c + d)
                if c + d
                else np.nan
            ),
        "risk_high_if_near":
            (
                a / (a + c)
                if a + c
                else np.nan
            ),
        "risk_high_if_not_near":
            (
                b / (b + d)
                if b + d
                else np.nan
            ),
        "odds_ratio":
            odds_ratio,
        "fisher_exact_p":
            fisher_p,
        "corrected_risk_ratio":
            risk_ratio,
        "corrected_risk_ratio_CI95_low":
            np.exp(
                np.log(risk_ratio)
                - 1.96 * log_rr_se
            ),
        "corrected_risk_ratio_CI95_high":
            np.exp(
                np.log(risk_ratio)
                + 1.96 * log_rr_se
            ),
    }


def summarize(
    task_manifest,
    simulation_replicates,
    model_fit_diagnostics,
    outcome_map,
):
    # ------------------------------------------------------------
    # Frozen Section 13 — schedule-specific LR calibration
    # ------------------------------------------------------------

    calibration = simulation_replicates.loc[
        simulation_replicates[
            "row_type"
        ].eq("lr_calibration")
        & simulation_replicates[
            "task_valid"
        ]
    ]

    LR_critical = (
        calibration.groupby("regime")[
            "restricted_LR"
        ]
        .quantile(0.95)
        .rename(
            "schedule_specific_LR_critical_95"
        )
    )

    simulation_analysis = (
        simulation_replicates.drop(
            columns=[
                "schedule_specific_LR_critical_95",
                "LR_reject_schedule_specific",
            ],
            errors="ignore",
        )
        .merge(
            LR_critical,
            on="regime",
            how="left",
        )
    )

    simulation_analysis[
        "LR_reject_schedule_specific"
    ] = (
        simulation_analysis[
            "restricted_LR"
        ]
        > simulation_analysis[
            "schedule_specific_LR_critical_95"
        ]
    )

    lr_rows = []

    for regime in REGIME_TO_OUTCOME:
        calibration_all = (
            simulation_analysis.loc[
                simulation_analysis[
                    "row_type"
                ].eq("lr_calibration")
                & simulation_analysis[
                    "regime"
                ].eq(regime)
            ]
        )

        null_all = (
            simulation_analysis.loc[
                simulation_analysis[
                    "row_type"
                ].eq("lr_null_evaluation")
                & simulation_analysis[
                    "regime"
                ].eq(regime)
            ]
        )

        null_valid = null_all.loc[
            null_all["task_valid"]
        ]

        lr_rows.append(
            {
                "regime":
                    regime,
                "source_outcome":
                    outcome_map[regime],
                "critical_95":
                    float(
                        LR_critical.get(
                            regime,
                            np.nan,
                        )
                    ),
                "calibration_requested":
                    len(calibration_all),
                "calibration_valid":
                    int(
                        calibration_all[
                            "task_valid"
                        ].sum()
                    ),
                "calibration_validity_rate":
                    float(
                        calibration_all[
                            "task_valid"
                        ].mean()
                    ),
                "null_evaluation_requested":
                    len(null_all),
                "null_evaluation_valid":
                    int(
                        null_all[
                            "task_valid"
                        ].sum()
                    ),
                "null_evaluation_validity_rate":
                    float(
                        null_all[
                            "task_valid"
                        ].mean()
                    ),
                "independent_null_rejection_rate":
                    (
                        float(
                            null_valid[
                                "LR_reject_schedule_specific"
                            ].mean()
                        )
                        if len(null_valid)
                        else np.nan
                    ),
            }
        )

    lr_calibration_summary = pd.DataFrame(
        lr_rows
    )

    # ------------------------------------------------------------
    # Frozen Section 14 — downstream-method performance
    # ------------------------------------------------------------

    evaluation = simulation_analysis.loc[
        simulation_analysis[
            "row_type"
        ].eq("evaluation")
    ].copy()

    method_rows = []

    for (
        regime,
        effect_condition,
    ), group in evaluation.groupby(
        [
            "regime",
            "effect_condition",
        ]
    ):
        truth = float(
            group["true_beta_xt"].iloc[0]
        )

        for method in METHODS:
            valid = (
                group[
                    f"{method}_valid"
                ]
                .astype("boolean")
                .fillna(False)
                .astype(bool)
            )

            subset = group.loc[valid]

            estimates = subset[
                f"{method}_beta"
            ]
            errors = subset[
                f"{method}_SE"
            ]
            pvalues = subset[
                f"{method}_p"
            ]

            method_rows.append(
                {
                    "regime":
                        regime,
                    "source_outcome":
                        outcome_map[regime],
                    "effect_condition":
                        effect_condition,
                    "method":
                        method,
                    "requested_replicates":
                        len(group),
                    "valid_replicates":
                        len(subset),
                    "validity_rate":
                        len(subset) / len(group),
                    "true_effect":
                        truth,
                    "bias":
                        (
                            float(
                                (
                                    estimates
                                    - truth
                                ).mean()
                            )
                            if len(subset)
                            else np.nan
                        ),
                    "RMSE":
                        (
                            float(
                                np.sqrt(
                                    np.mean(
                                        (
                                            estimates
                                            - truth
                                        )
                                        ** 2
                                    )
                                )
                            )
                            if len(subset)
                            else np.nan
                        ),
                    "coverage":
                        (
                            float(
                                (
                                    (
                                        estimates
                                        - 1.96
                                        * errors
                                        <= truth
                                    )
                                    & (
                                        truth
                                        <= estimates
                                        + 1.96
                                        * errors
                                    )
                                ).mean()
                            )
                            if len(subset)
                            else np.nan
                        ),
                    "null_false_positive_rate":
                        (
                            float(
                                (
                                    pvalues
                                    < ALPHA
                                ).mean()
                            )
                            if (
                                len(subset)
                                and truth == 0
                            )
                            else np.nan
                        ),
                    "power":
                        (
                            float(
                                (
                                    pvalues
                                    < ALPHA
                                ).mean()
                            )
                            if (
                                len(subset)
                                and truth != 0
                            )
                            else np.nan
                        ),
                    "mean_estimate":
                        (
                            float(
                                estimates.mean()
                            )
                            if len(subset)
                            else np.nan
                        ),
                    "mean_SE":
                        (
                            float(
                                errors.mean()
                            )
                            if len(subset)
                            else np.nan
                        ),
                }
            )

    simulation_method_summary = pd.DataFrame(
        method_rows
    )

    # ------------------------------------------------------------
    # Frozen Section 15 — true-zero primary diagnostic subset
    # ------------------------------------------------------------

    true_zero_primary = (
        model_fit_diagnostics.loc[
            model_fit_diagnostics[
                "model_role"
            ].eq("random_slope_primary")
            & model_fit_diagnostics[
                "true_slope_variance"
            ].eq(0)
        ]
        .copy()
    )

    true_zero_primary[
        "finite_full_predictability"
    ] = np.isfinite(
        true_zero_primary[
            "mean_full_posterior_predictability"
        ]
    )

    true_zero_primary[
        "false_high_full_predictability"
    ] = (
        true_zero_primary[
            "mean_full_posterior_predictability"
        ]
        >= HIGH_FALSE_FULL_THRESHOLD
    )

    true_zero_primary[
        "near_singular_primary"
    ] = (
        true_zero_primary[
            "absolute_intercept_slope_correlation"
        ]
        >= PRIMARY_NEAR_SINGULAR_ABS_RHO
    )

    true_zero_primary[
        "near_singular_any_definition"
    ] = (
        true_zero_primary[
            "near_singular_primary"
        ]
        | (
            true_zero_primary[
                "normalized_determinant"
            ]
            <= PRIMARY_SMALL_NORMALIZED_DETERMINANT
        )
        | (
            true_zero_primary[
                "minimum_eigenvalue_fraction_of_trace"
            ]
            <= PRIMARY_SMALL_EIGEN_FRACTION
        )
    )

    true_zero_denominator_audit = (
        true_zero_primary.groupby(
            [
                "row_type",
                "regime",
                "effect_condition",
            ]
        )
        .agg(
            fitted_replicates=(
                "replicate",
                "size",
            ),
            basic_validity_rate=(
                "basic_numerical_validity",
                "mean",
            ),
            strict_validity_rate=(
                "strict_numerical_validity",
                "mean",
            ),
            finite_full_rate=(
                "finite_full_predictability",
                "mean",
            ),
            slope_boundary_rate=(
                "slope_variance_boundary",
                "mean",
            ),
            covariance_boundary_rate=(
                "covariance_boundary",
                "mean",
            ),
            hessian_positive_rate=(
                "hessian_positive_definite",
                "mean",
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # Frozen Section 16 — concentration + logistic sensitivity
    # ------------------------------------------------------------

    primary_concentration_tests = pd.DataFrame(
        [
            concentration_test(
                true_zero_primary,
                "near_singular_primary",
            ),
            concentration_test(
                true_zero_primary,
                "near_singular_any_definition",
            ),
        ]
    )

    adjusted_logistic_result = pd.DataFrame(
        [
            {
                "status":
                    "not estimable"
            }
        ]
    )

    logistic_data = true_zero_primary.loc[
        true_zero_primary[
            "finite_full_predictability"
        ]
    ].copy()

    try:
        logistic_data[
            "near_singular_indicator"
        ] = logistic_data[
            "near_singular_primary"
        ].astype(int)

        logistic_data[
            "false_high_indicator"
        ] = logistic_data[
            "false_high_full_predictability"
        ].astype(int)

        if (
            logistic_data[
                "false_high_indicator"
            ].nunique()
            < 2
        ):
            raise ValueError(
                "The false-high-predictability "
                "outcome has no variation."
            )

        if (
            logistic_data[
                "near_singular_indicator"
            ].nunique()
            < 2
        ):
            raise ValueError(
                "The near-singular indicator "
                "has no variation."
            )

        logistic_warnings = []

        with warnings.catch_warnings(
            record=True
        ) as warning_records:
            warnings.simplefilter("always")

            logistic = smf.glm(
                "false_high_indicator ~ "
                "near_singular_indicator + "
                "C(regime) + C(row_type) + "
                "C(effect_condition)",
                data=logistic_data,
                family=sm.families.Binomial(),
            ).fit(
                cov_type="HC3"
            )

            logistic_warnings = [
                str(item.message)
                for item in warning_records
            ]

        term = "near_singular_indicator"

        adjusted_logistic_result = pd.DataFrame(
            [
                {
                    "status":
                        (
                            "estimated_with_warning"
                            if logistic_warnings
                            else "estimated"
                        ),
                    "term":
                        term,
                    "log_odds":
                        float(
                            logistic.params[
                                term
                            ]
                        ),
                    "SE":
                        float(
                            logistic.bse[
                                term
                            ]
                        ),
                    "adjusted_odds_ratio":
                        float(
                            np.exp(
                                logistic.params[
                                    term
                                ]
                            )
                        ),
                    "OR_CI95_low":
                        float(
                            np.exp(
                                logistic.params[
                                    term
                                ]
                                - 1.96
                                * logistic.bse[
                                    term
                                ]
                            )
                        ),
                    "OR_CI95_high":
                        float(
                            np.exp(
                                logistic.params[
                                    term
                                ]
                                + 1.96
                                * logistic.bse[
                                    term
                                ]
                            )
                        ),
                    "p_value":
                        float(
                            logistic.pvalues[
                                term
                            ]
                        ),
                    "n":
                        len(
                            logistic_data
                        ),
                    "warnings":
                        " | ".join(
                            sorted(
                                set(
                                    logistic_warnings
                                )
                            )
                        ),
                }
            ]
        )

    except Exception as exc:
        adjusted_logistic_result = (
            pd.DataFrame(
                [
                    {
                        "status":
                            "not estimable",
                        "error":
                            repr(exc),
                    }
                ]
            )
        )

    # ------------------------------------------------------------
    # Frozen Section 17 — threshold + continuous sensitivity
    # ------------------------------------------------------------

    threshold_rows = []

    finite = true_zero_primary.loc[
        true_zero_primary[
            "finite_full_predictability"
        ]
    ].copy()

    for rho_threshold in [
        0.90,
        0.95,
        0.99,
    ]:
        near = (
            finite[
                "absolute_intercept_slope_correlation"
            ]
            >= rho_threshold
        )

        for full_threshold in [
            0.25,
            0.40,
            0.75,
        ]:
            high = (
                finite[
                    "mean_full_posterior_predictability"
                ]
                >= full_threshold
            )

            table = (
                pd.crosstab(
                    high,
                    near,
                )
                .reindex(
                    index=[
                        False,
                        True,
                    ],
                    columns=[
                        False,
                        True,
                    ],
                    fill_value=0,
                )
            )

            odds_ratio, p_value = (
                stats.fisher_exact(
                    table.to_numpy()
                )
            )

            threshold_rows.append(
                {
                    "abs_rho_threshold":
                        rho_threshold,
                    "full_predictability_threshold":
                        full_threshold,
                    "n":
                        len(finite),
                    "high_full_n":
                        int(
                            high.sum()
                        ),
                    "near_singular_n":
                        int(
                            near.sum()
                        ),
                    "P_near_given_high":
                        (
                            float(
                                near.loc[
                                    high
                                ].mean()
                            )
                            if high.any()
                            else np.nan
                        ),
                    "P_near_given_not_high":
                        (
                            float(
                                near.loc[
                                    ~high
                                ].mean()
                            )
                            if (
                                ~high
                            ).any()
                            else np.nan
                        ),
                    "odds_ratio":
                        odds_ratio,
                    "fisher_exact_p":
                        p_value,
                }
            )

    singularity_threshold_sensitivity = (
        pd.DataFrame(
            threshold_rows
        )
    )

    continuous_rows = []

    for variable in [
        "absolute_intercept_slope_correlation",
        "one_minus_rho_squared",
        "normalized_determinant",
        "minimum_eigenvalue_fraction_of_trace",
        "innovation_fraction",
    ]:
        pair = (
            finite[
                [
                    variable,
                    "mean_full_posterior_predictability",
                ]
            ]
            .dropna()
        )

        (
            correlation,
            p_value,
        ) = stats.spearmanr(
            pair[variable],
            pair[
                "mean_full_posterior_predictability"
            ],
        )

        continuous_rows.append(
            {
                "geometry_variable":
                    variable,
                "n":
                    len(pair),
                "spearman_rho":
                    correlation,
                "p_value":
                    p_value,
            }
        )

    continuous_singularity_associations = (
        pd.DataFrame(
            continuous_rows
        )
    )

    # ------------------------------------------------------------
    # Frozen Section 18 — high vs non-high descriptive contrast
    # ------------------------------------------------------------

    true_zero_group_summary = (
        finite.groupby(
            "false_high_full_predictability"
        )
        .agg(
            replicates=(
                "replicate",
                "size",
            ),
            median_full_predictability=(
                "mean_full_posterior_predictability",
                "median",
            ),
            median_direct_predictability=(
                "mean_direct_predictability",
                "median",
            ),
            median_innovation_predictability=(
                "mean_innovation_predictability",
                "median",
            ),
            median_abs_rho=(
                "absolute_intercept_slope_correlation",
                "median",
            ),
            median_one_minus_rho_squared=(
                "one_minus_rho_squared",
                "median",
            ),
            median_normalized_determinant=(
                "normalized_determinant",
                "median",
            ),
            median_covariance_min_eigenvalue=(
                "covariance_minimum_eigenvalue",
                "median",
            ),
            median_min_eigen_fraction=(
                "minimum_eigenvalue_fraction_of_trace",
                "median",
            ),
            median_innovation_variance=(
                "innovation_variance",
                "median",
            ),
            median_innovation_fraction=(
                "innovation_fraction",
                "median",
            ),
            hessian_positive_rate=(
                "hessian_positive_definite",
                "mean",
            ),
            slope_boundary_rate=(
                "slope_variance_boundary",
                "mean",
            ),
            covariance_boundary_rate=(
                "covariance_boundary",
                "mean",
            ),
        )
        .reset_index()
    )

    true_zero_group_summary[
        "interpretation"
    ] = np.where(
        true_zero_group_summary[
            "false_high_full_predictability"
        ],
        (
            "false/illusory high full-posterior "
            "predictability under true-zero slope heterogeneity"
        ),
        "not high under the frozen threshold",
    )

    return {
        "simulation_analysis":
            simulation_analysis,
        "lr_calibration_summary":
            lr_calibration_summary,
        "simulation_method_summary":
            simulation_method_summary,
        "true_zero_denominator_audit":
            true_zero_denominator_audit,
        "primary_concentration_tests":
            primary_concentration_tests,
        "adjusted_logistic_result":
            adjusted_logistic_result,
        "singularity_threshold_sensitivity":
            singularity_threshold_sensitivity,
        "continuous_singularity_associations":
            continuous_singularity_associations,
        "true_zero_group_summary":
            true_zero_group_summary,
        "finite":
            finite,
    }


def make_singularity_figure(
    finite,
    output_path,
):
    plot_data = (
        finite.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=[
                "one_minus_rho_squared",
                "mean_full_posterior_predictability",
            ]
        )
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    for regime, color in zip(
        REGIME_TO_OUTCOME,
        [
            "#2b6cb0",
            "#38a169",
            "#dd6b20",
            "#c53030",
        ],
    ):
        subset = plot_data.loc[
            plot_data[
                "regime"
            ].eq(regime)
        ]

        axes[0].scatter(
            subset[
                "one_minus_rho_squared"
            ],
            subset[
                "mean_full_posterior_predictability"
            ],
            s=13,
            alpha=0.45,
            label=regime,
            color=color,
        )

    axes[0].axhline(
        HIGH_FALSE_FULL_THRESHOLD,
        color="black",
        ls="--",
        lw=0.8,
    )
    axes[0].axvline(
        PRIMARY_SMALL_NORMALIZED_DETERMINANT,
        color="black",
        ls=":",
        lw=0.8,
    )
    axes[0].set_xlabel(
        r"$1-\rho^2$ / normalized determinant"
    )
    axes[0].set_ylabel(
        "Mean full-posterior predictability"
    )
    axes[0].legend(
        frameon=False,
        fontsize=8,
    )

    axes[1].scatter(
        plot_data[
            "minimum_eigenvalue_fraction_of_trace"
        ],
        plot_data[
            "mean_full_posterior_predictability"
        ],
        c=np.where(
            plot_data[
                "false_high_full_predictability"
            ],
            "#c53030",
            "#718096",
        ),
        s=13,
        alpha=0.5,
    )
    axes[1].axhline(
        HIGH_FALSE_FULL_THRESHOLD,
        color="black",
        ls="--",
        lw=0.8,
    )
    axes[1].set_xlabel(
        "Minimum covariance eigenvalue / trace"
    )
    axes[1].set_ylabel(
        "Mean full-posterior predictability"
    )

    groups = [
        plot_data.loc[
            ~plot_data[
                "near_singular_primary"
            ],
            "mean_full_posterior_predictability",
        ].dropna(),
        plot_data.loc[
            plot_data[
                "near_singular_primary"
            ],
            "mean_full_posterior_predictability",
        ].dropna(),
    ]

    axes[2].boxplot(
        groups,
        tick_labels=[
            "not near-singular",
            "|rho| >= .95",
        ],
        showfliers=False,
    )
    axes[2].axhline(
        HIGH_FALSE_FULL_THRESHOLD,
        color="black",
        ls="--",
        lw=0.8,
    )
    axes[2].set_ylabel(
        "Mean full-posterior predictability"
    )
    axes[2].set_title(
        "True-zero conditions only"
    )

    fig.suptitle(
        (
            "False full-posterior predictability "
            "and estimated covariance singularity"
        ),
        y=1.02,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def build_completion_checklist(
    task_manifest,
    simulation_analysis,
    model_fit_diagnostics,
    summaries,
    figure_path,
    mode,
):
    required_fit_columns = {
        "random_intercept_variance",
        "random_slope_variance",
        "intercept_slope_covariance",
        "intercept_slope_correlation",
        "absolute_intercept_slope_correlation",
        "covariance_minimum_eigenvalue",
        "one_minus_rho_squared",
        "normalized_determinant",
        "innovation_variance",
        "innovation_fraction",
        "mean_full_posterior_predictability",
        "mean_direct_predictability",
        "mean_innovation_predictability",
        "hessian_positive_definite",
        "hessian_min_information_eigenvalue",
        "converged",
        "residual_variance_boundary",
        "intercept_variance_boundary",
        "slope_variance_boundary",
        "covariance_boundary",
        "any_boundary_flag",
    }

    expected_counts = (
        task_manifest.groupby(
            [
                "row_type",
                "regime",
                "effect_condition",
            ]
        )["replicate"]
        .nunique()
        .sort_index()
    )

    observed_counts = (
        simulation_analysis.groupby(
            [
                "row_type",
                "regime",
                "effect_condition",
            ]
        )["replicate"]
        .nunique()
        .sort_index()
    )

    requested_cells_complete = (
        expected_counts.equals(
            observed_counts
        )
    )

    if mode == "full":
        frozen_500_complete = bool(
            len(observed_counts) == 16
            and observed_counts.eq(
                FULL_REPS_PER_CELL
            ).all()
        )
    else:
        frozen_500_complete = None

    rows = [
        {
            "requirement":
                (
                    "all requested replicates present in "
                    "every calibration/evaluation cell"
                ),
            "complete":
                requested_cells_complete,
        },
        {
            "requirement":
                (
                    "500 replicates in every frozen "
                    "calibration/evaluation cell"
                ),
            "complete":
                frozen_500_complete,
        },
        {
            "requirement":
                "all tasks retained regardless of validity",
            "complete":
                (
                    len(simulation_analysis)
                    == len(task_manifest)
                ),
        },
        {
            "requirement":
                (
                    "requested covariance, reliability, "
                    "Hessian and boundary fields recorded"
                ),
            "complete":
                required_fit_columns.issubset(
                    model_fit_diagnostics.columns
                ),
        },
        {
            "requirement":
                (
                    "true-zero false-high-full "
                    "concentration test completed"
                ),
            "complete":
                (
                    len(
                        summaries[
                            "primary_concentration_tests"
                        ]
                    )
                    == 2
                ),
        },
        {
            "requirement":
                (
                    "threshold and continuous-association "
                    "sensitivities completed"
                ),
            "complete":
                (
                    len(
                        summaries[
                            "singularity_threshold_sensitivity"
                        ]
                    )
                    == 9
                    and len(
                        summaries[
                            "continuous_singularity_associations"
                        ]
                    )
                    == 5
                ),
        },
        {
            "requirement":
                "diagnostic figure saved",
            "complete":
                figure_path.exists(),
        },
    ]

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    outcome_map = load_outcome_map(
        args.anchor_manifest
    )

    task_manifest = pd.read_csv(
        require_file(
            args.task_manifest,
            "task manifest",
        )
    )

    simulation_replicates = pd.read_csv(
        require_file(
            args.simulation_replicates_raw,
            "raw simulation replicate results",
        )
    )

    model_fit_diagnostics = pd.read_csv(
        require_file(
            args.model_fit_diagnostics,
            "model-fit diagnostics",
        )
    )

    optimizer_attempts = pd.read_csv(
        require_file(
            args.optimizer_attempts,
            "optimizer attempts",
        )
    )

    validate_raw_inputs(
        task_manifest,
        simulation_replicates,
        model_fit_diagnostics,
        optimizer_attempts,
        args.mode,
    )

    summaries = summarize(
        task_manifest,
        simulation_replicates,
        model_fit_diagnostics,
        outcome_map,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        args.output_dir
        / "true_zero_false_full_predictability_singularity.png"
    )

    make_singularity_figure(
        summaries["finite"],
        figure_path,
    )

    exports = {
        "03_simulation_replicates.csv":
            summaries[
                "simulation_analysis"
            ],
        "06_LR_calibration_summary.csv":
            summaries[
                "lr_calibration_summary"
            ],
        "07_simulation_method_summary.csv":
            summaries[
                "simulation_method_summary"
            ],
        "08_true_zero_denominator_audit.csv":
            summaries[
                "true_zero_denominator_audit"
            ],
        "09_primary_singularity_concentration_tests.csv":
            summaries[
                "primary_concentration_tests"
            ],
        "10_adjusted_singularity_logistic_test.csv":
            summaries[
                "adjusted_logistic_result"
            ],
        "11_singularity_threshold_sensitivity.csv":
            summaries[
                "singularity_threshold_sensitivity"
            ],
        "12_continuous_singularity_associations.csv":
            summaries[
                "continuous_singularity_associations"
            ],
        "13_true_zero_high_full_group_summary.csv":
            summaries[
                "true_zero_group_summary"
            ],
    }

    for filename, frame in exports.items():
        frame.to_csv(
            args.output_dir / filename,
            index=False,
        )

    completion = build_completion_checklist(
        task_manifest,
        summaries[
            "simulation_analysis"
        ],
        model_fit_diagnostics,
        summaries,
        figure_path,
        args.mode,
    )

    completion.to_csv(
        args.output_dir
        / "summary_completion_checklist.csv",
        index=False,
    )

    required_complete = completion.loc[
        ~completion[
            "complete"
        ].isna()
    ]

    if not required_complete[
        "complete"
    ].astype(bool).all():
        raise RuntimeError(
            "One or more summarizer completion checks failed."
        )

    print(
        "Downstream simulation summarization complete."
    )
    print(
        f"  mode: {args.mode}"
    )
    print(
        f"  tasks retained: "
        f"{len(summaries['simulation_analysis'])}"
    )
    print(
        f"  method-summary rows: "
        f"{len(summaries['simulation_method_summary'])}"
    )
    print(
        f"  figure: {figure_path.name}"
    )
    print(
        "No simulation generation or mixed-model fitting "
        "was performed."
    )


if __name__ == "__main__":
    main()
