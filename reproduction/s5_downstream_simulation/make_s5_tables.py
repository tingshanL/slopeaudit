#!/usr/bin/env python3
"""Build the S5 downstream-simulation publication tables.

This is the isolated reviewer-facing table layer for the S5 downstream
simulation. It performs no simulation, no model fitting, no recursive file
discovery, and no participant-level data access.

Inputs are explicitly supplied semantic S5 result files. For empirical
AABC-derived results, the inputs/derived outputs remain HOLD pending written
AABC permission.

Generated tables:
    Main Table 4
    Supplementary Tables S6A--S6L

The table calculations, ordering, labels, Wilson intervals, captions, notes,
and LaTeX formatting are refactored from the frozen publication-table
notebook.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Frozen S5 labels and ordering
# ---------------------------------------------------------------------------

OUTCOME_ORDER = [
    "lateral_ventricles",
    "hippocampus",
    "global_cortical_cbf",
    "global_cortical_att",
    "fluid_cognition",
    "global_mean_thickness",
]

OUTCOME_LABELS = {
    "lateral_ventricles": "Ventricles",
    "hippocampus": "Hippocampus",
    "global_cortical_cbf": "CBF",
    "global_cortical_att": "ATT",
    "fluid_cognition": "Cognition",
    "global_mean_thickness": "Thickness",
}

METHOD_LABELS = {
    "one_stage_LMM": "One-stage LMM",
    "one_stage_GEE": "One-stage GEE",
    "uncertainty_aware_two_stage": "Uncertainty-aware two-stage",
    "BLUP_as_outcome_attenuation": "Naive EB-as-outcome",
    "predictor_double_use_overconfidence": "Shared-data predictor reuse",
}

MAIN_METHODS = [
    "one_stage_LMM",
    "uncertainty_aware_two_stage",
    "BLUP_as_outcome_attenuation",
    "predictor_double_use_overconfidence",
]

METHOD_ORDER_ALL = [
    "one_stage_LMM",
    "one_stage_GEE",
    "uncertainty_aware_two_stage",
    "BLUP_as_outcome_attenuation",
    "predictor_double_use_overconfidence",
]

REGIME_LABELS = {
    "high_slope_information": "High information (ventricles)",
    "intermediate": "Intermediate (hippocampus)",
    "near_boundary": "Near boundary (ATT)",
    "true_zero_slope_variance": "True-zero variance (cognition-based design)",
}

REGIME_ORDER = [
    "high_slope_information",
    "intermediate",
    "near_boundary",
    "true_zero_slope_variance",
]

POSITIVE_EFFECT = "beta_xt_0.10_SD_per_5y"
NULL_EFFECT = "beta_xt_zero"

TABLE_INVENTORY = []


# ---------------------------------------------------------------------------
# CLI / input validation
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--observed-source-fit-audit",
        required=True,
        type=Path,
        help=(
            "Observed source optimizer-fit audit from the restricted "
            "source-anchor preparer."
        ),
    )

    parser.add_argument(
        "--simulation-replicates",
        required=True,
        type=Path,
        help="Deterministically summarized 03_simulation_replicates.csv.",
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
        "--lr-calibration-summary",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--simulation-method-summary",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--true-zero-denominator-audit",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--singularity-concentration-tests",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--adjusted-logistic-test",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--threshold-sensitivity",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--continuous-associations",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--true-zero-group-summary",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} does not exist or is not a file: {path}"
        )

    return path


# ---------------------------------------------------------------------------
# Frozen formatting / Wilson helpers
# ---------------------------------------------------------------------------

def is_missing(value):
    return (
        value is None
        or (
            isinstance(
                value,
                (float, np.floating),
            )
            and not np.isfinite(value)
        )
        or pd.isna(value)
    )


def fmt_num(value, digits=3, leading_zero=True):
    if is_missing(value):
        return r"\textemdash"

    text = f"{float(value):.{digits}f}"

    if (
        not leading_zero
        and abs(float(value)) < 1
    ):
        text = (
            text.replace("-0.", "-.", 1)
            if text.startswith("-0.")
            else text.replace("0.", ".", 1)
        )

    return text


def fmt_prop(value, digits=3):
    return fmt_num(
        value,
        digits=digits,
        leading_zero=False,
    )


def fmt_int(value):
    return (
        r"\textemdash"
        if is_missing(value)
        else f"{int(round(float(value))):,}"
    )


def fmt_p(value):
    if is_missing(value):
        return r"\textemdash"

    value = float(value)

    if value < 0.001:
        return r"$<.001$"

    return "$" + fmt_prop(value, 3) + "$"


def fmt_interval(
    center,
    low,
    high,
    digits=3,
    probability=False,
):
    formatter = (
        fmt_prop
        if probability
        else (
            lambda x, digits=digits:
            fmt_num(x, digits)
        )
    )

    if any(
        is_missing(x)
        for x in [
            center,
            low,
            high,
        ]
    ):
        return r"\textemdash"

    return (
        f"{formatter(center, digits)} "
        f"[{formatter(low, digits)}, "
        f"{formatter(high, digits)}]"
    )


def latex_text(value):
    if is_missing(value):
        return r"\textemdash"

    text = str(value)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    return text


def wilson_interval(
    proportion,
    n,
    z=1.959963984540054,
):
    if (
        is_missing(proportion)
        or is_missing(n)
        or int(n) <= 0
    ):
        return np.nan, np.nan

    n = int(n)

    successes = int(
        np.rint(
            float(proportion)
            * n
        )
    )

    p = successes / n

    denominator = (
        1
        + z * z / n
    )

    center = (
        p
        + z * z / (2 * n)
    ) / denominator

    half = (
        z
        * math.sqrt(
            p * (1 - p) / n
            + z * z / (4 * n * n)
        )
        / denominator
    )

    return (
        max(
            0.0,
            center - half,
        ),
        min(
            1.0,
            center + half,
        ),
    )


def add_wilson_columns(
    frame,
    rate_column,
    n_column,
    prefix,
):
    frame = frame.copy()

    intervals = [
        wilson_interval(
            p,
            n,
        )
        for p, n in zip(
            frame[rate_column],
            frame[n_column],
        )
    ]

    frame[
        f"{prefix}_low"
    ] = [
        x[0]
        for x in intervals
    ]

    frame[
        f"{prefix}_high"
    ] = [
        x[1]
        for x in intervals
    ]

    return frame


def latex_rows(display_frame):
    header = (
        " & ".join(
            str(column)
            for column
            in display_frame.columns
        )
        + r" \\"
    )

    rows = []

    for values in display_frame.itertuples(
        index=False,
        name=None,
    ):
        cells = [
            (
                r"\textemdash"
                if is_missing(value)
                else str(value)
            )
            for value in values
        ]

        rows.append(
            " & ".join(cells)
            + r" \\"
        )

    return header, rows


# ---------------------------------------------------------------------------
# Frozen export behavior, isolated to an explicit output directory
# ---------------------------------------------------------------------------

def export_table(
    raw_frame,
    display_frame,
    *,
    output_dir,
    stem,
    section,
    caption,
    label,
    note,
    column_format=None,
    longtable=False,
    landscape=False,
    sources=(),
):
    raw_path = (
        output_dir
        / f"{stem}.csv"
    )

    tex_path = (
        output_dir
        / f"{stem}.tex"
    )

    raw_frame.to_csv(
        raw_path,
        index=False,
    )

    if column_format is None:
        column_format = (
            "l"
            + "c"
            * max(
                0,
                display_frame.shape[1]
                - 1,
            )
        )

    header, rows = latex_rows(
        display_frame
    )

    body = "\n".join(rows)

    if longtable:
        tex = (
            f"\\begin{{longtable}}{{{column_format}}}\n"
            + f"\\caption{{{caption}}}\\label{{{label}}}"
            + " \\\\"
            + "\n"
            + "\\toprule\n"
            + header
            + "\n\\midrule\n\\endfirsthead\n"
            + "\\toprule\n"
            + header
            + "\n\\midrule\n\\endhead\n"
            + body
            + "\n\\midrule\n"
            + (
                f"\\multicolumn{{{display_frame.shape[1]}}}"
                "{p{0.96\\linewidth}}"
            )
            + (
                f"{{\\footnotesize\\textit{{Note.}} "
                f"{note}}}"
            )
            + " \\\\"
            + "\n"
            + "\\bottomrule\n"
            + "\\end{longtable}\n"
        )

    else:
        tabular = (
            f"\\begin{{tabular}}{{{column_format}}}\n"
            "\\toprule\n"
            + header
            + "\n\\midrule\n"
            + body
            + "\n\\bottomrule\n"
            "\\end{tabular}\n"
        )

        tex = (
            "\\begin{table}[tbp]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\small\n"
            + tabular
            + "\\vspace{2pt}\n"
            "\\begin{minipage}{0.98\\linewidth}\n"
            + (
                f"\\footnotesize\\textit{{Note.}} "
                f"{note}\n"
            )
            + "\\end{minipage}\n"
            "\\end{table}\n"
        )

    if landscape:
        tex = (
            "\\begin{landscape}\n"
            + tex
            + "\\end{landscape}\n"
        )

    tex_path.write_text(
        tex,
        encoding="utf-8",
    )

    TABLE_INVENTORY.append(
        {
            "section":
                section,
            "stem":
                stem,
            "label":
                label,
            "rows":
                len(raw_frame),
            "csv":
                raw_path.relative_to(
                    output_dir
                ).as_posix(),
            "tex":
                tex_path.relative_to(
                    output_dir
                ).as_posix(),
            "sources":
                " | ".join(
                    sources
                ),
        }
    )

    return (
        raw_path,
        tex_path,
    )


def condition_label(value):
    return (
        "Null fixed effect"
        if value == NULL_EFFECT
        else "Positive fixed effect"
    )


# ---------------------------------------------------------------------------
# Main Table 4
# ---------------------------------------------------------------------------

def make_table_04(
    sim_methods,
    output_dir,
):
    positive = sim_methods.loc[
        sim_methods[
            "effect_condition"
        ].eq(POSITIVE_EFFECT)
        & sim_methods[
            "regime"
        ].isin(REGIME_ORDER[:-1])
        & sim_methods[
            "method"
        ].isin(MAIN_METHODS)
    ].copy()

    null = sim_methods.loc[
        sim_methods[
            "effect_condition"
        ].eq(NULL_EFFECT)
        & sim_methods[
            "regime"
        ].isin(REGIME_ORDER)
        & sim_methods[
            "method"
        ].isin(MAIN_METHODS)
    ].copy()

    compact_rows = []

    for regime in REGIME_ORDER:
        for method in MAIN_METHODS:
            p = positive.loc[
                positive[
                    "regime"
                ].eq(regime)
                & positive[
                    "method"
                ].eq(method)
            ]

            n = null.loc[
                null[
                    "regime"
                ].eq(regime)
                & null[
                    "method"
                ].eq(method)
            ]

            assert len(n) == 1, (
                regime,
                method,
                "null",
            )

            n = n.iloc[0]

            row = {
                "regime":
                    regime,
                "method":
                    method,
                "null_valid_replicates":
                    n.valid_replicates,
                "null_false_positive_rate":
                    n.null_false_positive_rate,
            }

            (
                fpr_low,
                fpr_high,
            ) = wilson_interval(
                n.null_false_positive_rate,
                n.valid_replicates,
            )

            row.update(
                {
                    "fpr_low":
                        fpr_low,
                    "fpr_high":
                        fpr_high,
                }
            )

            if (
                regime
                != "true_zero_slope_variance"
            ):
                assert len(p) == 1, (
                    regime,
                    method,
                    "positive",
                )

                p = p.iloc[0]

                (
                    coverage_low,
                    coverage_high,
                ) = wilson_interval(
                    p.coverage,
                    p.valid_replicates,
                )

                row.update(
                    {
                        "positive_valid_replicates":
                            p.valid_replicates,
                        "effect_ratio":
                            (
                                p.mean_estimate
                                / p.true_effect
                            ),
                        "positive_coverage":
                            p.coverage,
                        "coverage_low":
                            coverage_low,
                        "coverage_high":
                            coverage_high,
                    }
                )

            else:
                row.update(
                    {
                        "positive_valid_replicates":
                            np.nan,
                        "effect_ratio":
                            np.nan,
                        "positive_coverage":
                            np.nan,
                        "coverage_low":
                            np.nan,
                        "coverage_high":
                            np.nan,
                    }
                )

            compact_rows.append(
                row
            )

    table4 = pd.DataFrame(
        compact_rows
    )

    table4[
        "regime_order"
    ] = table4[
        "regime"
    ].map(
        {
            x: i
            for i, x
            in enumerate(
                REGIME_ORDER
            )
        }
    )

    table4[
        "method_order"
    ] = table4[
        "method"
    ].map(
        {
            x: i
            for i, x
            in enumerate(
                MAIN_METHODS
            )
        }
    )

    table4 = (
        table4.sort_values(
            [
                "regime_order",
                "method_order",
            ]
        )
        .drop(
            columns=[
                "regime_order",
                "method_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    table4_display = pd.DataFrame(
        {
            "Generating regime":
                table4.regime
                .map(REGIME_LABELS)
                .map(latex_text),

            "Analysis method":
                table4.method
                .map(METHOD_LABELS)
                .map(latex_text),

            "$n_{+}/n_{0}$":
                [
                    (
                        f"{fmt_int(p)}/"
                        f"{fmt_int(n)}"
                    )
                    for p, n in zip(
                        table4[
                            "positive_valid_replicates"
                        ],
                        table4[
                            "null_valid_replicates"
                        ],
                    )
                ],

            "Mean estimate / truth":
                table4[
                    "effect_ratio"
                ].map(
                    fmt_num
                ),

            r"Positive-effect coverage [95\% Wilson CI]":
                [
                    fmt_interval(
                        p,
                        lo,
                        hi,
                        probability=True,
                    )
                    for p, lo, hi
                    in zip(
                        table4[
                            "positive_coverage"
                        ],
                        table4[
                            "coverage_low"
                        ],
                        table4[
                            "coverage_high"
                        ],
                    )
                ],

            r"Null false-positive rate [95\% Wilson CI]":
                [
                    fmt_interval(
                        p,
                        lo,
                        hi,
                        probability=True,
                    )
                    for p, lo, hi
                    in zip(
                        table4[
                            "null_false_positive_rate"
                        ],
                        table4[
                            "fpr_low"
                        ],
                        table4[
                            "fpr_high"
                        ],
                    )
                ],
        }
    )

    export_table(
        table4,
        table4_display,
        output_dir=output_dir,
        stem="table_04_compact_simulation_performance",
        section="main text",
        caption=(
            r"\textit{Downstream Consequences of Treating "
            r"Estimated Slopes as Data}"
        ),
        label="tab:compact-simulation",
        note=(
            r"Positive-effect rows show coefficient recovery and 95\% interval coverage. Null rows show "
            r"false-positive rates. Shared-data predictor reuse is the double-use procedure in which a "
            r"predictor enters the model used to construct empirical-Bayes (EB) slopes and is then reused in "
            r"the second-stage regression; it is not a design in which EB slopes predict a separate later "
            r"outcome. The true-zero-variance cognition-based design has no oracle individual random slope, "
            r"so positive-effect recovery is not reported. Wilson intervals use the valid-replicate denominator."
        ),
        column_format="llcccc",
        longtable=True,
        landscape=True,
        sources=(
            "07_simulation_method_summary.csv",
        ),
    )


# ---------------------------------------------------------------------------
# Supplementary Tables S6A--S6L
# ---------------------------------------------------------------------------

def make_s6a(
    sim_methods,
    output_dir,
):
    table = sim_methods.copy()

    for rate, prefix in [
        (
            "coverage",
            "coverage",
        ),
        (
            "null_false_positive_rate",
            "fpr",
        ),
        (
            "power",
            "power",
        ),
    ]:
        table = add_wilson_columns(
            table,
            rate,
            "valid_replicates",
            prefix,
        )

    table[
        "regime_order"
    ] = table.regime.map(
        {
            x: i
            for i, x
            in enumerate(
                REGIME_ORDER
            )
        }
    )

    table[
        "effect_order"
    ] = table.effect_condition.map(
        {
            NULL_EFFECT: 0,
            POSITIVE_EFFECT: 1,
        }
    )

    table[
        "method_order"
    ] = table.method.map(
        {
            x: i
            for i, x
            in enumerate(
                METHOD_ORDER_ALL
            )
        }
    )

    table = (
        table.sort_values(
            [
                "regime_order",
                "effect_order",
                "method_order",
            ]
        )
        .drop(
            columns=[
                "regime_order",
                "effect_order",
                "method_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    display = pd.DataFrame(
        {
            "Regime":
                table.regime
                .map(REGIME_LABELS)
                .map(latex_text),

            "Condition":
                table.effect_condition
                .map(condition_label)
                .map(latex_text),

            "Method":
                table.method
                .map(METHOD_LABELS)
                .fillna(
                    table.method
                )
                .map(latex_text),

            "Valid/requested":
                [
                    (
                        f"{fmt_int(v)}/"
                        f"{fmt_int(r)}"
                    )
                    for v, r
                    in zip(
                        table.valid_replicates,
                        table.requested_replicates,
                    )
                ],

            "Mean estimate":
                table.mean_estimate.map(
                    fmt_num
                ),

            "Bias":
                table.bias.map(
                    fmt_num
                ),

            "RMSE":
                table.RMSE.map(
                    fmt_num
                ),

            "Mean SE":
                table.mean_SE.map(
                    fmt_num
                ),

            r"Coverage [95\% Wilson CI]":
                [
                    fmt_interval(
                        p,
                        lo,
                        hi,
                        probability=True,
                    )
                    for p, lo, hi
                    in zip(
                        table.coverage,
                        table.coverage_low,
                        table.coverage_high,
                    )
                ],

            r"False-positive rate [95\% Wilson CI]":
                [
                    fmt_interval(
                        p,
                        lo,
                        hi,
                        probability=True,
                    )
                    for p, lo, hi
                    in zip(
                        table.null_false_positive_rate,
                        table.fpr_low,
                        table.fpr_high,
                    )
                ],

            r"Power [95\% Wilson CI]":
                [
                    fmt_interval(
                        p,
                        lo,
                        hi,
                        probability=True,
                    )
                    for p, lo, hi
                    in zip(
                        table.power,
                        table.power_low,
                        table.power_high,
                    )
                ],
        }
    )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06a_complete_method_performance",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{Complete Simulation Performance "
            r"by Regime, Condition, and Method}"
        ),
        label="tab:supp-sim-methods",
        note=(
            r"Every calibration and evaluation cell requested 500 replicates. Wilson intervals are calculated "
            r"from the saved rates and valid-replicate denominators. Shared-data predictor reuse denotes the "
            r"double-use procedure, not use of EB slopes as predictors of an independent downstream outcome. "
            r"RMSE = root mean squared error; SE = standard error; GEE = generalized estimating equation."
        ),
        column_format="lllcccccccc",
        longtable=True,
        landscape=True,
        sources=(
            "07_simulation_method_summary.csv",
        ),
    )


def make_s6b(
    sim_replicates,
    output_dir,
):
    table = (
        sim_replicates.groupby(
            [
                "row_type",
                "regime",
                "effect_condition",
            ],
            as_index=False,
        )
        .agg(
            requested_replicates=(
                "replicate",
                "size",
            ),
            unique_replicates=(
                "replicate",
                "nunique",
            ),
            valid_replicates=(
                "task_valid",
                "sum",
            ),
            validity_rate=(
                "task_valid",
                "mean",
            ),
        )
    )

    table[
        "regime_order"
    ] = table.regime.map(
        {
            x: i
            for i, x
            in enumerate(
                REGIME_ORDER
            )
        }
    )

    table = (
        table.sort_values(
            [
                "row_type",
                "regime_order",
                "effect_condition",
            ]
        )
        .drop(
            columns="regime_order"
        )
        .reset_index(
            drop=True
        )
    )

    display = pd.DataFrame(
        {
            "Task type":
                table.row_type.map(
                    latex_text
                ),

            "Regime":
                table.regime
                .map(REGIME_LABELS)
                .map(latex_text),

            "Condition":
                table.effect_condition
                .map(condition_label)
                .map(latex_text),

            "Requested":
                table.requested_replicates.map(
                    fmt_int
                ),

            "Unique replicates":
                table.unique_replicates.map(
                    fmt_int
                ),

            "Valid":
                table.valid_replicates.map(
                    fmt_int
                ),

            "Validity rate":
                table.validity_rate.map(
                    fmt_prop
                ),
        }
    )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06b_task_completion_validity",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{Simulation Task Completion "
            r"and Validity Denominators}"
        ),
        label="tab:supp-sim-task-validity",
        note=(
            r"All requested tasks remain in the saved simulation file regardless of numerical validity. "
            r"Calibration and null evaluation are independent task sets."
        ),
        column_format="lllcccc",
        longtable=True,
        sources=(
            "03_simulation_replicates.csv",
        ),
    )


def make_s6c(
    sim_fit,
    output_dir,
):
    table = (
        sim_fit.groupby(
            [
                "row_type",
                "regime",
                "effect_condition",
                "model_role",
            ],
            as_index=False,
        )
        .agg(
            fitted_rows=(
                "replicate",
                "size",
            ),
            converged_rate=(
                "converged",
                "mean",
            ),
            basic_validity_rate=(
                "basic_numerical_validity",
                "mean",
            ),
            strict_validity_rate=(
                "strict_numerical_validity",
                "mean",
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
            any_boundary_rate=(
                "any_boundary_flag",
                "mean",
            ),
            median_one_minus_rho_squared=(
                "one_minus_rho_squared",
                "median",
            ),
            median_minimum_eigen_fraction=(
                "minimum_eigenvalue_fraction_of_trace",
                "median",
            ),
        )
    )

    table[
        "regime_order"
    ] = table.regime.map(
        {
            x: i
            for i, x
            in enumerate(
                REGIME_ORDER
            )
        }
    )

    table = (
        table.sort_values(
            [
                "row_type",
                "regime_order",
                "effect_condition",
                "model_role",
            ]
        )
        .drop(
            columns="regime_order"
        )
        .reset_index(
            drop=True
        )
    )

    display = pd.DataFrame(
        {
            "Task type":
                table.row_type.map(
                    latex_text
                ),

            "Regime":
                table.regime
                .map(REGIME_LABELS)
                .map(latex_text),

            "Condition":
                table.effect_condition
                .map(condition_label)
                .map(latex_text),

            "Model role":
                table.model_role.map(
                    latex_text
                ),

            "Fits":
                table.fitted_rows.map(
                    fmt_int
                ),

            "Converged":
                table.converged_rate.map(
                    fmt_prop
                ),

            "Basic valid":
                table.basic_validity_rate.map(
                    fmt_prop
                ),

            "Strict valid":
                table.strict_validity_rate.map(
                    fmt_prop
                ),

            "Hessian positive":
                table.hessian_positive_rate.map(
                    fmt_prop
                ),

            "Slope boundary":
                table.slope_boundary_rate.map(
                    fmt_prop
                ),

            "Covariance boundary":
                table.covariance_boundary_rate.map(
                    fmt_prop
                ),

            "Any boundary":
                table.any_boundary_rate.map(
                    fmt_prop
                ),

            r"Median $1-\hat\rho^2$":
                table[
                    "median_one_minus_rho_squared"
                ].map(
                    lambda x:
                    fmt_num(
                        x,
                        5,
                    )
                ),

            (
                r"Median $\lambda_{\min}(\hat G)/"
                r"\mathrm{tr}(\hat G)$"
            ):
                table[
                    "median_minimum_eigen_fraction"
                ].map(
                    lambda x:
                    fmt_num(
                        x,
                        5,
                    )
                ),
        }
    )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06c_model_fit_validity",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{Simulation Model-Fit Numerical "
            r"Validity and Boundary Diagnostics}"
        ),
        label="tab:supp-sim-fit-validity",
        note=(
            r"Rates retain all saved model-fit rows in the displayed denominator. Geometry summaries apply "
            r"only where the corresponding random-slope covariance quantities are finite."
        ),
        column_format="llllcccccccccc",
        longtable=True,
        landscape=True,
        sources=(
            "04_model_fit_covariance_diagnostics.csv",
        ),
    )


def make_s6d(
    sim_opt,
    output_dir,
):
    table = (
        sim_opt.groupby(
            [
                "row_type",
                "regime",
                "effect_condition",
                "model_role",
                "optimizer",
            ],
            as_index=False,
        )
        .agg(
            attempts=(
                "replicate",
                "size",
            ),
            fit_succeeded_rate=(
                "fit_succeeded",
                "mean",
            ),
            converged_rate=(
                "converged",
                "mean",
            ),
            basic_validity_rate=(
                "basic_numerical_validity",
                "mean",
            ),
            strict_validity_rate=(
                "strict_numerical_validity",
                "mean",
            ),
        )
    )

    table[
        "regime_order"
    ] = table.regime.map(
        {
            x: i
            for i, x
            in enumerate(
                REGIME_ORDER
            )
        }
    )

    table = (
        table.sort_values(
            [
                "row_type",
                "regime_order",
                "effect_condition",
                "model_role",
                "optimizer",
            ]
        )
        .drop(
            columns="regime_order"
        )
        .reset_index(
            drop=True
        )
    )

    display = pd.DataFrame(
        {
            "Task type":
                table.row_type.map(
                    latex_text
                ),

            "Regime":
                table.regime
                .map(REGIME_LABELS)
                .map(latex_text),

            "Condition":
                table.effect_condition
                .map(condition_label)
                .map(latex_text),

            "Model role":
                table.model_role.map(
                    latex_text
                ),

            "Optimizer":
                table.optimizer.map(
                    latex_text
                ),

            "Attempts":
                table.attempts.map(
                    fmt_int
                ),

            "Fit succeeded":
                table.fit_succeeded_rate.map(
                    fmt_prop
                ),

            "Converged":
                table.converged_rate.map(
                    fmt_prop
                ),

            "Basic valid":
                table.basic_validity_rate.map(
                    fmt_prop
                ),

            "Strict valid":
                table.strict_validity_rate.map(
                    fmt_prop
                ),
        }
    )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06d_optimizer_attempts",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{Simulation Optimizer-Attempt Diagnostics}"
        ),
        label="tab:supp-sim-optimizers",
        note=(
            r"Every recorded optimizer attempt remains represented. Basic and strict numerical validity use "
            r"the frozen simulation definitions."
        ),
        column_format="lllllccccc",
        longtable=True,
        landscape=True,
        sources=(
            "05_optimizer_attempt_audit.csv",
        ),
    )


def make_s6e(
    sim_lr,
    output_dir,
):
    table = sim_lr.copy()

    table = add_wilson_columns(
        table,
        "independent_null_rejection_rate",
        "null_evaluation_valid",
        "null_rejection",
    )

    table[
        "order"
    ] = table.regime.map(
        {
            x: i
            for i, x
            in enumerate(
                REGIME_ORDER
            )
        }
    )

    table = (
        table.sort_values(
            "order"
        )
        .drop(
            columns="order"
        )
        .reset_index(
            drop=True
        )
    )

    display = pd.DataFrame(
        {
            "Regime":
                table.regime
                .map(REGIME_LABELS)
                .map(latex_text),

            "Critical value":
                table.critical_95.map(
                    lambda x:
                    fmt_num(
                        x,
                        3,
                    )
                ),

            "Calibration valid/requested":
                [
                    (
                        f"{fmt_int(v)}/"
                        f"{fmt_int(r)}"
                    )
                    for v, r
                    in zip(
                        table.calibration_valid,
                        table.calibration_requested,
                    )
                ],

            "Null evaluation valid/requested":
                [
                    (
                        f"{fmt_int(v)}/"
                        f"{fmt_int(r)}"
                    )
                    for v, r
                    in zip(
                        table.null_evaluation_valid,
                        table.null_evaluation_requested,
                    )
                ],

            r"Independent null rejection [95\% Wilson CI]":
                [
                    fmt_interval(
                        p,
                        lo,
                        hi,
                        probability=True,
                    )
                    for p, lo, hi
                    in zip(
                        table[
                            "independent_null_rejection_rate"
                        ],
                        table[
                            "null_rejection_low"
                        ],
                        table[
                            "null_rejection_high"
                        ],
                    )
                ],
        }
    )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06e_lr_calibration",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{Schedule-Specific Likelihood-Ratio "
            r"Calibration and Independent Null Evaluation}"
        ),
        label="tab:supp-sim-lr",
        note=(
            r"The 95th-percentile critical value is estimated in a calibration sample and evaluated in an "
            r"independent null sample. The interval is a Wilson interval using the valid null-evaluation denominator."
        ),
        column_format="lcccc",
        sources=(
            "06_LR_calibration_summary.csv",
        ),
    )


def make_s6f(
    source,
    output_dir,
):
    table = source.copy()

    table[
        "order"
    ] = table.regime.map(
        {
            x: i
            for i, x
            in enumerate(
                REGIME_ORDER
            )
        }
    )

    table = (
        table.sort_values(
            [
                "row_type",
                "order",
                "effect_condition",
            ]
        )
        .drop(
            columns="order"
        )
        .reset_index(
            drop=True
        )
    )

    display = pd.DataFrame(
        {
            "Task type":
                table.row_type.map(
                    latex_text
                ),

            "Regime":
                table.regime
                .map(REGIME_LABELS)
                .map(latex_text),

            "Condition":
                table.effect_condition
                .map(condition_label)
                .map(latex_text),

            "Fitted replicates":
                table.fitted_replicates.map(
                    fmt_int
                ),

            "Basic valid":
                table.basic_validity_rate.map(
                    fmt_prop
                ),

            "Strict valid":
                table.strict_validity_rate.map(
                    fmt_prop
                ),

            "Finite full predictability":
                table.finite_full_rate.map(
                    fmt_prop
                ),

            "Slope boundary":
                table.slope_boundary_rate.map(
                    fmt_prop
                ),

            "Covariance boundary":
                table.covariance_boundary_rate.map(
                    fmt_prop
                ),

            "Hessian positive":
                table.hessian_positive_rate.map(
                    fmt_prop
                ),
        }
    )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06f_true_zero_denominator",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{Numerical Denominators in the "
            r"True-Zero Random-Slope-Variance Diagnostic}"
        ),
        label="tab:supp-sim-zero-denominator",
        note=(
            r"These rows describe correctly specified random-slope fits when the generating random-slope "
            r"variance is zero. They audit spurious fitted predictability; they do not assess recovery of an "
            r"oracle individual random slope."
        ),
        column_format="lllccccccc",
        longtable=True,
        landscape=True,
        sources=(
            "08_true_zero_denominator_audit.csv",
        ),
    )


def make_s6g(
    source,
    output_dir,
):
    table = source.copy()

    display = pd.DataFrame(
        {
            "Near-singular definition":
                table.near_definition.map(
                    latex_text
                ),

            "Finite replicates":
                table.finite_replicates.map(
                    fmt_int
                ),

            "High and near":
                table.high_and_near.map(
                    fmt_int
                ),

            "High and not near":
                table.high_and_not_near.map(
                    fmt_int
                ),

            r"$\Pr(\mathrm{near}\mid\mathrm{high})$":
                table.P_near_given_high.map(
                    fmt_prop
                ),

            r"$\Pr(\mathrm{high}\mid\mathrm{near})$":
                table.risk_high_if_near.map(
                    fmt_prop
                ),

            r"$\Pr(\mathrm{high}\mid\mathrm{not\ near})$":
                table.risk_high_if_not_near.map(
                    fmt_prop
                ),

            "Odds ratio":
                table.odds_ratio.map(
                    lambda x:
                    fmt_num(
                        x,
                        2,
                    )
                ),

            "Fisher $p$":
                table.fisher_exact_p.map(
                    fmt_p
                ),
        }
    )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06g_true_zero_concentration",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{Concentration of False High Full "
            r"Predictability Near Covariance Boundaries}"
        ),
        label="tab:supp-sim-concentration",
        note=(
            r"Tests use the frozen true-zero primary subset and the prespecified high-full-predictability "
            r"threshold. Near-singular definitions are named exactly as recorded by the simulation audit."
        ),
        column_format="lcccccccc",
        landscape=True,
        sources=(
            "09_primary_singularity_concentration_tests.csv",
        ),
    )


def make_s6h(
    source,
    output_dir,
):
    table = source.copy()

    required = {
        "adjusted_odds_ratio",
        "OR_CI95_low",
        "OR_CI95_high",
        "p_value",
        "n",
    }

    if required.issubset(
        table.columns
    ):
        display = pd.DataFrame(
            {
                "Status":
                    table.status.map(
                        latex_text
                    ),

                "Term":
                    table.get(
                        "term",
                        pd.Series(
                            [""] * len(table)
                        ),
                    ).map(
                        latex_text
                    ),

                "$n$":
                    table.n.map(
                        fmt_int
                    ),

                r"Adjusted odds ratio [95\% CI]":
                    [
                        fmt_interval(
                            m,
                            lo,
                            hi,
                            digits=2,
                        )
                        for m, lo, hi
                        in zip(
                            table.adjusted_odds_ratio,
                            table.OR_CI95_low,
                            table.OR_CI95_high,
                        )
                    ],

                "$p$":
                    table.p_value.map(
                        fmt_p
                    ),
            }
        )

    else:
        display = pd.DataFrame(
            {
                "Status":
                    table.status.map(
                        latex_text
                    ),

                "Reason":
                    table.get(
                        "error",
                        pd.Series(
                            ["Not reported"]
                            * len(table)
                        ),
                    ).map(
                        latex_text
                    ),
            }
        )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06h_adjusted_logistic",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{Adjusted Logistic Sensitivity "
            r"for False High Full Predictability}"
        ),
        label="tab:supp-sim-logistic",
        note=(
            r"The model adjusts the near-singular indicator for regime, task type, and effect condition. "
            r"If the model is not estimable, the saved reason is shown rather than substituting an unadjusted result."
        ),
        sources=(
            "10_adjusted_singularity_logistic_test.csv",
        ),
    )


def make_s6i(
    source,
    output_dir,
):
    table = (
        source.copy()
        .sort_values(
            [
                "abs_rho_threshold",
                "full_predictability_threshold",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    display = pd.DataFrame(
        {
            r"$|\hat\rho|$ threshold":
                table.abs_rho_threshold.map(
                    fmt_prop
                ),

            "Full-predictability threshold":
                table.full_predictability_threshold.map(
                    fmt_prop
                ),

            "$n$":
                table.n.map(
                    fmt_int
                ),

            "High full, $n$":
                table.high_full_n.map(
                    fmt_int
                ),

            "Near singular, $n$":
                table.near_singular_n.map(
                    fmt_int
                ),

            r"$\Pr(\mathrm{near}\mid\mathrm{high})$":
                table.P_near_given_high.map(
                    fmt_prop
                ),

            "Odds ratio":
                table.odds_ratio.map(
                    lambda x:
                    fmt_num(
                        x,
                        2,
                    )
                ),

            "Fisher $p$":
                table.fisher_exact_p.map(
                    fmt_p
                ),
        }
    )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06i_simulation_threshold_sensitivity",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{Threshold Sensitivity in the "
            r"True-Zero Simulation Diagnostic}"
        ),
        label="tab:supp-sim-thresholds",
        note=(
            r"This grid is the simulation-specific $|\hat\rho|$ by full-predictability sensitivity analysis; "
            r"it is distinct from the empirical full-499 threshold sweep in Table S3."
        ),
        column_format="cccccccc",
        sources=(
            "11_singularity_threshold_sensitivity.csv",
        ),
    )


def make_s6j(
    source,
    output_dir,
):
    table = source.copy()

    display = pd.DataFrame(
        {
            "Geometry variable":
                table.geometry_variable.map(
                    latex_text
                ),

            "$n$":
                table.n.map(
                    fmt_int
                ),

            r"Spearman $\rho$":
                table.spearman_rho.map(
                    fmt_prop
                ),

            "$p$":
                table.p_value.map(
                    fmt_p
                ),
        }
    )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06j_continuous_geometry_associations",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{Continuous Associations Between "
            r"Covariance Geometry and Full Predictability}"
        ),
        label="tab:supp-sim-continuous",
        note=(
            r"Spearman correlations are calculated in the finite true-zero primary subset using the saved "
            r"continuous geometry diagnostics."
        ),
        column_format="lccc",
        sources=(
            "12_continuous_singularity_associations.csv",
        ),
    )


def make_s6k(
    source,
    output_dir,
):
    table = (
        source.copy()
        .sort_values(
            "false_high_full_predictability"
        )
        .reset_index(
            drop=True
        )
    )

    display = pd.DataFrame(
        {
            "False high full predictability":
                np.where(
                    table[
                        "false_high_full_predictability"
                    ],
                    "Yes",
                    "No",
                ),

            "Replicates":
                table.replicates.map(
                    fmt_int
                ),

            "Median direct":
                table.median_direct_predictability.map(
                    fmt_prop
                ),

            "Median full":
                table.median_full_predictability.map(
                    fmt_prop
                ),

            "Median innovation":
                table.median_innovation_predictability.map(
                    fmt_prop
                ),

            r"Median $|\hat\rho|$":
                table.median_abs_rho.map(
                    fmt_prop
                ),

            r"Median $1-\hat\rho^2$":
                table.median_one_minus_rho_squared.map(
                    lambda x:
                    fmt_num(
                        x,
                        5,
                    )
                ),

            "Median eigenvalue fraction":
                table.median_min_eigen_fraction.map(
                    lambda x:
                    fmt_num(
                        x,
                        5,
                    )
                ),

            "Hessian positive":
                table.hessian_positive_rate.map(
                    fmt_prop
                ),

            "Slope boundary":
                table.slope_boundary_rate.map(
                    fmt_prop
                ),

            "Covariance boundary":
                table.covariance_boundary_rate.map(
                    fmt_prop
                ),
        }
    )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06k_true_zero_group_summary",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{True-Zero Fits With and Without "
            r"False High Full Predictability}"
        ),
        label="tab:supp-sim-zero-groups",
        note=(
            r"Groups are defined by the frozen false-high-full-predictability threshold. Summaries are "
            r"descriptive and retain the fitted covariance-boundary context."
        ),
        column_format="lcccccccccc",
        landscape=True,
        sources=(
            "13_true_zero_high_full_group_summary.csv",
        ),
    )


def make_s6l(
    source,
    output_dir,
):
    table = source.copy()

    table[
        "outcome_order"
    ] = table.outcome.map(
        {
            x: i
            for i, x
            in enumerate(
                OUTCOME_ORDER
            )
        }
    )

    table = (
        table.sort_values(
            [
                "outcome_order",
                "model_role",
                "optimizer",
            ]
        )
        .drop(
            columns="outcome_order"
        )
        .reset_index(
            drop=True
        )
    )

    display = pd.DataFrame(
        {
            "Outcome":
                table.outcome
                .map(OUTCOME_LABELS)
                .fillna(
                    table.outcome
                )
                .map(latex_text),

            "Model role":
                table.model_role.map(
                    latex_text
                ),

            "Optimizer":
                table.optimizer.map(
                    latex_text
                ),

            "Selected":
                np.where(
                    table.selected,
                    "Yes",
                    "No",
                ),

            "Fit succeeded":
                np.where(
                    table.fit_succeeded,
                    "Yes",
                    "No",
                ),

            "Converged":
                np.where(
                    table.converged,
                    "Yes",
                    "No",
                ),

            "Basic valid":
                np.where(
                    table.basic_numerical_validity,
                    "Yes",
                    "No",
                ),

            "Strict valid":
                np.where(
                    table.strict_numerical_validity,
                    "Yes",
                    "No",
                ),
        }
    )

    export_table(
        table,
        display,
        output_dir=output_dir,
        stem="supp_table_S06l_observed_source_fits",
        section="supplement simulation diagnostics",
        caption=(
            r"\textit{Observed Source Optimizer Fits "
            r"Used to Parameterize Simulation Regimes}"
        ),
        label="tab:supp-sim-source-fits",
        note=(
            r"Three optimizers were attempted for the observed source models. Selected fits satisfy the "
            r"frozen agreement rule and provide the regime-specific generating parameters."
        ),
        column_format="lllccccc",
        longtable=True,
        sources=(
            "01_observed_source_fit_audit.csv",
        ),
    )


# ---------------------------------------------------------------------------
# Inventory / include files
# ---------------------------------------------------------------------------

def write_inventory_files(
    output_dir,
):
    inventory = pd.DataFrame(
        TABLE_INVENTORY
    )

    assert len(inventory) == 13, (
        inventory[
            [
                "section",
                "stem",
            ]
        ]
    )

    assert (
        inventory.stem.is_unique
        and inventory.label.is_unique
    )

    inventory.to_csv(
        output_dir
        / "table_inventory.csv",
        index=False,
    )

    include_lines = [
        "% Generated by make_s5_tables.py.",
        "% Requires: booktabs, longtable, array, and pdflscape.",
    ]

    for row in inventory.itertuples(
        index=False
    ):
        include_lines.append(
            "\\input{"
            + row.stem
            + ".tex}"
        )

    (
        output_dir
        / "all_table_includes.tex"
    ).write_text(
        "\n\n".join(
            include_lines
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir
        / "table_preamble.tex"
    ).write_text(
        "\\usepackage{booktabs}\n"
        "\\usepackage{longtable}\n"
        "\\usepackage{array}\n"
        "\\usepackage{pdflscape}\n",
        encoding="utf-8",
    )

    return inventory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    inputs = {
        "source_fits":
            pd.read_csv(
                require_file(
                    args.observed_source_fit_audit,
                    "observed source-fit audit",
                )
            ),

        "sim_replicates":
            pd.read_csv(
                require_file(
                    args.simulation_replicates,
                    "simulation replicates",
                )
            ),

        "sim_fit":
            pd.read_csv(
                require_file(
                    args.model_fit_diagnostics,
                    "model-fit diagnostics",
                )
            ),

        "sim_opt":
            pd.read_csv(
                require_file(
                    args.optimizer_attempts,
                    "optimizer attempts",
                )
            ),

        "sim_lr":
            pd.read_csv(
                require_file(
                    args.lr_calibration_summary,
                    "LR calibration summary",
                )
            ),

        "sim_methods":
            pd.read_csv(
                require_file(
                    args.simulation_method_summary,
                    "simulation method summary",
                )
            ),

        "sim_zero_denominator":
            pd.read_csv(
                require_file(
                    args.true_zero_denominator_audit,
                    "true-zero denominator audit",
                )
            ),

        "sim_concentration":
            pd.read_csv(
                require_file(
                    args.singularity_concentration_tests,
                    "singularity concentration tests",
                )
            ),

        "sim_logistic":
            pd.read_csv(
                require_file(
                    args.adjusted_logistic_test,
                    "adjusted logistic test",
                )
            ),

        "sim_thresholds":
            pd.read_csv(
                require_file(
                    args.threshold_sensitivity,
                    "threshold sensitivity",
                )
            ),

        "sim_continuous":
            pd.read_csv(
                require_file(
                    args.continuous_associations,
                    "continuous associations",
                )
            ),

        "sim_zero_group":
            pd.read_csv(
                require_file(
                    args.true_zero_group_summary,
                    "true-zero group summary",
                )
            ),
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_INVENTORY.clear()

    # Main Table 4.
    make_table_04(
        inputs["sim_methods"],
        args.output_dir,
    )

    # Supplementary Tables S6A--S6L.
    make_s6a(
        inputs["sim_methods"],
        args.output_dir,
    )

    make_s6b(
        inputs["sim_replicates"],
        args.output_dir,
    )

    make_s6c(
        inputs["sim_fit"],
        args.output_dir,
    )

    make_s6d(
        inputs["sim_opt"],
        args.output_dir,
    )

    make_s6e(
        inputs["sim_lr"],
        args.output_dir,
    )

    make_s6f(
        inputs["sim_zero_denominator"],
        args.output_dir,
    )

    make_s6g(
        inputs["sim_concentration"],
        args.output_dir,
    )

    make_s6h(
        inputs["sim_logistic"],
        args.output_dir,
    )

    make_s6i(
        inputs["sim_thresholds"],
        args.output_dir,
    )

    make_s6j(
        inputs["sim_continuous"],
        args.output_dir,
    )

    make_s6k(
        inputs["sim_zero_group"],
        args.output_dir,
    )

    make_s6l(
        inputs["source_fits"],
        args.output_dir,
    )

    inventory = write_inventory_files(
        args.output_dir
    )

    print(
        "S5 publication tables complete."
    )

    print(
        f"  tables: {len(inventory)}"
    )

    print(
        "  main: table_04_compact_simulation_performance"
    )

    print(
        "  supplement: S6A-S6L"
    )

    print(
        "  no simulation, model fitting, or "
        "source-data discovery performed."
    )


if __name__ == "__main__":
    main()
