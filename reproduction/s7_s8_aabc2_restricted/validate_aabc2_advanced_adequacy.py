#!/usr/bin/env python3
"""Validate AABC2 spline and residual-adequacy diagnostics.

This notebook-independent restricted-data validation imports the exact
validated advanced core, recreates the frozen two-dimensional nonlinear spline
basis from the observed outcomes, and refits the observed ML comparisons plus
four selected adequacy datasets.

Only aggregate validation tables are written.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import patsy
from scipy import stats
from tqdm.auto import tqdm

import aabc2_common as common
import _aabc2_advanced_rlrt_core as core
from _aabc2_advanced_rlrt_core import (
    fit_optimizer_panel,
    attach_frozen_spline_basis,
    spline_time_formula,
    fit_adequacy_dataset,
)

parser=argparse.ArgumentParser()
parser.add_argument("--input-config",required=True)
parser.add_argument("--smoke-manifest",required=True)
parser.add_argument("--reference-results",required=True)
parser.add_argument("--output-dir",required=True)
args=parser.parse_args()

SMOKE_PATH=Path(args.smoke_manifest).expanduser().resolve()
FROZEN=Path(args.reference_results).expanduser().resolve()
OUT=Path(args.output_dir).expanduser().resolve()
if not SMOKE_PATH.is_file(): raise FileNotFoundError(SMOKE_PATH)
if not FROZEN.is_dir(): raise FileNotFoundError(FROZEN)
if OUT.exists(): raise FileExistsError(f"Output already exists: {OUT}")
OUT.mkdir(parents=True)

raw=common.build_raw_cohorts(args.input_config)
OUTCOMES, ANALYSIS_DATA, cohort_check=common.build_analysis_data(raw)
core.configure_analysis(OUTCOMES, ANALYSIS_DATA)

def fixed_formula(key):
    return common.fixed_formula(key, OUTCOMES)

TARGET_GEOMETRY_OUTCOMES=("global_mean_thickness","fluid_cognition")

# Exact frozen spline-state construction from the validated adequacy helper.
for outcome in TARGET_GEOMETRY_OUTCOMES:
    observed=ANALYSIS_DATA[outcome]
    raw_basis=patsy.dmatrix(
        "cr(t5_mid, df=3, constraints='center') - 1",
        observed, return_type="dataframe",
    )
    linear_design=np.column_stack([
        np.ones(len(observed)), observed.t5_mid.to_numpy(float)
    ])
    projection_coefficients=np.linalg.lstsq(
        linear_design, raw_basis.to_numpy(float), rcond=None
    )[0]
    residual_basis=raw_basis.to_numpy(float)-linear_design@projection_coefficients
    _, singular_values, right_vectors=np.linalg.svd(
        residual_basis, full_matrices=False
    )
    nonlinear_rank=int(np.sum(
        singular_values > 1e-10 * max(1.0, singular_values[0])
    ))
    if nonlinear_rank != 2:
        raise RuntimeError(f"{outcome}: unexpected nonlinear spline rank {nonlinear_rank}")
    core.SPLINE_STATES[outcome]={
        "design_info": raw_basis.design_info,
        "projection_coefficients": projection_coefficients,
        "nonlinear_loadings": right_vectors[:2].T,
    }
# ======================================================================
# Final endpoint-covariance naming normalization
# ======================================================================

def apply_final_geometry_names(frame):
    """
    Final manuscript/release convention:
      raw endpoint covariance: cov_L0_L5
      normalized determinant: d_0_5 = 1 - gamma_0_5^2

    Historical notebook helpers may still emit g_0_5.
    """
    frame = frame.copy()

    if "g_0_5" in frame.columns:
        if "cov_L0_L5" in frame.columns:
            a = pd.to_numeric(
                frame["g_0_5"],
                errors="coerce",
            )
            b = pd.to_numeric(
                frame["cov_L0_L5"],
                errors="coerce",
            )

            if not np.allclose(
                a.to_numpy(float),
                b.to_numpy(float),
                atol=1e-12,
                rtol=1e-10,
                equal_nan=True,
            ):
                raise AssertionError(
                    "g_0_5 and cov_L0_L5 disagree."
                )

            frame = frame.drop(
                columns=["g_0_5"]
            )

        else:
            frame = frame.rename(
                columns={
                    "g_0_5": "cov_L0_L5"
                }
            )

    if (
        "gamma_0_5" in frame.columns
        and
        "d_0_5" not in frame.columns
    ):
        gamma = pd.to_numeric(
            frame["gamma_0_5"],
            errors="coerce",
        )

        frame["d_0_5"] = np.maximum(
            0.0,
            1.0 - gamma**2,
        )

    return frame


# ======================================================================
# Comparator
# ======================================================================

JSON_COLUMNS = {
    "fixed_effects_json",
    "fixed_effect_SE_json",
}


def compare_frames(
    label,
    new,
    old,
    keys,
    *,
    atol=2e-8,
    rtol=2e-8,
):
    if set(new.columns) != set(old.columns):
        raise AssertionError(
            f"{label}: schema mismatch\n"
            f"new-only="
            f"{sorted(set(new.columns)-set(old.columns))}\n"
            f"old-only="
            f"{sorted(set(old.columns)-set(new.columns))}"
        )

    new = (
        new[old.columns]
        .sort_values(
            keys,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    old = (
        old
        .sort_values(
            keys,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if len(new) != len(old):
        raise AssertionError(
            f"{label}: row count "
            f"{len(new)} != {len(old)}"
        )

    numeric_fail = []
    text_fail = []
    json_fail = []
    max_abs = 0.0

    for column in old.columns:
        a = new[column]
        b = old[column]

        if column in JSON_COLUMNS:
            for i, (av, bv) in enumerate(
                zip(a, b)
            ):
                da = json.loads(str(av))
                db = json.loads(str(bv))

                if set(da) != set(db):
                    json_fail.append(
                        (column, i, "keys")
                    )
                    continue

                for key in da:
                    if not np.isclose(
                        float(da[key]),
                        float(db[key]),
                        atol=atol,
                        rtol=rtol,
                        equal_nan=True,
                    ):
                        json_fail.append(
                            (column, i, key)
                        )

            continue

        an = pd.to_numeric(
            a,
            errors="coerce",
        )

        bn = pd.to_numeric(
            b,
            errors="coerce",
        )

        numeric_candidate = (
            (an.notna() | a.isna()).all()
            and
            (bn.notna() | b.isna()).all()
            and
            (
                an.notna().any()
                or
                bn.notna().any()
            )
        )

        if numeric_candidate:
            av = an.to_numpy(float)
            bv = bn.to_numpy(float)

            finite = (
                np.isfinite(av)
                &
                np.isfinite(bv)
            )

            if finite.any():
                max_abs = max(
                    max_abs,
                    float(
                        np.max(
                            np.abs(
                                av[finite]
                                -
                                bv[finite]
                            )
                        )
                    ),
                )

            if not np.allclose(
                av,
                bv,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            ):
                numeric_fail.append(
                    column
                )

        else:
            aa = (
                a.astype("string")
                .fillna("")
            )

            bb = (
                b.astype("string")
                .fillna("")
            )

            if not aa.equals(bb):
                text_fail.append(
                    column
                )

    passed = not (
        numeric_fail
        or
        text_fail
        or
        json_fail
    )

    print()
    print(label)
    print(
        "rows:",
        len(new),
    )
    print(
        "maximum absolute numeric difference:",
        f"{max_abs:.3g}",
    )
    print(
        "numeric mismatch columns:",
        numeric_fail,
    )
    print(
        "text mismatch columns:",
        text_fail,
    )
    print(
        "JSON mismatch count:",
        len(json_fail),
    )
    print(
        "EQUIVALENT:",
        passed,
    )

    if not passed:
        raise AssertionError(
            f"{label}: mismatch"
        )

    return passed


# ======================================================================
# 13–14: full observed ML spline comparisons
# ======================================================================

observed_spline_ml_rows = []
observed_spline_ml_optimizer_rows = []


for outcome in tqdm(
    TARGET_GEOMETRY_OUTCOMES,
    desc="Observed spline ML",
    unit="outcome",
):
    data = ANALYSIS_DATA[
        outcome
    ]

    spline_data = (
        attach_frozen_spline_basis(
            data,
            outcome,
        )
    )

    linear_ml = fit_optimizer_panel(
        data,
        fixed_formula(outcome),
        "1+t5_mid",
        reml=False,
        stop_after_strict_pair=False,
    )

    spline_ml = fit_optimizer_panel(
        spline_data,
        spline_time_formula(outcome),
        "1+t5_mid",
        reml=False,
        stop_after_strict_pair=False,
    )

    degrees_of_freedom = (
        int(
            spline_ml[
                "selected_record"
            ][
                "fixed_design_columns"
            ]
        )
        -
        int(
            linear_ml[
                "selected_record"
            ][
                "fixed_design_columns"
            ]
        )
        if (
            spline_ml[
                "selected_record"
            ]
            is not None
            and
            linear_ml[
                "selected_record"
            ]
            is not None
        )
        else np.nan
    )

    likelihood_ratio = (
        max(
            0.0,
            2
            *
            (
                float(
                    spline_ml[
                        "result"
                    ].llf
                )
                -
                float(
                    linear_ml[
                        "result"
                    ].llf
                )
            ),
        )
        if (
            spline_ml[
                "result"
            ]
            is not None
            and
            linear_ml[
                "result"
            ]
            is not None
        )
        else np.nan
    )

    observed_spline_ml_rows.append({
        "outcome":
            outcome,

        "comparison":
            (
                "linear vs fixed natural "
                "cubic spline (3 df)"
            ),

        "estimation":
            "ML",

        "likelihood_ratio":
            likelihood_ratio,

        "degrees_of_freedom":
            degrees_of_freedom,

        "chi_square_reference_p_descriptive":
            (
                stats.chi2.sf(
                    likelihood_ratio,
                    degrees_of_freedom,
                )
                if (
                    np.isfinite(
                        likelihood_ratio
                    )
                    and
                    degrees_of_freedom > 0
                )
                else np.nan
            ),

        "linear_strict_panel_valid":
            linear_ml[
                "strict_panel_valid"
            ],

        "spline_strict_panel_valid":
            spline_ml[
                "strict_panel_valid"
            ],
    })

    for model, panel in (
        (
            "linear_ML",
            linear_ml,
        ),
        (
            "spline3_ML",
            spline_ml,
        ),
    ):
        audit = (
            panel[
                "audit"
            ]
            .copy()
        )

        audit.insert(
            0,
            "outcome",
            outcome,
        )

        audit.insert(
            1,
            "model",
            model,
        )

        observed_spline_ml_optimizer_rows.append(
            audit
        )


new13 = pd.DataFrame(
    observed_spline_ml_rows
)

new14 = pd.concat(
    observed_spline_ml_optimizer_rows,
    ignore_index=True,
)

new14 = apply_final_geometry_names(
    new14
)


new13.to_csv(
    OUT
    / "13_observed_spline_ML_comparison_validation.csv",
    index=False,
)

new14.to_csv(
    OUT
    / "14_observed_spline_ML_optimizer_audit_validation.csv",
    index=False,
)


frozen13 = pd.read_csv(
    FROZEN
    / "13_observed_spline_ML_comparison.csv",
    low_memory=False,
)

frozen14 = pd.read_csv(
    FROZEN
    / "14_observed_spline_ML_optimizer_audit.csv",
    low_memory=False,
)


pass13 = compare_frames(
    "13 observed spline ML comparison",
    new13,
    frozen13,
    ["outcome"],
)

pass14 = compare_frames(
    "14 observed spline ML optimizer audit",
    new14,
    frozen14,
    [
        "outcome",
        "model",
        "optimizer",
    ],
)


# ======================================================================
# Four selected adequacy datasets
# ======================================================================

smoke = pd.read_csv(
    SMOKE_PATH,
    low_memory=False,
)

assert len(smoke) == 4


tasks = []

for row in smoke.itertuples(
    index=False
):
    tasks.append({
        "dataset_id":
            str(row.dataset_id),

        "outcome":
            str(row.outcome),

        "dataset_role":
            str(row.dataset_role),

        "gap_regime":
            str(row.gap_regime),

        "replicate":
            int(row.replicate),

        "seed":
            int(row.seed),
    })


summary_parts = []
optimizer_parts = []


for task in tqdm(
    tasks,
    desc="Adequacy fitting smoke",
    unit="dataset",
):
    summary_rows, optimizer_rows = (
        fit_adequacy_dataset(
            task
        )
    )

    summary_parts.extend(
        summary_rows
    )

    optimizer_parts.extend(
        optimizer_rows
    )


new15 = pd.DataFrame(
    summary_parts
)

new16 = pd.DataFrame(
    optimizer_parts
)

new15 = apply_final_geometry_names(
    new15
)

new16 = apply_final_geometry_names(
    new16
)


assert len(new15) == 12, len(new15)
assert len(new16) == 32, len(new16)


new15.to_csv(
    OUT
    / "15_four_dataset_adequacy_summary_validation.csv",
    index=False,
)

new16.to_csv(
    OUT
    / "16_four_dataset_adequacy_optimizer_validation.csv",
    index=False,
)


# ======================================================================
# Select exact frozen comparator rows
# ======================================================================

frozen15_all = pd.read_csv(
    FROZEN
    / "15_minimal_adequacy_summary.csv",
    low_memory=False,
)

frozen16_all = pd.read_csv(
    FROZEN
    / "16_minimal_adequacy_optimizer_audit.csv",
    low_memory=False,
)


keys = smoke[
    [
        "dataset_id",
        "outcome",
    ]
].drop_duplicates()


frozen15 = (
    frozen15_all.merge(
        keys,
        on=[
            "dataset_id",
            "outcome",
        ],
        how="inner",
        validate="many_to_one",
    )
)

frozen16 = (
    frozen16_all.merge(
        keys,
        on=[
            "dataset_id",
            "outcome",
        ],
        how="inner",
        validate="many_to_one",
    )
)


assert len(frozen15) == 12, len(frozen15)
assert len(frozen16) == 32, len(frozen16)


pass15 = compare_frames(
    "15 four-dataset adequacy summaries",
    new15,
    frozen15,
    [
        "dataset_id",
        "model",
    ],
)

pass16 = compare_frames(
    "16 four-dataset adequacy optimizer audit",
    new16,
    frozen16,
    [
        "dataset_id",
        "model",
        "optimizer",
    ],
)


# ======================================================================
# Compact scientific status
# ======================================================================

print()
print(
    "=== OBSERVED SPLINE ML RESULTS ==="
)

print(
    new13[
        [
            "outcome",
            "likelihood_ratio",
            "degrees_of_freedom",
            "chi_square_reference_p_descriptive",
            "linear_strict_panel_valid",
            "spline_strict_panel_valid",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "=== FOUR-DATASET ADEQUACY RESULTS ==="
)

display_cols = [
    c
    for c in (
        "dataset_id",
        "outcome",
        "gap_regime",
        "model",
        "basic_panel_valid",
        "strict_panel_valid",
        "fit_succeeded",
        "random_slope_variance",
        "median_direct",
        "median_full",
        "median_innovation",
        "median_gap",
        "later_to_first_residual_variance_ratio",
    )
    if c in new15.columns
]

print(
    new15[
        display_cols
    ]
    .sort_values(
        [
            "outcome",
            "gap_regime",
            "model",
        ],
        kind="stable",
    )
    .to_string(
        index=False
    )
)


print()
print(
    "=== OPTIMIZER ROW COUNTS ==="
)

counts = (
    new16.groupby(
        [
            "dataset_id",
            "model",
        ]
    )
    .optimizer
    .nunique()
    .reset_index(
        name="optimizers"
    )
)

print(
    counts.to_string(
        index=False
    )
)


expected_count = {
    "linear_homoscedastic_REML": 3,
    "natural_cubic_spline_3df_homoscedastic_REML": 3,
    "linear_first_vs_later_residual_REML": 2,
}

for row in counts.itertuples(
    index=False
):
    assert (
        row.optimizers
        ==
        expected_count[
            row.model
        ]
    )


print()
print("=" * 80)
print(
    "ADVANCED RLRT ADEQUACY FITTING VALIDATION"
)
print("=" * 80)

print(
    "13 OBSERVED SPLINE COMPARISON PASS:",
    pass13,
)

print(
    "14 OBSERVED ML OPTIMIZER AUDIT PASS:",
    pass14,
)

print(
    "FOUR × THREE ADEQUACY SUMMARIES PASS:",
    pass15,
)

print(
    "FOUR-DATASET ADEQUACY OPTIMIZER AUDIT PASS:",
    pass16,
)

print(
    "SELECTED ADEQUACY DATASETS REFIT:",
    len(tasks),
)

print(
    "FULL 60-SELECTED-DATASET ADEQUACY RERUN:",
    False,
)

print(
    "ALL ADEQUACY FITTING CHECKS PASS:",
    (
        pass13
        and
        pass14
        and
        pass15
        and
        pass16
    ),
)


if not (
    pass13
    and
    pass14
    and
    pass15
    and
    pass16
):
    raise SystemExit(1)
