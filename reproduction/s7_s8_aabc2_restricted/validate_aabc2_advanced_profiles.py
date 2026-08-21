#!/usr/bin/env python3
"""Validate the final AABC2 anchored-d profile at selected and boundary coordinates.

This notebook-independent restricted-data validation reconstructs the four
scientific profile datasets and refits the selected fitted coordinate and exact
PSD boundary for each dataset. It imports the exact computational functions
from the validated advanced core.

The supplied reference directory contains aggregate final profile products.
No participant-level tables are exported.
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
from tqdm.auto import tqdm

import aabc2_common as common
import _aabc2_advanced_rlrt_core as core
from _aabc2_advanced_rlrt_core import (
    reconstruct_bootstrap_dataset,
    prepare_reml_problem,
    fit_profile1d_d_value,
)

parser=argparse.ArgumentParser()
parser.add_argument("--input-config",required=True)
parser.add_argument("--smoke-manifest",required=True)
parser.add_argument("--reference-results",required=True)
parser.add_argument("--output-dir",required=True)
args=parser.parse_args()

MANIFEST_PATH=Path(args.smoke_manifest).expanduser().resolve()
FROZEN=Path(args.reference_results).expanduser().resolve()
OUT=Path(args.output_dir).expanduser().resolve()
if not MANIFEST_PATH.is_file(): raise FileNotFoundError(MANIFEST_PATH)
if not FROZEN.is_dir(): raise FileNotFoundError(FROZEN)
if OUT.exists(): raise FileExistsError(f"Output already exists: {OUT}")
OUT.mkdir(parents=True)

raw=common.build_raw_cohorts(args.input_config)
OUTCOMES, ANALYSIS_DATA, cohort_check=common.build_analysis_data(raw)
core.configure_analysis(OUTCOMES, ANALYSIS_DATA)

def fixed_formula(key):
    return common.fixed_formula(key, OUTCOMES)

PROFILE1D_BOUNDARY_PLOT_VALUE=core.PROFILE1D_BOUNDARY_PLOT_VALUE
# ======================================================================
# Frozen final profile state
# ======================================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    low_memory=False,
)

profile_frozen = pd.read_csv(
    FROZEN
    / "07c_anchored_d_1d_profile.csv",
    low_memory=False,
)

audit_frozen = pd.read_csv(
    FROZEN
    / "08c_anchored_d_1d_optimizer_audit.csv",
    low_memory=False,
)

quality_frozen = pd.read_csv(
    FROZEN
    / "09c_anchored_d_1d_profile_quality.csv",
    low_memory=False,
)

resample_manifest = pd.read_csv(
    FROZEN
    / "03_profile_resample_manifest.csv",
    low_memory=False,
)


for frame in (
    manifest,
    profile_frozen,
    audit_frozen,
):
    frame[
        "d_0_5"
    ] = pd.to_numeric(
        frame["d_0_5"],
        errors="raise",
    )


# ======================================================================
# Reconstruct the four profile datasets
# ======================================================================

def profile_dataset(
    dataset_id,
    outcome,
    dataset_role,
):
    if dataset_role == "observed":
        return (
            ANALYSIS_DATA[
                outcome
            ]
            .copy()
        )

    if (
        dataset_role
        ==
        "high_gap_resample_nearest_high_regime_median"
    ):
        source = (
            resample_manifest.loc[
                resample_manifest
                .outcome
                .eq(outcome)
            ]
        )

        if len(source) != 1:
            raise RuntimeError(
                f"{outcome}: expected one "
                "profile-resample row."
            )

        seed = int(
            source.iloc[0].seed
        )

        data = reconstruct_bootstrap_dataset(
            outcome,
            seed,
        )

        expected_rep = int(
            source.iloc[0].replicate
        )

        if (
            f"rep{expected_rep}"
            not in dataset_id
        ):
            raise RuntimeError(
                f"{dataset_id}: does not match "
                f"frozen replicate {expected_rep}."
            )

        return data

    raise ValueError(
        f"Unknown dataset_role: "
        f"{dataset_role}"
    )


# ======================================================================
# Helper: coordinate-specific frozen start
#
# This is used ONLY as an optimization starting value.
# The objective and both optimizers are rerun from scratch.
# ======================================================================

def coordinate_start(
    dataset_id,
    d_0_5,
):
    rows = (
        audit_frozen.loc[
            audit_frozen
            .dataset_id
            .eq(dataset_id)
            &
            np.isclose(
                audit_frozen.d_0_5,
                float(d_0_5),
                atol=1e-15,
                rtol=0,
            )
        ]
        .copy()
    )

    if rows.empty:
        raise RuntimeError(
            f"{dataset_id}: no frozen optimizer "
            f"rows at d={d_0_5}"
        )

    finite = rows.loc[
        rows[
            "finite_solution"
        ]
        .astype(str)
        .str.lower()
        .isin(
            ["true", "1"]
        )
    ].copy()

    if finite.empty:
        raise RuntimeError(
            f"{dataset_id}: no finite frozen "
            f"start at d={d_0_5}"
        )

    best = (
        finite
        .sort_values(
            "log_likelihood",
            ascending=False,
        )
        .iloc[0]
    )

    return np.array(
        [
            float(
                best.log_slope_sd
            ),
            float(
                best.log_sd5_over_sd0
            ),
            float(
                best.log_residual_sd
            ),
        ],
        dtype=float,
    )


# ======================================================================
# Fit selected coordinate -> boundary for each dataset
# ======================================================================

summary_rows = []
audit_rows_all = []


dataset_meta = (
    manifest[
        [
            "dataset_id",
            "outcome",
            "dataset_role",
        ]
    ]
    .drop_duplicates()
)


for meta in tqdm(
    dataset_meta.itertuples(
        index=False
    ),
    total=len(dataset_meta),
    desc="Profile datasets",
    unit="dataset",
):

    dataset_id = str(
        meta.dataset_id
    )

    outcome = str(
        meta.outcome
    )

    dataset_role = str(
        meta.dataset_role
    )

    data = profile_dataset(
        dataset_id,
        outcome,
        dataset_role,
    )

    problem = prepare_reml_problem(
        data,
        fixed_formula(outcome),
    )


    # Frozen fitted coordinate.
    selected_rows = (
        profile_frozen.loc[
            profile_frozen
            .dataset_id
            .eq(dataset_id)
        ]
    )

    selected_values = (
        selected_rows[
            "selected_fitted_d_0_5"
        ]
        .dropna()
        .astype(float)
        .unique()
    )

    if len(selected_values) != 1:
        raise RuntimeError(
            f"{dataset_id}: ambiguous "
            "selected fitted d."
        )

    selected_d = float(
        selected_values[0]
    )


    # Use the final selected-coordinate optimizer state as the
    # high-quality starting value. The likelihood is still recomputed.
    frozen_start = coordinate_start(
        dataset_id,
        selected_d,
    )

    selected_log_residual_sd = float(
        frozen_start[2]
    )

    # Exact the validated final 1D profile bounds.
    bounds = [
        (
            max(
                -12.0,
                float(
                    frozen_start[0]
                ) - 5.0,
            ),
            min(
                8.0,
                float(
                    frozen_start[0]
                ) + 5.0,
            ),
        ),
        (
            -8.0,
            8.0,
        ),
        (
            max(
                -12.0,
                selected_log_residual_sd
                - 4.0,
            ),
            min(
                8.0,
                selected_log_residual_sd
                + 4.0,
            ),
        ),
    ]


    warm = {
        1.0:
            frozen_start.copy(),

        -1.0:
            frozen_start.copy(),
    }


    # Match the source path:
    # selected fitted coordinate first, then descend to d=0.
    coordinate_sequence = [
        (
            "selected_fitted_coordinate",
            selected_d,
        ),
        (
            "exact_PSD_boundary",
            0.0,
        ),
    ]


    for coordinate_role, d_value in (
        coordinate_sequence
    ):
        start_here = coordinate_start(
            dataset_id,
            d_value,
        )

        summary, audit_rows, next_warm = (
            fit_profile1d_d_value(
                problem,
                float(d_value),
                frozen_start,
                warm,
                start_here,
                bounds,
            )
        )

        summary.update({
            "dataset_id":
                dataset_id,

            "outcome":
                outcome,

            "dataset_role":
                dataset_role,

            "coordinate_role":
                coordinate_role,

            "selected_fitted_d_0_5":
                selected_d,
        })

        summary_rows.append(
            summary
        )

        for audit_row in audit_rows:
            audit_row = dict(
                audit_row
            )

            audit_row.update({
                "dataset_id":
                    dataset_id,

                "outcome":
                    outcome,

                "dataset_role":
                    dataset_role,

                "coordinate_role":
                    coordinate_role,
            })

            audit_rows_all.append(
                audit_row
            )

        # Continue the lower warm path after the selected point,
        # matching the validated final 1D profile.
        for sign, value in (
            next_warm.items()
        ):
            warm[
                float(sign)
            ] = np.asarray(
                value,
                float,
            )


smoke_summary = pd.DataFrame(
    summary_rows
)

smoke_audit = pd.DataFrame(
    audit_rows_all
)


smoke_summary.to_csv(
    OUT
    / "profile_8_coordinate_smoke_summary.csv",
    index=False,
)

smoke_audit.to_csv(
    OUT
    / "profile_8_coordinate_optimizer_audit.csv",
    index=False,
)


# ======================================================================
# Compare all eight scientific profile coordinates
# ======================================================================

comparison = (
    manifest.merge(
        smoke_summary,
        on=[
            "dataset_id",
            "outcome",
            "dataset_role",
            "coordinate_role",
            "d_0_5",
        ],
        how="left",
        validate="one_to_one",
        suffixes=(
            "__expected",
            "__new",
        ),
    )
)


if len(comparison) != 8:
    raise AssertionError(
        f"Expected 8 comparison rows; "
        f"found {len(comparison)}"
    )


comparison[
    "LL_absolute_difference"
] = np.abs(
    comparison[
        "profile_log_likelihood"
    ]
    -
    comparison[
        "expected_profile_log_likelihood"
    ]
)


# Final frozen reference likelihood is used only to reconstruct
# relative deviance for comparison.
reference_lookup = (
    profile_frozen[
        [
            "dataset_id",
            "reference_log_likelihood",
        ]
    ]
    .drop_duplicates()
    .set_index(
        "dataset_id"
    )[
        "reference_log_likelihood"
    ]
    .to_dict()
)


comparison[
    "new_relative_deviance"
] = [
    2
    * (
        float(
            reference_lookup[
                dataset_id
            ]
        )
        -
        float(ll)
    )
    for dataset_id, ll
    in zip(
        comparison.dataset_id,
        comparison.profile_log_likelihood,
    )
]


comparison[
    "relative_deviance_absolute_difference"
] = np.abs(
    comparison[
        "new_relative_deviance"
    ]
    -
    comparison[
        "expected_relative_deviance"
    ]
)


def bool_value(x):
    return (
        str(x)
        .strip()
        .lower()
        in {
            "true",
            "1",
        }
    )


comparison[
    "optimizer_agreement_matches"
] = [
    (
        bool_value(new)
        ==
        bool_value(expected)
    )
    for new, expected
    in zip(
        comparison[
            "optimizer_agreement"
        ],
        comparison[
            "expected_optimizer_agreement"
        ],
    )
]


# The fitter itself licenses optimizer agreement up to 1e-3
# in log likelihood. Here we ask for a tighter scientific-coordinate
# reproduction than that.
LL_TOL = 5e-4
DEVIANCE_TOL = 1e-3


comparison[
    "coordinate_pass"
] = (
    comparison[
        "LL_absolute_difference"
    ]
    .le(
        LL_TOL
    )
    &
    comparison[
        "relative_deviance_absolute_difference"
    ]
    .le(
        DEVIANCE_TOL
    )
    &
    comparison[
        "optimizer_agreement_matches"
    ]
)


print()
print(
    "=== 8-COORDINATE PROFILE FITTING CHECK ==="
)

print(
    comparison[
        [
            "dataset_id",
            "coordinate_role",
            "d_0_5",
            "expected_profile_log_likelihood",
            "profile_log_likelihood",
            "LL_absolute_difference",
            "expected_relative_deviance",
            "new_relative_deviance",
            "relative_deviance_absolute_difference",
            "optimizer_agreement",
            "coordinate_pass",
        ]
    ].to_string(
        index=False
    )
)


# ======================================================================
# Optimizer-audit structure at the eight fitted coordinates
# ======================================================================

audit_counts = (
    smoke_audit
    .groupby(
        [
            "dataset_id",
            "coordinate_role",
        ]
    )
    .agg(
        rows=(
            "optimizer",
            "size",
        ),

        gamma_signs=(
            "gamma_sign",
            "nunique",
        ),

        optimizers=(
            "optimizer",
            "nunique",
        ),

        finite_solutions=(
            "finite_solution",
            lambda x:
                sum(
                    bool_value(v)
                    for v in x
                ),
        ),

        agreeing_rows=(
            "optimizer_agreement_for_sign",
            lambda x:
                sum(
                    bool_value(v)
                    for v in x
                ),
        ),
    )
    .reset_index()
)


audit_counts[
    "structure_pass"
] = (
    audit_counts.rows.eq(4)
    &
    audit_counts.gamma_signs.eq(2)
    &
    audit_counts.optimizers.eq(2)
    &
    audit_counts.finite_solutions.eq(4)
    &
    audit_counts.agreeing_rows.eq(4)
)


print()
print(
    "=== OPTIMIZER STRUCTURE ==="
)

print(
    audit_counts.to_string(
        index=False
    )
)


# ======================================================================
# Cross-check against frozen 08 at same eight coordinates
# ======================================================================

frozen_selected = []

for row in (
    manifest.itertuples(
        index=False
    )
):
    subset = (
        audit_frozen.loc[
            audit_frozen
            .dataset_id
            .eq(
                row.dataset_id
            )
            &
            np.isclose(
                audit_frozen.d_0_5,
                float(
                    row.d_0_5
                ),
                atol=1e-15,
                rtol=0,
            )
        ]
        .copy()
    )

    if len(subset) != 4:
        raise AssertionError(
            f"{row.dataset_id} "
            f"{row.coordinate_role}: "
            f"expected 4 frozen audit rows, "
            f"found {len(subset)}"
        )

    frozen_selected.append(
        subset
    )


frozen_selected = pd.concat(
    frozen_selected,
    ignore_index=True,
)


audit_compare = (
    smoke_audit.merge(
        frozen_selected[
            [
                "dataset_id",
                "d_0_5",
                "gamma_sign",
                "optimizer",
                "optimizer_agreement_for_sign",
                "log_likelihood",
            ]
        ],
        on=[
            "dataset_id",
            "d_0_5",
            "gamma_sign",
            "optimizer",
        ],
        how="left",
        validate="one_to_one",
        suffixes=(
            "__new",
            "__frozen",
        ),
    )
)


audit_compare[
    "LL_absolute_difference"
] = np.abs(
    audit_compare[
        "log_likelihood__new"
    ]
    -
    audit_compare[
        "log_likelihood__frozen"
    ]
)


audit_compare[
    "agreement_matches"
] = [
    bool_value(a)
    ==
    bool_value(b)
    for a, b
    in zip(
        audit_compare[
            "optimizer_agreement_for_sign__new"
        ],
        audit_compare[
            "optimizer_agreement_for_sign__frozen"
        ],
    )
]


audit_compare[
    "audit_pass"
] = (
    audit_compare[
        "LL_absolute_difference"
    ]
    .le(
        1e-3
    )
    &
    audit_compare[
        "agreement_matches"
    ]
)


print()
print(
    "=== FROZEN 08 AUDIT CROSS-CHECK ==="
)

print(
    audit_compare[
        [
            "dataset_id",
            "d_0_5",
            "gamma_sign",
            "optimizer",
            "LL_absolute_difference",
            "agreement_matches",
            "audit_pass",
        ]
    ].to_string(
        index=False
    )
)


# ======================================================================
# Final status
# ======================================================================

coordinate_pass = bool(
    comparison[
        "coordinate_pass"
    ].all()
)

structure_pass = bool(
    audit_counts[
        "structure_pass"
    ].all()
)

audit_pass = bool(
    audit_compare[
        "audit_pass"
    ].all()
)


print()
print("=" * 80)
print(
    "ADVANCED RLRT 1D PROFILE FITTING SMOKE"
)
print("=" * 80)

print(
    "8 SCIENTIFIC COORDINATES PASS:",
    coordinate_pass,
)

print(
    "8 × 4 OPTIMIZER STRUCTURE PASS:",
    structure_pass,
)

print(
    "FROZEN 08 OPTIMIZER CROSS-CHECK PASS:",
    audit_pass,
)

print(
    "PROFILE COORDINATES REFIT:",
    len(
        smoke_summary
    ),
)

print(
    "FULL 172-COORDINATE PROFILE RERUN:",
    False,
)

print(
    "ALL PROFILE FITTING CHECKS PASS:",
    (
        coordinate_pass
        and
        structure_pass
        and
        audit_pass
    ),
)


if not (
    coordinate_pass
    and
    structure_pass
    and
    audit_pass
):
    raise SystemExit(1)
