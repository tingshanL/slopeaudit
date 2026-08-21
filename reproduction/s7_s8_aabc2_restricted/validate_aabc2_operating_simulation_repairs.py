#!/usr/bin/env python3
"""Refit five targeted AABC2 operating-simulation repair cases.

This notebook-independent restricted-data validation imports the exact
validated operating-simulation and direct-REML repair functions from the
advanced core. It refits only the five preselected repair cases and compares
their scientific outputs with the frozen 758-row repair overlay.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import aabc2_common as common
import _aabc2_advanced_rlrt_core as core
from _aabc2_advanced_rlrt_core import (
    build_schedule_template,
    repair_one_operating_replicate,
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

SEED=core.SEED
PB_OUTCOMES=core.PB_OUTCOMES
SIM_BALANCED_N=600
SIM_ANCHORED_NEAR_THRESHOLD=core.SIM_ANCHORED_NEAR_THRESHOLD
PSD_REL_TOL=core.PSD_REL_TOL
SLOPE_BOUNDARY_REL_TOL=core.SLOPE_BOUNDARY_REL_TOL
LLF_REL_TOL=core.LLF_REL_TOL
FIXED_AGREEMENT_SE_TOL=core.FIXED_AGREEMENT_SE_TOL
ROUNDING_TOL=core.ROUNDING_TOL

balanced_times=[
    np.array([-0.5,0.0,0.5])
    for _ in range(SIM_BALANCED_N)
]
SCHEDULE_TEMPLATES={
    "balanced_three_wave": build_schedule_template(
        "balanced_three_wave", balanced_times
    )
}
for outcome in PB_OUTCOMES:
    participant_times=[
        group.t5_mid.to_numpy(float)
        for _,group in ANALYSIS_DATA[outcome].groupby("pid",sort=False)
    ]
    SCHEDULE_TEMPLATES[f"empirical__{outcome}"]=build_schedule_template(
        f"empirical__{outcome}", participant_times
    )
core.SCHEDULE_TEMPLATES=SCHEDULE_TEMPLATES

simulation_manifest=pd.read_csv(
    FROZEN/"18_simulation_cell_manifest.csv",low_memory=False
)
if len(simulation_manifest)!=46 or simulation_manifest["cell_id"].duplicated().any():
    raise RuntimeError("Unexpected simulation-cell manifest")
core.simulation_manifest=simulation_manifest
core.SIMULATION_CELL_LOOKUP=(
    simulation_manifest.set_index("cell_id").to_dict("index")
)

print("constructed schedule templates:",sorted(SCHEDULE_TEMPLATES))
print("loaded exact operating-simulation repair functions from validated core")
# ================================================================
# Smoke cases
# ================================================================

smoke = pd.read_csv(
    SMOKE_PATH,
    low_memory=False,
)

smoke[
    "replicate"
] = pd.to_numeric(
    smoke["replicate"],
    errors="raise",
).astype(int)

assert len(smoke) == 5


# Confirm every smoke case is one of the frozen 758 repairs.
frozen_repair_all = pd.read_csv(
    FROZEN
    / "19a_targeted_758_simulation_repairs.csv",
    low_memory=False,
)

frozen_repair_all[
    "replicate"
] = pd.to_numeric(
    frozen_repair_all["replicate"],
    errors="raise",
).astype(int)

selected_frozen = (
    frozen_repair_all.merge(
        smoke[
            [
                "cell_id",
                "replicate",
                "smoke_role",
            ]
        ],
        on=[
            "cell_id",
            "replicate",
        ],
        how="inner",
        validate="one_to_one",
    )
)

assert len(
    selected_frozen
) == 5


# ================================================================
# Refit exactly the five repair cases
# ================================================================

new_rows = []

for row in tqdm(
    smoke.itertuples(
        index=False
    ),
    total=len(smoke),
    desc="Operating repair smoke",
    unit="dataset",
):
    repaired = (
        repair_one_operating_replicate(
            str(row.cell_id),
            int(row.replicate),
        )
    )

    repaired[
        "smoke_role"
    ] = str(
        row.smoke_role
    )

    new_rows.append(
        repaired
    )


new = pd.DataFrame(
    new_rows
)

assert len(new) == 5

new.to_csv(
    OUT
    / "19a_five_case_targeted_repair_validation.csv",
    index=False,
)


# ================================================================
# Comparison helpers
# ================================================================

def bool_value(value):
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
        }
    )


# Compare the actual targeted-repair scientific output fields.
CORE_COLUMNS = [
    "cell_id",
    "replicate",
    "seed",
    "fit_valid",
    "strict_fit_valid",
    "restricted_LR",
    "null_converged",
    "alternative_converged",
    "alternative_optimizers_attempted",
    "optimizer_agreement",
    "null_optimizer_agreement",
    "repair_attempted",
    "repair_succeeded",
    "repair_method",
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
    "anchor_random_level_variance",
    "anchor_level_slope_covariance",
    "five_year_random_level_variance",
    "anchor_to_five_year_covariance",
    "g_0_5",
    "gamma_0_5",
    "absolute_gamma_0_5",
    "one_minus_gamma_0_5_squared",
    "anchored_endpoint_minimum_eigenvalue",
    "anchored_endpoint_minimum_eigenvalue_fraction",
    "residual_variance",
    "hessian_positive_definite",
    "hessian_minimum_information_eigenvalue",
    "boundary_flag",
    "mean_Rdirect",
    "mean_Pfull",
    "mean_Rinnovation",
    "mean_Delta",
    "estimated_anchored_near",
]


missing_new = [
    c for c in CORE_COLUMNS
    if c not in new.columns
]

missing_old = [
    c for c in CORE_COLUMNS
    if c not in selected_frozen.columns
]

if missing_new:
    raise KeyError(
        f"New targeted-repair rows missing: {missing_new}"
    )

if missing_old:
    raise KeyError(
        f"Frozen 19a missing: {missing_old}"
    )


a = (
    new[
        CORE_COLUMNS
        +
        ["smoke_role"]
    ]
    .sort_values(
        [
            "cell_id",
            "replicate",
        ],
        kind="stable",
    )
    .reset_index(drop=True)
)

b = (
    selected_frozen[
        CORE_COLUMNS
        +
        ["smoke_role"]
    ]
    .sort_values(
        [
            "cell_id",
            "replicate",
        ],
        kind="stable",
    )
    .reset_index(drop=True)
)


BOOL_COLUMNS = {
    "fit_valid",
    "strict_fit_valid",
    "null_converged",
    "alternative_converged",
    "optimizer_agreement",
    "null_optimizer_agreement",
    "repair_attempted",
    "repair_succeeded",
    "hessian_positive_definite",
    "boundary_flag",
    "estimated_anchored_near",
}

TEXT_COLUMNS = {
    "cell_id",
    "repair_method",
    "smoke_role",
}


numeric_fail = []
boolean_fail = []
text_fail = []

maximum_absolute_difference = 0.0


# Numerical-equivalence policy for independently rerun targeted-repair fits.
#
# Exact discrete quantities remain exact.  The restricted LR is the
# inferential quantity and is required to agree much more tightly than
# optimizer-dependent variance-component estimates.  Fitted covariance
# quantities and deterministic functions of them may differ slightly
# across numerically equivalent optimizer solutions.  The Hessian
# minimum eigenvalue is based on finite-difference numerical curvature
# and therefore receives its own tolerance.
EXACT_NUMERIC_COLUMNS = {
    "replicate",
    "seed",
    "alternative_optimizers_attempted",
}

LIKELIHOOD_NUMERIC_COLUMNS = {
    "restricted_LR",
}

HESSIAN_NUMERIC_COLUMNS = {
    "hessian_minimum_information_eigenvalue",
}

FIT_NUMERIC_ATOL = 5e-6
FIT_NUMERIC_RTOL = 5e-6

LIKELIHOOD_NUMERIC_ATOL = 1e-8
LIKELIHOOD_NUMERIC_RTOL = 1e-8

HESSIAN_NUMERIC_ATOL = 5e-4
HESSIAN_NUMERIC_RTOL = 2e-4


for column in a.columns:
    av = a[column]
    bv = b[column]

    if column in BOOL_COLUMNS:
        equal = all(
            bool_value(x)
            ==
            bool_value(y)
            for x, y
            in zip(
                av,
                bv,
            )
        )

        if not equal:
            boolean_fail.append(
                column
            )

        continue

    if column in TEXT_COLUMNS:
        equal = (
            av.astype("string")
            .fillna("")
            .equals(
                bv.astype("string")
                .fillna("")
            )
        )

        if not equal:
            text_fail.append(
                column
            )

        continue

    an = pd.to_numeric(
        av,
        errors="coerce",
    )

    bn = pd.to_numeric(
        bv,
        errors="coerce",
    )

    aa = an.to_numpy(float)
    bb = bn.to_numpy(float)

    finite = (
        np.isfinite(aa)
        &
        np.isfinite(bb)
    )

    if finite.any():
        maximum_absolute_difference = max(
            maximum_absolute_difference,
            float(
                np.max(
                    np.abs(
                        aa[finite]
                        -
                        bb[finite]
                    )
                )
            ),
        )

    if column in EXACT_NUMERIC_COLUMNS:
        numeric_equal = np.array_equal(
            aa,
            bb,
            equal_nan=True,
        )
    elif column in LIKELIHOOD_NUMERIC_COLUMNS:
        numeric_equal = np.allclose(
            aa,
            bb,
            atol=LIKELIHOOD_NUMERIC_ATOL,
            rtol=LIKELIHOOD_NUMERIC_RTOL,
            equal_nan=True,
        )
    elif column in HESSIAN_NUMERIC_COLUMNS:
        numeric_equal = np.allclose(
            aa,
            bb,
            atol=HESSIAN_NUMERIC_ATOL,
            rtol=HESSIAN_NUMERIC_RTOL,
            equal_nan=True,
        )
    else:
        numeric_equal = np.allclose(
            aa,
            bb,
            atol=FIT_NUMERIC_ATOL,
            rtol=FIT_NUMERIC_RTOL,
            equal_nan=True,
        )

    if not numeric_equal:
        numeric_fail.append(
            column
        )


comparison_pass = not (
    numeric_fail
    or
    boolean_fail
    or
    text_fail
)


# ================================================================
# Explicitly test the expected final states
# ================================================================

merged_state = (
    new.merge(
        smoke[
            [
                "cell_id",
                "replicate",
                "smoke_role",
                "fit_valid",
                "strict_fit_valid",
                "repair_succeeded",
            ]
        ],
        on=[
            "cell_id",
            "replicate",
            "smoke_role",
        ],
        how="left",
        validate="one_to_one",
        suffixes=(
            "__new",
            "__expected",
        ),
    )
)


state_pass = True

for row in merged_state.itertuples(
    index=False
):
    for name in (
        "fit_valid",
        "strict_fit_valid",
        "repair_succeeded",
    ):
        actual = getattr(
            row,
            name + "__new",
        )

        expected = getattr(
            row,
            name + "__expected",
        )

        if (
            bool_value(actual)
            !=
            bool_value(expected)
        ):
            state_pass = False


# ================================================================
# Seed regeneration check
# ================================================================

seed_pass = True

for row in smoke.itertuples(
    index=False
):
    _, regenerated_seed = (
        generate_simulation_outcome(
            str(row.cell_id),
            int(row.replicate),
        )
    )

    frozen_seed = int(
        selected_frozen.loc[
            selected_frozen.cell_id.eq(
                str(row.cell_id)
            )
            &
            selected_frozen.replicate.eq(
                int(row.replicate)
            ),
            "seed",
        ].iloc[0]
    )

    if int(
        regenerated_seed
    ) != frozen_seed:
        seed_pass = False


# ================================================================
# Report, without huge numerical dumps
# ================================================================

display = (
    new.merge(
        smoke[
            [
                "cell_id",
                "replicate",
                "smoke_role",
            ]
        ],
        on=[
            "cell_id",
            "replicate",
            "smoke_role",
        ],
        how="left",
    )
    [
        [
            "smoke_role",
            "cell_id",
            "replicate",
            "fit_valid",
            "strict_fit_valid",
            "repair_succeeded",
            "restricted_LR",
            "random_slope_variance",
            "gamma_0_5",
            "hessian_positive_definite",
        ]
    ]
)


print()
print(
    "=== FIVE TARGETED REPAIR REFITS ==="
)

print(
    display.to_string(
        index=False
    )
)


print()
print(
    "=== targeted-repair FROZEN COMPARISON ==="
)

print(
    "rows:",
    len(a),
)

print(
    "maximum absolute numeric difference:",
    f"{maximum_absolute_difference:.3g}",
)

print(
    "numeric mismatch columns:",
    numeric_fail,
)

print(
    "boolean mismatch columns:",
    boolean_fail,
)

print(
    "text mismatch columns:",
    text_fail,
)

print(
    "EQUIVALENT:",
    comparison_pass,
)


print()
print("=" * 80)
print(
    "ADVANCED RLRT OPERATING-SIMULATION FITTING SMOKE"
)
print("=" * 80)

print(
    "DETERMINISTIC SEEDS PASS:",
    seed_pass,
)

print(
    "FIVE targeted-repair REPAIR ROWS PASS:",
    comparison_pass,
)

print(
    "EXPECTED VALID/STRICT STATES PASS:",
    state_pass,
)

print(
    "TARGETED REPAIR DATASETS REFIT:",
    len(new),
)

print(
    "FULL 758-REPAIR RERUN:",
    False,
)

print(
    "FULL 62,000-DATASET RERUN:",
    False,
)

print(
    "ALL OPERATING-SIMULATION FITTING CHECKS PASS:",
    (
        seed_pass
        and
        comparison_pass
        and
        state_pass
    ),
)


if not (
    seed_pass
    and
    comparison_pass
    and
    state_pass
):
    raise SystemExit(1)
