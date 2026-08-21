#!/usr/bin/env python3
"""Validate AABC2 covariance influence and selected exact deletion refits.

This notebook-independent restricted-data validation imports the exact
validated advanced core. The full one-step influence ranking and selected
exact-deletion refits are compared with restricted reference outputs.

All outputs from this script belong in a restricted validation directory.
Participant identifiers are never printed to the console.
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
    fit_optimizer_panel,
    summarize_panel_metrics,
    cluster_covariance_influence,
    replace_raw_endpoint_covariance_names,
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
# ======================================================================
# Fit the two observed alternative models
# ======================================================================

OBSERVED_PANELS = {}
INFLUENCE_BASELINES = {}
influence_parts = []


for outcome in tqdm(
    TARGET_GEOMETRY_OUTCOMES,
    desc="Observed influence baselines",
    unit="outcome",
):
    data = ANALYSIS_DATA[
        outcome
    ]

    panel = fit_optimizer_panel(
        data,
        fixed_formula(outcome),
        "1+t5_mid",
        reml=True,
        stop_after_strict_pair=False,
    )

    if panel["result"] is None:
        raise RuntimeError(
            f"{outcome}: no selected "
            "observed alternative fit."
        )

    OBSERVED_PANELS[
        (
            outcome,
            "alternative",
        )
    ] = panel

    INFLUENCE_BASELINES[
        outcome
    ] = summarize_panel_metrics(
        panel,
        data,
    )

    influence = (
        cluster_covariance_influence(
            data,
            fixed_formula(outcome),
            panel["result"],
            outcome,
        )
    )

    influence[
        "influence_rank"
    ] = np.arange(
        1,
        len(influence) + 1,
    )

    influence_parts.append(
        influence
    )


cluster_influence = pd.concat(
    influence_parts,
    ignore_index=True,
)

cluster_influence = (
    replace_raw_endpoint_covariance_names(
        cluster_influence
    )
)

assert len(cluster_influence) == 1313

assert (
    cluster_influence
    .groupby("outcome")
    .size()
    .to_dict()
    ==
    {
        "global_mean_thickness": 622,
        "fluid_cognition": 691,
    }
)

cluster_influence.to_csv(
    OUT
    / "10_cluster_covariance_influence_validation.csv",
    index=False,
)


# ======================================================================
# Generic comparator
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
    atol=1e-10,
    rtol=1e-8,
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
        new[
            old.columns
        ]
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
                da = json.loads(
                    str(av)
                )
                db = json.loads(
                    str(bv)
                )

                if set(da) != set(db):
                    json_fail.append(
                        (
                            column,
                            i,
                            "keys",
                        )
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
                            (
                                column,
                                i,
                                key,
                            )
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
            (
                an.notna()
                |
                a.isna()
            ).all()
            and
            (
                bn.notna()
                |
                b.isna()
            ).all()
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
# Compare full one-step influence table
# ======================================================================

frozen10 = pd.read_csv(
    FROZEN
    / "10_cluster_covariance_influence.csv",
    low_memory=False,
)

influence_pass = compare_frames(
    "10 full one-step influence ranking",
    cluster_influence,
    frozen10,
    [
        "outcome",
        "influence_rank",
        "pid",
    ],
    atol=2e-9,
    rtol=2e-8,
)


# Verify complete ranking identity without printing PIDs.
new_rank_keys = (
    cluster_influence[
        [
            "outcome",
            "influence_rank",
            "pid",
        ]
    ]
    .astype({
        "outcome": str,
        "pid": str,
    })
    .sort_values(
        [
            "outcome",
            "influence_rank",
        ]
    )
    .reset_index(drop=True)
)

old_rank_keys = (
    frozen10[
        [
            "outcome",
            "influence_rank",
            "pid",
        ]
    ]
    .astype({
        "outcome": str,
        "pid": str,
    })
    .sort_values(
        [
            "outcome",
            "influence_rank",
        ]
    )
    .reset_index(drop=True)
)

ranking_identical = (
    new_rank_keys.equals(
        old_rank_keys
    )
)

print()
print(
    "FULL 1,313-PARTICIPANT RANKING IDENTICAL:",
    ranking_identical,
)

if not ranking_identical:
    raise AssertionError(
        "Influence ranking changed."
    )


# ======================================================================
# Private four deletion cases
# ======================================================================

smoke = pd.read_csv(
    SMOKE_PATH,
    low_memory=False,
)

smoke["pid_removed"] = (
    smoke[
        "pid_removed"
    ]
    .astype(str)
)

smoke[
    "influence_rank"
] = pd.to_numeric(
    smoke[
        "influence_rank"
    ],
    errors="raise",
).astype(int)

assert len(smoke) == 4


# Confirm the saved manifest corresponds exactly to the newly generated rank.
rank_lookup = (
    cluster_influence[
        [
            "outcome",
            "pid",
            "influence_rank",
            "influence_norm",
        ]
    ]
    .rename(
        columns={
            "pid":
                "pid_removed",

            "influence_norm":
                "new_one_step_influence_norm",
        }
    )
)

manifest_check = smoke.merge(
    rank_lookup,
    on=[
        "outcome",
        "pid_removed",
        "influence_rank",
    ],
    how="left",
    validate="one_to_one",
)

if (
    manifest_check[
        "new_one_step_influence_norm"
    ]
    .isna()
    .any()
):
    raise AssertionError(
        "Private smoke manifest no longer "
        "matches generated influence ranking."
    )

assert np.allclose(
    manifest_check[
        "one_step_influence_norm"
    ].to_numpy(float),
    manifest_check[
        "new_one_step_influence_norm"
    ].to_numpy(float),
    atol=2e-9,
    rtol=2e-8,
)


# ======================================================================
# Four exact deletion refits
# ======================================================================

summary_rows = []
audit_parts = []


for task in tqdm(
    smoke.itertuples(index=False),
    total=len(smoke),
    desc="Exact deletion smoke",
    unit="dataset",
):
    outcome = str(
        task.outcome
    )

    pid_removed = str(
        task.pid_removed
    )

    rank = int(
        task.influence_rank
    )

    data = ANALYSIS_DATA[
        outcome
    ]

    reduced = (
        data.loc[
            ~data.pid.eq(
                pid_removed
            )
        ]
        .copy()
    )

    panel = fit_optimizer_panel(
        reduced,
        fixed_formula(outcome),
        "1+t5_mid",
        reml=True,
        stop_after_strict_pair=False,
    )

    metrics = summarize_panel_metrics(
        panel,
        reduced,
    )

    baseline = INFLUENCE_BASELINES[
        outcome
    ]

    dataset_id = (
        f"{outcome}__remove_pid_"
        f"{pid_removed}"
    )

    summary_row = {
        "dataset_id":
            dataset_id,

        "outcome":
            outcome,

        "pid_removed":
            pid_removed,

        "influence_rank":
            rank,

        "one_step_influence_norm":
            float(
                task.one_step_influence_norm
            ),

        "participants_removed":
            1,

        "participants_retained":
            int(
                reduced.pid.nunique()
            ),

        "basic_panel_valid":
            panel[
                "basic_panel_valid"
            ],

        "strict_panel_valid":
            panel[
                "strict_panel_valid"
            ],

        **metrics,
    }

    for metric in (
        "random_slope_variance",
        "median_full",
        "median_gap",
        "g_0_5",
        "gamma_0_5",
    ):
        summary_row[
            f"change_in_{metric}"
        ] = (
            summary_row[
                metric
            ]
            -
            baseline[
                metric
            ]
        )

    summary_rows.append(
        summary_row
    )

    audit = (
        panel[
            "audit"
        ]
        .copy()
    )

    audit.insert(
        0,
        "dataset_id",
        dataset_id,
    )

    audit.insert(
        1,
        "outcome",
        outcome,
    )

    audit.insert(
        2,
        "pid_removed",
        pid_removed,
    )

    audit.insert(
        3,
        "influence_rank",
        rank,
    )

    audit_parts.append(
        audit
    )


deletion_summary = pd.DataFrame(
    summary_rows
)

deletion_audit = pd.concat(
    audit_parts,
    ignore_index=True,
)


# Apply final endpoint-covariance naming repair.
deletion_summary = (
    replace_raw_endpoint_covariance_names(
        deletion_summary
    )
)

deletion_audit = (
    replace_raw_endpoint_covariance_names(
        deletion_audit
    )
)


# Final output adds change_in_d_0_5 after name repair.
baseline_d = pd.Series({
    outcome: max(
        0.0,
        1
        -
        float(
            INFLUENCE_BASELINES[
                outcome
            ][
                "gamma_0_5"
            ]
        ) ** 2,
    )
    for outcome
    in TARGET_GEOMETRY_OUTCOMES
})


deletion_summary[
    "change_in_d_0_5"
] = deletion_summary.apply(
    lambda row:
        row.d_0_5
        -
        baseline_d.loc[
            row.outcome
        ],
    axis=1,
)


deletion_summary.to_csv(
    OUT
    / "10b_four_case_exact_deletion_validation.csv",
    index=False,
)

deletion_audit.to_csv(
    OUT
    / "12b_four_case_optimizer_validation.csv",
    index=False,
)


# ======================================================================
# Frozen four-case comparators
# ======================================================================

frozen10b_all = pd.read_csv(
    FROZEN
    / "10b_exact_top10_individual_deletion_refits.csv",
    low_memory=False,
)

frozen12b_all = pd.read_csv(
    FROZEN
    / "12b_exact_top10_individual_deletion_optimizer_audit.csv",
    low_memory=False,
)


selected_keys = smoke[
    [
        "outcome",
        "pid_removed",
        "influence_rank",
    ]
].copy()

selected_keys[
    "pid_removed"
] = selected_keys[
    "pid_removed"
].astype(str)


frozen10b_all[
    "pid_removed"
] = frozen10b_all[
    "pid_removed"
].astype(str)

frozen12b_all[
    "pid_removed"
] = frozen12b_all[
    "pid_removed"
].astype(str)


frozen10b = frozen10b_all.merge(
    selected_keys,
    on=[
        "outcome",
        "pid_removed",
        "influence_rank",
    ],
    how="inner",
    validate="one_to_one",
)


frozen12b = frozen12b_all.merge(
    selected_keys,
    on=[
        "outcome",
        "pid_removed",
        "influence_rank",
    ],
    how="inner",
    validate="many_to_one",
)


assert len(
    frozen10b
) == 4

assert len(
    frozen12b
) == 12


deletion_summary_pass = compare_frames(
    "10b four exact deletion summaries",
    deletion_summary,
    frozen10b,
    [
        "outcome",
        "influence_rank",
        "pid_removed",
    ],
    atol=2e-9,
    rtol=2e-8,
)


deletion_audit_pass = compare_frames(
    "12b four deletion optimizer panels",
    deletion_audit,
    frozen12b,
    [
        "outcome",
        "influence_rank",
        "pid_removed",
        "optimizer",
    ],
    atol=2e-9,
    rtol=2e-8,
)


# ======================================================================
# Public-safe status only — never print participant IDs
# ======================================================================

status = (
    deletion_summary[
        [
            "outcome",
            "influence_rank",
            "participants_retained",
            "basic_panel_valid",
            "strict_panel_valid",
            "random_slope_variance",
            "median_direct",
            "median_full",
            "median_innovation",
            "median_gap",
            "gamma_0_5",
            "d_0_5",
        ]
    ]
    .sort_values(
        [
            "outcome",
            "influence_rank",
        ]
    )
)


print()
print(
    "=== FOUR EXACT DELETION SMOKE RESULTS ==="
)

print(
    status.to_string(
        index=False
    )
)


print()
print(
    "=== OPTIMIZER PANEL COUNTS ==="
)

panel_counts = (
    deletion_audit
    .groupby(
        [
            "outcome",
            "influence_rank",
        ]
    )
    .agg(
        optimizer_rows=(
            "optimizer",
            "size",
        ),
        optimizers=(
            "optimizer",
            "nunique",
        ),
        selected_solutions=(
            "selected_panel_solution",
            lambda x:
                int(
                    pd.Series(x)
                    .astype(str)
                    .str.lower()
                    .isin(
                        ["true", "1"]
                    )
                    .sum()
                ),
        ),
    )
    .reset_index()
)

print(
    panel_counts.to_string(
        index=False
    )
)

assert (
    panel_counts.optimizer_rows
    .eq(3)
    .all()
)

assert (
    panel_counts.optimizers
    .eq(3)
    .all()
)

assert (
    panel_counts.selected_solutions
    .eq(1)
    .all()
)


print()
print("=" * 80)
print(
    "ADVANCED RLRT INFLUENCE / DELETION VALIDATION"
)
print("=" * 80)

print(
    "FULL 1,313-ROW INFLUENCE TABLE PASS:",
    influence_pass,
)

print(
    "FULL INFLUENCE RANKING IDENTICAL:",
    ranking_identical,
)

print(
    "FOUR EXACT DELETION SUMMARIES PASS:",
    deletion_summary_pass,
)

print(
    "FOUR × THREE OPTIMIZER PANELS PASS:",
    deletion_audit_pass,
)

print(
    "EXACT INDIVIDUAL DELETION REFITS PERFORMED:",
    len(deletion_summary),
)

print(
    "FULL 20-DELETION RERUN:",
    False,
)

print(
    "PARTICIPANT IDs PRINTED:",
    False,
)

print(
    "ALL INFLUENCE FITTING CHECKS PASS:",
    (
        influence_pass
        and
        ranking_identical
        and
        deletion_summary_pass
        and
        deletion_audit_pass
    ),
)


if not (
    influence_pass
    and
    ranking_identical
    and
    deletion_summary_pass
    and
    deletion_audit_pass
):
    raise SystemExit(1)
