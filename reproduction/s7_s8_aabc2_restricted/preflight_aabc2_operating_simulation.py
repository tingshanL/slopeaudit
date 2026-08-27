#!/usr/bin/env python3
"""Preflight the completed AABC2 advanced operating simulation.

This performs structural and overlay checks on the validated 46-cell,
62,000-replicate operating-study state without refitting any dataset. It also
writes a five-case targeted-repair smoke manifest for optional fitting
validation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument("--reference-results",required=True)
parser.add_argument("--smoke-manifest-out",required=True)
args=parser.parse_args()

ROOT=Path(args.reference_results).expanduser().resolve()
OUT=Path(args.smoke_manifest_out).expanduser().resolve()
if not ROOT.is_dir(): raise FileNotFoundError(ROOT)
if OUT.exists(): raise FileExistsError(f"Output already exists: {OUT}")
OUT.parent.mkdir(parents=True,exist_ok=True)

# ================================================================
# Helpers
# ================================================================

def as_bool(series):
    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .fillna(False)
        .astype(bool)
    )


def first_existing(frame, candidates, label):
    for c in candidates:
        if c in frame.columns:
            return c

    raise KeyError(
        f"{label}: none of {candidates} found. "
        f"Columns={list(frame.columns)}"
    )


def normalize_historical_geometry_names(frame):
    frame = frame.copy()

    renames = {}

    if (
        "g_0_5" in frame.columns
        and
        "cov_L0_L5" not in frame.columns
    ):
        renames["g_0_5"] = "cov_L0_L5"

    if (
        "oracle_g_0_5" in frame.columns
        and
        "oracle_cov_L0_L5" not in frame.columns
    ):
        renames[
            "oracle_g_0_5"
        ] = "oracle_cov_L0_L5"

    if renames:
        frame = frame.rename(
            columns=renames
        )

    return frame


# ================================================================
# Load final/frozen state
# ================================================================

f18 = pd.read_csv(
    ROOT / "18_simulation_cell_manifest.csv",
    low_memory=False,
)

f19 = pd.read_csv(
    ROOT / "19_operating_simulation_replicates_repaired.csv",
    low_memory=False,
)

f19a = pd.read_csv(
    ROOT / "19a_targeted_758_simulation_repairs.csv",
    low_memory=False,
)

f20 = pd.read_csv(
    ROOT / "20_simulation_LR_calibration_repaired.csv",
    low_memory=False,
)

f21 = pd.read_csv(
    ROOT / "21_operating_characteristics_strict_repaired.csv",
    low_memory=False,
)

f22 = pd.read_csv(
    ROOT / "22_operating_metric_quantiles_strict_repaired.csv",
    low_memory=False,
)

f23 = pd.read_csv(
    ROOT / "23_true_zero_spurious_predictability_repaired.csv",
    low_memory=False,
)

f23b = pd.read_csv(
    ROOT / "23b_simulation_cell_validity_repaired.csv",
    low_memory=False,
)

f24 = pd.read_csv(
    ROOT / "24_completion_checklist.csv",
    low_memory=False,
)


# ================================================================
# 1. Global structure
# ================================================================

assert len(f18) == 46
assert len(f19) == 62000
assert len(f19a) == 758
assert len(f20) == 4
assert len(f21) == 46
assert len(f22) == 690
assert len(f23) == 4
assert len(f23b) == 46
assert len(f24) == 8

assert not f18["cell_id"].duplicated().any()

assert not f19.duplicated(
    ["cell_id", "replicate"]
).any()

assert not f19a.duplicated(
    ["cell_id", "replicate"]
).any()


f18["requested_reps"] = pd.to_numeric(
    f18["requested_reps"],
    errors="raise",
).astype(int)

assert int(
    f18["requested_reps"].sum()
) == 62000


actual_counts = (
    f19.groupby("cell_id")
    ["replicate"]
    .nunique()
    .rename("actual_reps")
)

expected_counts = (
    f18.set_index("cell_id")
    ["requested_reps"]
)

counts = pd.concat(
    [
        expected_counts,
        actual_counts,
    ],
    axis=1,
)

assert counts.notna().all().all()

assert counts[
    "requested_reps"
].eq(
    counts[
        "actual_reps"
    ]
).all()


# ================================================================
# 2. Final repair-state flags
# ================================================================

required_final_flags = (
    "fit_valid",
    "strict_fit_valid",
    "repair_attempted",
    "repair_succeeded",
)

for c in required_final_flags:
    if c not in f19.columns:
        raise KeyError(
            f"Final 19 missing required column: {c}"
        )

    f19[c] = as_bool(
        f19[c]
    )


assert int(
    f19["repair_attempted"].sum()
) == 758


# Use the final 19 state as the source of truth for repaired cases.
repair_keys = f19a[
    [
        "cell_id",
        "replicate",
    ]
].copy()

repaired_final = repair_keys.merge(
    f19,
    on=[
        "cell_id",
        "replicate",
    ],
    how="left",
    validate="one_to_one",
)

assert len(repaired_final) == 758

assert repaired_final[
    "repair_attempted"
].all()


strict_valid_repairs = int(
    repaired_final[
        "strict_fit_valid"
    ].sum()
)

strict_invalid_repairs = int(
    (
        ~repaired_final[
            "strict_fit_valid"
        ]
    ).sum()
)

fit_invalid_repairs = int(
    (
        ~repaired_final[
            "fit_valid"
        ]
    ).sum()
)

final_strict_invalid = int(
    (
        ~f19[
            "strict_fit_valid"
        ]
    ).sum()
)


# ================================================================
# 3. Exact 758-row repair overlay
# ================================================================

repair_cmp = (
    normalize_historical_geometry_names(
        f19a
    )
)

final_cmp = (
    normalize_historical_geometry_names(
        f19
    )
)


shared_columns = [
    c
    for c in repair_cmp.columns
    if c in final_cmp.columns
]


overlay = repair_cmp.merge(
    final_cmp[
        [
            "cell_id",
            "replicate",
        ]
        +
        [
            c
            for c in shared_columns
            if c not in (
                "cell_id",
                "replicate",
            )
        ]
    ],
    on=[
        "cell_id",
        "replicate",
    ],
    how="inner",
    suffixes=(
        "__repair",
        "__final",
    ),
    validate="one_to_one",
)

assert len(overlay) == 758


overlay_fail = []
overlay_max_abs = 0.0

for c in shared_columns:
    if c in (
        "cell_id",
        "replicate",
    ):
        continue

    ca = c + "__repair"
    cb = c + "__final"

    if (
        ca not in overlay.columns
        or
        cb not in overlay.columns
    ):
        continue

    a = overlay[ca]
    b = overlay[cb]

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
            overlay_max_abs = max(
                overlay_max_abs,
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

        equal = np.allclose(
            av,
            bv,
            atol=1e-12,
            rtol=1e-10,
            equal_nan=True,
        )

    else:
        equal = (
            a.astype("string")
            .fillna("")
            .equals(
                b.astype("string")
                .fillna("")
            )
        )

    if not equal:
        overlay_fail.append(c)


assert not overlay_fail, (
    "Repair overlay mismatch columns: "
    f"{overlay_fail}"
)


# ================================================================
# 4. Final naming convention
# ================================================================

for required in (
    "cov_L0_L5",
    "oracle_cov_L0_L5",
    "d_0_5",
):
    if required not in f19.columns:
        raise KeyError(
            f"Final 19 missing {required}"
        )

assert "g_0_5" not in f19.columns
assert "oracle_g_0_5" not in f19.columns


# ================================================================
# 5. 23b exact deterministic reconstruction from 21
# ================================================================

validity_columns = [
    "cell_id",
    "requested_reps",
    "completed_reps",
    "strict_valid_reps",
    "invalid_fit_denominator",
    "strict_validity_rate",
    "inference_licensed_at_95pct_validity",
]

missing = [
    c for c in validity_columns
    if (
        c not in f21.columns
        or
        c not in f23b.columns
    )
]

if missing:
    raise KeyError(
        f"21/23b missing validity columns: "
        f"{missing}"
    )


rebuilt23b = f21[
    validity_columns
].copy()

a = (
    rebuilt23b
    .sort_values(
        "cell_id",
        kind="stable",
    )
    .reset_index(drop=True)
)

b = (
    f23b
    .sort_values(
        "cell_id",
        kind="stable",
    )
    .reset_index(drop=True)
)

assert list(a.columns) == list(
    b.columns
)


max23b = 0.0
fail23b = []

for c in a.columns:
    an = pd.to_numeric(
        a[c],
        errors="coerce",
    )

    bn = pd.to_numeric(
        b[c],
        errors="coerce",
    )

    numeric_candidate = (
        (
            an.notna()
            |
            a[c].isna()
        ).all()
        and
        (
            bn.notna()
            |
            b[c].isna()
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
            max23b = max(
                max23b,
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

        equal = np.allclose(
            av,
            bv,
            atol=1e-12,
            rtol=1e-10,
            equal_nan=True,
        )

    else:
        equal = (
            a[c]
            .astype("string")
            .fillna("")
            .equals(
                b[c]
                .astype("string")
                .fillna("")
            )
        )

    if not equal:
        fail23b.append(c)


assert not fail23b, fail23b


licensed = as_bool(
    f23b[
        "inference_licensed_at_95pct_validity"
    ]
)

assert licensed.all()


# ================================================================
# 6. Completion checklist
# ================================================================

completion_col = first_existing(
    f24,
    (
        "complete",
        "completed",
        "passed",
    ),
    "24 completion checklist",
)

completion_flags = as_bool(
    f24[
        completion_col
    ]
)

assert completion_flags.all()


# ================================================================
# 7. Attach cell-family metadata
# ================================================================

family_col = first_existing(
    f18,
    (
        "cell_family",
        "family",
        "simulation_family",
    ),
    "18 simulation manifest",
)


metadata_candidates = (
    "cell_id",
    family_col,
    "schedule",
    "q1",
    "q0",
    "rho",
    "misspecification",
)

# Final 19 already carries much of the simulation-cell metadata.
# Merge only metadata fields that are genuinely absent, otherwise
# pandas would create *_x / *_y duplicates such as cell_family_x/y.
metadata_columns = [
    "cell_id"
] + [
    c
    for c in metadata_candidates
    if (
        c != "cell_id"
        and
        c in f18.columns
        and
        c not in repaired_final.columns
    )
]


if len(metadata_columns) > 1:
    repaired_final = repaired_final.merge(
        f18[
            metadata_columns
        ],
        on="cell_id",
        how="left",
        validate="many_to_one",
    )

assert family_col in repaired_final.columns

assert repaired_final[
    family_col
].notna().all()


repair_family = (
    repaired_final
    .groupby(
        family_col,
        dropna=False,
    )
    .agg(
        repairs=(
            "replicate",
            "size",
        ),
        fit_valid=(
            "fit_valid",
            "sum",
        ),
        strict_valid=(
            "strict_fit_valid",
            "sum",
        ),
        strict_invalid=(
            "strict_fit_valid",
            lambda x:
                int(
                    (~x).sum()
                ),
        ),
        repair_succeeded=(
            "repair_succeeded",
            "sum",
        ),
    )
    .reset_index()
)


# ================================================================
# 8. Private fitting-smoke manifest
#
# One repaired case per represented simulation family, preferring
# strict-valid cases, plus one explicit strict-invalid repaired case.
# If a fully fit-invalid repaired case exists, include one too.
# ================================================================

chosen = []


def append_case(row, role):
    record = row.copy()
    record[
        "smoke_role"
    ] = role

    chosen.append(record)


for family, group in repaired_final.groupby(
    family_col,
    sort=True,
):
    strict = group.loc[
        group[
            "strict_fit_valid"
        ]
    ]

    source = (
        strict
        if not strict.empty
        else group
    )

    row = (
        source
        .sort_values(
            [
                "cell_id",
                "replicate",
            ],
            kind="stable",
        )
        .iloc[0]
    )

    append_case(
        row,
        (
            f"family_repair__{family}"
        ),
    )


strict_invalid = repaired_final.loc[
    ~repaired_final[
        "strict_fit_valid"
    ]
]

if not strict_invalid.empty:
    row = (
        strict_invalid
        .sort_values(
            [
                "cell_id",
                "replicate",
            ],
            kind="stable",
        )
        .iloc[0]
    )

    append_case(
        row,
        "strict_invalid_after_repair",
    )


fit_invalid = repaired_final.loc[
    ~repaired_final[
        "fit_valid"
    ]
]

if not fit_invalid.empty:
    row = (
        fit_invalid
        .sort_values(
            [
                "cell_id",
                "replicate",
            ],
            kind="stable",
        )
        .iloc[0]
    )

    append_case(
        row,
        "fit_invalid_after_repair",
    )


smoke = pd.DataFrame(
    chosen
)

smoke = (
    smoke
    .drop_duplicates(
        [
            "cell_id",
            "replicate",
        ],
        keep="first",
    )
    .sort_values(
        [
            "smoke_role",
            "cell_id",
            "replicate",
        ],
        kind="stable",
    )
    .reset_index(drop=True)
)


smoke_columns = [
    "smoke_role",
    "cell_id",
    "replicate",
]

for c in (
    "seed",
    family_col,
    "schedule",
    "q1",
    "q0",
    "rho",
    "misspecification",
    "fit_valid",
    "strict_fit_valid",
    "repair_succeeded",
    "restricted_LR",
):
    if c in smoke.columns:
        smoke_columns.append(c)


smoke = smoke[
    smoke_columns
]


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

smoke.to_csv(
    OUT,
    index=False,
)


# ================================================================
# Report
# ================================================================

print(
    "=== OPERATING SIMULATION STRUCTURE ==="
)

print(
    "simulation cells:",
    len(f18),
)

print(
    "cell families:"
)

print(
    f18.groupby(
        family_col
    )
    .size()
    .rename("cells")
    .to_string()
)

print(
    "requested simulation rows:",
    int(
        f18[
            "requested_reps"
        ].sum()
    ),
)

print(
    "final simulation rows:",
    len(f19),
)

print(
    "all 46 per-cell requested counts exact:",
    True,
)


print()
print(
    "=== TARGETED REPAIR STATE ==="
)

print(
    "targeted repair rows:",
    len(f19a),
)

print(
    "strict-valid repaired rows:",
    strict_valid_repairs,
)

print(
    "strict-invalid repaired rows:",
    strict_invalid_repairs,
)

print(
    "fit-invalid repaired rows:",
    fit_invalid_repairs,
)

print(
    "final strict-invalid rows:",
    final_strict_invalid,
)

print(
    "repair-attempted rows in final 19:",
    int(
        f19[
            "repair_attempted"
        ].sum()
    ),
)


print()
print(
    "=== REPAIR DISTRIBUTION BY FAMILY ==="
)

print(
    repair_family.to_string(
        index=False
    )
)


print()
print(
    "=== 758-ROW REPAIR OVERLAY ==="
)

print(
    "rows matched:",
    len(overlay),
)

print(
    "maximum absolute numeric difference:",
    f"{overlay_max_abs:.3g}",
)

print(
    "mismatch columns:",
    overlay_fail,
)

print(
    "EQUIVALENT:",
    True,
)


print()
print(
    "=== FINAL DERIVED OUTPUT STRUCTURE ==="
)

print(
    "20 LR calibration rows:",
    len(f20),
)

print(
    "21 operating-characteristic rows:",
    len(f21),
)

print(
    "22 quantile rows:",
    len(f22),
)

print(
    "23 true-zero rows:",
    len(f23),
)

print(
    "23b validity rows:",
    len(f23b),
)

print(
    "23b reconstruction maximum difference:",
    f"{max23b:.3g}",
)

print(
    "all 46 cells inference-licensed at >=95% strict validity:",
    bool(
        licensed.all()
    ),
)

print(
    "24 completion checklist all complete:",
    bool(
        completion_flags.all()
    ),
)


print()
print(
    "=== REPAIR FITTING SMOKE MANIFEST ==="
)

display_columns = [
    c
    for c in (
        "smoke_role",
        "cell_id",
        "replicate",
        family_col,
        "schedule",
        "q1",
        "q0",
        "rho",
        "misspecification",
        "fit_valid",
        "strict_fit_valid",
        "repair_succeeded",
    )
    if c in smoke.columns
]

print(
    smoke[
        display_columns
    ].to_string(
        index=False
    )
)


print()
print(
    "WROTE:",
    OUT.resolve(),
)

print(
    "REFITS PERFORMED:",
    0,
)

print(
    "FULL 62,000-DATASET RERUN:",
    False,
)

print(
    "ALL OPERATING-SIMULATION PREFLIGHT CHECKS PASS:",
    True,
)
