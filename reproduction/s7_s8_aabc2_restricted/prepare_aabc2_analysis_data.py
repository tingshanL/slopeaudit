#!/usr/bin/env python3
"""Prepare and validate the restricted AABC2 analysis cohorts.

This public entry point reconstructs the frozen structural, ASL, and
cognition cohorts from explicitly supplied restricted inputs, then
prepares the six outcome-specific analysis datasets in memory.

Only aggregate cohort summaries are written. Participant identifiers,
event identifiers, exact visit schedules, prepared participant-level
analysis tables, and other restricted row-level data are never exported.

Restricted source data are not distributed with this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import aabc2_common as common


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
    help="Fresh directory for aggregate cohort-validation outputs.",
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


# ------------------------------------------------------------------
# Reconstruct the frozen restricted cohorts.
# ------------------------------------------------------------------

raw = common.build_raw_cohorts(
    args.input_config
)

OUTCOMES, ANALYSIS_DATA, analysis_cohort_summary = (
    common.build_analysis_data(raw)
)

raw_cohort_summary = (
    raw["cohort_summary"]
    .copy()
)


# ------------------------------------------------------------------
# Public-release safety checks.
# ------------------------------------------------------------------

EXPECTED_RAW_COLUMNS = {
    "cohort",
    "participants",
    "observations",
}

EXPECTED_ANALYSIS_COLUMNS = {
    "outcome",
    "label",
    "modality",
    "participants",
    "observations",
    "median_scans",
    "median_followup_years",
}

if set(raw_cohort_summary.columns) != EXPECTED_RAW_COLUMNS:
    raise RuntimeError(
        "Unexpected raw cohort-summary schema: "
        f"{list(raw_cohort_summary.columns)}"
    )

if set(analysis_cohort_summary.columns) != EXPECTED_ANALYSIS_COLUMNS:
    raise RuntimeError(
        "Unexpected analysis cohort-summary schema: "
        f"{list(analysis_cohort_summary.columns)}"
    )

if len(OUTCOMES) != 6:
    raise RuntimeError(
        f"Expected six outcome specifications; found {len(OUTCOMES)}."
    )

if set(ANALYSIS_DATA) != set(OUTCOMES):
    raise RuntimeError(
        "Prepared analysis-data keys do not match outcome specifications."
    )

for key, data in ANALYSIS_DATA.items():
    if "pid" not in data.columns:
        raise RuntimeError(
            f"{key}: prepared restricted analysis table lacks pid."
        )

    if data["pid"].isna().any():
        raise RuntimeError(
            f"{key}: missing participant identifier after preparation."
        )

    if data.groupby("pid").size().min() < 2:
        raise RuntimeError(
            f"{key}: participant with fewer than two observations."
        )

    span = (
        data.groupby("pid")["t5"].max()
        - data.groupby("pid")["t5"].min()
    )

    if not (span > 0).all():
        raise RuntimeError(
            f"{key}: participant with non-positive retained time span."
        )


# ------------------------------------------------------------------
# Aggregate exports only.
# ------------------------------------------------------------------

raw_path = (
    OUTPUT_DIR
    / "00_raw_cohort_summary.csv"
)

analysis_path = (
    OUTPUT_DIR
    / "00_analysis_cohort_summary.csv"
)

raw_cohort_summary.to_csv(
    raw_path,
    index=False,
)

analysis_cohort_summary.to_csv(
    analysis_path,
    index=False,
)


print("Restricted AABC2 cohorts reconstructed successfully.")
print()
print("=== RAW COHORT SUMMARY ===")
print(
    raw_cohort_summary.to_string(
        index=False
    )
)
print()
print("=== SIX-OUTCOME ANALYSIS COHORT SUMMARY ===")
print(
    analysis_cohort_summary.to_string(
        index=False
    )
)
print()
print("PARTICIPANT-LEVEL TABLES EXPORTED: 0")
print(f"WROTE: {raw_path}")
print(f"WROTE: {analysis_path}")
