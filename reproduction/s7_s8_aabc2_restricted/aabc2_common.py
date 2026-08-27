"""Shared restricted-data loader for the AMPPS AABC analyses.

This module reconstructs the AABC analysis cohorts used by the
six-outcome reanalysis and subsequent restricted-data diagnostics.

It provides restricted-input validation, cohort construction, prespecified
transformations and time variables, six-outcome data preparation, and common
fixed formulas. Model fitting, bootstrap execution, participant-level exports,
filesystem discovery, and machine-specific path audits are handled elsewhere.

Restricted source data remain local and must not be committed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Prespecified constants from the six-outcome source notebook
# ---------------------------------------------------------------------------

FIVE_YEAR_DAYS = 365.25 * 5
AGE_REFERENCE_YEARS = 60.0
AGE_SCALE_YEARS = 10.0
MIN_THICKNESS_COMPLETENESS = 0.95

EXPECTED_STRUCT_ROSTER_N = 875
EXPECTED_FINAL_STRUCT_N = 622
EXPECTED_STRUCT_VISITS = 1629

EXPECTED_ASL_N = 802
EXPECTED_ASL_VISITS = 2047

EXPECTED_COGNITION_N = 691
EXPECTED_COGNITION_VISITS = 1805

EXPECTED_CORTICAL_ROIS = 360


REQUIRED_CONFIG_KEYS = (
    "structural_idps",
    "asl_idps_zip",
    "idp_info_zip",
    "asl_audit_xlsx",
    "struct_roster_csv",
    "subjects_csv",
)

OPTIONAL_CONFIG_KEYS = (
    "final622_ids_csv",
)


# ---------------------------------------------------------------------------
# Explicit input contract
# ---------------------------------------------------------------------------

def _resolved_path(value: Any, *, key: str, required: bool = True) -> Path | None:
    """Resolve and validate one path from the restricted-input configuration."""

    if value is None or str(value).strip() == "":
        if required:
            raise ValueError(f"Required restricted-input path {key!r} is missing.")
        return None

    path = Path(str(value)).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Restricted-input path for {key!r} does not exist: {path}"
        )

    return path


def load_input_config(config_path: str | Path) -> dict[str, Path | None]:
    """Load and validate the explicit restricted-input JSON contract."""

    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Input configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise TypeError("Restricted-input configuration must be a JSON object.")

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in raw]
    if missing_keys:
        raise KeyError(
            "Restricted-input configuration is missing required keys: "
            + ", ".join(missing_keys)
        )

    config: dict[str, Path | None] = {}

    for key in REQUIRED_CONFIG_KEYS:
        config[key] = _resolved_path(raw[key], key=key, required=True)

    for key in OPTIONAL_CONFIG_KEYS:
        config[key] = _resolved_path(
            raw.get(key),
            key=key,
            required=False,
        )

    structural = config["structural_idps"]
    assert structural is not None

    if not structural.is_dir() and not structural.is_file():
        raise ValueError(
            "'structural_idps' must be either the extracted structural-IDP "
            "directory or its ZIP archive."
        )

    if structural.is_file() and structural.suffix.lower() != ".zip":
        raise ValueError(
            "When 'structural_idps' is a file, it must be a ZIP archive."
        )

    for key in ("asl_idps_zip", "idp_info_zip"):
        path = config[key]
        assert path is not None
        if not path.is_file() or path.suffix.lower() != ".zip":
            raise ValueError(f"{key!r} must point to a ZIP archive: {path}")

    for key in ("asl_audit_xlsx", "struct_roster_csv", "subjects_csv"):
        path = config[key]
        assert path is not None
        if not path.is_file():
            raise ValueError(f"{key!r} must point to a file: {path}")

    final622 = config["final622_ids_csv"]
    if final622 is not None and not final622.is_file():
        raise ValueError(
            "'final622_ids_csv', when supplied, must point to a file."
        )

    return config


# ---------------------------------------------------------------------------
# Archive/table helpers
# ---------------------------------------------------------------------------

def read_zip_csv(
    zip_path: str | Path,
    suffix: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read the unique non-__MACOSX CSV member ending in ``suffix``."""

    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        hits = [
            name
            for name in archive.namelist()
            if name.endswith(suffix) and not name.startswith("__MACOSX/")
        ]

        if len(hits) != 1:
            raise ValueError(
                f"Expected exactly one member ending {suffix!r} in "
                f"{zip_path.name}; found {hits}"
            )

        with archive.open(hits[0]) as handle:
            return pd.read_csv(handle, **kwargs)


def read_structural_csv(
    structural_source: str | Path,
    filename: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a structural-IDP CSV from an extracted directory or ZIP."""

    structural_source = Path(structural_source)

    if structural_source.is_dir():
        path = structural_source / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Structural-IDP file not found: {path}"
            )

        return pd.read_csv(path, **kwargs)

    return read_zip_csv(structural_source, filename, **kwargs)


def canonical_event_id(values: pd.Series) -> pd.Series:
    """Canonicalize structural IDP event identifiers exactly as frozen."""

    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"_MR$", "", regex=True, case=False)
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


# ---------------------------------------------------------------------------
# Frozen raw-cohort reconstruction
# ---------------------------------------------------------------------------

def build_raw_cohorts(
    config_path: str | Path,
) -> dict[str, Any]:
    """Reconstruct the frozen structural, ASL, and cognition cohorts."""

    config = load_input_config(config_path)

    structural_source = config["structural_idps"]
    asl_zip = config["asl_idps_zip"]
    info_zip = config["idp_info_zip"]
    asl_audit = config["asl_audit_xlsx"]
    struct_roster_path = config["struct_roster_csv"]
    final622_path = config["final622_ids_csv"]
    subjects_path = config["subjects_csv"]

    assert structural_source is not None
    assert asl_zip is not None
    assert info_zip is not None
    assert asl_audit is not None
    assert struct_roster_path is not None
    assert subjects_path is not None

    # ------------------------------------------------------------------
    # Frozen structural cohort
    # ------------------------------------------------------------------

    struct_roster = pd.read_csv(struct_roster_path, low_memory=False)

    _require_columns(
        struct_roster,
        [
            "participant_id",
            "in_final_622",
            "anchor_age",
            "anchor_days_from_V1",
            "all_valid_id_events",
        ],
        label="structural frozen roster",
    )

    struct_roster["participant_id"] = (
        struct_roster["participant_id"].astype(str)
    )

    if struct_roster["participant_id"].nunique() != EXPECTED_STRUCT_ROSTER_N:
        raise RuntimeError(
            "Frozen structural roster mismatch: expected "
            f"{EXPECTED_STRUCT_ROSTER_N} unique participants, found "
            f"{struct_roster['participant_id'].nunique()}."
        )

    final_flag_n = int(struct_roster["in_final_622"].sum())

    if final_flag_n != EXPECTED_FINAL_STRUCT_N:
        raise RuntimeError(
            "Frozen structural cohort mismatch: expected "
            f"{EXPECTED_FINAL_STRUCT_N} flagged participants, found "
            f"{final_flag_n}."
        )

    frozen_roster_ids = set(
        struct_roster.loc[
            struct_roster["in_final_622"].eq(1),
            "participant_id",
        ]
    )

    if final622_path is not None:
        final622 = pd.read_csv(final622_path)

        _require_columns(
            final622,
            ["participant_id"],
            label="optional final-622 ID cross-check",
        )

        frozen_file_ids = set(
            final622["participant_id"].astype(str)
        )

        if frozen_file_ids != frozen_roster_ids:
            raise RuntimeError(
                "Optional final-622 ID file does not match the frozen "
                "structural-roster membership."
            )

    subject_columns = [
        "id_event",
        "study",
        "days_from_V1",
        "age_open",
        "sex",
        "site",
        "scanner",
        "tlbx_cfc_uncorrected_standard_score",
    ]

    subject_events = pd.read_csv(
        subjects_path,
        usecols=subject_columns,
        low_memory=False,
    )

    subject_events["id_event"] = (
        subject_events["id_event"].astype(str).str.strip()
    )
    subject_events["id"] = subject_events["id_event"].str.replace(
        r"_[^_]+$",
        "",
        regex=True,
    )

    if not subject_events["id_event"].is_unique:
        raise RuntimeError(
            "Expected subject-event identifiers to be unique."
        )

    valid_events = (
        struct_roster[
            [
                "participant_id",
                "in_final_622",
                "anchor_age",
                "anchor_days_from_V1",
                "all_valid_id_events",
            ]
        ]
        .assign(
            id_event=lambda frame: (
                frame["all_valid_id_events"].astype(str).str.split(";")
            )
        )
        .explode("id_event", ignore_index=True)
        .drop(columns="all_valid_id_events")
    )

    struct_visits = valid_events.merge(
        subject_events[
            [
                "id_event",
                "study",
                "days_from_V1",
                "age_open",
                "sex",
                "site",
                "scanner",
            ]
        ],
        on="id_event",
        how="left",
        validate="one_to_one",
    )

    struct_visits = struct_visits.loc[
        struct_visits["in_final_622"].eq(1)
    ].copy()

    if (
        struct_visits["participant_id"].nunique()
        != EXPECTED_FINAL_STRUCT_N
        or len(struct_visits) != EXPECTED_STRUCT_VISITS
    ):
        raise RuntimeError(
            "Frozen structural visit cohort mismatch: expected "
            f"{EXPECTED_FINAL_STRUCT_N} participants / "
            f"{EXPECTED_STRUCT_VISITS} observations, found "
            f"{struct_visits['participant_id'].nunique()} / "
            f"{len(struct_visits)}."
        )

    aseg = read_structural_csv(
        structural_source,
        "asegstats.csv",
        low_memory=False,
    )
    thick = read_structural_csv(
        structural_source,
        "Cortical_Areal_Thicknesses.csv",
        low_memory=False,
    )
    area = read_structural_csv(
        structural_source,
        "Cortical_Areal_Surface_Areas.csv",
        low_memory=False,
    )

    for frame in (aseg, thick, area):
        key = frame.columns[0]
        frame.rename(columns={key: "id_event"}, inplace=True)
        frame["id_event"] = canonical_event_id(frame["id_event"])

    _require_columns(
        aseg,
        [
            "id_event",
            "FS_L_Lateral-Ventricle",
            "FS_R_Lateral-Ventricle",
            "FS_L_Hippocampus",
            "FS_R_Hippocampus",
            "EstimatedTotalIntraCranialVol",
        ],
        label="asegstats.csv",
    )

    parcel_fields = [
        column
        for column in thick.columns
        if column != "id_event" and column in set(area.columns)
    ]

    if len(parcel_fields) != EXPECTED_CORTICAL_ROIS:
        raise RuntimeError(
            f"Expected {EXPECTED_CORTICAL_ROIS} HCP-MMP parcels; "
            f"found {len(parcel_fields)}."
        )

    struct_idps = aseg[
        [
            "id_event",
            "FS_L_Lateral-Ventricle",
            "FS_R_Lateral-Ventricle",
            "FS_L_Hippocampus",
            "FS_R_Hippocampus",
            "EstimatedTotalIntraCranialVol",
        ]
    ].merge(
        thick[["id_event"] + parcel_fields],
        on="id_event",
        validate="one_to_one",
        suffixes=("", "__thick"),
    )

    struct_idps = struct_idps.rename(
        columns={
            column: f"thick__{column}"
            for column in parcel_fields
        }
    )

    area_part = area[
        ["id_event"] + parcel_fields
    ].rename(
        columns={
            column: f"area__{column}"
            for column in parcel_fields
        }
    )

    struct_idps = struct_idps.merge(
        area_part,
        on="id_event",
        validate="one_to_one",
    )

    struct_visits = struct_visits.merge(
        struct_idps,
        on="id_event",
        how="left",
        validate="one_to_one",
    )

    struct_visits["ventricles_raw"] = struct_visits[
        [
            "FS_L_Lateral-Ventricle",
            "FS_R_Lateral-Ventricle",
        ]
    ].sum(axis=1, min_count=2)

    struct_visits["hippocampus_raw"] = struct_visits[
        [
            "FS_L_Hippocampus",
            "FS_R_Hippocampus",
        ]
    ].sum(axis=1, min_count=2)

    thickness_values = struct_visits[
        [f"thick__{column}" for column in parcel_fields]
    ].to_numpy(float)

    area_values = struct_visits[
        [f"area__{column}" for column in parcel_fields]
    ].to_numpy(float)

    valid = (
        np.isfinite(thickness_values)
        & np.isfinite(area_values)
        & (area_values > 0)
    )

    completeness = valid.mean(axis=1)

    denominator = np.where(
        valid,
        area_values,
        0,
    ).sum(axis=1)

    struct_visits["thickness_raw"] = np.divide(
        np.where(
            valid,
            thickness_values * area_values,
            0,
        ).sum(axis=1),
        denominator,
        out=np.full(len(struct_visits), np.nan),
        where=denominator > 0,
    )

    struct_visits.loc[
        completeness < MIN_THICKNESS_COMPLETENESS,
        "thickness_raw",
    ] = np.nan

    struct_visits = struct_visits.sort_values(
        ["participant_id", "days_from_V1", "id_event"]
    ).reset_index(drop=True)

    baseline_icv = (
        struct_visits
        .dropna(subset=["EstimatedTotalIntraCranialVol"])
        .groupby("participant_id", as_index=False)
        .head(1)[
            [
                "participant_id",
                "EstimatedTotalIntraCranialVol",
            ]
        ]
        .rename(
            columns={
                "EstimatedTotalIntraCranialVol": "baseline_icv"
            }
        )
    )

    icv_mean = baseline_icv["baseline_icv"].mean()
    icv_sd = baseline_icv["baseline_icv"].std(ddof=1)

    baseline_icv["baseline_icv_z"] = (
        baseline_icv["baseline_icv"] - icv_mean
    ) / icv_sd

    struct_visits = struct_visits.merge(
        baseline_icv,
        on="participant_id",
        how="left",
        validate="many_to_one",
    )

    structural_transform_rows = []

    for raw_column, outcome, transform, direction in [
        (
            "ventricles_raw",
            "lateral_ventricles",
            "log1p",
            -1.0,
        ),
        (
            "hippocampus_raw",
            "hippocampus",
            "raw",
            1.0,
        ),
        (
            "thickness_raw",
            "global_mean_thickness",
            "raw",
            1.0,
        ),
    ]:
        values = (
            np.log1p(struct_visits[raw_column])
            if transform == "log1p"
            else struct_visits[raw_column].astype(float)
        )

        baseline = (
            struct_visits[
                ["participant_id", "days_from_V1"]
            ]
            .assign(value=values)
            .dropna(subset=["value"])
            .groupby("participant_id", as_index=False)
            .head(1)["value"]
        )

        center = baseline.mean()
        scale = baseline.std(ddof=1)

        struct_visits[outcome] = (
            direction * (values - center) / scale
        )

        structural_transform_rows.append(
            {
                "outcome": outcome,
                "transform": transform,
                "direction": direction,
                "center": center,
                "scale": scale,
            }
        )

    struct_visits["t5"] = (
        pd.to_numeric(struct_visits["days_from_V1"])
        - pd.to_numeric(struct_visits["anchor_days_from_V1"])
    ) / FIVE_YEAR_DAYS

    struct_visits["t5_mid"] = struct_visits["t5"] - 0.5

    struct_visits["age0"] = pd.to_numeric(
        struct_visits["anchor_age"],
        errors="coerce",
    )

    struct_visits["age0_c"] = (
        struct_visits["age0"] - AGE_REFERENCE_YEARS
    ) / AGE_SCALE_YEARS

    struct_visits["age0_sq_c"] = (
        struct_visits["age0_c"] ** 2
    )

    # ------------------------------------------------------------------
    # Frozen ASL cohort
    # ------------------------------------------------------------------

    asl_roster = pd.read_excel(
        asl_audit,
        sheet_name="Participant Roster",
        header=2,
    )
    session_qc = pd.read_excel(
        asl_audit,
        sheet_name="Session QC",
        header=2,
    )

    _require_columns(
        asl_roster,
        [
            "id",
            "anchor_day",
            "anchor_age",
            "anchor_site",
            "anchor_scanner",
            "primary_score_joint_ge2_repeated_cognition",
        ],
        label="ASL audit participant roster",
    )

    _require_columns(
        session_qc,
        [
            "id_event",
            "Measure",
            "qc_pass",
            "joint_qc_pass",
        ],
        label="ASL audit session QC",
    )

    cbf = read_zip_csv(
        asl_zip,
        "PVEc_ASL_CBF.csv",
        low_memory=False,
    )
    att = read_zip_csv(
        asl_zip,
        "PVEc_ASL_ATT.csv",
        low_memory=False,
    )
    atlas = read_zip_csv(
        info_zip,
        "AreaNamesAndIndicesAndNetworks.csv",
    )

    for frame in (cbf, att):
        frame.rename(
            columns={frame.columns[0]: "id_event"},
            inplace=True,
        )
        frame["id_event"] = (
            frame["id_event"].astype(str).str.strip()
        )

    _require_columns(
        atlas,
        ["Brain Area", "Ji Network Label"],
        label="ASL atlas",
    )

    atlas["Brain Area"] = (
        atlas["Brain Area"].astype(str).str.strip()
    )

    cortical_rois = atlas.loc[
        atlas["Ji Network Label"].notna(),
        "Brain Area",
    ].tolist()

    if len(cortical_rois) != EXPECTED_CORTICAL_ROIS:
        raise RuntimeError(
            f"Expected {EXPECTED_CORTICAL_ROIS} cortical ASL ROIs; "
            f"found {len(cortical_rois)}."
        )

    if not set(cortical_rois).issubset(cbf.columns):
        raise RuntimeError(
            "Not all frozen cortical ROIs are present in the CBF table."
        )

    if not set(cortical_rois).issubset(att.columns):
        raise RuntimeError(
            "Not all frozen cortical ROIs are present in the ATT table."
        )

    qc_measure = session_qc.pivot_table(
        index="id_event",
        columns="Measure",
        values="qc_pass",
        aggfunc="max",
    )

    qc_joint = (
        session_qc.groupby("id_event")["joint_qc_pass"]
        .max()
        .rename("joint_qc_pass")
    )

    qc = qc_measure.join(qc_joint).reset_index()

    asl_meta = subject_events[
        [
            "id_event",
            "id",
            "study",
            "days_from_V1",
            "age_open",
            "sex",
            "site",
            "scanner",
        ]
    ].copy()

    asl_visits = cbf[
        ["id_event"] + cortical_rois
    ].merge(
        att[["id_event"] + cortical_rois],
        on="id_event",
        validate="one_to_one",
        suffixes=("__cbf", "__att"),
    )

    asl_visits = asl_visits.merge(
        qc,
        on="id_event",
        how="left",
        validate="one_to_one",
    )

    asl_visits = asl_visits.merge(
        asl_meta,
        on="id_event",
        how="left",
        validate="one_to_one",
    )

    asl_visits = asl_visits.merge(
        asl_roster[
            [
                "id",
                "anchor_day",
                "anchor_age",
                "anchor_site",
                "anchor_scanner",
            ]
        ],
        on="id",
        how="left",
        validate="many_to_one",
    )

    asl_visits = asl_visits.loc[
        asl_visits["joint_qc_pass"].fillna(False)
    ].copy()

    cbf_columns = [
        f"{roi}__cbf"
        for roi in cortical_rois
    ]
    att_columns = [
        f"{roi}__att"
        for roi in cortical_rois
    ]

    cbf_array = asl_visits[
        cbf_columns
    ].to_numpy(float)

    att_array = asl_visits[
        att_columns
    ].to_numpy(float)

    asl_visits["global_cortical_cbf"] = np.log(
        np.nanmean(cbf_array, axis=1)
    )

    asl_visits["global_cortical_att"] = np.nanmean(
        att_array,
        axis=1,
    )

    asl_counts = asl_visits.groupby("id").size()
    asl_long_ids = asl_counts[
        asl_counts >= 2
    ].index

    asl_visits = asl_visits.loc[
        asl_visits["id"].isin(asl_long_ids)
    ].copy()

    if (
        asl_visits["id"].nunique() != EXPECTED_ASL_N
        or len(asl_visits) != EXPECTED_ASL_VISITS
    ):
        raise RuntimeError(
            "Frozen ASL cohort mismatch: expected "
            f"{EXPECTED_ASL_N} participants / "
            f"{EXPECTED_ASL_VISITS} observations, found "
            f"{asl_visits['id'].nunique()} / "
            f"{len(asl_visits)}."
        )

    asl_visits["t5"] = (
        pd.to_numeric(asl_visits["days_from_V1"])
        - pd.to_numeric(asl_visits["anchor_day"])
    ) / FIVE_YEAR_DAYS

    asl_visits["t5_mid"] = asl_visits["t5"] - 0.5

    asl_visits["age0"] = pd.to_numeric(
        asl_visits["anchor_age"],
        errors="coerce",
    )

    asl_visits["age0_c"] = (
        asl_visits["age0"] - AGE_REFERENCE_YEARS
    ) / AGE_SCALE_YEARS

    asl_visits["age0_sq_c"] = (
        asl_visits["age0_c"] ** 2
    )

    asl_visits["site_scanner"] = (
        asl_visits["site"].astype(str)
        + " | "
        + asl_visits["scanner"].astype(str)
    )

    # ------------------------------------------------------------------
    # Frozen cognition cohort
    # ------------------------------------------------------------------

    cognition_ids = set(
        asl_roster.loc[
            asl_roster[
                "primary_score_joint_ge2_repeated_cognition"
            ].fillna(False),
            "id",
        ].astype(str)
    )

    cognition = subject_events.loc[
        subject_events["id"].astype(str).isin(cognition_ids)
    ].copy()

    cognition["fluid_cognition"] = pd.to_numeric(
        cognition["tlbx_cfc_uncorrected_standard_score"],
        errors="coerce",
    )

    cognition = cognition.loc[
        cognition["fluid_cognition"].between(1, 200)
    ].copy()

    cognition = cognition.merge(
        asl_roster[
            [
                "id",
                "anchor_day",
                "anchor_age",
                "anchor_site",
                "anchor_scanner",
            ]
        ],
        on="id",
        how="left",
        validate="many_to_one",
    )

    cognition["elapsed_days"] = (
        pd.to_numeric(cognition["days_from_V1"])
        - pd.to_numeric(cognition["anchor_day"])
    )

    cognition = (
        cognition.loc[cognition["elapsed_days"] >= 0]
        .sort_values(["id", "days_from_V1"])
        .drop_duplicates(["id", "days_from_V1"])
    )

    cognition_support = (
        cognition.groupby("id")["elapsed_days"]
        .agg(["count", "min", "max"])
    )

    valid_cognition_ids = cognition_support.index[
        (cognition_support["count"] >= 2)
        & (
            cognition_support["max"]
            - cognition_support["min"]
            >= 365
        )
    ]

    cognition = cognition.loc[
        cognition["id"].isin(valid_cognition_ids)
    ].copy()

    if (
        cognition["id"].nunique() != EXPECTED_COGNITION_N
        or len(cognition) != EXPECTED_COGNITION_VISITS
    ):
        raise RuntimeError(
            "Frozen cognition cohort mismatch: expected "
            f"{EXPECTED_COGNITION_N} participants / "
            f"{EXPECTED_COGNITION_VISITS} observations, found "
            f"{cognition['id'].nunique()} / "
            f"{len(cognition)}."
        )

    cognition["t5"] = (
        cognition["elapsed_days"] / FIVE_YEAR_DAYS
    )

    cognition["t5_mid"] = cognition["t5"] - 0.5

    cognition["age0"] = pd.to_numeric(
        cognition["anchor_age"],
        errors="coerce",
    )

    cognition["age0_c"] = (
        cognition["age0"] - AGE_REFERENCE_YEARS
    ) / AGE_SCALE_YEARS

    cognition["age0_sq_c"] = (
        cognition["age0_c"] ** 2
    )

    cognition["site_scanner"] = (
        cognition["anchor_site"].astype(str)
        + " | "
        + cognition["anchor_scanner"].astype(str)
    )

    cohort_summary = pd.DataFrame(
        [
            {
                "cohort": "structural",
                "participants": int(
                    struct_visits["participant_id"].nunique()
                ),
                "observations": int(len(struct_visits)),
            },
            {
                "cohort": "ASL",
                "participants": int(
                    asl_visits["id"].nunique()
                ),
                "observations": int(len(asl_visits)),
            },
            {
                "cohort": "cognition",
                "participants": int(
                    cognition["id"].nunique()
                ),
                "observations": int(len(cognition)),
            },
        ]
    )

    return {
        "struct_roster": struct_roster,
        "subject_events": subject_events,
        "struct_visits": struct_visits,
        "asl_roster": asl_roster,
        "session_qc": session_qc,
        "asl_visits": asl_visits,
        "cognition": cognition,
        "cortical_rois": cortical_rois,
        "structural_transform_manifest": pd.DataFrame(
            structural_transform_rows
        ),
        "cohort_summary": cohort_summary,
    }


# ---------------------------------------------------------------------------
# Common six-outcome analysis-data preparation
# ---------------------------------------------------------------------------

def outcome_specs(
    raw: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the frozen six-outcome specifications."""

    return {
        "lateral_ventricles": {
            "label": "Lateral ventricles",
            "modality": "structural MRI",
            "source": raw["struct_visits"],
            "source_col": "lateral_ventricles",
            "id_col": "participant_id",
            "icv": True,
        },
        "hippocampus": {
            "label": "Hippocampus",
            "modality": "structural MRI",
            "source": raw["struct_visits"],
            "source_col": "hippocampus",
            "id_col": "participant_id",
            "icv": True,
        },
        "global_mean_thickness": {
            "label": "Global mean cortical thickness",
            "modality": "structural MRI",
            "source": raw["struct_visits"],
            "source_col": "global_mean_thickness",
            "id_col": "participant_id",
            "icv": False,
        },
        "global_cortical_cbf": {
            "label": "Global cortical CBF",
            "modality": "ASL",
            "source": raw["asl_visits"],
            "source_col": "global_cortical_cbf",
            "id_col": "id",
            "icv": False,
        },
        "global_cortical_att": {
            "label": "Global cortical ATT",
            "modality": "ASL",
            "source": raw["asl_visits"],
            "source_col": "global_cortical_att",
            "id_col": "id",
            "icv": False,
        },
        "fluid_cognition": {
            "label": "Fluid Cognition Composite",
            "modality": "cognition",
            "source": raw["cognition"],
            "source_col": "fluid_cognition",
            "id_col": "id",
            "icv": False,
        },
    }


def prepare_outcome_data(
    key: str,
    specs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Prepare one frozen outcome-specific analysis table."""

    spec = specs[key]

    columns = [
        spec["id_col"],
        spec["source_col"],
        "t5",
        "t5_mid",
        "age0",
        "age0_c",
        "age0_sq_c",
        "sex",
        "site",
        "study",
    ]

    if spec["modality"] in ("ASL", "cognition"):
        columns += ["site_scanner"]

    if spec["icv"]:
        columns += ["baseline_icv_z"]

    data = (
        spec["source"][columns]
        .rename(
            columns={
                spec["id_col"]: "pid",
                spec["source_col"]: "y",
            }
        )
        .copy()
    )

    data["pid"] = data["pid"].astype(str)

    data = (
        data
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    category_columns = ["sex", "site", "study"]

    if "site_scanner" in data.columns:
        category_columns += ["site_scanner"]

    for column in category_columns:
        data[column] = data[column].astype(str)

    count = data.groupby("pid")["y"].transform("size")
    span = (
        data.groupby("pid")["t5"].transform("max")
        - data.groupby("pid")["t5"].transform("min")
    )

    data = (
        data.loc[(count >= 2) & (span > 0)]
        .sort_values(["pid", "t5"])
        .reset_index(drop=True)
    )

    data["t5_person_centered"] = (
        data["t5"]
        - data.groupby("pid")["t5"].transform("mean")
    )

    return data


def build_analysis_data(
    raw: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """Prepare all six frozen outcome datasets and a cohort summary."""

    specs = outcome_specs(raw)

    if len(specs) != 6:
        raise RuntimeError(
            f"Expected six outcome specifications; found {len(specs)}."
        )

    analysis_data = {
        key: prepare_outcome_data(key, specs)
        for key in specs
    }

    cohort_check = pd.DataFrame(
        [
            {
                "outcome": key,
                "label": specs[key]["label"],
                "modality": specs[key]["modality"],
                "participants": int(data["pid"].nunique()),
                "observations": int(len(data)),
                "median_scans": float(
                    data.groupby("pid").size().median()
                ),
                "median_followup_years": float(
                    (
                        data.groupby("pid")["t5"].max()
                        - data.groupby("pid")["t5"].min()
                    ).median()
                    * 5
                ),
            }
            for key, data in analysis_data.items()
        ]
    )

    return specs, analysis_data, cohort_check


# ---------------------------------------------------------------------------
# Common formula construction
# ---------------------------------------------------------------------------

def formula_from_time(
    key: str,
    specs: dict[str, dict[str, Any]],
    time_name: str = "t5_mid",
    *,
    omit_age_time: bool = False,
) -> str:
    """Construct the frozen common fixed-effects formula."""

    spec = specs[key]

    terms = [
        time_name,
        "age0_c",
        "age0_sq_c",
        "C(sex)",
        "C(study)",
    ]

    if not omit_age_time:
        terms += [
            f"{time_name}:age0_c",
            f"{time_name}:age0_sq_c",
        ]

    terms += [
        f"{time_name}:C(sex)",
        f"{time_name}:C(site)",
    ]

    if spec["modality"] == "structural MRI":
        terms += ["C(site)"]
    else:
        terms += ["C(site_scanner)"]

    if spec["icv"]:
        terms += [
            "baseline_icv_z",
            f"{time_name}:baseline_icv_z",
        ]

    return "y ~ " + " + ".join(terms)


def fixed_formula(
    key: str,
    specs: dict[str, dict[str, Any]],
) -> str:
    return formula_from_time(
        key,
        specs,
        "t5_mid",
        omit_age_time=False,
    )


def predictor_omitted_formula(
    key: str,
    specs: dict[str, dict[str, Any]],
) -> str:
    return formula_from_time(
        key,
        specs,
        "t5_mid",
        omit_age_time=True,
    )


def original_frozen_formula(
    key: str,
    specs: dict[str, dict[str, Any]],
) -> str:
    """Construct the original-frozen comparator formula."""

    original_time = (
        "t5_person_centered"
        if specs[key]["modality"] == "structural MRI"
        else "t5"
    )

    return formula_from_time(
        key,
        specs,
        original_time,
        omit_age_time=False,
    )
