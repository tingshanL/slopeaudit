#!/usr/bin/env python3
"""Prepare the exact AABC-derived schedules used by the S4 operating study.

This workflow requires authorized AABC inputs.

This script contains no AABC participant rows or identifiers. It reconstructs
the analysis cohorts from explicitly supplied authorized AABC inputs
and writes the empirical visit schedules required by the operating-study
generator.

The resulting schedule JSON is AABC-derived and remains in restricted storage.
"""

from __future__ import annotations

import json
import zipfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


FIVE_YEAR_DAYS = 365.25 * 5
AGE_REFERENCE_YEARS = 60.0
AGE_SCALE_YEARS = 10.0
MIN_THICKNESS_COMPLETENESS = 0.95


def parse_args():
    parser = ArgumentParser()

    structural = parser.add_mutually_exclusive_group(required=True)
    structural.add_argument(
        "--structural-idp-dir",
        type=Path,
        help="Authorized extracted AABC structural-IDP directory.",
    )
    structural.add_argument(
        "--structural-idp-zip",
        type=Path,
        help="Authorized AABC structural-IDP zip archive.",
    )

    parser.add_argument("--asl-idp-zip", required=True, type=Path)
    parser.add_argument("--idp-info-zip", required=True, type=Path)
    parser.add_argument("--asl-audit-xlsx", required=True, type=Path)
    parser.add_argument("--structural-roster-csv", required=True, type=Path)
    parser.add_argument("--final622-ids-csv", type=Path, default=None)
    parser.add_argument("--subjects-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file.")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_dir():
        raise NotADirectoryError(f"{label} does not exist or is not a directory.")
    return path


def read_zip_csv(zip_path: Path, suffix: str, **kwargs):
    with zipfile.ZipFile(zip_path) as zf:
        hits = [
            name
            for name in zf.namelist()
            if name.endswith(suffix)
            and not name.startswith("__MACOSX/")
        ]
        if len(hits) != 1:
            raise ValueError(
                f"Expected exactly one archive member ending "
                f"{suffix!r}; found {len(hits)}."
            )
        with zf.open(hits[0]) as handle:
            return pd.read_csv(handle, **kwargs)


def canonical_event_id(values):
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"_MR$", "", regex=True, case=False)
    )

def prepare_restricted_schedules(args):
    # Validate all explicitly supplied inputs without discovering local files.
    STRUCT_DIR = (
        require_dir(args.structural_idp_dir, "structural IDP directory")
        if args.structural_idp_dir is not None
        else None
    )
    STRUCT_ZIP = (
        require_file(args.structural_idp_zip, "structural IDP zip")
        if args.structural_idp_zip is not None
        else None
    )
    ASL_ZIP = require_file(args.asl_idp_zip, "ASL IDP zip")
    INFO_ZIP = require_file(args.idp_info_zip, "IDP info zip")
    ASL_AUDIT = require_file(args.asl_audit_xlsx, "ASL audit workbook")
    STRUCT_ROSTER = require_file(
        args.structural_roster_csv, "structural roster CSV"
    )
    FINAL622_IDS = (
        require_file(args.final622_ids_csv, "final-622 ID CSV")
        if args.final622_ids_csv is not None
        else None
    )
    SUBJECTS_CSV = require_file(args.subjects_csv, "subjects CSV")

    def read_structural_csv(filename, **kwargs):
        if STRUCT_DIR is not None:
            source = STRUCT_DIR / filename
            if not source.is_file():
                raise FileNotFoundError(
                    f"Required structural-IDP member is absent: {filename}"
                )
            return pd.read_csv(source, **kwargs)

        return read_zip_csv(STRUCT_ZIP, filename, **kwargs)

    # Replace notebook display calls with a no-op in this script.
    def display(*_args, **_kwargs):
        return None
    struct_roster = pd.read_csv(STRUCT_ROSTER, low_memory=False)
    struct_roster["participant_id"] = struct_roster["participant_id"].astype(str)
    assert struct_roster["participant_id"].nunique() == 875
    assert int(struct_roster["in_final_622"].sum()) == 622
    if FINAL622_IDS is not None:
        frozen_file = set(pd.read_csv(FINAL622_IDS)["participant_id"].astype(str))
        frozen_roster = set(struct_roster.loc[struct_roster["in_final_622"].eq(1), "participant_id"])
        assert frozen_file == frozen_roster

    subject_cols = ["id_event", "study", "days_from_V1", "age_open", "sex", "site", "scanner", "tlbx_cfc_uncorrected_standard_score"]
    subject_events = pd.read_csv(SUBJECTS_CSV, usecols=subject_cols, low_memory=False)
    subject_events["id_event"] = subject_events["id_event"].astype(str).str.strip()
    subject_events["id"] = subject_events["id_event"].str.replace(r"_[^_]+$", "", regex=True)
    assert subject_events["id_event"].is_unique

    valid_events = (
        struct_roster[[
            "participant_id", "in_final_622", "anchor_age", "anchor_days_from_V1", "all_valid_id_events"
        ]]
        .assign(id_event=lambda d: d["all_valid_id_events"].astype(str).str.split(";"))
        .explode("id_event", ignore_index=True).drop(columns="all_valid_id_events")
    )
    struct_visits = valid_events.merge(
        subject_events[["id_event", "study", "days_from_V1", "age_open", "sex", "site", "scanner"]],
        on="id_event", how="left", validate="one_to_one",
    )
    struct_visits = struct_visits.loc[struct_visits["in_final_622"].eq(1)].copy()
    assert struct_visits["participant_id"].nunique() == 622 and len(struct_visits) == 1629

    aseg = read_structural_csv("asegstats.csv", low_memory=False)
    thick = read_structural_csv("Cortical_Areal_Thicknesses.csv", low_memory=False)
    area = read_structural_csv("Cortical_Areal_Surface_Areas.csv", low_memory=False)
    for frame in (aseg, thick, area):
        key = frame.columns[0]
        frame.rename(columns={key: "id_event"}, inplace=True)
        frame["id_event"] = canonical_event_id(frame["id_event"])

    parcel_fields = [c for c in thick.columns if c != "id_event" and c in set(area.columns)]
    assert len(parcel_fields) == 360, f"Expected 360 HCP-MMP parcels; found {len(parcel_fields)}"

    struct_idps = aseg[[
        "id_event", "FS_L_Lateral-Ventricle", "FS_R_Lateral-Ventricle",
        "FS_L_Hippocampus", "FS_R_Hippocampus", "EstimatedTotalIntraCranialVol",
    ]].merge(thick[["id_event"] + parcel_fields], on="id_event", validate="one_to_one", suffixes=("", "__thick"))

    # The merge above preserves thickness names. Rename before attaching identically named area fields.
    struct_idps = struct_idps.rename(columns={c: f"thick__{c}" for c in parcel_fields})
    area_part = area[["id_event"] + parcel_fields].rename(columns={c: f"area__{c}" for c in parcel_fields})
    struct_idps = struct_idps.merge(area_part, on="id_event", validate="one_to_one")
    struct_visits = struct_visits.merge(struct_idps, on="id_event", how="left", validate="one_to_one")

    struct_visits["ventricles_raw"] = struct_visits[["FS_L_Lateral-Ventricle", "FS_R_Lateral-Ventricle"]].sum(axis=1, min_count=2)
    struct_visits["hippocampus_raw"] = struct_visits[["FS_L_Hippocampus", "FS_R_Hippocampus"]].sum(axis=1, min_count=2)

    tvals = struct_visits[[f"thick__{c}" for c in parcel_fields]].to_numpy(float)
    avals = struct_visits[[f"area__{c}" for c in parcel_fields]].to_numpy(float)
    valid = np.isfinite(tvals) & np.isfinite(avals) & (avals > 0)
    complete = valid.mean(axis=1)
    denom = np.where(valid, avals, 0).sum(axis=1)
    struct_visits["thickness_raw"] = np.divide(
        np.where(valid, tvals * avals, 0).sum(axis=1), denom,
        out=np.full(len(struct_visits), np.nan), where=denom > 0,
    )
    struct_visits.loc[complete < MIN_THICKNESS_COMPLETENESS, "thickness_raw"] = np.nan

    # Baseline ICV and outcome standardization are frozen within the n=622 cohort.
    struct_visits = struct_visits.sort_values(["participant_id", "days_from_V1", "id_event"]).reset_index(drop=True)
    baseline_icv = (struct_visits.dropna(subset=["EstimatedTotalIntraCranialVol"])
                    .groupby("participant_id", as_index=False).head(1)
                    [["participant_id", "EstimatedTotalIntraCranialVol"]]
                    .rename(columns={"EstimatedTotalIntraCranialVol": "baseline_icv"}))
    icv_mean, icv_sd = baseline_icv["baseline_icv"].mean(), baseline_icv["baseline_icv"].std(ddof=1)
    baseline_icv["baseline_icv_z"] = (baseline_icv["baseline_icv"] - icv_mean) / icv_sd
    struct_visits = struct_visits.merge(baseline_icv, on="participant_id", how="left", validate="many_to_one")

    struct_transform_manifest = []
    for raw_col, outcome, transform, direction in [
        ("ventricles_raw", "lateral_ventricles", "log1p", -1.0),
        ("hippocampus_raw", "hippocampus", "raw", 1.0),
        ("thickness_raw", "global_mean_thickness", "raw", 1.0),
    ]:
        x = np.log1p(struct_visits[raw_col]) if transform == "log1p" else struct_visits[raw_col].astype(float)
        baseline = (struct_visits[["participant_id", "days_from_V1"]].assign(value=x)
                    .dropna(subset=["value"]).groupby("participant_id", as_index=False).head(1)["value"])
        center, scale = baseline.mean(), baseline.std(ddof=1)
        struct_visits[outcome] = direction * (x - center) / scale
        struct_transform_manifest.append({"outcome": outcome, "transform": transform, "direction": direction, "center": center, "scale": scale})

    struct_visits["t5"] = (pd.to_numeric(struct_visits["days_from_V1"]) - pd.to_numeric(struct_visits["anchor_days_from_V1"])) / FIVE_YEAR_DAYS
    struct_visits["t5_mid"] = struct_visits["t5"] - 0.5
    struct_visits["age0"] = pd.to_numeric(struct_visits["anchor_age"], errors="coerce")
    struct_visits["age0_c"] = (struct_visits["age0"] - AGE_REFERENCE_YEARS) / AGE_SCALE_YEARS
    struct_visits["age0_sq_c"] = struct_visits["age0_c"] ** 2

    display(pd.DataFrame(struct_transform_manifest))

    asl_roster = pd.read_excel(ASL_AUDIT, sheet_name="Participant Roster", header=2)
    session_qc = pd.read_excel(ASL_AUDIT, sheet_name="Session QC", header=2)
    cbf = read_zip_csv(ASL_ZIP, "PVEc_ASL_CBF.csv", low_memory=False)
    att = read_zip_csv(ASL_ZIP, "PVEc_ASL_ATT.csv", low_memory=False)
    atlas = read_zip_csv(INFO_ZIP, "AreaNamesAndIndicesAndNetworks.csv")

    for frame in (cbf, att):
        frame.rename(columns={frame.columns[0]: "id_event"}, inplace=True)
        frame["id_event"] = frame["id_event"].astype(str).str.strip()
    atlas["Brain Area"] = atlas["Brain Area"].astype(str).str.strip()
    cortical_rois = atlas.loc[atlas["Ji Network Label"].notna(), "Brain Area"].tolist()
    assert len(cortical_rois) == 360
    assert set(cortical_rois).issubset(cbf.columns) and set(cortical_rois).issubset(att.columns)

    qc_measure = session_qc.pivot_table(index="id_event", columns="Measure", values="qc_pass", aggfunc="max")
    qc_joint = session_qc.groupby("id_event")["joint_qc_pass"].max().rename("joint_qc_pass")
    qc = qc_measure.join(qc_joint).reset_index()

    asl_meta = subject_events[["id_event", "id", "study", "days_from_V1", "age_open", "sex", "site", "scanner"]].copy()
    asl_visits = cbf[["id_event"] + cortical_rois].merge(
        att[["id_event"] + cortical_rois], on="id_event", validate="one_to_one", suffixes=("__cbf", "__att")
    )
    asl_visits = asl_visits.merge(qc, on="id_event", how="left", validate="one_to_one")
    asl_visits = asl_visits.merge(asl_meta, on="id_event", how="left", validate="one_to_one")
    asl_visits = asl_visits.merge(
        asl_roster[["id", "anchor_day", "anchor_age", "anchor_site", "anchor_scanner"]],
        on="id", how="left", validate="many_to_one",
    )
    asl_visits = asl_visits.loc[asl_visits["joint_qc_pass"].fillna(False)].copy()

    cbf_cols = [f"{r}__cbf" for r in cortical_rois]
    att_cols = [f"{r}__att" for r in cortical_rois]
    cbf_array = asl_visits[cbf_cols].to_numpy(float)
    att_array = asl_visits[att_cols].to_numpy(float)
    asl_visits["global_cortical_cbf"] = np.log(np.nanmean(cbf_array, axis=1))
    asl_visits["global_cortical_att"] = np.nanmean(att_array, axis=1)

    counts = asl_visits.groupby("id").size()
    asl_long_ids = counts[counts >= 2].index
    asl_visits = asl_visits.loc[asl_visits["id"].isin(asl_long_ids)].copy()
    assert asl_visits["id"].nunique() == 802 and len(asl_visits) == 2047
    asl_visits["t5"] = (pd.to_numeric(asl_visits["days_from_V1"]) - pd.to_numeric(asl_visits["anchor_day"])) / FIVE_YEAR_DAYS
    asl_visits["t5_mid"] = asl_visits["t5"] - 0.5
    asl_visits["age0"] = pd.to_numeric(asl_visits["anchor_age"], errors="coerce")
    asl_visits["age0_c"] = (asl_visits["age0"] - AGE_REFERENCE_YEARS) / AGE_SCALE_YEARS
    asl_visits["age0_sq_c"] = asl_visits["age0_c"] ** 2
    asl_visits["site_scanner"] = asl_visits["site"].astype(str) + " | " + asl_visits["scanner"].astype(str)

    cog_ids = set(asl_roster.loc[asl_roster["primary_score_joint_ge2_repeated_cognition"].fillna(False), "id"].astype(str))
    cognition = subject_events.loc[subject_events["id"].astype(str).isin(cog_ids)].copy()
    cognition["fluid_cognition"] = pd.to_numeric(cognition["tlbx_cfc_uncorrected_standard_score"], errors="coerce")
    cognition = cognition.loc[cognition["fluid_cognition"].between(1, 200)].copy()
    cognition = cognition.merge(
        asl_roster[["id", "anchor_day", "anchor_age", "anchor_site", "anchor_scanner"]],
        on="id", how="left", validate="many_to_one",
    )
    cognition["elapsed_days"] = pd.to_numeric(cognition["days_from_V1"]) - pd.to_numeric(cognition["anchor_day"])
    cognition = cognition.loc[cognition["elapsed_days"] >= 0].sort_values(["id", "days_from_V1"]).drop_duplicates(["id", "days_from_V1"])
    cog_support = cognition.groupby("id")["elapsed_days"].agg(["count", "min", "max"])
    valid_cog = cog_support.index[(cog_support["count"] >= 2) & ((cog_support["max"] - cog_support["min"]) >= 365)]
    cognition = cognition.loc[cognition["id"].isin(valid_cog)].copy()
    assert cognition["id"].nunique() == 691 and len(cognition) == 1805
    cognition["t5"] = cognition["elapsed_days"] / FIVE_YEAR_DAYS
    cognition["t5_mid"] = cognition["t5"] - 0.5
    cognition["age0"] = pd.to_numeric(cognition["anchor_age"], errors="coerce")
    cognition["age0_c"] = (cognition["age0"] - AGE_REFERENCE_YEARS) / AGE_SCALE_YEARS
    cognition["age0_sq_c"] = cognition["age0_c"] ** 2
    cognition["site_scanner"] = cognition["anchor_site"].astype(str) + " | " + cognition["anchor_scanner"].astype(str)

    display(pd.DataFrame([
        {"cohort": "structural", "participants": struct_visits.participant_id.nunique(), "observations": len(struct_visits)},
        {"cohort": "ASL", "participants": asl_visits.id.nunique(), "observations": len(asl_visits)},
        {"cohort": "cognition", "participants": cognition.id.nunique(), "observations": len(cognition)},
    ]))

    OUTCOMES = {
        "lateral_ventricles": {"label": "Lateral ventricles", "modality": "structural MRI", "source": struct_visits, "source_col": "lateral_ventricles", "id_col": "participant_id", "icv": True},
        "hippocampus": {"label": "Hippocampus", "modality": "structural MRI", "source": struct_visits, "source_col": "hippocampus", "id_col": "participant_id", "icv": True},
        "global_mean_thickness": {"label": "Global mean cortical thickness", "modality": "structural MRI", "source": struct_visits, "source_col": "global_mean_thickness", "id_col": "participant_id", "icv": False},
        "global_cortical_cbf": {"label": "Global cortical CBF", "modality": "ASL", "source": asl_visits, "source_col": "global_cortical_cbf", "id_col": "id", "icv": False},
        "global_cortical_att": {"label": "Global cortical ATT", "modality": "ASL", "source": asl_visits, "source_col": "global_cortical_att", "id_col": "id", "icv": False},
        "fluid_cognition": {"label": "Fluid Cognition Composite", "modality": "cognition", "source": cognition, "source_col": "fluid_cognition", "id_col": "id", "icv": False},
    }

    def prepare_outcome_data(key):
        spec = OUTCOMES[key]
        cols = [spec["id_col"], spec["source_col"], "t5", "t5_mid", "age0", "age0_c", "age0_sq_c", "sex", "site", "study"]
        if spec["modality"] in ("ASL", "cognition"):
            cols += ["site_scanner"]
        if spec["icv"]:
            cols += ["baseline_icv_z"]
        d = spec["source"][cols].rename(columns={spec["id_col"]: "pid", spec["source_col"]: "y"}).copy()
        d["pid"] = d["pid"].astype(str)
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        for c in ["sex", "site", "study"] + (["site_scanner"] if "site_scanner" in d else []):
            d[c] = d[c].astype(str)
        n = d.groupby("pid")["y"].transform("size")
        span = d.groupby("pid")["t5"].transform("max") - d.groupby("pid")["t5"].transform("min")
        d = d.loc[(n >= 2) & (span > 0)].sort_values(["pid", "t5"]).reset_index(drop=True)
        d["t5_person_centered"] = d["t5"] - d.groupby("pid")["t5"].transform("mean")
        return d

    ANALYSIS_DATA = {k: prepare_outcome_data(k) for k in OUTCOMES}


    # This reproduces the exact schedule construction used by the original
    # operating-study cell:
    #
    #   ANALYSIS_DATA[outcome]
    #       -> groupby("pid", sort=False)
    #       -> each participant's t5_mid vector
    #
    # Export schedules without participant identifiers.
    schedules = {}

    for outcome, data in ANALYSIS_DATA.items():
        participant_times = [
            [float(value) for value in group["t5_mid"].to_numpy(float)]
            for _, group in data.groupby("pid", sort=False)
        ]

        schedules[f"empirical__{outcome}"] = {
            "participants": int(len(participant_times)),
            "observations": int(sum(map(len, participant_times))),
            "participant_times": participant_times,
        }

    payload = {
        "schema_version": 1,
        "data_class": "restricted-derived schedule",
        "release_status": "restricted_input_not_distributed",
        "time_variable": "t5_mid",
        "time_unit_years": 5.0,
        "time_definition": "t5_mid = t5 - 0.5",
        "schedules": schedules,
    }

    return payload


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = prepare_restricted_schedules(args)

    output_path = (
        args.output_dir
        / "operating_study_empirical_schedules_restricted.json"
    )
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Prepared restricted operating-study schedule file.")
    print(f"Output filename: {output_path.name}")
    for name, schedule in payload["schedules"].items():
        print(
            f"  {name}: "
            f"{schedule['participants']} participants, "
            f"{schedule['observations']} observations"
        )


if __name__ == "__main__":
    main()
