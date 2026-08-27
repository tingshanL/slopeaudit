#!/usr/bin/env python3
"""Prepare restricted source anchors for the S5 downstream simulation.

This workflow requires authorized AABC inputs.

This script contains no embedded AABC participant rows or identifiers. It
reconstructs the analysis cohorts from explicitly supplied authorized
AABC inputs, fits the four observed source regimes used by the downstream
simulation, and writes:

1. a non-participant-level source-anchor manifest;
2. anonymous exact schedule fixtures containing only within-participant times;
3. an optimizer-fit audit for validating the source-model refactor.

All resulting files are AABC-derived and remain in restricted storage.
"""

from __future__ import annotations

import os
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import json
import warnings
import zipfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf


FIVE_YEAR_DAYS = 365.25 * 5
AGE_REFERENCE_YEARS = 60.0
AGE_SCALE_YEARS = 10.0
MIN_THICKNESS_COMPLETENESS = 0.95

OPTIMIZERS = ("lbfgs", "powell", "bfgs")
MAXITER = 1200

SLOPE_BOUNDARY_REL_TOL = 1e-8
PSD_REL_TOL = 1e-8
LLF_REL_TOL = 1e-5
FIXED_AGREEMENT_SE_TOL = 0.25

PRIMARY_NEAR_SINGULAR_ABS_RHO = 0.95
PRIMARY_SMALL_NORMALIZED_DETERMINANT = 0.10
PRIMARY_SMALL_EIGEN_FRACTION = 1e-4


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

def prepare_source_anchors(args):
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


    # ------------------------------------------------------------------
    # S5 source regimes and observed-source model definitions
    # ------------------------------------------------------------------

    SIM_REGIMES = {
        "high_slope_information": "lateral_ventricles",
        "intermediate": "hippocampus",
        "near_boundary": "global_cortical_att",
        "true_zero_slope_variance": "fluid_cognition",
    }

    def formula_from_time(key, time_name="t5_mid", omit_age_time=False):
        spec = OUTCOMES[key]
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

        terms += (
            ["C(site)"]
            if spec["modality"] == "structural MRI"
            else ["C(site_scanner)"]
        )

        if spec["icv"]:
            terms += [
                "baseline_icv_z",
                f"{time_name}:baseline_icv_z",
            ]

        return "y ~ " + " + ".join(terms)

    def fixed_formula(key):
        return formula_from_time(
            key,
            "t5_mid",
            omit_age_time=False,
        )

    # Preflight the four source formulas before fitting.
    formula_preflight = []

    for outcome in SIM_REGIMES.values():
        y_matrix, x_matrix = patsy.dmatrices(
            fixed_formula(outcome),
            ANALYSIS_DATA[outcome],
            return_type="dataframe",
        )
        rank = int(np.linalg.matrix_rank(x_matrix.to_numpy()))

        formula_preflight.append(
            {
                "outcome": outcome,
                "rows": len(y_matrix),
                "columns": x_matrix.shape[1],
                "rank": rank,
                "full_rank": rank == x_matrix.shape[1],
            }
        )

    formula_preflight = pd.DataFrame(formula_preflight)

    if len(formula_preflight) != 4:
        raise RuntimeError("Expected four S5 source-formula preflight rows.")

    if not formula_preflight["full_rank"].all():
        raise RuntimeError(
            "At least one S5 source formula is rank deficient."
        )

    # ------------------------------------------------------------------
    # Numerical diagnostics copied from the frozen S5 analysis
    # ------------------------------------------------------------------

    def hessian_diagnostic(result):
        try:
            hessian = result.model.hessian(result.params_object)

            if isinstance(hessian, tuple):
                hessian = hessian[0]

            information = -(
                np.asarray(hessian, float)
                + np.asarray(hessian, float).T
            ) / 2

            eigenvalues = np.linalg.eigvalsh(information)

            tolerance = 1e-8 * max(
                1.0,
                float(np.max(np.abs(eigenvalues))),
            )

            return (
                bool(np.min(eigenvalues) > tolerance),
                float(np.min(eigenvalues)),
                float(np.max(eigenvalues)),
            )

        except Exception:
            return False, np.nan, np.nan

    def fixed_finite(result):
        try:
            estimates = np.asarray(result.fe_params, float)
            errors = np.asarray(result.bse_fe, float)

            return bool(
                np.isfinite(estimates).all()
                and np.isfinite(errors).all()
                and (errors >= 0).all()
            )

        except Exception:
            return False

    def compatibility(first, second):
        if first is None or second is None:
            return False

        common = first.fe_params.index.intersection(
            second.fe_params.index
        )

        if not len(common):
            return False

        scale = np.maximum(
            first.bse_fe.loc[common].to_numpy(float),
            second.bse_fe.loc[common].to_numpy(float),
        )

        fixed_distance = float(
            np.max(
                np.abs(
                    first.fe_params.loc[common].to_numpy(float)
                    - second.fe_params.loc[common].to_numpy(float)
                )
                / np.maximum(scale, 1e-8)
            )
        )

        likelihood_scale = max(
            1.0,
            abs(float(first.llf)),
            abs(float(second.llf)),
        )

        return bool(
            abs(float(first.llf) - float(second.llf))
            <= LLF_REL_TOL * likelihood_scale
            and fixed_distance <= FIXED_AGREEMENT_SE_TOL
        )

    def covariance_geometry(result, has_random_slope):
        residual = float(result.scale)

        random_covariance = np.asarray(
            result.cov_re,
            float,
        )

        covariance_eigenvalues = np.linalg.eigvalsh(
            (
                random_covariance
                + random_covariance.T
            )
            / 2
        )

        covariance_minimum = float(
            np.min(covariance_eigenvalues)
        )
        covariance_trace = float(
            np.trace(random_covariance)
        )

        covariance_scale = max(
            1.0,
            float(
                np.max(
                    np.abs(covariance_eigenvalues)
                )
            ),
            abs(residual),
        )

        covariance_psd = bool(
            np.isfinite(random_covariance).all()
            and residual > 0
            and covariance_minimum
            >= -PSD_REL_TOL * covariance_scale
        )

        (
            hessian_pd,
            hessian_minimum,
            hessian_maximum,
        ) = hessian_diagnostic(result)

        output = {
            "converged": bool(result.converged),
            "reml_log_likelihood": float(result.llf),
            "fixed_parameters_finite": fixed_finite(result),
            "hessian_positive_definite": hessian_pd,
            "hessian_min_information_eigenvalue":
                hessian_minimum,
            "hessian_max_information_eigenvalue":
                hessian_maximum,
            "residual_variance": residual,
            "random_intercept_variance":
                float(random_covariance[0, 0]),
            "random_slope_variance": np.nan,
            "intercept_slope_covariance": np.nan,
            "intercept_slope_correlation": np.nan,
            "absolute_intercept_slope_correlation": np.nan,
            "covariance_minimum_eigenvalue":
                covariance_minimum,
            "covariance_trace": covariance_trace,
            "minimum_eigenvalue_fraction_of_trace": (
                covariance_minimum / covariance_trace
                if covariance_trace > 0
                else np.nan
            ),
            "covariance_psd": covariance_psd,
            "one_minus_rho_squared": np.nan,
            "normalized_determinant": np.nan,
            "innovation_variance": np.nan,
            "innovation_fraction": np.nan,
            "residual_variance_boundary": bool(
                not np.isfinite(residual)
                or residual <= PSD_REL_TOL
            ),
            "intercept_variance_boundary": bool(
                not np.isfinite(
                    random_covariance[0, 0]
                )
                or random_covariance[0, 0]
                <= PSD_REL_TOL * covariance_scale
            ),
            "slope_variance_boundary": False,
            "covariance_boundary": bool(
                covariance_minimum
                <= PSD_REL_TOL * covariance_scale
            ),
            "near_singular_abs_rho_095": False,
            "near_singular_normalized_determinant_010":
                False,
            "near_singular_eigen_fraction_1e4": bool(
                covariance_trace > 0
                and covariance_minimum / covariance_trace
                <= PRIMARY_SMALL_EIGEN_FRACTION
            ),
        }

        if (
            has_random_slope
            and random_covariance.shape == (2, 2)
        ):
            intercept_variance = float(
                random_covariance[0, 0]
            )
            slope_variance = float(
                random_covariance[1, 1]
            )
            covariance = float(
                random_covariance[0, 1]
            )

            correlation = (
                covariance
                / np.sqrt(
                    intercept_variance
                    * slope_variance
                )
                if intercept_variance > 0
                and slope_variance > 0
                else np.nan
            )

            normalized_determinant = (
                float(
                    np.linalg.det(random_covariance)
                    / (
                        intercept_variance
                        * slope_variance
                    )
                )
                if intercept_variance > 0
                and slope_variance > 0
                else np.nan
            )

            one_minus_rho_squared = (
                1 - correlation**2
                if np.isfinite(correlation)
                else np.nan
            )

            innovation_variance = (
                slope_variance
                - covariance**2 / intercept_variance
                if intercept_variance > 0
                else np.nan
            )

            innovation_fraction = (
                innovation_variance / slope_variance
                if (
                    slope_variance > 0
                    and np.isfinite(
                        innovation_variance
                    )
                )
                else np.nan
            )

            slope_boundary_tolerance = (
                SLOPE_BOUNDARY_REL_TOL
                * max(1.0, residual)
            )

            output.update(
                {
                    "random_slope_variance":
                        slope_variance,
                    "intercept_slope_covariance":
                        covariance,
                    "intercept_slope_correlation":
                        correlation,
                    "absolute_intercept_slope_correlation":
                        (
                            abs(correlation)
                            if np.isfinite(correlation)
                            else np.nan
                        ),
                    "one_minus_rho_squared":
                        one_minus_rho_squared,
                    "normalized_determinant":
                        normalized_determinant,
                    "innovation_variance":
                        innovation_variance,
                    "innovation_fraction":
                        innovation_fraction,
                    "slope_variance_boundary":
                        bool(
                            not np.isfinite(
                                slope_variance
                            )
                            or slope_variance
                            <= slope_boundary_tolerance
                        ),
                    "near_singular_abs_rho_095":
                        bool(
                            np.isfinite(correlation)
                            and abs(correlation)
                            >= PRIMARY_NEAR_SINGULAR_ABS_RHO
                        ),
                    "near_singular_normalized_determinant_010":
                        bool(
                            np.isfinite(
                                normalized_determinant
                            )
                            and normalized_determinant
                            <= PRIMARY_SMALL_NORMALIZED_DETERMINANT
                        ),
                }
            )

        output["any_boundary_flag"] = bool(
            output["residual_variance_boundary"]
            or output["intercept_variance_boundary"]
            or output["slope_variance_boundary"]
            or output["covariance_boundary"]
        )

        output["basic_numerical_validity"] = bool(
            output["converged"]
            and output["fixed_parameters_finite"]
            and output["covariance_psd"]
            and np.isfinite(
                output["reml_log_likelihood"]
            )
        )

        output["strict_numerical_validity"] = bool(
            output["basic_numerical_validity"]
            and output["hessian_positive_definite"]
            and not output["covariance_boundary"]
            and (
                not has_random_slope
                or not output["slope_variance_boundary"]
            )
        )

        return output

    # ------------------------------------------------------------------
    # Frozen three-optimizer observed-source panels
    # ------------------------------------------------------------------

    def fit_observed_panel(outcome, model_role):
        data = ANALYSIS_DATA[outcome]
        formula = fixed_formula(outcome)

        re_formula = (
            "1+t5_mid"
            if model_role == "reference_random_slope"
            else "1"
        )

        attempts = []
        results = {}

        for optimizer in OPTIMIZERS:
            caught = []

            try:
                with warnings.catch_warnings(
                    record=True
                ) as warning_records:
                    warnings.simplefilter("always")

                    result = smf.mixedlm(
                        formula,
                        data=data,
                        groups=data["pid"],
                        re_formula=re_formula,
                    ).fit(
                        reml=True,
                        method=optimizer,
                        maxiter=MAXITER,
                        disp=False,
                    )

                    caught = [
                        str(item.message)
                        for item in warning_records
                    ]

                geometry = covariance_geometry(
                    result,
                    has_random_slope=(
                        model_role
                        == "reference_random_slope"
                    ),
                )

                results[optimizer] = result

                attempts.append(
                    {
                        "outcome": outcome,
                        "model_role": model_role,
                        "optimizer": optimizer,
                        "fit_succeeded": True,
                        "warning_count": len(caught),
                        "warnings": " | ".join(
                            sorted(set(caught))
                        ),
                        "error": "",
                        **geometry,
                    }
                )

            except Exception as exc:
                results[optimizer] = None

                attempts.append(
                    {
                        "outcome": outcome,
                        "model_role": model_role,
                        "optimizer": optimizer,
                        "fit_succeeded": False,
                        "converged": False,
                        "basic_numerical_validity":
                            False,
                        "strict_numerical_validity":
                            False,
                        "reml_log_likelihood":
                            np.nan,
                        "warning_count":
                            len(caught),
                        "warnings": " | ".join(
                            sorted(set(caught))
                        ),
                        "error": repr(exc),
                    }
                )

        for row in attempts:
            optimizer = row["optimizer"]

            row["common_solution_count"] = (
                sum(
                    results[other] is not None
                    and compatibility(
                        results[optimizer],
                        results[other],
                    )
                    for other in OPTIMIZERS
                )
                if (
                    results[optimizer] is not None
                    and row.get(
                        "basic_numerical_validity",
                        False,
                    )
                )
                else 0
            )

            row["selected"] = False

        usable = [
            row
            for row in attempts
            if row.get(
                "basic_numerical_validity",
                False,
            )
        ]

        if not usable:
            raise RuntimeError(
                "No usable observed source fit: "
                f"{outcome}, {model_role}"
            )

        maximum_agreement = max(
            row["common_solution_count"]
            for row in usable
        )

        if maximum_agreement < 2:
            raise RuntimeError(
                "Observed source optimizers do not agree: "
                f"{outcome}, {model_role}"
            )

        selected = max(
            [
                row
                for row in usable
                if row["common_solution_count"]
                == maximum_agreement
            ],
            key=lambda row:
                row["reml_log_likelihood"],
        )

        selected["selected"] = True

        return (
            results[selected["optimizer"]],
            pd.DataFrame(attempts),
        )

    observed_source_fits = {}
    observed_audits = []

    for outcome in SIM_REGIMES.values():
        for model_role in (
            "reference_random_slope",
            "random_intercept_null",
        ):
            result, audit = fit_observed_panel(
                outcome,
                model_role,
            )

            observed_source_fits[
                (outcome, model_role)
            ] = result

            observed_audits.append(audit)

    observed_source_fit_audit = pd.concat(
        observed_audits,
        ignore_index=True,
    )

    if len(observed_source_fit_audit) != 24:
        raise RuntimeError(
            "Expected exactly 24 observed-source optimizer rows."
        )

    # ------------------------------------------------------------------
    # Restricted aggregate anchors + anonymous exact schedules
    # ------------------------------------------------------------------

    anchor_manifest = {
        "schema_version": 1,
        "data_class":
            "restricted-derived source anchors",
        "release_status":
            "restricted_input_not_distributed",
        "source_model":
            "frozen S5 downstream simulation source fits",
        "regimes": {},
    }

    schedule_payloads = {}

    for regime, outcome in SIM_REGIMES.items():
        reference = observed_source_fits[
            (outcome, "reference_random_slope")
        ]
        intercept_only = observed_source_fits[
            (outcome, "random_intercept_null")
        ]

        source_data = (
            ANALYSIS_DATA[outcome][
                ["pid", "t5_mid", "y"]
            ]
            .rename(columns={"t5_mid": "t"})
            .copy()
        )

        random_covariance = np.asarray(
            reference.cov_re,
            float,
        )

        values, vectors = np.linalg.eigh(
            (
                random_covariance
                + random_covariance.T
            )
            / 2
        )

        # Frozen source code projected rounding-level negative
        # eigenvalues to the PSD boundary before simulation.
        random_covariance = (
            vectors
            @ np.diag(np.maximum(values, 0))
            @ vectors.T
        )

        participant_times = [
            [
                float(value)
                for value in group[
                    "t"
                ].to_numpy(float)
            ]
            for _, group in source_data.groupby(
                "pid",
                sort=False,
            )
        ]

        n_participants = len(participant_times)
        n_observations = sum(
            len(times)
            for times in participant_times
        )

        if n_participants != source_data["pid"].nunique():
            raise RuntimeError(
                f"Schedule participant mismatch: {regime}"
            )

        if n_observations != len(source_data):
            raise RuntimeError(
                f"Schedule observation mismatch: {regime}"
            )

        schedule_filename = (
            f"source_schedule__{regime}"
            "__restricted.json"
        )

        schedule_payloads[
            schedule_filename
        ] = {
            "schema_version": 1,
            "data_class":
                "restricted-derived exact schedule",
            "release_status":
                "restricted_input_not_distributed",
            "regime": regime,
            "source_outcome": outcome,
            "time_variable": "t",
            "time_source": "t5_mid",
            "time_scale":
                "1 unit = 5 years",
            "time_origin":
                "t5_mid = t5 - 0.5 relative to the "
                "outcome-specific anchor",
            "n_participants":
                int(n_participants),
            "n_observations":
                int(n_observations),
            "participant_times":
                participant_times,
        }

        anchor_manifest["regimes"][regime] = {
            "outcome": outcome,
            "n_participants":
                int(n_participants),
            "n_observations":
                int(n_observations),
            "outcome_sd":
                float(
                    source_data["y"].std(ddof=1)
                ),
            "fixed_intercept":
                float(
                    source_data["y"].mean()
                ),
            "fixed_time_slope":
                float(
                    reference.fe_params.get(
                        "t5_mid",
                        0.0,
                    )
                ),
            "random_effect_covariance_2x2":
                [
                    [
                        float(
                            random_covariance[
                                0,
                                0,
                            ]
                        ),
                        float(
                            random_covariance[
                                0,
                                1,
                            ]
                        ),
                    ],
                    [
                        float(
                            random_covariance[
                                1,
                                0,
                            ]
                        ),
                        float(
                            random_covariance[
                                1,
                                1,
                            ]
                        ),
                    ],
                ],
            "residual_variance":
                float(reference.scale),
            "random_intercept_only_variance":
                float(
                    np.asarray(
                        intercept_only.cov_re
                    )[0, 0]
                ),
            "random_intercept_only_residual_variance":
                float(intercept_only.scale),
            "time_scale":
                "1 unit = 5 years",
            "time_origin":
                "t5_mid = t5 - 0.5 relative to the "
                "outcome-specific anchor",
            "schedule_input":
                schedule_filename,
        }

    return (
        anchor_manifest,
        schedule_payloads,
        observed_source_fit_audit,
        formula_preflight,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        anchor_manifest,
        schedule_payloads,
        observed_source_fit_audit,
        formula_preflight,
    ) = prepare_source_anchors(args)

    manifest_path = (
        args.output_dir
        / "source_anchor_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            anchor_manifest,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    for filename, payload in schedule_payloads.items():
        (
            args.output_dir
            / filename
        ).write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    observed_source_fit_audit.to_csv(
        args.output_dir
        / "observed_source_fit_audit_restricted.csv",
        index=False,
    )

    formula_preflight.to_csv(
        args.output_dir
        / "source_formula_preflight_restricted.csv",
        index=False,
    )

    print(
        "Prepared restricted S5 source anchors and schedules."
    )
    print(
        "Anchor manifest: source_anchor_manifest.json"
    )
    print(
        "Optimizer audit: "
        "observed_source_fit_audit_restricted.csv"
    )
    print(
        "Formula preflight: "
        "source_formula_preflight_restricted.csv"
    )

    for regime, anchor in (
        anchor_manifest["regimes"].items()
    ):
        print(
            f"  {regime}: "
            f"{anchor['n_participants']} participants, "
            f"{anchor['n_observations']} observations, "
            f"schedule={anchor['schedule_input']}"
        )


if __name__ == "__main__":
    main()
