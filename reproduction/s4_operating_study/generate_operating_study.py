#!/usr/bin/env python3
"""
Generate the S4 operating-characteristic simulation.

This entry point owns:
- explicit schedule/design inputs;
- deterministic oracle-manifest reconstruction;
- historical deterministic seed assignment;
- resumable replicate generation;
- the documented targeted repair of 758 Hessian-nonpositive fits.

AABC-derived schedules and empirical generating anchors are supplied as
external inputs and are not embedded in this script.

LR calibration, operating-characteristic summaries, manuscript exports,
and Figure 2 are handled by separate deterministic scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import pandas as pd

from joblib import Parallel, delayed, parallel_config
from tqdm.auto import tqdm

from simulation_core import (
    SLOPE_BOUNDARY_REL_TOL,
    anchored_endpoint_geometry,
    covariance_metrics_from_matrix,
    fit_direct_multistart,
    fit_specialized_reml,
    schedule_predictability_summary,
    simulation_problem_from_template,
)


SEED = 20260718
SIM_ANCHORED_NEAR_THRESHOLD = 0.01

# Runtime state retains the global names used by the source notebook functions.
# Values are populated only from explicit inputs.
SCHEDULE_TEMPLATES = {}
SIMULATION_CELL_LOOKUP = {}
simulation_manifest = pd.DataFrame()
operating_replicates = pd.DataFrame()

OUTPUT_DIR = None
REPAIR_OUTPUT_DIR = None
CHECKPOINTS = {}
SIMULATION_SIGNATURE = ""

N_JOBS = 1
SIM_CHECKPOINT_BATCH = 100


def display(_object):
    """Suppress notebook-only display calls."""
    return None



def coerce_boolean(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
    }).fillna(False).astype(bool)


def build_schedule_template(name, participant_times):
    participant_times = [np.asarray(times, float) for times in participant_times]
    flat_times = np.concatenate(participant_times)
    group_index = np.concatenate([
        np.repeat(index, len(times))
        for index, times in enumerate(participant_times)
    ])
    positions_by_length = {}
    offset = 0
    for index, times in enumerate(participant_times):
        positions = np.arange(offset, offset + len(times))
        positions_by_length.setdefault(len(times), []).append((index, positions, times))
        offset += len(times)
    by_length = {}
    for length, records in positions_by_length.items():
        positions = np.vstack([record[1] for record in records])
        times = np.vstack([record[2] for record in records])
        design = np.stack([np.ones_like(times), times], axis=2)
        by_length[length] = {
            "positions": positions, "times": times, "design": design,
        }
    information = np.array([
        np.sum((times - times.mean())**2) for times in participant_times
    ])
    return {
        "name": name,
        "participant_times": participant_times,
        "flat_times": flat_times,
        "group_index": group_index,
        "participants": len(participant_times),
        "observations": len(flat_times),
        "by_length": by_length,
        "schedule_information": information,
        "reference_information": float(np.median(information)),
    }



ORIGINAL_MANIFEST_COLUMNS = [
    "cell_id",
    "cell_family",
    "schedule",
    "q1",
    "q0",
    "rho",
    "misspecification",
    "requested_reps",
    "source_outcome",
]

EXPECTED_FULL_FAMILY_COUNTS = {
    "balanced_core": 24,
    "critical_near_boundary": 12,
    "empirical_AABC": 6,
    "true_zero": 4,
}


def load_base_design_manifest(path, mode, smoke_reps):
    frame = pd.read_csv(path)

    required = [
        "cell_order",
        "cell_id",
        "cell_family",
        "schedule",
        "q0",
        "q1",
        "rho",
        "misspecification",
        "requested_reps",
    ]

    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(
            f"Design manifest is missing required columns: {missing}"
        )

    if "source_outcome" not in frame.columns:
        frame["source_outcome"] = np.nan

    frame["cell_order"] = pd.to_numeric(
        frame["cell_order"], errors="raise"
    ).astype(int)

    if frame["cell_order"].duplicated().any():
        raise ValueError("cell_order contains duplicates.")

    frame = (
        frame
        .sort_values("cell_order", kind="stable")
        .reset_index(drop=True)
    )

    expected_order = np.arange(len(frame), dtype=int)

    if not np.array_equal(
        frame["cell_order"].to_numpy(int),
        expected_order,
    ):
        raise ValueError(
            "cell_order must be consecutive 0,...,n-1 because historical "
            "cell position determines the random-number seed."
        )

    if frame["cell_id"].astype(str).duplicated().any():
        raise ValueError("cell_id contains duplicates.")

    for column in ("q0", "q1", "rho"):
        frame[column] = pd.to_numeric(
            frame[column], errors="raise"
        )

    frame["requested_reps"] = pd.to_numeric(
        frame["requested_reps"], errors="raise"
    ).astype(int)

    if (frame["requested_reps"] <= 0).any():
        raise ValueError("requested_reps must be positive.")

    if mode == "full":
        if len(frame) != 46:
            raise ValueError(
                f"Historical full manifest must contain 46 cells; "
                f"found {len(frame)}."
            )

        if int(frame["requested_reps"].sum()) != 62000:
            raise ValueError(
                "Historical full manifest must request exactly "
                "62,000 replicates."
            )

        observed_counts = (
            frame.groupby("cell_family")["cell_id"]
            .nunique()
            .to_dict()
        )

        if observed_counts != EXPECTED_FULL_FAMILY_COUNTS:
            raise ValueError(
                "Unexpected full-study family counts: "
                f"{observed_counts}"
            )

    else:
        frame["requested_reps"] = np.minimum(
            frame["requested_reps"].to_numpy(int),
            int(smoke_reps),
        )

    return frame[ORIGINAL_MANIFEST_COLUMNS].copy()


def load_schedule_templates(schedule_json, manifest, mode):
    balanced_n = 600 if mode == "full" else 40

    balanced_times = np.array(
        [-0.5, 0.0, 0.5],
        dtype=float,
    )

    templates = {
        "balanced_three_wave": build_schedule_template(
            "balanced_three_wave",
            [
                balanced_times.copy()
                for _ in range(balanced_n)
            ],
        )
    }

    needed = (
        set(manifest["schedule"].astype(str))
        - {"balanced_three_wave"}
    )

    if not needed:
        return templates

    if schedule_json is None:
        raise ValueError(
            "This manifest references empirical schedules, so "
            "--schedule-json is required."
        )

    payload = json.loads(
        Path(schedule_json).read_text(encoding="utf-8")
    )

    schedules = payload.get("schedules")
    if not isinstance(schedules, dict):
        raise ValueError(
            "Schedule JSON must contain a 'schedules' object."
        )

    missing = sorted(needed - set(schedules))
    if missing:
        raise ValueError(
            f"Schedule JSON is missing schedules: {missing}"
        )

    for name in sorted(needed):
        spec = schedules[name]

        participant_times = spec.get("participant_times")

        if not isinstance(participant_times, list):
            raise ValueError(
                f"Schedule {name!r} has invalid participant_times."
            )

        if not participant_times:
            raise ValueError(
                f"Schedule {name!r} has no participant_times."
            )

        template = build_schedule_template(
            name,
            participant_times,
        )

        if "participants" in spec:
            if int(spec["participants"]) != int(
                template["participants"]
            ):
                raise ValueError(
                    f"Participant count mismatch for {name!r}."
                )

        if "observations" in spec:
            if int(spec["observations"]) != int(
                template["observations"]
            ):
                raise ValueError(
                    f"Observation count mismatch for {name!r}."
                )

        templates[name] = template

    return templates


def add_oracle_manifest_fields(base_manifest, templates):
    oracle_rows = []

    for row in base_manifest.itertuples(index=False):
        template = templates[row.schedule]

        residual_variance = 1.0

        slope_variance = (
            row.q1
            * residual_variance
            / template["reference_information"]
            if row.q1 > 0
            else 0.0
        )

        intercept_variance = row.q0 * residual_variance

        covariance = (
            row.rho
            * np.sqrt(
                intercept_variance * slope_variance
            )
            if slope_variance > 0
            else 0.0
        )

        random_covariance = np.array(
            [
                [intercept_variance, covariance],
                [covariance, slope_variance],
            ],
            dtype=float,
        )

        oracle_metrics = schedule_predictability_summary(
            random_covariance,
            residual_variance,
            template,
        )

        if slope_variance > 0:
            geometry = anchored_endpoint_geometry(
                random_covariance
            )
        else:
            geometry = {
                "g_0_5": intercept_variance,
                "gamma_0_5": 1.0,
                "anchored_endpoint_minimum_eigenvalue_fraction": 0.0,
            }

        oracle_rows.append(
            {
                "cell_id": row.cell_id,
                "oracle_intercept_variance":
                    intercept_variance,
                "oracle_slope_variance":
                    slope_variance,
                "oracle_covariance":
                    covariance,
                "oracle_g_0_5":
                    geometry["g_0_5"],
                "oracle_gamma_0_5":
                    geometry["gamma_0_5"],
                "oracle_anchored_endpoint_minimum_eigenvalue_fraction":
                    geometry[
                        "anchored_endpoint_minimum_eigenvalue_fraction"
                    ],
                "oracle_anchored_near":
                    bool(
                        geometry[
                            "anchored_endpoint_minimum_eigenvalue_fraction"
                        ]
                        <= SIM_ANCHORED_NEAR_THRESHOLD
                    ),
                **{
                    f"oracle_{key}": value
                    for key, value
                    in oracle_metrics.items()
                },
            }
        )

    return base_manifest.merge(
        pd.DataFrame(oracle_rows),
        on="cell_id",
        validate="one_to_one",
    )


def simulation_signature_for(manifest):
    payload = {
        "version":
            "compact-operating-characteristic-v2",
        "seed":
            SEED,
        "manifest":
            json.loads(
                manifest.to_json(
                    orient="records",
                    double_precision=15,
                )
            ),
        "anchored_threshold":
            SIM_ANCHORED_NEAR_THRESHOLD,
    }

    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def load_or_initialize_checkpoint():
    checkpoint_path = CHECKPOINTS["simulation"]

    if not checkpoint_path.exists():
        return pd.DataFrame()

    checkpoint = pd.read_pickle(checkpoint_path)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Simulation checkpoint is not a dictionary."
        )

    if checkpoint.get("signature") != SIMULATION_SIGNATURE:
        raise ValueError(
            "Existing checkpoint does not match the current "
            "design/schedule/signature. Use another output directory."
        )

    frame = checkpoint.get(
        "replicates",
        pd.DataFrame(),
    ).copy()

    if len(frame):
        if frame.duplicated(
            ["cell_id", "replicate"]
        ).any():
            raise ValueError(
                "Existing checkpoint contains duplicate keys."
            )

    return frame



def generate_simulation_outcome(cell_id, replicate):
    cell = SIMULATION_CELL_LOOKUP[cell_id]
    template = SCHEDULE_TEMPLATES[cell["schedule"]]
    seed = SEED + 47_000_000 + 100_000 * simulation_manifest.index[
        simulation_manifest.cell_id.eq(cell_id)
    ][0] + int(replicate)
    rng = np.random.default_rng(seed)
    residual_variance = 1.0
    slope_variance = (
        cell["q1"] / template["reference_information"]
        if cell["q1"] > 0 else 0.0
    )
    intercept_variance = cell["q0"]
    covariance = (
        cell["rho"] * np.sqrt(intercept_variance * slope_variance)
        if slope_variance > 0 else 0.0
    )
    random_covariance = np.array([
        [intercept_variance, covariance], [covariance, slope_variance]
    ])
    random_effects = rng.multivariate_normal(
        np.zeros(2), random_covariance, size=template["participants"],
        check_valid="ignore",
    )
    times = template["flat_times"]
    group_index = template["group_index"]
    fixed_mean = 0.10 * times
    if cell["misspecification"] == "mild_curvature_max_0p25_residual_SD":
        curvature = times**2
        curvature = curvature - np.mean(curvature)
        curvature = 0.25 * curvature / np.max(np.abs(curvature))
        fixed_mean = fixed_mean + curvature
    random_mean = (
        random_effects[group_index, 0]
        + random_effects[group_index, 1] * times
    )
    errors = np.zeros(template["observations"])
    if cell["misspecification"] == "residual_AR1_rho_0p4":
        offset = 0
        for participant_times in template["participant_times"]:
            count = len(participant_times)
            orders = np.arange(count)
            correlation = 0.4 ** np.abs(orders[:, None] - orders[None, :])
            errors[offset:offset + count] = rng.multivariate_normal(
                np.zeros(count), correlation
            )
            offset += count
    else:
        error_sd = np.ones(template["observations"])
        if cell["misspecification"] == "late_visit_residual_variance_doubled":
            offset = 0
            for participant_times in template["participant_times"]:
                count = len(participant_times)
                latest = np.argmax(participant_times)
                error_sd[offset + latest] = np.sqrt(2.0)
                offset += count
        errors = rng.normal(0, error_sd)
    y_values = fixed_mean + random_mean + errors
    return y_values, seed


def run_operating_replicate(cell_id, replicate):
    cell = SIMULATION_CELL_LOOKUP[cell_id]
    template = SCHEDULE_TEMPLATES[cell["schedule"]]
    y_values, seed = generate_simulation_outcome(cell_id, replicate)
    alternative = fit_specialized_reml(template, y_values, alternative=True)
    null = fit_specialized_reml(template, y_values, alternative=False)
    fit_valid = bool(alternative.get("fit_valid", False) and null.get("fit_valid", False))
    restricted_lr = (
        max(0.0, 2 * (alternative["log_likelihood"] - null["log_likelihood"]))
        if fit_valid else np.nan
    )
    row = {
        "cell_id": cell_id, "replicate": int(replicate), "seed": int(seed),
        "fit_valid": fit_valid, "restricted_LR": restricted_lr,
        "null_converged": null.get("converged", False),
        "null_optimizer": null.get("optimizer", ""),
        "alternative_converged": alternative.get("converged", False),
        "alternative_optimizer": alternative.get("optimizer", ""),
        "alternative_optimizers_attempted": alternative.get("optimizers_attempted", 0),
        "hessian_positive_definite": alternative.get("hessian_positive_definite", False),
        "hessian_minimum_information_eigenvalue": alternative.get(
            "hessian_minimum_information_eigenvalue", np.nan
        ),
        "boundary_flag": alternative.get("boundary_flag", False),
        "error": " | ".join(filter(None, [
            null.get("error", ""), alternative.get("error", "")
        ])),
    }
    for key in (
        "random_intercept_variance", "random_slope_variance",
        "intercept_slope_covariance", "intercept_slope_correlation",
        "absolute_intercept_slope_correlation", "covariance_minimum_eigenvalue",
        "one_minus_rho_squared", "normalized_determinant",
        "innovation_variance", "innovation_fraction", "residual_variance",
        "g_0_5", "gamma_0_5", "absolute_gamma_0_5",
        "one_minus_gamma_0_5_squared",
        "anchored_endpoint_minimum_eigenvalue",
        "anchored_endpoint_minimum_eigenvalue_fraction",
    ):
        row[key] = alternative.get(key, np.nan)
    if alternative.get("fit_valid", False):
        fitted_covariance = np.array([
            [row["random_intercept_variance"], row["intercept_slope_covariance"]],
            [row["intercept_slope_covariance"], row["random_slope_variance"]],
        ])
        metrics = schedule_predictability_summary(
            fitted_covariance, row["residual_variance"], template
        )
    else:
        metrics = {
            "mean_Rdirect": np.nan, "mean_Pfull": np.nan,
            "mean_Rinnovation": np.nan, "mean_Delta": np.nan,
        }
    row.update(metrics)
    row["estimated_anchored_near"] = bool(
        np.isfinite(row["anchored_endpoint_minimum_eigenvalue_fraction"])
        and row["anchored_endpoint_minimum_eigenvalue_fraction"]
        <= SIM_ANCHORED_NEAR_THRESHOLD
    )
    return row


def save_simulation_checkpoint(frame):
    frame = frame.drop_duplicates(["cell_id", "replicate"], keep="last")
    pd.to_pickle({
        "signature": SIMULATION_SIGNATURE, "replicates": frame,
    }, CHECKPOINTS["simulation"])
    return frame


def run_simulation_cell(cell_id, requested_reps):
    global operating_replicates
    completed = set(
        operating_replicates.loc[
            operating_replicates.cell_id.eq(cell_id), "replicate"
        ].astype(int)
    ) if len(operating_replicates) else set()
    missing = [
        replicate for replicate in range(1, int(requested_reps) + 1)
        if replicate not in completed
    ]
    if not missing:
        return
    buffer = []
    with tqdm(total=len(missing), desc=f"Simulation — {cell_id}", unit="rep") as progress:
        if N_JOBS > 1:
            completed_locally = set()
            try:
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    stream = Parallel(
                        n_jobs=N_JOBS, batch_size=1, pre_dispatch=N_JOBS,
                        return_as="generator_unordered",
                    )(
                        delayed(run_operating_replicate)(cell_id, replicate)
                        for replicate in missing
                    )
                    for row in stream:
                        completed_locally.add(int(row["replicate"]))
                        buffer.append(row)
                        progress.update(1)
                        if len(buffer) >= SIM_CHECKPOINT_BATCH:
                            operating_replicates = pd.concat([
                                operating_replicates, pd.DataFrame(buffer)
                            ], ignore_index=True)
                            operating_replicates = save_simulation_checkpoint(
                                operating_replicates
                            )
                            buffer = []
            except Exception as exc:
                warnings.warn(
                    f"Parallel simulation execution stopped ({exc!r}); "
                    "only unfinished replicates will continue serially."
                )
                for replicate in missing:
                    if replicate in completed_locally:
                        continue
                    buffer.append(run_operating_replicate(cell_id, replicate))
                    progress.update(1)
                    if len(buffer) >= SIM_CHECKPOINT_BATCH:
                        operating_replicates = pd.concat([
                            operating_replicates, pd.DataFrame(buffer)
                        ], ignore_index=True)
                        operating_replicates = save_simulation_checkpoint(
                            operating_replicates
                        )
                        buffer = []
        else:
            for replicate in missing:
                buffer.append(run_operating_replicate(cell_id, replicate))
                progress.update(1)
                if len(buffer) >= SIM_CHECKPOINT_BATCH:
                    operating_replicates = pd.concat([
                        operating_replicates, pd.DataFrame(buffer)
                    ], ignore_index=True)
                    operating_replicates = save_simulation_checkpoint(
                        operating_replicates
                    )
                    buffer = []
        if buffer:
            operating_replicates = pd.concat([
                operating_replicates, pd.DataFrame(buffer)
            ], ignore_index=True)
            operating_replicates = save_simulation_checkpoint(operating_replicates)


def repair_one_operating_replicate(cell_id, replicate):
    try:
        cell = SIMULATION_CELL_LOOKUP[cell_id]
        template = SCHEDULE_TEMPLATES[cell["schedule"]]
        y_values, seed = generate_simulation_outcome(
            cell_id, int(replicate)
        )
        problem = simulation_problem_from_template(template, y_values)
        alternative_fit = fit_direct_multistart(
            problem, alternative=True, maxiter=700
        )
        null_fit = fit_direct_multistart(
            problem, alternative=False, maxiter=700
        )
        alternative_selected = alternative_fit["selected"]
        null_selected = null_fit["selected"]
        basic_fit_valid = bool(
            alternative_fit["basic_panel_valid"]
            and null_fit["basic_panel_valid"]
            and alternative_selected is not None
            and null_selected is not None
        )
        strict_fit_valid = bool(
            alternative_fit["strict_panel_valid"]
            and null_fit["strict_panel_valid"]
        )
        row = {
            "cell_id": cell_id,
            "replicate": int(replicate),
            "seed": int(seed),
            "fit_valid": basic_fit_valid,
            "strict_fit_valid": strict_fit_valid,
            "restricted_LR": (
                max(0.0, 2 * (
                    alternative_selected["log_likelihood"]
                    - null_selected["log_likelihood"]
                )) if basic_fit_valid else np.nan
            ),
            "null_converged": bool(null_fit["basic_panel_valid"]),
            "null_optimizer": "L-BFGS-B+Powell multistart",
            "alternative_converged": bool(
                alternative_fit["basic_panel_valid"]
            ),
            "alternative_optimizer": "L-BFGS-B+Powell multistart",
            "alternative_optimizers_attempted": 2,
            "optimizer_agreement": bool(
                alternative_fit["optimizer_agreement"]
            ),
            "null_optimizer_agreement": bool(
                null_fit["optimizer_agreement"]
            ),
            "repair_attempted": True,
            "repair_succeeded": strict_fit_valid,
            "repair_method": "multistart_vectorized_direct_REML_LBFGSB_and_Powell",
            "error": "",
        }
        if alternative_selected is None:
            row.update({
                "hessian_positive_definite": False,
                "hessian_minimum_information_eigenvalue": np.nan,
                "boundary_flag": True,
            })
            return row

        method_hessians = [
            item["hessian_minimum_information_eigenvalue"]
            for item in alternative_fit["audit"]
        ]
        random_covariance = alternative_selected["random_covariance"]
        residual_variance = float(
            alternative_selected["residual_variance"]
        )
        covariance_metrics = covariance_metrics_from_matrix(
            random_covariance
        )
        row.update(covariance_metrics)
        row["residual_variance"] = residual_variance
        row["hessian_positive_definite"] = bool(
            alternative_fit["strict_panel_valid"]
        )
        row["hessian_minimum_information_eigenvalue"] = float(
            np.nanmin(method_hessians)
        )
        row["boundary_flag"] = bool(
            random_covariance[1, 1]
            <= SLOPE_BOUNDARY_REL_TOL * max(1.0, residual_variance)
            or covariance_metrics["covariance_minimum_eigenvalue"]
            <= SLOPE_BOUNDARY_REL_TOL * max(
                1.0, float(np.trace(random_covariance))
            )
        )
        row.update(schedule_predictability_summary(
            random_covariance, residual_variance, template
        ))
        row["estimated_anchored_near"] = bool(
            np.isfinite(row[
                "anchored_endpoint_minimum_eigenvalue_fraction"
            ])
            and row[
                "anchored_endpoint_minimum_eigenvalue_fraction"
            ] <= SIM_ANCHORED_NEAR_THRESHOLD
        )
        return row
    except Exception as exc:
        return {
            "cell_id": cell_id,
            "replicate": int(replicate),
            "fit_valid": False,
            "strict_fit_valid": False,
            "hessian_positive_definite": False,
            "optimizer_agreement": False,
            "repair_attempted": True,
            "repair_succeeded": False,
            "repair_method": "multistart_vectorized_direct_REML_LBFGSB_and_Powell",
            "error": repr(exc),
        }


def run_historical_targeted_repair(
    operating_base,
    simulation_suspects,
):
    # Targeted refit — exactly the 758 Hessian-nonpositive simulation fits
    SIMULATION_REPAIR_CHECKPOINT = (
        REPAIR_OUTPUT_DIR / "targeted_refit_audit.csv"
    )
    suspect_keys = set(zip(
        simulation_suspects.cell_id.astype(str),
        simulation_suspects.replicate.astype(int),
    ))
    assert len(suspect_keys) == 758
    if SIMULATION_REPAIR_CHECKPOINT.exists():
        simulation_repairs = pd.read_csv(SIMULATION_REPAIR_CHECKPOINT)
    else:
        simulation_repairs = pd.DataFrame()
    completed_repair_keys = set(zip(
        simulation_repairs.get(
            "cell_id", pd.Series(dtype=str)
        ).astype(str),
        pd.to_numeric(
            simulation_repairs.get("replicate", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(-1).astype(int),
    ))
    pending_simulation_repairs = sorted(
        suspect_keys - completed_repair_keys
    )

    def save_simulation_repairs(frame):
        frame = (
            frame.drop_duplicates(["cell_id", "replicate"], keep="last")
            .sort_values(["cell_id", "replicate"])
            .reset_index(drop=True)
        )
        frame.to_csv(SIMULATION_REPAIR_CHECKPOINT, index=False)
        return frame

    simulation_repair_buffer = []
    with tqdm(
        total=len(pending_simulation_repairs),
        desc="Repair 758 suspect simulation fits",
        unit="fit",
    ) as progress:
        if N_JOBS > 1 and pending_simulation_repairs:
            completed_locally = set()
            try:
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    stream = Parallel(
                        n_jobs=N_JOBS, batch_size=1, pre_dispatch=N_JOBS,
                        return_as="generator_unordered",
                    )(
                        delayed(repair_one_operating_replicate)(
                            cell_id, replicate
                        )
                        for cell_id, replicate in pending_simulation_repairs
                    )
                    for row in stream:
                        completed_locally.add((
                            str(row["cell_id"]), int(row["replicate"])
                        ))
                        simulation_repair_buffer.append(row)
                        progress.update(1)
                        if len(simulation_repair_buffer) >= max(2, N_JOBS):
                            simulation_repairs = pd.concat([
                                simulation_repairs,
                                pd.DataFrame(simulation_repair_buffer),
                            ], ignore_index=True, sort=False)
                            simulation_repairs = save_simulation_repairs(
                                simulation_repairs
                            )
                            simulation_repair_buffer = []
            except Exception as exc:
                warnings.warn(
                    f"Parallel simulation repair stopped ({exc!r}); "
                    "unfinished fits will continue serially."
                )
                for cell_id, replicate in pending_simulation_repairs:
                    if (cell_id, replicate) in completed_locally:
                        continue
                    simulation_repair_buffer.append(
                        repair_one_operating_replicate(cell_id, replicate)
                    )
                    progress.update(1)
                    if len(simulation_repair_buffer) >= max(2, N_JOBS):
                        simulation_repairs = pd.concat([
                            simulation_repairs,
                            pd.DataFrame(simulation_repair_buffer),
                        ], ignore_index=True, sort=False)
                        simulation_repairs = save_simulation_repairs(
                            simulation_repairs
                        )
                        simulation_repair_buffer = []
        else:
            for cell_id, replicate in pending_simulation_repairs:
                simulation_repair_buffer.append(
                    repair_one_operating_replicate(cell_id, replicate)
                )
                progress.update(1)
                if len(simulation_repair_buffer) >= max(2, N_JOBS):
                    simulation_repairs = pd.concat([
                        simulation_repairs,
                        pd.DataFrame(simulation_repair_buffer),
                    ], ignore_index=True, sort=False)
                    simulation_repairs = save_simulation_repairs(
                        simulation_repairs
                    )
                    simulation_repair_buffer = []
        if simulation_repair_buffer:
            simulation_repairs = pd.concat([
                simulation_repairs,
                pd.DataFrame(simulation_repair_buffer),
            ], ignore_index=True, sort=False)
            simulation_repairs = save_simulation_repairs(
                simulation_repairs
            )

    repaired_keys = set(zip(
        simulation_repairs.cell_id.astype(str),
        simulation_repairs.replicate.astype(int),
    ))
    assert suspect_keys.issubset(repaired_keys)

    operating_replicates_repaired = operating_base.copy()
    operating_replicates_repaired["repair_attempted"] = False
    operating_replicates_repaired["repair_succeeded"] = False
    operating_replicates_repaired["optimizer_agreement"] = np.nan
    operating_replicates_repaired["strict_fit_valid"] = (
        coerce_boolean(operating_replicates_repaired.fit_valid)
        & coerce_boolean(
            operating_replicates_repaired.hessian_positive_definite
        )
    )
    operating_replicates_repaired = (
        operating_replicates_repaired
        .set_index(["cell_id", "replicate"], drop=False)
    )
    replacement_columns = [
        column for column in simulation_repairs.columns
        if column not in ("cell_id", "replicate")
    ]
    for repair_row in simulation_repairs.itertuples(index=False):
        key = (str(repair_row.cell_id), int(repair_row.replicate))
        if key not in operating_replicates_repaired.index:
            raise KeyError(f"Repair key absent from original simulation: {key}")
        for column in replacement_columns:
            if column not in operating_replicates_repaired.columns:
                operating_replicates_repaired[column] = np.nan
            operating_replicates_repaired.at[key, column] = getattr(
                repair_row, column
            )
    operating_replicates_repaired = (
        operating_replicates_repaired.reset_index(drop=True)
    )
    assert len(operating_replicates_repaired) == 62000
    assert not operating_replicates_repaired.duplicated(
        ["cell_id", "replicate"]
    ).any()
    assert int(coerce_boolean(
        operating_replicates_repaired.repair_attempted
    ).sum()) == 758

    simulation_repairs.to_csv(
        OUTPUT_DIR / "targeted_refit_audit.csv",
        index=False,
    )
    operating_replicates_repaired.to_csv(
        OUTPUT_DIR / "operating_study_replicate_results.csv",
        index=False,
    )
    display(pd.DataFrame([{
        "targeted_fits": 758,
        "repairs_completed": len(simulation_repairs),
        "strict_repairs": int(coerce_boolean(
            simulation_repairs.strict_fit_valid
        ).sum()),
        "remaining_strict_invalid_total": int((
            ~coerce_boolean(operating_replicates_repaired.strict_fit_valid)
        ).sum()),
    }]))
    return operating_replicates_repaired




def run_generation(args):
    global OUTPUT_DIR
    global REPAIR_OUTPUT_DIR
    global CHECKPOINTS
    global N_JOBS
    global SIM_CHECKPOINT_BATCH
    global SCHEDULE_TEMPLATES
    global simulation_manifest
    global SIMULATION_CELL_LOOKUP
    global SIMULATION_SIGNATURE
    global operating_replicates

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPAIR_OUTPUT_DIR = (
        OUTPUT_DIR / "repair_checkpoints"
    )
    REPAIR_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    N_JOBS = max(
        1,
        min(int(args.n_jobs), 3),
    )

    SIM_CHECKPOINT_BATCH = (
        100 if args.mode == "full" else 2
    )

    CHECKPOINTS = {
        "simulation":
            OUTPUT_DIR
            / "operating_study_generation_checkpoint.pkl",
    }

    base_manifest = load_base_design_manifest(
        Path(args.design_manifest),
        args.mode,
        args.smoke_reps,
    )

    SCHEDULE_TEMPLATES = load_schedule_templates(
        args.schedule_json,
        base_manifest,
        args.mode,
    )

    simulation_manifest = (
        add_oracle_manifest_fields(
            base_manifest,
            SCHEDULE_TEMPLATES,
        )
        .reset_index(drop=True)
    )

    SIMULATION_CELL_LOOKUP = (
        simulation_manifest
        .set_index("cell_id")
        .to_dict("index")
    )

    SIMULATION_SIGNATURE = simulation_signature_for(
        simulation_manifest
    )

    simulation_manifest.to_csv(
        OUTPUT_DIR
        / "operating_study_design.csv",
        index=False,
    )

    operating_replicates = (
        load_or_initialize_checkpoint()
    )

    for cell in simulation_manifest.itertuples(
        index=False
    ):
        run_simulation_cell(
            cell.cell_id,
            int(cell.requested_reps),
        )

    expected_total = int(
        simulation_manifest[
            "requested_reps"
        ].sum()
    )

    if len(operating_replicates) != expected_total:
        raise RuntimeError(
            "Generation ended without all requested "
            f"replicate rows: "
            f"{len(operating_replicates):,}/"
            f"{expected_total:,}."
        )

    if operating_replicates.duplicated(
        ["cell_id", "replicate"]
    ).any():
        raise RuntimeError(
            "Generated replicate table contains "
            "duplicate keys."
        )

    operating_base = operating_replicates.merge(
        simulation_manifest,
        on="cell_id",
        how="left",
        validate="many_to_one",
    )

    operating_base.to_csv(
        OUTPUT_DIR
        / "operating_study_replicate_results_before_targeted_refit.csv",
        index=False,
    )

    if args.mode == "smoke":
        smoke_repaired = operating_base.copy()
        smoke_repaired["repair_attempted"] = False
        smoke_repaired["repair_succeeded"] = False
        smoke_repaired["optimizer_agreement"] = np.nan
        smoke_repaired["strict_fit_valid"] = (
            coerce_boolean(smoke_repaired["fit_valid"])
            & coerce_boolean(
                smoke_repaired["hessian_positive_definite"]
            )
        )
        smoke_repaired.to_csv(
            OUTPUT_DIR / "operating_study_replicate_results.csv",
            index=False,
        )
        smoke_manifest = {
            "schema_version": 1,
            "study": "S4 operating-characteristic simulation",
            "mode": "smoke",
            "seed": SEED,
            "design_cells": int(
                simulation_manifest["cell_id"].nunique()
            ),
            "requested_replicates": expected_total,
            "completed_replicates": int(len(smoke_repaired)),
            "targeted_refit_rows": 0,
            "repair_attempted_rows": 0,
            "design_manifest_file": Path(args.design_manifest).name,
            "schedule_file": (
                Path(args.schedule_json).name
                if args.schedule_json is not None
                else None
            ),
            "outputs": {
                "design": "operating_study_design.csv",
                "pre_repair_replicates":
                    "operating_study_replicate_results_before_targeted_refit.csv",
                "replicate_results":
                    "operating_study_replicate_results.csv",
            },
            "reportable": False,
        }
        (OUTPUT_DIR / "operating_study_run_manifest.json").write_text(
            json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "Smoke generation complete: "
            f"{len(smoke_repaired):,} rows. "
            "The historical 758-fit repair was not run in smoke mode. "
            "The replicate file is compatible with the summarizer but is "
            "not suitable for scientific reporting."
        )
        return

    simulation_suspects = operating_base.loc[
        ~coerce_boolean(
            operating_base[
                "hessian_positive_definite"
            ]
        )
    ].copy()

    if len(simulation_suspects) != 758:
        raise RuntimeError(
            "Full historical run did not reproduce "
            "the documented 758 Hessian-nonpositive "
            "fits before repair; "
            f"found {len(simulation_suspects)}."
        )

    repaired = run_historical_targeted_repair(
        operating_base,
        simulation_suspects,
    )

    if len(repaired) != 62000:
        raise RuntimeError(
            f"Repaired table has {len(repaired):,} "
            "rows; expected 62,000."
        )

    remaining_strict_invalid = int(
        (
            ~coerce_boolean(
                repaired["strict_fit_valid"]
            )
        ).sum()
    )

    repair_attempted_count = int(
        coerce_boolean(
            repaired["repair_attempted"]
        ).sum()
    )

    if repair_attempted_count != 758:
        raise RuntimeError(
            "Historical full run must mark exactly 758 "
            "rows as repair_attempted; found "
            f"{repair_attempted_count}."
        )

    if remaining_strict_invalid != 82:
        raise RuntimeError(
            "Historical full run must end with exactly "
            "82 strict-invalid rows; found "
            f"{remaining_strict_invalid}."
        )

    targeted_refit_path = (
        OUTPUT_DIR / "targeted_refit_audit.csv"
    )

    if not targeted_refit_path.is_file():
        raise RuntimeError(
            "Targeted-refit audit was not written."
        )

    targeted_refit_audit = pd.read_csv(
        targeted_refit_path,
        low_memory=False,
    )

    if len(targeted_refit_audit) != 758:
        raise RuntimeError(
            "Targeted-refit audit must contain exactly "
            f"758 rows; found {len(targeted_refit_audit)}."
        )

    repair_completion_checks = pd.DataFrame(
        [
            {
                "check": "requested_replicates",
                "expected": expected_total,
                "observed": int(
                    simulation_manifest[
                        "requested_reps"
                    ].sum()
                ),
                "passed": True,
            },
            {
                "check": "completed_replicate_rows",
                "expected": expected_total,
                "observed": len(repaired),
                "passed": len(repaired) == expected_total,
            },
            {
                "check": "unique_replicate_keys",
                "expected": expected_total,
                "observed": repaired[
                    ["cell_id", "replicate"]
                ].drop_duplicates().shape[0],
                "passed": (
                    repaired[
                        ["cell_id", "replicate"]
                    ].drop_duplicates().shape[0]
                    == expected_total
                ),
            },
            {
                "check": "targeted_refit_rows",
                "expected": 758,
                "observed": len(targeted_refit_audit),
                "passed": len(targeted_refit_audit) == 758,
            },
            {
                "check": "repair_attempted_rows",
                "expected": 758,
                "observed": repair_attempted_count,
                "passed": repair_attempted_count == 758,
            },
            {
                "check": "remaining_strict_invalid_rows",
                "expected": 82,
                "observed": remaining_strict_invalid,
                "passed": remaining_strict_invalid == 82,
            },
        ]
    )

    if not repair_completion_checks["passed"].all():
        raise RuntimeError(
            "One or more repair completion checks failed."
        )

    repair_completion_checks.to_csv(
        OUTPUT_DIR / "repair_completion_checks.csv",
        index=False,
    )

    run_manifest = {
        "schema_version": 1,
        "study": "S4 operating-characteristic simulation",
        "mode": args.mode,
        "seed": SEED,
        "simulation_signature": SIMULATION_SIGNATURE,
        "design_cells": int(
            simulation_manifest["cell_id"].nunique()
        ),
        "requested_replicates": expected_total,
        "completed_replicates": int(len(repaired)),
        "targeted_refit_rows": int(
            len(targeted_refit_audit)
        ),
        "repair_attempted_rows": repair_attempted_count,
        "remaining_strict_invalid_rows":
            remaining_strict_invalid,
        "design_manifest_file":
            Path(args.design_manifest).name,
        "schedule_file": (
            Path(args.schedule_json).name
            if args.schedule_json is not None
            else None
        ),
        "outputs": {
            "design":
                "operating_study_design.csv",
            "pre_repair_replicates":
                "operating_study_replicate_results_before_targeted_refit.csv",
            "replicate_results":
                "operating_study_replicate_results.csv",
            "targeted_refit_audit":
                "targeted_refit_audit.csv",
            "repair_completion_checks":
                "repair_completion_checks.csv",
        },
        "historical_repair_note": (
            "The original run contained 758 "
            "Hessian-nonpositive fits that were targeted "
            "for multistart refitting. The final repaired "
            "62,000-row export retains 82 strict-invalid fits."
        ),
    }

    (
        OUTPUT_DIR
        / "operating_study_run_manifest.json"
    ).write_text(
        json.dumps(
            run_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Full operating-study generation complete: "
        f"{len(repaired):,} rows; "
        "758 fits targeted for historical repair; "
        f"{remaining_strict_invalid} strict-invalid "
        "rows remain."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate S4 operating-study replicates "
            "from explicit design and schedule inputs."
        )
    )

    parser.add_argument(
        "--design-manifest",
        type=Path,
        required=True,
        help=(
            "Ordered design manifest. Historical full "
            "reproduction requires cell_order 0..45."
        ),
    )

    parser.add_argument(
        "--schedule-json",
        type=Path,
        default=None,
        help=(
            "Restricted-derived empirical schedule JSON. "
            "Required when the manifest references "
            "empirical schedules."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        required=True,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--smoke-reps",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    if args.smoke_reps < 1:
        parser.error(
            "--smoke-reps must be >= 1"
        )

    return args


def main():
    run_generation(parse_args())


if __name__ == "__main__":
    main()
