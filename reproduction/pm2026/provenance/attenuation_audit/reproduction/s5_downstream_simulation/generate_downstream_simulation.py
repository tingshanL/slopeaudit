#!/usr/bin/env python3
"""Generate raw results for the S5 downstream simulation.

This entry point contains no AABC file discovery and no observed-data reads.
It consumes an explicitly supplied source-anchor manifest plus the exact
schedule files named by that manifest.

Empirical AABC-derived anchors and schedules are restricted inputs and are not
distributed with the repository.

The full design uses 500 replicates per cell:
    4 regimes x
    [500 LR calibration
     + 500 independent-null LR evaluation
     + 2 x 500 downstream evaluation]
    = 8,000 tasks.

This generator stops at raw task/model diagnostics. LR calibration,
method-performance summaries, singularity analyses, figures, and final
semantic exports are handled by summarize_downstream_simulation.py.
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

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from simulation_core import run_evaluation_task, run_lr_task


SEED = 20260718
FULL_REPS_PER_CELL = 500

REGIME_TO_OUTCOME = {
    "high_slope_information": "lateral_ventricles",
    "intermediate": "hippocampus",
    "near_boundary": "global_cortical_att",
    "true_zero_slope_variance": "fluid_cognition",
}

OPTIMIZERS = ("lbfgs", "powell", "bfgs")

HIGH_FALSE_FULL_THRESHOLD = 0.40


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--anchor-manifest",
        required=True,
        type=Path,
        help=(
            "Explicit source_anchor_manifest.json. Empirical AABC-derived "
            "manifests are restricted inputs and are not distributed."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help=(
            "smoke: small fixed-seed run; "
            "full: completed 500-replicate-per-cell study design."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel workers. The reference analysis used 2; choose a value "
            "within available memory."
        ),
    )

    parser.add_argument(
        "--reps-per-cell",
        type=int,
        default=None,
        help=(
            "Override replicate count. Normally omit. "
            "full defaults to 500; smoke defaults to 1."
        ),
    )

    parser.add_argument(
        "--regime",
        action="append",
        choices=tuple(REGIME_TO_OUTCOME),
        default=None,
        help=(
            "Restrict execution to named regime(s). May be repeated. "
            "Full mode requires all four regimes."
        ),
    )

    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} does not exist or is not a file."
        )

    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_source_parameters(
    anchor_manifest_path: Path,
    selected_regimes: list[str],
):
    manifest_path = require_file(
        anchor_manifest_path,
        "source-anchor manifest",
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if "regimes" not in manifest:
        raise ValueError(
            "Anchor manifest lacks top-level 'regimes'."
        )

    manifest_regimes = manifest["regimes"]

    missing = (
        set(selected_regimes)
        - set(manifest_regimes)
    )

    if missing:
        raise ValueError(
            "Requested regimes absent from anchor manifest: "
            + ", ".join(sorted(missing))
        )

    source_parameters = {}
    schedule_hashes = {}

    required_anchor_fields = {
        "outcome",
        "n_participants",
        "n_observations",
        "outcome_sd",
        "fixed_intercept",
        "fixed_time_slope",
        "random_effect_covariance_2x2",
        "residual_variance",
        "random_intercept_only_variance",
        "random_intercept_only_residual_variance",
        "time_scale",
        "time_origin",
        "schedule_input",
    }

    for regime in selected_regimes:
        anchor = manifest_regimes[regime]

        missing_fields = (
            required_anchor_fields
            - set(anchor)
        )

        if missing_fields:
            raise ValueError(
                f"{regime}: missing anchor fields: "
                + ", ".join(sorted(missing_fields))
            )

        expected_outcome = REGIME_TO_OUTCOME[regime]

        if anchor["outcome"] != expected_outcome:
            raise ValueError(
                f"{regime}: expected source outcome "
                f"{expected_outcome!r}, got "
                f"{anchor['outcome']!r}."
            )

        schedule_name = anchor["schedule_input"]
        schedule_relative = Path(schedule_name)

        if (
            schedule_relative.is_absolute()
            or schedule_relative.name
            != str(schedule_relative)
        ):
            raise ValueError(
                f"{regime}: schedule_input must be a "
                "filename relative to the manifest directory."
            )

        schedule_path = require_file(
            manifest_path.parent
            / schedule_relative,
            f"{regime} schedule",
        )

        schedule = json.loads(
            schedule_path.read_text(
                encoding="utf-8"
            )
        )

        if schedule.get("regime") != regime:
            raise ValueError(
                f"{regime}: schedule regime mismatch."
            )

        if (
            schedule.get("source_outcome")
            != anchor["outcome"]
        ):
            raise ValueError(
                f"{regime}: schedule outcome mismatch."
            )

        participant_times = schedule.get(
            "participant_times"
        )

        if not isinstance(
            participant_times,
            list,
        ):
            raise ValueError(
                f"{regime}: participant_times must be a list."
            )

        rows = []

        for participant_index, times in enumerate(
            participant_times,
            start=1,
        ):
            values = np.asarray(
                times,
                dtype=float,
            )

            if (
                values.ndim != 1
                or len(values) < 2
                or not np.isfinite(values).all()
                or np.ptp(values) <= 0
            ):
                raise ValueError(
                    f"{regime}: invalid schedule for "
                    f"anonymous participant "
                    f"{participant_index}."
                )

            if not np.all(
                np.diff(values) >= 0
            ):
                raise ValueError(
                    f"{regime}: participant times "
                    "must be nondecreasing."
                )

            pid = (
                f"anon_"
                f"{participant_index:04d}"
            )

            for value in values:
                rows.append(
                    {
                        "pid": pid,
                        "t": float(value),
                    }
                )

        data = pd.DataFrame(
            rows,
            columns=["pid", "t"],
        )

        if (
            data["pid"].nunique()
            != int(anchor["n_participants"])
        ):
            raise ValueError(
                f"{regime}: participant count differs "
                "between anchor and schedule."
            )

        if (
            len(data)
            != int(anchor["n_observations"])
        ):
            raise ValueError(
                f"{regime}: observation count differs "
                "between anchor and schedule."
            )

        random_covariance = np.asarray(
            anchor[
                "random_effect_covariance_2x2"
            ],
            dtype=float,
        )

        if random_covariance.shape != (2, 2):
            raise ValueError(
                f"{regime}: expected a 2x2 "
                "random-effect covariance matrix."
            )

        if not np.isfinite(
            random_covariance
        ).all():
            raise ValueError(
                f"{regime}: covariance contains "
                "non-finite values."
            )

        if not np.allclose(
            random_covariance,
            random_covariance.T,
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{regime}: covariance is not symmetric."
            )

        source_parameters[regime] = {
            "outcome":
                anchor["outcome"],
            "data":
                data,
            "outcome_SD":
                float(
                    anchor["outcome_sd"]
                ),
            "beta0":
                float(
                    anchor["fixed_intercept"]
                ),
            "beta_t":
                float(
                    anchor["fixed_time_slope"]
                ),
            "reference_random_covariance":
                random_covariance,
            "reference_residual_variance":
                float(
                    anchor[
                        "residual_variance"
                    ]
                ),
            "RI_intercept_variance":
                float(
                    anchor[
                        "random_intercept_only_variance"
                    ]
                ),
            "RI_residual_variance":
                float(
                    anchor[
                        "random_intercept_only_residual_variance"
                    ]
                ),
        }

        schedule_hashes[
            regime
        ] = sha256_file(schedule_path)

    return (
        source_parameters,
        sha256_file(manifest_path),
        schedule_hashes,
    )


def build_task_manifest(
    regimes: list[str],
    reps_per_cell: int,
):
    rows = []

    # Preserve the recorded regime indices when a smoke run selects only a
    # subset so each regime retains its original random seeds.
    regime_index = {
        regime: index
        for index, regime in enumerate(
            REGIME_TO_OUTCOME
        )
    }

    for regime in regimes:
        index = regime_index[regime]

        for replicate in range(
            1,
            reps_per_cell + 1,
        ):
            rows.append(
                {
                    "row_type":
                        "lr_calibration",
                    "regime":
                        regime,
                    "effect_condition":
                        "beta_xt_zero",
                    "replicate":
                        replicate,
                    "seed":
                        SEED
                        + 3_000_000
                        + 100_000 * index
                        + replicate,
                }
            )

            rows.append(
                {
                    "row_type":
                        "lr_null_evaluation",
                    "regime":
                        regime,
                    "effect_condition":
                        "beta_xt_zero",
                    "replicate":
                        replicate,
                    "seed":
                        SEED
                        + 4_000_000
                        + 100_000 * index
                        + replicate,
                }
            )

        for effect_index, effect_condition in enumerate(
            (
                "beta_xt_zero",
                "beta_xt_0.10_SD_per_5y",
            )
        ):
            for replicate in range(
                1,
                reps_per_cell + 1,
            ):
                rows.append(
                    {
                        "row_type":
                            "evaluation",
                        "regime":
                            regime,
                        "effect_condition":
                            effect_condition,
                        "replicate":
                            replicate,
                        "seed":
                            SEED
                            + 5_000_000
                            + 100_000 * index
                            + 10_000
                            * effect_index
                            + replicate,
                    }
                )

    manifest = pd.DataFrame(rows)

    manifest["task_id"] = (
        manifest["row_type"]
        + "__"
        + manifest["regime"]
        + "__"
        + manifest["effect_condition"]
        + "__"
        + manifest["replicate"].astype(str)
    )

    expected = (
        len(regimes)
        * 4
        * reps_per_cell
    )

    if len(manifest) != expected:
        raise RuntimeError(
            "Task-manifest row count mismatch."
        )

    if (
        manifest["task_id"].nunique()
        != len(manifest)
    ):
        raise RuntimeError(
            "Task IDs are not unique."
        )

    return manifest


def execute_task(
    task: dict,
    source_parameters: dict,
):
    if task["row_type"] == "evaluation":
        result = run_evaluation_task(
            source_parameters,
            task["regime"],
            task["effect_condition"],
            task["replicate"],
            task["seed"],
        )
    else:
        result = run_lr_task(
            source_parameters,
            task["row_type"],
            task["regime"],
            task["replicate"],
            task["seed"],
        )

    (
        replicate_row,
        diagnostics,
        attempts,
    ) = result

    replicate_row["task_id"] = task[
        "task_id"
    ]

    for row in diagnostics:
        row["task_id"] = task[
            "task_id"
        ]

    for row in attempts:
        row["task_id"] = task[
            "task_id"
        ]

    return (
        replicate_row,
        diagnostics,
        attempts,
    )


def checkpoint_signature(
    reps_per_cell: int,
    regimes: list[str],
    manifest_sha256: str,
    schedule_sha256: dict,
):
    payload = {
        "seed":
            SEED,
        "reps_per_cell":
            reps_per_cell,
        "optimizers":
            OPTIMIZERS,
        "regimes":
            {
                regime:
                    REGIME_TO_OUTCOME[
                        regime
                    ]
                for regime in regimes
            },
        "high_full_threshold":
            HIGH_FALSE_FULL_THRESHOLD,
        "anchor_manifest_sha256":
            manifest_sha256,
        "schedule_sha256":
            schedule_sha256,
    }

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def run_generator(
    source_parameters: dict,
    task_manifest: pd.DataFrame,
    output_dir: Path,
    workers: int,
    signature: str,
):
    checkpoint_file = (
        output_dir
        / "_downstream_simulation_checkpoint.pkl"
    )

    if checkpoint_file.exists():
        checkpoint = pd.read_pickle(
            checkpoint_file
        )

        if (
            checkpoint.get("signature")
            != signature
        ):
            raise RuntimeError(
                "Checkpoint configuration differs. "
                "Move or delete the checkpoint before restarting."
            )

        simulation_replicates = (
            checkpoint.get(
                "simulation_replicates",
                pd.DataFrame(),
            )
        )

        model_fit_diagnostics = (
            checkpoint.get(
                "model_fit_diagnostics",
                pd.DataFrame(),
            )
        )

        optimizer_attempt_audit = (
            checkpoint.get(
                "optimizer_attempt_audit",
                pd.DataFrame(),
            )
        )

    else:
        simulation_replicates = (
            pd.DataFrame()
        )
        model_fit_diagnostics = (
            pd.DataFrame()
        )
        optimizer_attempt_audit = (
            pd.DataFrame()
        )

    completed = (
        set(
            simulation_replicates[
                "task_id"
            ].astype(str)
        )
        if not simulation_replicates.empty
        else set()
    )

    pending = (
        task_manifest.loc[
            ~task_manifest[
                "task_id"
            ].isin(completed)
        ]
        .to_dict("records")
    )

    progress = tqdm(
        total=len(pending),
        desc="Downstream simulation",
        unit="task",
    )

    batch_size = max(
        2,
        2 * workers,
    )

    for start in range(
        0,
        len(pending),
        batch_size,
    ):
        batch = pending[
            start:
            start + batch_size
        ]

        if workers == 1:
            batch_results = [
                execute_task(
                    task,
                    source_parameters,
                )
                for task in batch
            ]
        else:
            batch_results = Parallel(
                n_jobs=workers,
                backend="loky",
                batch_size=1,
                pre_dispatch=workers,
            )(
                delayed(execute_task)(
                    task,
                    source_parameters,
                )
                for task in batch
            )

        replicate_rows = []
        diagnostic_rows = []
        attempt_rows = []

        for (
            replicate_row,
            diagnostics,
            attempts,
        ) in batch_results:
            replicate_rows.append(
                replicate_row
            )
            diagnostic_rows.extend(
                diagnostics
            )
            attempt_rows.extend(
                attempts
            )

        simulation_replicates = pd.concat(
            [
                simulation_replicates,
                pd.DataFrame(
                    replicate_rows
                ),
            ],
            ignore_index=True,
        )

        model_fit_diagnostics = pd.concat(
            [
                model_fit_diagnostics,
                pd.DataFrame(
                    diagnostic_rows
                ),
            ],
            ignore_index=True,
        )

        optimizer_attempt_audit = pd.concat(
            [
                optimizer_attempt_audit,
                pd.DataFrame(
                    attempt_rows
                ),
            ],
            ignore_index=True,
        )

        pd.to_pickle(
            {
                "signature":
                    signature,
                "simulation_replicates":
                    simulation_replicates,
                "model_fit_diagnostics":
                    model_fit_diagnostics,
                "optimizer_attempt_audit":
                    optimizer_attempt_audit,
            },
            checkpoint_file,
        )

        progress.update(
            len(batch)
        )

    progress.close()

    simulation_replicates = (
        simulation_replicates
        .drop_duplicates(
            "task_id",
            keep="last",
        )
    )

    model_fit_diagnostics = (
        model_fit_diagnostics
        .drop_duplicates(
            [
                "task_id",
                "model_role",
            ],
            keep="last",
        )
    )

    optimizer_attempt_audit = (
        optimizer_attempt_audit
        .drop_duplicates(
            [
                "task_id",
                "model_role",
                "optimizer",
            ],
            keep="last",
        )
    )

    if (
        len(simulation_replicates)
        != len(task_manifest)
    ):
        raise RuntimeError(
            "Completed replicate count differs "
            "from task manifest."
        )

    return (
        simulation_replicates,
        model_fit_diagnostics,
        optimizer_attempt_audit,
    )


def write_raw_outputs(
    output_dir: Path,
    task_manifest: pd.DataFrame,
    simulation_replicates: pd.DataFrame,
    model_fit_diagnostics: pd.DataFrame,
    optimizer_attempt_audit: pd.DataFrame,
):
    # These are raw generator products. Calibration and final semantic
    # summaries are produced by summarize_downstream_simulation.py.
    outputs = {
        "task_manifest.csv":
            task_manifest,
        "simulation_replicate_results_raw.csv":
            simulation_replicates,
        "model_fit_diagnostics.csv":
            model_fit_diagnostics,
        "optimizer_attempts.csv":
            optimizer_attempt_audit,
    }

    for filename, frame in outputs.items():
        frame.to_csv(
            output_dir / filename,
            index=False,
        )


def main():
    args = parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers must be >= 1."
        )

    if args.workers > 3:
        raise ValueError(
            "--workers > 3 is not supported by the validated low-memory "
            "workflow."
        )

    reps_per_cell = (
        args.reps_per_cell
        if args.reps_per_cell is not None
        else (
            FULL_REPS_PER_CELL
            if args.mode == "full"
            else 1
        )
    )

    if reps_per_cell < 1:
        raise ValueError(
            "--reps-per-cell must be >= 1."
        )

    selected_regimes = (
        list(REGIME_TO_OUTCOME)
        if args.regime is None
        else list(dict.fromkeys(args.regime))
    )

    if args.mode == "full":
        if set(selected_regimes) != set(
            REGIME_TO_OUTCOME
        ):
            raise ValueError(
                "Full mode requires all four frozen regimes."
            )

        if reps_per_cell != FULL_REPS_PER_CELL:
            raise ValueError(
                "Full mode requires exactly 500 "
                "replicates per cell."
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        source_parameters,
        manifest_hash,
        schedule_hashes,
    ) = load_source_parameters(
        args.anchor_manifest,
        selected_regimes,
    )

    task_manifest = build_task_manifest(
        selected_regimes,
        reps_per_cell,
    )

    signature = checkpoint_signature(
        reps_per_cell,
        selected_regimes,
        manifest_hash,
        schedule_hashes,
    )

    print(
        "Downstream simulation generator"
    )
    print(
        f"  mode: {args.mode}"
    )
    print(
        f"  regimes: "
        f"{', '.join(selected_regimes)}"
    )
    print(
        f"  replicates per cell: "
        f"{reps_per_cell}"
    )
    print(
        f"  requested tasks: "
        f"{len(task_manifest)}"
    )
    print(
        f"  workers: {args.workers}"
    )

    (
        simulation_replicates,
        model_fit_diagnostics,
        optimizer_attempt_audit,
    ) = run_generator(
        source_parameters,
        task_manifest,
        args.output_dir,
        args.workers,
        signature,
    )

    write_raw_outputs(
        args.output_dir,
        task_manifest,
        simulation_replicates,
        model_fit_diagnostics,
        optimizer_attempt_audit,
    )

    print()
    print(
        "Generator complete."
    )
    print(
        "Raw semantic outputs:"
    )
    print(
        "  task_manifest.csv"
    )
    print(
        "  simulation_replicate_results_raw.csv"
    )
    print(
        "  model_fit_diagnostics.csv"
    )
    print(
        "  optimizer_attempts.csv"
    )
    print(
        "No LR calibration, method summary, "
        "singularity analysis, or figure was "
        "computed by this generator."
    )


if __name__ == "__main__":
    main()
