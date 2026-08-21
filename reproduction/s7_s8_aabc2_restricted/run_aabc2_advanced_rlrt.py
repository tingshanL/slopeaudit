#!/usr/bin/env python3
"""Run observed and parametric-bootstrap AABC2 random-slope tests.

Smoke mode runs explicitly requested replicate numbers for all six outcomes and
optionally applies the validated direct-REML ventricular repair to specified
replicates. Full mode reproduces the final advanced state: 999 replicates for
five outcomes, 4,999 for cortical thickness, with failed ventricular strict
panels repaired by the validated multistart direct-REML route.
"""
from __future__ import annotations

import os

# Keep every numerical fit single-threaded. Parallelism, when requested, is
# across bootstrap replicates rather than nested BLAS/Accelerate threads.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import aabc2_common as common
import _aabc2_advanced_rlrt_core as core
import _aabc2_advanced_rlrt_worker as worker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
    )
    parser.add_argument(
        "--replicates",
        nargs="+",
        type=int,
        default=(1, 2, 3),
        help="Smoke replicate numbers for every outcome",
    )
    parser.add_argument(
        "--ventricular-repair-replicates",
        nargs="*",
        type=int,
        default=(),
        help=(
            "Smoke-mode ventricular replicates to recompute with "
            "direct-REML repair"
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=2)
    return parser.parse_args()


def compact_null_state(observed_panels):
    """Keep only the observed-null quantities used by bootstrap workers."""
    state = {}
    for outcome in core.PB_OUTCOMES:
        result = observed_panels[(outcome, "null")]["result"]
        if result is None:
            state[outcome] = None
            continue
        state[outcome] = {
            "exog": np.asarray(result.model.exog, float),
            "fe_params": np.asarray(result.fe_params, float),
            "cov_re": np.asarray(result.cov_re, float),
            "scale": float(result.scale),
        }
    return state


def parallel_pairs(
    tasks,
    desc,
    n_jobs,
    outcomes,
    analysis_data,
    null_state,
):
    if n_jobs == 1:
        return [
            core.simulate_one_null_rlrt(outcome, replicate)
            for outcome, replicate in tqdm(
                tasks,
                desc=desc,
                unit="rep",
            )
        ]

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=worker.initialize,
        initargs=(outcomes, analysis_data, null_state),
    ) as executor:
        results = executor.map(
            worker.simulate_pair,
            tasks,
            chunksize=1,
        )
        return list(
            tqdm(
                results,
                total=len(tasks),
                desc=desc,
                unit="rep",
            )
        )


def parallel_repairs(
    repair_reps,
    desc,
    n_jobs,
    outcomes,
    analysis_data,
    null_state,
):
    """Run the expensive direct-REML repair path serially.

    Each repair performs a large multistart optimizer panel. Running multiple
    repairs concurrently causes severe nested numerical-library contention on
    modest-memory machines and is unnecessary for deterministic validation.
    The scientific repair function itself is unchanged.
    """
    if n_jobs != 1:
        print(
            "Ventricular direct-REML repairs are intentionally serial "
            "to avoid nested optimizer/BLAS contention."
        )

    return [
        core.repair_one_ventricular_rlrt(replicate)
        for replicate in tqdm(
            repair_reps,
            desc=desc,
            unit="rep",
        )
    ]


def main():
    args = parse_args()
    n_jobs = max(1, min(int(args.n_jobs), 3))

    out = Path(args.output_dir).expanduser().resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    raw = common.build_raw_cohorts(args.input_config)
    outcomes, analysis_data, _ = common.build_analysis_data(raw)
    core.configure_analysis(outcomes, analysis_data)

    # Observed RI and RI+slope panels.
    observed_audits = []
    test_rows = []
    for outcome in tqdm(
        core.PB_OUTCOMES,
        desc="Observed RLRT panels",
        unit="outcome",
    ):
        data = analysis_data[outcome]
        formula = core.fixed_formula(outcome)
        null = core.fit_optimizer_panel(
            data,
            formula,
            "1",
            reml=True,
            stop_after_strict_pair=False,
        )
        alternative = core.fit_optimizer_panel(
            data,
            formula,
            "1+t5_mid",
            reml=True,
            stop_after_strict_pair=False,
        )
        core.OBSERVED_PANELS[(outcome, "null")] = null
        core.OBSERVED_PANELS[(outcome, "alternative")] = alternative

        for model_name, panel in (
            ("RI_null", null),
            ("RI_RS_alternative", alternative),
        ):
            audit = panel["audit"].copy()
            audit.insert(0, "outcome", outcome)
            audit.insert(1, "model", model_name)
            observed_audits.append(audit)

        lr = (
            max(
                0.0,
                2
                * (
                    float(alternative["result"].llf)
                    - float(null["result"].llf)
                ),
            )
            if null["result"] is not None
            and alternative["result"] is not None
            else np.nan
        )

        test_rows.append({
            "outcome": outcome,
            "label": outcomes[outcome]["label"],
            "fixed_formula": formula,
            "null_random_effects": "random intercept only",
            "alternative_random_effects": (
                "unstructured random intercept + t5_mid slope"
            ),
            "estimation": "REML",
            "observed_restricted_LR": lr,
            "null_basic_panel_valid": null["basic_panel_valid"],
            "null_strict_panel_valid": null["strict_panel_valid"],
            "alternative_basic_panel_valid": (
                alternative["basic_panel_valid"]
            ),
            "alternative_strict_panel_valid": (
                alternative["strict_panel_valid"]
            ),
        })

    observed_audit = pd.concat(observed_audits, ignore_index=True)
    observed_tests = pd.DataFrame(test_rows)
    core.observed_rlrt_tests = observed_tests

    # Final release geometry naming. Historical fitting helpers emit g_0_5
    # for the raw endpoint covariance; the frozen final state names that
    # quantity cov_L0_L5 and reserves d_0_5 for the dimensionless normalized
    # determinant 1 - gamma_0_5^2. This is an export-only normalization.
    if "g_0_5" in observed_audit.columns:
        observed_audit = observed_audit.rename(
            columns={"g_0_5": "cov_L0_L5"}
        )

    observed_audit["d_0_5"] = (
        observed_audit["one_minus_gamma_0_5_squared"]
    )

    observed_audit.to_csv(
        out / "04_observed_RLRT_optimizer_audit.csv",
        index=False,
    )
    observed_tests.to_csv(
        out / "04a_observed_RLRT_tests.csv",
        index=False,
    )

    null_state = compact_null_state(core.OBSERVED_PANELS)

    if args.mode == "full":
        targets = {
            outcome: (
                core.PB_MAX_B
                if outcome == "global_mean_thickness"
                else core.PB_INITIAL_B
            )
            for outcome in core.PB_OUTCOMES
        }
        tasks = [
            (outcome, replicate)
            for outcome in core.PB_OUTCOMES
            for replicate in range(1, targets[outcome] + 1)
        ]
    else:
        replicates = tuple(
            dict.fromkeys(int(x) for x in args.replicates)
        )
        if (
            not replicates
            or any(
                x < 1 or x > core.PB_MAX_B
                for x in replicates
            )
        ):
            raise ValueError(
                "Smoke replicates must be unique integers in 1..4999"
            )
        tasks = [
            (outcome, replicate)
            for outcome in core.PB_OUTCOMES
            for replicate in replicates
        ]

    rows = parallel_pairs(
        tasks,
        "Parametric RLRT",
        n_jobs,
        outcomes,
        analysis_data,
        null_state,
    )
    bootstrap = pd.DataFrame(rows)

    # Final ventricular repair route: strict-panel failures in full mode;
    # explicit replicate numbers in smoke mode.
    core.rlrt_bootstrap_replicates = bootstrap

    if args.mode == "full":
        ventricular = bootstrap.loc[
            bootstrap.outcome.eq("lateral_ventricles")
        ].copy()
        ventricular["strict_pair_valid"] = core.coerce_boolean(
            ventricular["strict_pair_valid"]
        )
        repair_reps = sorted(
            ventricular.loc[
                ~ventricular.strict_pair_valid,
                "replicate",
            ].astype(int).unique()
        )
    else:
        repair_reps = sorted(
            set(
                int(x)
                for x in args.ventricular_repair_replicates
            )
        )

    repairs = []
    if repair_reps:
        repairs = parallel_repairs(
            repair_reps,
            "Ventricular direct-REML repairs",
            n_jobs,
            outcomes,
            analysis_data,
            null_state,
        )

    repairs = pd.DataFrame(repairs)

    if len(repairs):
        keep = ~(
            bootstrap.outcome.eq("lateral_ventricles")
            & bootstrap.replicate.astype(int).isin(
                set(repairs.replicate.astype(int))
            )
        )
        bootstrap = (
            pd.concat(
                [bootstrap.loc[keep], repairs],
                ignore_index=True,
                sort=False,
            )
            .drop_duplicates(
                ["outcome", "replicate"],
                keep="last",
            )
            .sort_values(["outcome", "replicate"])
            .reset_index(drop=True)
        )

    # Normalize the replicate-table schema to the final repaired release
    # contract even when a smoke run contains no repaired rows. In the frozen
    # final table these fields are populated only for rows replaced by the
    # direct-REML ventricular repair; ordinary rows carry missing values.
    repair_metadata_columns = (
        "repair_method",
        "null_optimizer_agreement",
        "alternative_optimizer_agreement",
        "repair_error",
    )
    for column in repair_metadata_columns:
        if column not in bootstrap.columns:
            bootstrap[column] = np.nan

    final_replicate_columns = [
        "outcome",
        "replicate",
        "seed",
        "basic_pair_valid",
        "strict_pair_valid",
        "restricted_LR",
        "null_optimizers_attempted",
        "alternative_optimizers_attempted",
        "null_maximum_basic_agreement",
        "alternative_maximum_basic_agreement",
        "null_maximum_strict_agreement",
        "alternative_maximum_strict_agreement",
        "alternative_slope_variance",
        "alternative_hessian_minimum",
        "error",
        "repair_method",
        "null_optimizer_agreement",
        "alternative_optimizer_agreement",
        "repair_error",
    ]
    missing_columns = [
        column for column in final_replicate_columns
        if column not in bootstrap.columns
    ]
    if missing_columns:
        raise RuntimeError(
            "Final RLRT replicate table is missing required columns: "
            + ", ".join(missing_columns)
        )
    bootstrap = bootstrap[final_replicate_columns]

    core.rlrt_bootstrap_replicates = bootstrap

    bootstrap.to_csv(
        out
        / (
            "06_parametric_RLRT_replicates_repaired.csv"
            if args.mode == "full"
            else "06_parametric_RLRT_smoke.csv"
        ),
        index=False,
    )

    if len(repairs):
        repairs.to_csv(
            out
            / (
                "05a_ventricular_RLRT_multistart_repaired_replicates.csv"
                if args.mode == "full"
                else "05a_ventricular_RLRT_repair_smoke.csv"
            ),
            index=False,
        )

    if args.mode == "full":
        # Final denominator rule and Holm adjustment.
        requested = {
            outcome: (
                core.PB_MAX_B
                if outcome == "global_mean_thickness"
                else core.PB_INITIAL_B
            )
            for outcome in core.PB_OUTCOMES
        }

        summary = pd.DataFrame([
            core.summarize_one_rlrt_final(
                outcome,
                requested[outcome],
                frame=bootstrap,
            )
            for outcome in core.PB_OUTCOMES
        ])

        raw_p = summary[
            "bootstrap_p_value_plus_one"
        ].to_numpy(float)
        adjusted = np.full(len(raw_p), np.nan)
        finite = np.flatnonzero(np.isfinite(raw_p))

        if len(finite):
            order = finite[
                np.argsort(
                    raw_p[finite],
                    kind="mergesort",
                )
            ]
            values = raw_p[order]
            ordered = np.maximum.accumulate(
                (len(values) - np.arange(len(values))) * values
            )
            adjusted[order] = np.minimum(ordered, 1.0)

        summary["Holm_adjusted_p_across_six"] = adjusted
        summary["reporting_interpretation"] = summary.apply(
            lambda row: (
                "not licensed: incomplete requested B, invalid observed "
                "panel, or <95% finite optimizer agreement"
                if not row.p_value_licensed
                else "supported after Holm adjustment"
                if row.Holm_adjusted_p_across_six < core.ALPHA
                else "borderline: raw p<.05 but not Holm-significant"
                if row.bootstrap_p_value_plus_one < core.ALPHA
                else "not supported at raw .05"
            ),
            axis=1,
        )
        summary.to_csv(
            out / "05_parametric_RLRT_summary_repaired.csv",
            index=False,
        )

        pd.DataFrame([
            core.summarize_one_rlrt_final(
                "lateral_ventricles",
                core.PB_INITIAL_B,
                frame=bootstrap,
            )
        ]).to_csv(
            out / "05b_ventricular_RLRT_adjudication.csv",
            index=False,
        )

        counts = (
            bootstrap.groupby("outcome")
            .replicate.nunique()
            .reindex(core.PB_OUTCOMES)
        )
        expected = pd.Series({
            outcome: requested[outcome]
            for outcome in core.PB_OUTCOMES
        })
        if not counts.equals(expected):
            raise RuntimeError(
                "Unexpected final RLRT replicate counts:\n"
                + counts.to_string()
            )

    print("mode:", args.mode)
    print("observed audit rows:", len(observed_audit))
    print("bootstrap rows:", len(bootstrap))
    print("ventricular repair rows:", len(repairs))
    print("PARTICIPANT-LEVEL TABLES EXPORTED: 0")


if __name__ == "__main__":
    main()
