#!/usr/bin/env python3
"""
Deterministically summarize the repaired S4 operating-study results.

This script performs no model fitting and generates no simulated data.

Inputs
------
1. The repaired replicate-level operating-study CSV.
2. The corresponding simulation-cell manifest.

Outputs
-------
The final repaired LR calibration, strict-valid operating characteristics,
metric quantiles, true-zero apparent-determinacy summaries, and cell-validity
table used by the manuscript pipeline.

The numerical summary logic is the final strict-valid repair logic from the
authoritative notebook. It supersedes the original pre-repair operating
summary.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


STRICT_VALIDITY_TARGET = 0.95


def display(*_args, **_kwargs):
    """Suppress notebook-only display calls."""
    return None


def coerce_boolean(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
    }).fillna(False).astype(bool)



def summarize_operating_study(
    replicates_csv,
    manifest_csv,
    output_dir,
    mode,
):
    replicates_csv = Path(replicates_csv)
    manifest_csv = Path(manifest_csv)
    OUTPUT_DIR = Path(output_dir)

    if not replicates_csv.is_file():
        raise FileNotFoundError(
            "Repaired replicate CSV does not exist."
        )

    if not manifest_csv.is_file():
        raise FileNotFoundError(
            "Simulation manifest CSV does not exist."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    operating_replicates_repaired = pd.read_csv(
        replicates_csv,
        low_memory=False,
        float_precision="round_trip",
    )

    simulation_manifest = pd.read_csv(
        manifest_csv,
        low_memory=False,
        float_precision="round_trip",
    )

    required_replicate_columns = {
        "cell_id",
        "replicate",
        "cell_family",
        "strict_fit_valid",
        "restricted_LR",
    }

    missing = sorted(
        required_replicate_columns
        - set(operating_replicates_repaired.columns)
    )

    if missing:
        raise ValueError(
            "Repaired replicate table is missing required columns: "
            + ", ".join(missing)
        )

    required_manifest_columns = {
        "cell_id",
        "requested_reps",
    }

    missing = sorted(
        required_manifest_columns
        - set(simulation_manifest.columns)
    )

    if missing:
        raise ValueError(
            "Simulation manifest is missing required columns: "
            + ", ".join(missing)
        )

    operating_replicates_repaired["replicate"] = (
        pd.to_numeric(
            operating_replicates_repaired["replicate"],
            errors="raise",
        ).astype(int)
    )

    if operating_replicates_repaired.duplicated(
        ["cell_id", "replicate"]
    ).any():
        raise ValueError(
            "Repaired replicate table contains duplicate "
            "(cell_id, replicate) keys."
        )

    if simulation_manifest["cell_id"].duplicated().any():
        raise ValueError(
            "Simulation manifest contains duplicate cell_id values."
        )

    unknown_cells = sorted(
        set(
            operating_replicates_repaired[
                "cell_id"
            ].astype(str)
        )
        - set(
            simulation_manifest[
                "cell_id"
            ].astype(str)
        )
    )

    if unknown_cells:
        raise ValueError(
            "Repaired replicate table contains cells absent "
            "from the supplied manifest."
        )

    if mode == "full":
        if len(simulation_manifest) != 46:
            raise ValueError(
                "Full S4 manifest must contain 46 cells; "
                f"found {len(simulation_manifest)}."
            )

        requested_total = int(
            pd.to_numeric(
                simulation_manifest["requested_reps"],
                errors="raise",
            ).sum()
        )

        if requested_total != 62000:
            raise ValueError(
                "Full S4 manifest must request 62,000 "
                f"replicates; found {requested_total:,}."
            )

        if len(operating_replicates_repaired) != 62000:
            raise ValueError(
                "Full repaired S4 table must contain "
                "62,000 rows; found "
                f"{len(operating_replicates_repaired):,}."
            )
    # Final strict-valid operating characteristics and true-zero reporting repair
    strict_operating = operating_replicates_repaired.loc[
        coerce_boolean(operating_replicates_repaired.strict_fit_valid)
    ].copy()

    # Independent calibration/evaluation halves are assigned by frozen replicate
    # number, so invalid fits do not move a row from one half to the other.
    calibration_rows = []
    calibration_keys = set()
    evaluation_keys = set()
    true_zero_strict = strict_operating.loc[
        strict_operating.cell_family.eq("true_zero")
    ].copy()
    for cell_id, group in true_zero_strict.groupby("cell_id"):
        requested = int(
            simulation_manifest.set_index("cell_id").loc[
                cell_id, "requested_reps"
            ]
        )
        split_replicate = requested // 2
        calibration = group.loc[group.replicate.le(split_replicate)]
        evaluation = group.loc[group.replicate.gt(split_replicate)]
        calibration_keys.update(
            (cell_id, int(value)) for value in calibration.replicate
        )
        evaluation_keys.update(
            (cell_id, int(value)) for value in evaluation.replicate
        )
        critical_value = (
            float(calibration.restricted_LR.quantile(0.95))
            if len(calibration) else np.nan
        )
        evaluation_indicator = (
            evaluation.restricted_LR >= critical_value
        )
        evaluation_rate = (
            float(evaluation_indicator.mean())
            if len(evaluation) else np.nan
        )
        calibration_rows.append({
            "cell_id": cell_id,
            "schedule_class": (
                "balanced"
                if group.schedule.iloc[0] == "balanced_three_wave"
                else "irregular"
            ),
            "q0": float(group.q0.iloc[0]),
            "calibration_requested_n": split_replicate,
            "calibration_strict_n": len(calibration),
            "evaluation_requested_n": requested - split_replicate,
            "evaluation_strict_n": len(evaluation),
            "LR_95_critical_value": critical_value,
            "evaluation_false_support_rate": evaluation_rate,
            "evaluation_false_support_MCSE": (
                float(np.sqrt(
                    evaluation_rate * (1 - evaluation_rate) / len(evaluation)
                )) if len(evaluation) and np.isfinite(evaluation_rate)
                else np.nan
            ),
        })
    lr_calibration_repaired = pd.DataFrame(calibration_rows)

    operating_replicates_repaired["LR_calibration_replicate"] = [
        (cell_id, int(replicate)) in calibration_keys
        for cell_id, replicate in zip(
            operating_replicates_repaired.cell_id,
            operating_replicates_repaired.replicate,
        )
    ]
    operating_replicates_repaired["LR_evaluation_replicate"] = [
        (cell_id, int(replicate)) in evaluation_keys
        for cell_id, replicate in zip(
            operating_replicates_repaired.cell_id,
            operating_replicates_repaired.replicate,
        )
    ]

    def matched_repaired_critical_value(row):
        schedule_class = (
            "balanced" if row.schedule == "balanced_three_wave"
            else "irregular"
        )
        candidates = lr_calibration_repaired.loc[
            lr_calibration_repaired.schedule_class.eq(schedule_class)
        ].copy()
        if candidates.empty:
            return np.nan
        candidates["q0_distance"] = (
            candidates.q0 - float(row.q0)
        ).abs()
        return float(candidates.sort_values(
            ["q0_distance", "q0"]
        ).iloc[0].LR_95_critical_value)

    operating_replicates_repaired[
        "matched_LR_critical_value"
    ] = operating_replicates_repaired.apply(
        matched_repaired_critical_value, axis=1
    )
    operating_replicates_repaired[
        "slope_heterogeneity_supported"
    ] = (
        operating_replicates_repaired.restricted_LR
        >= operating_replicates_repaired.matched_LR_critical_value
    )
    operating_replicates_repaired[
        "false_high_full_when_oracle_low"
    ] = (
        operating_replicates_repaired.oracle_mean_Pfull.lt(0.20)
        & operating_replicates_repaired.mean_Pfull.ge(0.40)
    )
    operating_replicates_repaired[
        "anchored_geometry_correct"
    ] = (
        coerce_boolean(
            operating_replicates_repaired.estimated_anchored_near
        )
        == coerce_boolean(
            operating_replicates_repaired.oracle_anchored_near
        )
    )

    strict_summary_rows = []
    strict_quantile_rows = []
    manifest_lookup = simulation_manifest.set_index("cell_id")
    for cell_id, group in operating_replicates_repaired.groupby(
        "cell_id", sort=False
    ):
        cell = manifest_lookup.loc[cell_id]
        strict = group.loc[coerce_boolean(group.strict_fit_valid)].copy()
        slope_evaluation = (
            strict.loc[coerce_boolean(strict.LR_evaluation_replicate)]
            if float(cell.q1) == 0 else strict
        )
        completed_reps = int(group.replicate.nunique())
        strict_validity_rate = (
            len(strict) / completed_reps if completed_reps else np.nan
        )
        row = {
            "cell_id": cell_id,
            "cell_family": cell.cell_family,
            "schedule": cell.schedule,
            "misspecification": cell.misspecification,
            "q1": cell.q1,
            "q0": cell.q0,
            "rho": cell.rho,
            "requested_reps": int(cell.requested_reps),
            "completed_reps": completed_reps,
            "strict_valid_reps": len(strict),
            "invalid_fit_denominator": completed_reps - len(strict),
            "strict_validity_rate": strict_validity_rate,
            "inference_licensed_at_95pct_validity": bool(
                strict_validity_rate >= STRICT_VALIDITY_TARGET
            ),
            "convergence_rate": float(coerce_boolean(
                group.alternative_converged
            ).mean()),
            "Hessian_positive_rate": float(coerce_boolean(
                group.hessian_positive_definite
            ).mean()),
            "boundary_rate": float(coerce_boolean(
                group.boundary_flag
            ).mean()),
            "trait_reliability_defined": bool(float(cell.q1) > 0),
            "false_slope_heterogeneity_support_rate": (
                float(slope_evaluation.slope_heterogeneity_supported.mean())
                if float(cell.q1) == 0 and len(slope_evaluation) else np.nan
            ),
            "slope_heterogeneity_support_rate": (
                float(slope_evaluation.slope_heterogeneity_supported.mean())
                if len(slope_evaluation) else np.nan
            ),
            "false_high_Pfull_rate_when_oracle_Pfull_lt_0p20": (
                float(strict.false_high_full_when_oracle_low.mean())
                if float(cell.oracle_mean_Pfull) < 0.20 and len(strict)
                else np.nan
            ),
            "anchored_geometry_sensitivity": (
                float(coerce_boolean(
                    strict.estimated_anchored_near
                ).mean())
                if bool(cell.oracle_anchored_near) and len(strict)
                else np.nan
            ),
            "anchored_geometry_specificity": (
                float((~coerce_boolean(
                    strict.estimated_anchored_near
                )).mean())
                if not bool(cell.oracle_anchored_near) and len(strict)
                else np.nan
            ),
        }
        for metric in ("mean_Rdirect", "mean_Pfull", "mean_Delta"):
            oracle_value = float(cell[f"oracle_{metric}"])
            errors = strict[metric] - oracle_value
            if float(cell.q1) > 0:
                row[f"bias_{metric}"] = (
                    float(errors.mean()) if len(errors) else np.nan
                )
                row[f"RMSE_{metric}"] = (
                    float(np.sqrt(np.mean(errors**2)))
                    if len(errors) else np.nan
                )
            else:
                # At true zero, no slope trait exists; bias/RMSE of its
                # reliability is undefined.
                row[f"bias_{metric}"] = np.nan
                row[f"RMSE_{metric}"] = np.nan
            row[f"MCSE_mean_{metric}"] = (
                float(strict[metric].std(ddof=1) / np.sqrt(len(strict)))
                if len(strict) > 1 else np.nan
            )
            for probability in (0.025, 0.25, 0.5, 0.75, 0.975):
                strict_quantile_rows.append({
                    "cell_id": cell_id,
                    "metric": metric,
                    "quantile": probability,
                    "empirical_value": (
                        float(strict[metric].quantile(probability))
                        if len(strict) else np.nan
                    ),
                    "oracle_value": (
                        oracle_value if float(cell.q1) > 0 else np.nan
                    ),
                    "interpretation": (
                        "estimation error relative to defined oracle"
                        if float(cell.q1) > 0
                        else "spurious fitted predictability distribution; no oracle reliability"
                    ),
                })
        for proportion_name in (
            "slope_heterogeneity_supported",
            "false_high_full_when_oracle_low",
            "anchored_geometry_correct",
        ):
            proportion = (
                float(strict[proportion_name].mean())
                if len(strict) else np.nan
            )
            row[f"MCSE_{proportion_name}"] = (
                float(np.sqrt(
                    proportion * (1 - proportion) / len(strict)
                )) if len(strict) and np.isfinite(proportion)
                else np.nan
            )
        strict_summary_rows.append(row)

    operating_characteristics_repaired = pd.DataFrame(
        strict_summary_rows
    )
    operating_quantiles_repaired = pd.DataFrame(strict_quantile_rows)

    true_zero_spurious_rows = []
    for cell_id, group in operating_replicates_repaired.loc[
        operating_replicates_repaired.cell_family.eq("true_zero")
        & coerce_boolean(operating_replicates_repaired.strict_fit_valid)
    ].groupby("cell_id"):
        row = {
            "cell_id": cell_id,
            "strict_valid_reps": len(group),
            "frequency_spurious_Pfull_ge_0p40": float(
                group.mean_Pfull.ge(0.40).mean()
            ),
            "frequency_spurious_Rdirect_ge_0p40": float(
                group.mean_Rdirect.ge(0.40).mean()
            ),
            "frequency_spurious_Rinnovation_ge_0p40": float(
                group.mean_Rinnovation.ge(0.40).mean()
            ),
        }
        for metric in (
            "mean_Rdirect", "mean_Pfull", "mean_Rinnovation", "mean_Delta"
        ):
            for probability in (0.025, 0.25, 0.5, 0.75, 0.975):
                row[f"{metric}_q{str(probability).replace('.', 'p')}"] = float(
                    group[metric].quantile(probability)
                )
        true_zero_spurious_rows.append(row)
    true_zero_spurious_predictability = pd.DataFrame(
        true_zero_spurious_rows
    )

    simulation_cell_validity = operating_characteristics_repaired[[
        "cell_id", "requested_reps", "completed_reps",
        "strict_valid_reps", "invalid_fit_denominator",
        "strict_validity_rate", "inference_licensed_at_95pct_validity",
    ]].copy()

    operating_replicates_repaired.to_csv(
        OUTPUT_DIR / "operating_study_replicate_results.csv",
        index=False,
    )
    lr_calibration_repaired.to_csv(
        OUTPUT_DIR / "likelihood_ratio_calibration.csv", index=False
    )
    operating_characteristics_repaired.to_csv(
        OUTPUT_DIR / "operating_characteristics.csv",
        index=False,
    )
    operating_quantiles_repaired.to_csv(
        OUTPUT_DIR / "operating_metric_quantiles.csv",
        index=False,
    )
    true_zero_spurious_predictability.to_csv(
        OUTPUT_DIR / "true_zero_predictability_diagnostics.csv",
        index=False,
    )
    simulation_cell_validity.to_csv(
        OUTPUT_DIR / "simulation_cell_validity.csv",
        index=False,
    )
    display(simulation_cell_validity.sort_values(
        "strict_validity_rate"
    ).head(10))
    display(true_zero_spurious_predictability)
    if not simulation_cell_validity[
        "inference_licensed_at_95pct_validity"
    ].all():
        warnings.warn(
            "At least one simulation cell remains below 95% strict validity. "
            "Its operating characteristics are diagnostic, not licensed."
        )

    output_names = [
        "operating_study_replicate_results.csv",
        "likelihood_ratio_calibration.csv",
        "operating_characteristics.csv",
        "operating_metric_quantiles.csv",
        "true_zero_predictability_diagnostics.csv",
        "simulation_cell_validity.csv",
    ]

    missing_outputs = [
        name
        for name in output_names
        if not (OUTPUT_DIR / name).is_file()
    ]

    if missing_outputs:
        raise RuntimeError(
            "Summary completed without expected outputs: "
            + ", ".join(missing_outputs)
        )

    print(
        "Operating-study summarization complete."
    )
    print(
        f"Replicate rows summarized: "
        f"{len(operating_replicates_repaired):,}"
    )
    print(
        f"Manifest cells: "
        f"{simulation_manifest['cell_id'].nunique()}"
    )
    if "repair_attempted" in operating_replicates_repaired.columns:
        repair_attempted_count = int(
            coerce_boolean(
                operating_replicates_repaired["repair_attempted"]
            ).sum()
        )
        print(
            f"Rows marked repair_attempted: "
            f"{repair_attempted_count:,}"
        )

    strict_invalid_count = int(
        (
            ~coerce_boolean(
                operating_replicates_repaired["strict_fit_valid"]
            )
        ).sum()
    )
    print(
        f"Strict-invalid rows: "
        f"{strict_invalid_count:,}"
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically summarize repaired S4 "
            "operating-study results. No fitting is performed."
        )
    )

    parser.add_argument(
        "--replicates-csv",
        required=True,
        type=Path,
        help=(
            "Repaired operating-study replicate CSV."
        ),
    )

    parser.add_argument(
        "--manifest-csv",
        required=True,
        type=Path,
        help=(
            "Corresponding simulation-cell manifest CSV."
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
        required=True,
        help=(
            "full enforces the frozen 46-cell/62,000-row "
            "contract; smoke permits smaller validation inputs."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    summarize_operating_study(
        replicates_csv=args.replicates_csv,
        manifest_csv=args.manifest_csv,
        output_dir=args.output_dir,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
