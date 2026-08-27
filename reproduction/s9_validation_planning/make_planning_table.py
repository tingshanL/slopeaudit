#!/usr/bin/env python3
"""Generate the manuscript prospective-design sensitivity table.

This is the script refactor of ``AABC_practical_design_table.ipynb``.  It is
fully synthetic/analytic: it reads no participant data and performs no model
fitting.  Every displayed value is conditional on the design, random-slope
scale, residual covariance, and sample size encoded below.

The Question 1 power calculation is exact for the stated equal-schedule
Gaussian direct-contrast variance test with known residual covariance.  It is
a transparent planning reference, not exact power for the fitted-model
parametric-bootstrap procedure used after model fitting.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2


ALPHA = 0.05

DESIGNS = [
    {
        "design_id": "three_waves_two_years",
        "design_label": "3 waves / 2 years",
        "time_unit": "years",
        "times": np.array([0.0, 1.0, 2.0]),
        "n_participants": 100,
        "slope_sd_per_time_unit": 0.10,
        "residual_model": "independent",
        "ar1_phi": 0.0,
        "ar1_interval": None,
        "schedule_note": "baseline, year 1, year 2",
    },
    {
        "design_id": "five_waves_five_years",
        "design_label": "5 waves / 5 years",
        "time_unit": "years",
        "times": np.linspace(0.0, 5.0, 5),
        "n_participants": 100,
        "slope_sd_per_time_unit": 0.10,
        "residual_model": "independent",
        "ar1_phi": 0.0,
        "ar1_interval": None,
        "schedule_note": "equally spaced over 5 years",
    },
    {
        "design_id": "ten_consecutive_days",
        "design_label": "10 daily observations",
        "time_unit": "days",
        "times": np.arange(10, dtype=float),
        "n_participants": 100,
        "slope_sd_per_time_unit": 0.05,
        "residual_model": "ar1",
        "ar1_phi": 0.30,
        "ar1_interval": 1.0,
        "schedule_note": "one observation/day; AR(1) phi=.30 per day",
    },
    {
        "design_id": "thirty_day_ema",
        "design_label": "30-day EMA burst",
        "time_unit": "days",
        "times": np.arange(120, dtype=float) / 4.0,
        "n_participants": 100,
        "slope_sd_per_time_unit": 0.01,
        "residual_model": "ar1",
        "ar1_phi": 0.50,
        "ar1_interval": 0.25,
        "schedule_note": "4 prompts/day; AR(1) phi=.50 per adjacent prompt",
    },
]

PRECISION_SCENARIOS = [
    {
        "precision_id": "higher",
        "precision_label": "Higher precision",
        "residual_sd": 0.75,
    },
    {
        "precision_id": "reference",
        "precision_label": "Reference precision",
        "residual_sd": 1.00,
    },
    {
        "precision_id": "lower",
        "precision_label": "Lower precision",
        "residual_sd": 1.50,
    },
]


OUTPUT_FILENAMES = (
    "practical_design_grid_full_S9.csv",
    "practical_design_assumptions.csv",
    "practical_design_table_main.csv",
    "practical_design_table_main.tex",
    "practical_design_contract.json",
)


def residual_covariance(
    times: np.ndarray,
    residual_sd: float,
    residual_model: str = "independent",
    ar1_phi: float = 0.0,
    ar1_interval: Optional[float] = None,
) -> np.ndarray:
    times = np.asarray(times, float)
    if residual_sd <= 0:
        raise ValueError("residual_sd must be positive")
    if residual_model == "independent":
        return residual_sd ** 2 * np.eye(len(times))
    if residual_model == "ar1":
        if not (0 <= ar1_phi < 1):
            raise ValueError(
                "This planning implementation requires AR(1) phi in [0, 1)"
            )
        if ar1_interval is None:
            unique_times = np.unique(times)
            positive_gaps = np.diff(np.sort(unique_times))
            positive_gaps = positive_gaps[positive_gaps > 0]
            if len(positive_gaps) == 0:
                raise ValueError(
                    "Cannot infer an AR(1) interval from fewer than two times"
                )
            ar1_interval = float(positive_gaps.min())
        if ar1_interval <= 0:
            raise ValueError("ar1_interval must be positive")
        # Preserve planned lags when an observation is missing: phi is per
        # ar1_interval rather than per adjacent retained row.
        lags = np.abs(times[:, None] - times[None, :]) / ar1_interval
        return residual_sd ** 2 * np.power(ar1_phi, lags)
    raise ValueError(f"Unknown residual_model={residual_model!r}")


def default_nuisance_basis(times: np.ndarray) -> np.ndarray:
    return np.ones((len(times), 1), dtype=float)


def direct_slope_error_variance(
    times: np.ndarray,
    residual: np.ndarray,
    nuisance_basis: Callable[[np.ndarray], np.ndarray] = default_nuisance_basis,
) -> float:
    times = np.asarray(times, float)
    residual = np.asarray(residual, float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("At least two planned times are required")
    if residual.shape != (len(times), len(times)):
        raise ValueError("Residual covariance has the wrong shape")
    if not np.allclose(residual, residual.T, atol=1e-10):
        raise ValueError("Residual covariance must be symmetric")
    if np.linalg.eigvalsh(residual).min() <= 0:
        raise ValueError("Residual covariance must be positive definite")

    nuisance = np.asarray(nuisance_basis(times), float)
    if nuisance.ndim == 1:
        nuisance = nuisance[:, None]
    centered_target = times - times.mean()
    design = np.column_stack([nuisance, centered_target])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(
            "The target slope is not estimable after removing nuisance columns"
        )

    information = design.T @ np.linalg.solve(residual, design)
    covariance = np.linalg.inv(information)
    value = float(covariance[-1, -1])
    if not np.isfinite(value) or value <= 0:
        raise FloatingPointError(
            "Direct slope error variance is not finite and positive"
        )
    return value


def q1_power_equal_schedule(
    direct_reliability: float,
    n_participants: int,
    alpha: float = ALPHA,
) -> float:
    if n_participants < 3:
        return np.nan
    if not (0 <= direct_reliability < 1):
        if math.isclose(direct_reliability, 1.0):
            return 1.0
        raise ValueError("direct_reliability must be in [0, 1]")
    degrees_freedom = n_participants - 1
    critical = chi2.ppf(1 - alpha, degrees_freedom)
    return float(
        chi2.sf(
            critical * (1 - direct_reliability),
            degrees_freedom,
        )
    )


def normalize_schedule_patterns(
    full_times: np.ndarray,
    schedule_patterns: Optional[List[Dict]],
) -> List[Dict]:
    full_times = np.asarray(full_times, float)
    if schedule_patterns is None:
        return [
            {
                "probability": 1.0,
                "times": full_times,
                "label": "complete",
            }
        ]
    normalized = []
    for index, pattern in enumerate(schedule_patterns):
        probability = float(pattern["probability"])
        if probability < 0:
            raise ValueError("Schedule probabilities cannot be negative")
        if "times" in pattern:
            times = np.asarray(pattern["times"], float)
        elif "indices" in pattern:
            times = full_times[np.asarray(pattern["indices"], int)]
        else:
            raise ValueError("Every schedule pattern needs times or indices")
        normalized.append(
            {
                "probability": probability,
                "times": times,
                "label": pattern.get("label", f"pattern_{index + 1}"),
            }
        )
    total = sum(pattern["probability"] for pattern in normalized)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError(
            f"Schedule probabilities must sum to 1; found {total}"
        )
    return normalized


def slope_design_sensitivity(
    *,
    times: Iterable[float],
    n_participants: int,
    slope_sd: float,
    residual_sd: float,
    residual_model: str = "independent",
    ar1_phi: float = 0.0,
    ar1_interval: Optional[float] = None,
    nuisance_basis: Callable[[np.ndarray], np.ndarray] = default_nuisance_basis,
    schedule_patterns: Optional[List[Dict]] = None,
    alpha: float = ALPHA,
) -> Dict:
    times = np.asarray(list(times), float)
    if slope_sd <= 0:
        raise ValueError(
            "slope_sd must be positive for prospective Question 1 power"
        )
    patterns = normalize_schedule_patterns(times, schedule_patterns)

    pattern_rows = []
    for pattern in patterns:
        pattern_times = np.asarray(pattern["times"], float)
        analyzable = len(np.unique(pattern_times)) >= 2
        effective_error = np.inf
        reliability = 0.0
        if analyzable:
            residual = residual_covariance(
                pattern_times,
                residual_sd,
                residual_model,
                ar1_phi,
                ar1_interval=ar1_interval,
            )
            try:
                effective_error = direct_slope_error_variance(
                    pattern_times,
                    residual,
                    nuisance_basis,
                )
                reliability = slope_sd ** 2 / (
                    slope_sd ** 2 + effective_error
                )
            except (
                ValueError,
                np.linalg.LinAlgError,
                FloatingPointError,
            ):
                analyzable = False
                effective_error = np.inf
                reliability = 0.0
        pattern_rows.append(
            {
                **pattern,
                "analyzable": analyzable,
                "effective_slope_error_variance": effective_error,
                "R_direct": reliability,
            }
        )

    analyzable_probability = sum(
        row["probability"] for row in pattern_rows if row["analyzable"]
    )
    unconditional_r = sum(
        row["probability"] * row["R_direct"] for row in pattern_rows
    )
    conditional_r = (
        unconditional_r / analyzable_probability
        if analyzable_probability > 0
        else np.nan
    )
    effective_n = int(round(n_participants * analyzable_probability))
    power = q1_power_equal_schedule(conditional_r, effective_n, alpha)
    exact_equal_schedule = (
        len(pattern_rows) == 1
        and pattern_rows[0]["analyzable"]
        and math.isclose(pattern_rows[0]["probability"], 1.0)
    )

    return {
        "planned_n": int(n_participants),
        "effective_analyzable_n": effective_n,
        "analyzable_schedule_probability": analyzable_probability,
        "slope_sd": float(slope_sd),
        "slope_variance": float(slope_sd ** 2),
        "residual_sd": float(residual_sd),
        "expected_R_direct_unconditional": unconditional_r,
        "expected_R_direct_analyzable": conditional_r,
        "question1_power": power,
        "power_method": (
            "exact known-residual direct-contrast variance test"
            if exact_equal_schedule
            else "equal-information-equivalent direct-contrast approximation"
        ),
        "alpha": alpha,
        "pattern_details": pattern_rows,
    }


def run_mathematical_checks() -> None:
    # 1. Balanced independent design equals sigma_e^2 / sum(t - mean(t))^2.
    test_times = np.array([0.0, 1.0, 2.0, 3.0])
    test_sigma = 1.7
    test_residual = residual_covariance(
        test_times,
        test_sigma,
        "independent",
        0.0,
    )
    matrix_value = direct_slope_error_variance(
        test_times,
        test_residual,
    )
    closed_form = test_sigma ** 2 / np.sum(
        (test_times - test_times.mean()) ** 2
    )
    assert np.isclose(
        matrix_value,
        closed_form,
        rtol=1e-11,
        atol=1e-12,
    )

    # 2. AR(1) with phi=0 equals independent residuals.
    ar0 = direct_slope_error_variance(
        test_times,
        residual_covariance(
            test_times,
            test_sigma,
            "ar1",
            0.0,
        ),
    )
    assert np.isclose(
        ar0,
        matrix_value,
        rtol=1e-11,
        atol=1e-12,
    )

    # 3. Time-unit transformation leaves reliability unchanged if slope SD
    # transforms with it.
    base = slope_design_sensitivity(
        times=[0, 1, 2],
        n_participants=100,
        slope_sd=0.10,
        residual_sd=1.0,
    )
    rescaled = slope_design_sensitivity(
        times=[0, 12, 24],
        n_participants=100,
        slope_sd=0.10 / 12,
        residual_sd=1.0,
    )
    assert np.isclose(
        base["expected_R_direct_analyzable"],
        rescaled["expected_R_direct_analyzable"],
        rtol=1e-11,
        atol=1e-12,
    )

    # 4. Power is nominal at zero reliability and rises with reliability.
    assert np.isclose(
        q1_power_equal_schedule(0.0, 100),
        ALPHA,
        atol=1e-12,
    )
    power_sequence = [
        q1_power_equal_schedule(value, 100)
        for value in [0.01, 0.10, 0.30]
    ]
    assert power_sequence == sorted(power_sequence)


def build_assumption_preview() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "design": design["design_label"],
                "n": design["n_participants"],
                "planned_observations": len(design["times"]),
                "follow_up": float(np.ptp(design["times"])),
                "time_unit": design["time_unit"],
                "slope_SD_per_unit": design["slope_sd_per_time_unit"],
                "slope_SD_over_window": (
                    design["slope_sd_per_time_unit"]
                    * float(np.ptp(design["times"]))
                ),
                "residual_model": design["residual_model"],
                "AR1_phi": design["ar1_phi"],
                "AR1_interval": design["ar1_interval"],
                "schedule": design["schedule_note"],
            }
            for design in DESIGNS
        ]
    )


def build_design_grid() -> pd.DataFrame:
    grid_rows = []
    for design in DESIGNS:
        times = np.asarray(design["times"], float)
        for precision in PRECISION_SCENARIOS:
            result = slope_design_sensitivity(
                times=times,
                n_participants=design["n_participants"],
                slope_sd=design["slope_sd_per_time_unit"],
                residual_sd=precision["residual_sd"],
                residual_model=design["residual_model"],
                ar1_phi=design["ar1_phi"],
                ar1_interval=design["ar1_interval"],
                schedule_patterns=None,
                alpha=ALPHA,
            )
            assert result["power_method"].startswith("exact")
            pattern = result.pop("pattern_details")[0]
            row = {
                "design_id": design["design_id"],
                "design": design["design_label"],
                "schedule": design["schedule_note"],
                "time_unit": design["time_unit"],
                "planned_observations": len(times),
                "follow_up": float(np.ptp(times)),
                "n_participants": design["n_participants"],
                "slope_sd_per_time_unit": design[
                    "slope_sd_per_time_unit"
                ],
                "slope_sd_over_window": (
                    design["slope_sd_per_time_unit"]
                    * float(np.ptp(times))
                ),
                "residual_model": design["residual_model"],
                "ar1_phi": design["ar1_phi"],
                "ar1_interval": design["ar1_interval"],
                "precision_id": precision["precision_id"],
                "precision": precision["precision_label"],
                "residual_sd": precision["residual_sd"],
                "effective_slope_error_variance": pattern[
                    "effective_slope_error_variance"
                ],
                **result,
            }
            grid_rows.append(row)

    design_grid = pd.DataFrame(grid_rows)
    assert len(design_grid) == len(DESIGNS) * len(PRECISION_SCENARIOS) == 12
    assert design_grid[
        "expected_R_direct_analyzable"
    ].between(0, 1).all()
    assert design_grid["question1_power"].between(ALPHA, 1).all()

    precision_order = {
        "higher": 0,
        "reference": 1,
        "lower": 2,
    }
    for _, group in design_grid.groupby("design_id"):
        ordered = group.assign(
            precision_order=group["precision_id"].map(precision_order)
        ).sort_values("precision_order")
        assert ordered[
            "expected_R_direct_analyzable"
        ].is_monotonic_decreasing
        assert ordered["question1_power"].is_monotonic_decreasing

    return design_grid


def latex_escape(value) -> str:
    text = str(value)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
    ]:
        text = text.replace(old, new)
    return text


def build_main_table(design_grid: pd.DataFrame) -> pd.DataFrame:
    return design_grid[
        [
            "design",
            "precision",
            "n_participants",
            "planned_observations",
            "expected_R_direct_analyzable",
            "question1_power",
        ]
    ].rename(
        columns={
            "n_participants": "N",
            "planned_observations": "Planned observations",
            "expected_R_direct_analyzable": "Expected R_direct",
            "question1_power": "Question 1 power",
        }
    )


def build_latex_table(main_table: pd.DataFrame) -> str:
    main_display = main_table.copy()
    main_display["Expected R_direct"] = main_display[
        "Expected R_direct"
    ].map(lambda value: f"{value:.3f}")
    main_display["Question 1 power"] = main_display[
        "Question 1 power"
    ].map(lambda value: f"{value:.3f}")

    body_rows = []
    for row in main_display.itertuples(index=False, name=None):
        body_rows.append(
            " & ".join(latex_escape(value) for value in row) + r" \\"
        )

    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            (
                r"\caption{Assumption-dependent direct-change reliability "
                r"and Question 1 planning power across illustrative designs.}"
            ),
            r"\label{tab:practical_design_sensitivity}",
            r"\begin{tabular}{llrrcc}",
            r"\toprule",
            (
                r"Design & Precision & $N$ & Planned observations & "
                r"Expected $R_{\mathrm{direct}}$ & Question 1 power \\"
            ),
            r"\midrule",
            *body_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def build_contract() -> Dict:
    # Store repository-relative filenames rather than absolute local paths so
    # the contract is portable across machines.
    return {
        "alpha": ALPHA,
        "interpretation": (
            "Every value is conditional on the stated planning assumptions."
        ),
        "question1_power_scope": (
            "Exact for the stated equal-schedule Gaussian direct-contrast "
            "variance test with known residual covariance; a planning "
            "approximation to the full estimated-residual mixed-model "
            "parametric-bootstrap test."
        ),
        "designs": [
            {
                **{
                    key: value
                    for key, value in design.items()
                    if key != "times"
                },
                "times": np.asarray(design["times"], float).tolist(),
            }
            for design in DESIGNS
        ],
        "precision_scenarios": PRECISION_SCENARIOS,
        "outputs": {
            "main_csv": "practical_design_table_main.csv",
            "main_tex": "practical_design_table_main.tex",
            "full_grid_csv": "practical_design_grid_full_S9.csv",
            "assumptions_csv": "practical_design_assumptions.csv",
        },
    }


def write_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    run_mathematical_checks()
    design_grid = build_design_grid()
    assumptions = build_assumption_preview()
    main_table = build_main_table(design_grid)
    latex_table = build_latex_table(main_table)
    contract = build_contract()

    design_grid.to_csv(
        output_dir / "practical_design_grid_full_S9.csv",
        index=False,
    )
    assumptions.to_csv(
        output_dir / "practical_design_assumptions.csv",
        index=False,
    )
    main_table.to_csv(
        output_dir / "practical_design_table_main.csv",
        index=False,
    )
    (output_dir / "practical_design_table_main.tex").write_text(
        latex_table,
        encoding="utf-8",
    )
    (output_dir / "practical_design_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n",
        encoding="utf-8",
    )

    print("All direct-contrast, invariance, and power checks passed.")
    print("planning rows:", len(design_grid))
    print("designs:", design_grid["design_id"].nunique())
    print("precision scenarios:", design_grid["precision_id"].nunique())
    for filename in OUTPUT_FILENAMES:
        print("WROTE:", output_dir / filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for the five deterministic planning-table outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_outputs(args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
