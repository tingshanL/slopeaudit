#!/usr/bin/env python3
"""
Reproduce manuscript Figure 2 from frozen analysis outputs.

This script performs deterministic post-processing and plotting only.
It does not fit mixed models, generate simulations, or search the local
filesystem for analysis results.

Inputs are supplied explicitly:
- repaired S4 operating-study replicates;
- frozen downstream simulation replicates;
- frozen observed source-fit audit.

The provenance manifest records source filenames and SHA-256 hashes,
but never absolute machine-specific paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, t as student_t

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


def display(*_args, **_kwargs):
    """Suppress notebook-only display output."""
    return None


POSITIVE_REGIMES = [
    "high_slope_information",
    "intermediate",
    "near_boundary",
]

REGIME_LABELS = {
    "high_slope_information": "More direct\nslope information",
    "intermediate": "Intermediate",
    "near_boundary": "Near-boundary",
}

REGIME_SOURCE_OUTCOMES = {
    "high_slope_information": "lateral_ventricles",
    "intermediate": "hippocampus",
    "near_boundary": "global_cortical_att",
}

FROZEN_SOURCE_N = {
    "high_slope_information": 622,
    "intermediate": 622,
    "near_boundary": 802,
}

POSITIVE_EFFECT = "beta_xt_0.10_SD_per_5y"

SCORE_THRESHOLD = 0.40

NOMINAL_COVERAGE = 0.95

Z_95 = float(norm.ppf(0.975))

COLORS = {
    "full": "#76518A",          # D_slope^2 / total fitted-slope determinacy
    "direct": "#24557A",        # R_direct / direct-change reliability
    "heterogeneity": "#A44A52", # boundary-aware test
    "oracle": "#333333",
    "naive": "#D55E00",
    "one_stage": "#0072B2",
    "uncertainty": "#009E73",
    "ink": "#263238",
    "muted": "#6E7781",
    "grid": "#DCE1E5",
    "white": "#FFFFFF",
    "balanced_band": "#F4F7FA",
    "irregular_band": "#FAF6F2",
}

METHODS = {
    "oracle": {
        "label": "Oracle",
        "color": COLORS["oracle"],
        "marker": "D",
    },
    "naive": {
        "label": "Naive EB-as-outcome",
        "color": COLORS["naive"],
        "marker": "^",
        "prefix": "BLUP_as_outcome_attenuation",
    },
    "one_stage": {
        "label": "One-stage mixed model",
        "color": COLORS["one_stage"],
        "marker": "o",
        "prefix": "one_stage_LMM",
    },
    "uncertainty": {
        "label": "Uncertainty-aware two-stage",
        "color": COLORS["uncertainty"],
        "marker": "s",
        "prefix": "uncertainty_aware_two_stage",
    },
}

METHOD_ORDER = ["oracle", "naive", "one_stage", "uncertainty"]


mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.titlesize": 10.4,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9.4,
    "axes.edgecolor": COLORS["ink"],
    "axes.labelcolor": COLORS["ink"],
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3,
    "legend.fontsize": 8.2,
    "legend.frameon": False,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.65,
    "grid.alpha": 0.82,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False).astype(bool)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


OPERATING_REQUIRED = {
    "cell_id", "cell_family", "schedule", "q0", "replicate",
    "strict_fit_valid", "mean_Pfull", "mean_Rdirect",
    "LR_evaluation_replicate", "restricted_LR",
    "matched_LR_critical_value",
}

DOWNSTREAM_REQUIRED = {
    "row_type", "regime", "effect_condition", "replicate", "seed",
    "true_beta_xt",
    "one_stage_LMM_beta", "one_stage_LMM_SE", "one_stage_LMM_valid",
    "uncertainty_aware_two_stage_beta",
    "uncertainty_aware_two_stage_SE",
    "uncertainty_aware_two_stage_valid",
    "BLUP_as_outcome_attenuation_beta",
    "BLUP_as_outcome_attenuation_SE",
    "BLUP_as_outcome_attenuation_valid",
}

SOURCE_FIT_REQUIRED = {
    "outcome", "model_role", "selected",
    "random_intercept_variance", "random_slope_variance",
    "intercept_slope_covariance",
}



def require_columns(frame, required, label):
    missing = sorted(
        set(required) - set(frame.columns)
    )
    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            + ", ".join(missing)
        )


def make_figure_02(
    operating_csv,
    downstream_csv,
    source_fits_csv,
    output_dir,
):
    operating_path = Path(operating_csv)
    downstream_path = Path(downstream_csv)
    source_fits_path = Path(source_fits_csv)
    OUTPUT_DIR = Path(output_dir)

    for path, label in [
        (operating_path, "operating CSV"),
        (downstream_path, "downstream CSV"),
        (source_fits_path, "source-fit CSV"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} does not exist."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    operating = pd.read_csv(
        operating_path,
        low_memory=False,
    )

    downstream = pd.read_csv(
        downstream_path,
        low_memory=False,
    )

    source_fits = pd.read_csv(
        source_fits_path,
        low_memory=False,
    )

    require_columns(
        operating,
        OPERATING_REQUIRED,
        "operating CSV",
    )
    require_columns(
        downstream,
        DOWNSTREAM_REQUIRED,
        "downstream CSV",
    )
    require_columns(
        source_fits,
        SOURCE_FIT_REQUIRED,
        "source-fit CSV",
    )

    operating["strict_fit_valid"] = coerce_bool(
        operating["strict_fit_valid"]
    )

    operating["LR_evaluation_replicate"] = coerce_bool(
        operating["LR_evaluation_replicate"]
    )

    source_fits["selected"] = coerce_bool(
        source_fits["selected"]
    )

    for method in METHOD_ORDER[1:]:
        prefix = METHODS[method]["prefix"]
        downstream[f"{prefix}_valid"] = coerce_bool(
            downstream[f"{prefix}_valid"]
        )

    if operating.duplicated(
        ["cell_id", "replicate"]
    ).any():
        raise ValueError(
            "The repaired operating file contains "
            "duplicate cell/replicate rows."
        )

    if downstream.duplicated(
        [
            "row_type",
            "regime",
            "effect_condition",
            "replicate",
        ]
    ).any():
        raise ValueError(
            "The downstream raw file contains "
            "duplicate simulation tasks."
        )

    source_paths = {
        "operating": operating_path,
        "downstream": downstream_path,
        "source_fits": source_fits_path,
    }

    SOURCE_SHA256 = {
        name: sha256_file(path)
        for name, path in source_paths.items()
    }


    def wilson_interval(successes: int, trials: int, z: float = Z_95):
        if trials <= 0:
            return np.nan, np.nan
        p = successes / trials
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        half = z * math.sqrt(
            p * (1 - p) / trials + z**2 / (4 * trials**2)
        ) / denominator
        return max(0.0, center - half), min(1.0, center + half)


    true_zero = operating.loc[
        operating["cell_family"].eq("true_zero")
        & operating["strict_fit_valid"]
    ].copy()
    for column in [
        "q0", "mean_Pfull", "mean_Rdirect",
        "restricted_LR", "matched_LR_critical_value",
    ]:
        true_zero[column] = pd.to_numeric(true_zero[column], errors="coerce")
    true_zero = true_zero.replace([np.inf, -np.inf], np.nan)

    assert len(true_zero) == 7_985, (
        f"Expected exactly 7,985 strict-valid true-zero fits; found {len(true_zero):,}."
    )
    assert true_zero[["mean_Pfull", "mean_Rdirect"]].notna().all().all()
    assert true_zero["cell_id"].nunique() == 4
    assert true_zero[["mean_Pfull", "mean_Rdirect"]].ge(-1e-8).all().all()
    assert true_zero[["mean_Pfull", "mean_Rdirect"]].le(1 + 1e-8).all().all()


    def schedule_label(schedule: str) -> str:
        return (
            "Balanced 3-wave schedule"
            if schedule == "balanced_three_wave"
            else "Irregular AABC schedule"
        )


    def variance_label(q0: float) -> str:
        return "Smaller level variance" if math.isclose(float(q0), 0.5) else "Larger level variance"


    def q0_label(q0: float) -> str:
        return r"$q_0=.5$" if math.isclose(float(q0), 0.5) else r"$q_0=4$"


    design_manifest = (
        true_zero[["cell_id", "schedule", "q0"]]
        .drop_duplicates()
        .assign(
            schedule_label=lambda d: d["schedule"].map(schedule_label),
            variance_label=lambda d: d["q0"].map(variance_label),
            schedule_order=lambda d: d["schedule"].map(
                lambda value: 0 if value == "balanced_three_wave" else 1
            ),
        )
        .sort_values(["schedule_order", "q0"])
        .reset_index(drop=True)
    )
    DESIGN_ORDER = design_manifest["cell_id"].tolist()
    SCORE_COUNTS = (
        true_zero.groupby("cell_id").size().reindex(DESIGN_ORDER).astype(int)
    )
    EXPECTED_SCORE_COUNTS = [1_997, 1_994, 1_998, 1_996]
    assert SCORE_COUNTS.tolist() == EXPECTED_SCORE_COUNTS, (
        f"Unexpected score-fit denominators: {SCORE_COUNTS.to_dict()}"
    )

    metric_specs = [
        (
            "D_slope_squared", "mean_Pfull",
            r"Total fitted-slope determinacy ($D_{\mathrm{slope}}^2$)",
            COLORS["full"],
        ),
        (
            "R_direct", "mean_Rdirect",
            r"Direct-change reliability ($R_{\mathrm{direct}}$)",
            COLORS["direct"],
        ),
    ]

    score_summary_rows = []
    for row in design_manifest.itertuples(index=False):
        group = true_zero.loc[true_zero["cell_id"].eq(row.cell_id)]
        for metric, column, metric_label, color in metric_specs:
            values = group[column].to_numpy(float)
            q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
            score_summary_rows.append({
                "cell_id": row.cell_id,
                "schedule": row.schedule,
                "schedule_label": row.schedule_label,
                "variance_label": row.variance_label,
                "q0": float(row.q0),
                "metric": metric,
                "metric_label": metric_label,
                "n": len(values),
                "q25": q25,
                "median": median,
                "q75": q75,
                "iqr": q75 - q25,
                "color": color,
            })
    score_summary = pd.DataFrame(score_summary_rows)

    evaluation = true_zero.loc[true_zero["LR_evaluation_replicate"]].dropna(
        subset=["restricted_LR", "matched_LR_critical_value"]
    ).copy()
    evaluation["false_support"] = (
        evaluation["restricted_LR"] >= evaluation["matched_LR_critical_value"]
    )

    heterogeneity_rows = []
    for row in design_manifest.itertuples(index=False):
        group = evaluation.loc[evaluation["cell_id"].eq(row.cell_id)]
        n = int(len(group))
        k = int(group["false_support"].sum())
        low, high = wilson_interval(k, n)
        heterogeneity_rows.append({
            "cell_id": row.cell_id,
            "schedule_label": row.schedule_label,
            "variance_label": row.variance_label,
            "q0": float(row.q0),
            "k": k,
            "n": n,
            "false_positive_rate": k / n,
            "low": low,
            "high": high,
        })
    heterogeneity = pd.DataFrame(heterogeneity_rows)

    EXPECTED_HETEROGENEITY_KN = [(47, 999), (48, 997), (41, 998), (45, 999)]
    assert list(zip(heterogeneity["k"], heterogeneity["n"])) == EXPECTED_HETEROGENEITY_KN, (
        "The bootstrap heterogeneity-test k/n values do not match the frozen results: "
        f"{list(zip(heterogeneity['k'], heterogeneity['n']))}"
    )

    determinacy_proceeds = int(true_zero["mean_Pfull"].ge(SCORE_THRESHOLD).sum())
    direct_proceeds = int(true_zero["mean_Rdirect"].ge(SCORE_THRESHOLD).sum())
    assert determinacy_proceeds == 4_014
    assert direct_proceeds == 0

    score_summary.to_csv(
        OUTPUT_DIR / "figure_02_panel_A_score_median_IQR.csv", index=False
    )
    heterogeneity.to_csv(
        OUTPUT_DIR / "figure_02_panel_A_heterogeneity_false_positive_rates.csv",
        index=False,
    )

    print("Panel A contract verified")
    print({
        "score_fit_denominators": SCORE_COUNTS.tolist(),
        "heterogeneity_test_k_over_n": EXPECTED_HETEROGENEITY_KN,
        "illustrative_threshold_consequence": f"{determinacy_proceeds:,}/{len(true_zero):,}",
    })


    selected_source = source_fits.loc[
        source_fits.selected
        & source_fits.model_role.eq("reference_random_slope")
        & source_fits.outcome.isin(REGIME_SOURCE_OUTCOMES.values())
    ].copy()
    assert selected_source.outcome.nunique() == len(POSITIVE_REGIMES), (
        "Expected one selected reference random-slope source fit per positive regime."
    )
    assert selected_source.groupby("outcome").size().eq(1).all()


    def projected_covariance(row: pd.Series) -> np.ndarray:
        covariance = np.array([
            [
                float(row.random_intercept_variance),
                float(row.intercept_slope_covariance),
            ],
            [
                float(row.intercept_slope_covariance),
                float(row.random_slope_variance),
            ],
        ])
        values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
        return vectors @ np.diag(np.maximum(values, 0)) @ vectors.T


    SOURCE_G = {}
    for regime, outcome in REGIME_SOURCE_OUTCOMES.items():
        row = selected_source.loc[selected_source.outcome.eq(outcome)].iloc[0]
        SOURCE_G[regime] = projected_covariance(row)


    positive = downstream.loc[
        downstream.row_type.eq("evaluation")
        & downstream.effect_condition.eq(POSITIVE_EFFECT)
        & downstream.regime.isin(POSITIVE_REGIMES)
    ].copy()
    counts = positive.groupby("regime").replicate.nunique().reindex(POSITIVE_REGIMES)
    assert counts.eq(500).all(), (
        f"Expected 500 downstream replicates per positive regime; found {counts.to_dict()}."
    )


    def hc3_oracle_fit(x: np.ndarray, y: np.ndarray):
        design = np.column_stack([np.ones(len(x)), x])
        inverse = np.linalg.inv(design.T @ design)
        beta = inverse @ (design.T @ y)
        residual = y - design @ beta
        leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
        adjusted_squared = (residual / np.maximum(1 - leverage, 1e-12)) ** 2
        meat = design.T @ (adjusted_squared[:, None] * design)
        covariance = inverse @ meat @ inverse
        standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
        return float(beta[1]), standard_error


    oracle_rows = []
    for row in tqdm(
        list(positive[[
            "regime", "replicate", "seed", "true_beta_xt"
        ]].itertuples(index=False)),
        desc="Reconstructing oracle regressions",
        unit="replicate",
    ):
        n_participants = FROZEN_SOURCE_N[row.regime]
        rng = np.random.default_rng(int(row.seed))
        predictor = rng.normal(size=n_participants)
        random_effects = rng.multivariate_normal(
            np.zeros(2),
            SOURCE_G[row.regime],
            size=n_participants,
            check_valid="ignore",
        )
        true_total_slope = (
            float(row.true_beta_xt) * predictor + random_effects[:, 1]
        )
        beta, standard_error = hc3_oracle_fit(predictor, true_total_slope)
        oracle_rows.append({
            "regime": row.regime,
            "replicate": int(row.replicate),
            "oracle_beta": beta,
            "oracle_SE": standard_error,
            "oracle_valid": bool(np.isfinite(beta) and standard_error > 0),
        })

    oracle = pd.DataFrame(oracle_rows)
    assert len(oracle) == 1_500
    assert oracle.oracle_valid.all()
    oracle.to_csv(OUTPUT_DIR / "figure_02_reconstructed_oracle.csv", index=False)


    analysis = positive.merge(
        oracle,
        on=["regime", "replicate"],
        how="left",
        validate="one_to_one",
    )

    performance_rows = []
    for regime in POSITIVE_REGIMES:
        group = analysis.loc[analysis.regime.eq(regime)].copy()
        truth_values = pd.to_numeric(group.true_beta_xt, errors="coerce")
        assert truth_values.notna().all() and truth_values.gt(0).all()
        assert truth_values.nunique() == 1
        truth = float(truth_values.iloc[0])

        for method in METHOD_ORDER:
            if method == "oracle":
                beta_column, se_column, valid_column = (
                    "oracle_beta", "oracle_SE", "oracle_valid"
                )
            else:
                prefix = METHODS[method]["prefix"]
                beta_column, se_column, valid_column = (
                    f"{prefix}_beta", f"{prefix}_SE", f"{prefix}_valid"
                )

            valid = coerce_bool(group[valid_column])
            subset = group.loc[valid, [beta_column, se_column]].copy()
            subset[beta_column] = pd.to_numeric(
                subset[beta_column], errors="coerce"
            )
            subset[se_column] = pd.to_numeric(subset[se_column], errors="coerce")
            subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
            subset = subset.loc[subset[se_column].gt(0)]
            n = int(len(subset))
            assert n == 500, (
                f"{regime}/{method} has {n} valid positive-effect replicates; expected 500."
            )

            estimates = subset[beta_column].to_numpy(float)
            standard_errors = subset[se_column].to_numpy(float)
            ratios = estimates / truth
            mean_ratio = float(ratios.mean())
            critical = float(student_t.ppf(0.975, n - 1))
            mc_half = critical * float(ratios.std(ddof=1)) / math.sqrt(n)

            covered = (
                (estimates - Z_95 * standard_errors <= truth)
                & (truth <= estimates + Z_95 * standard_errors)
            )
            covered_k = int(covered.sum())
            coverage = covered_k / n
            coverage_low, coverage_high = wilson_interval(covered_k, n)

            performance_rows.append({
                "regime": regime,
                "regime_label": REGIME_LABELS[regime],
                "method": method,
                "method_label": METHODS[method]["label"],
                "valid_n": n,
                "true_effect": truth,
                "mean_estimate": float(estimates.mean()),
                "mean_estimate_ratio": mean_ratio,
                "ratio_mc_low": mean_ratio - mc_half,
                "ratio_mc_high": mean_ratio + mc_half,
                "coverage_successes": covered_k,
                "coverage": coverage,
                "coverage_low": coverage_low,
                "coverage_high": coverage_high,
                "attenuation": 1 - mean_ratio,
            })

    performance = pd.DataFrame(performance_rows)
    performance.to_csv(
        OUTPUT_DIR / "figure_02_downstream_plotting_values.csv", index=False
    )

    print("Panels B-C audit")
    display(performance[[
        "regime", "method_label", "valid_n", "true_effect",
        "mean_estimate", "mean_estimate_ratio",
        "coverage_successes", "coverage", "coverage_low", "coverage_high",
    ]])

    naive = performance.loc[performance.method.eq("naive")].set_index("regime")
    # Broad guards catch source/path mistakes without freezing rounded orientation values.
    assert naive.loc["high_slope_information", "attenuation"] > 0.15
    assert naive.loc["intermediate", "attenuation"] > 0.40
    assert naive.loc["near_boundary", "attenuation"] > 0.75
    assert naive.coverage.max() < 0.02


    def clean_axis(axis: mpl.axes.Axes, grid_axis: str = "y"):
        axis.grid(True, axis=grid_axis)
        axis.set_axisbelow(True)
        axis.tick_params(length=3, width=0.65, color=COLORS["ink"])


    def panel_label(axis: mpl.axes.Axes, label: str, x=-0.10, y=1.06):
        axis.text(
            x, y, label,
            transform=axis.transAxes,
            ha="left", va="bottom",
            fontsize=12, fontweight="bold",
            color=COLORS["ink"], clip_on=False,
        )


    def draw_violin_with_numeric_summary(axis, values, position, color, width=0.30):
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        if len(np.unique(np.round(values, 12))) > 1 and np.std(values) > 1e-10:
            violin = axis.violinplot(
                [np.clip(values, 0, 1)],
                positions=[position],
                widths=width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                points=180,
                bw_method=0.22,
            )
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_linewidth(0.85)
                body.set_alpha(0.44)

        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        axis.plot(
            [position, position], [q25, q75],
            color=color, lw=5.0, solid_capstyle="round", zorder=5,
        )
        axis.scatter(
            [position], [median], s=74, color=color,
            edgecolor=COLORS["white"], linewidth=1.0, zorder=6,
        )
        label_y = float(np.clip(median, 0.065, 0.925))
        axis.text(
            position, label_y,
            f"M {median:.2f}\nIQR {q25:.2f}–{q75:.2f}",
            ha="center", va="center", fontsize=6.6, color=COLORS["ink"],
            linespacing=1.04, zorder=7,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": COLORS["white"],
                "edgecolor": color,
                "linewidth": 0.55,
                "alpha": 0.90,
            },
        )
        return q25, median, q75


    fig = plt.figure(figsize=(11.6, 8.1))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.18, 1.0],
        hspace=0.49,
        left=0.070,
        right=0.985,
        top=0.945,
        bottom=0.175,
    )
    top = outer[0].subgridspec(
        1, 2, width_ratios=[2.55, 1.0], wspace=0.27,
    )
    bottom = outer[1].subgridspec(1, 2, wspace=0.27)

    ax_a = fig.add_subplot(top[0, 0])
    ax_a_test = fig.add_subplot(top[0, 1])
    ax_b = fig.add_subplot(bottom[0, 0])
    ax_c = fig.add_subplot(bottom[0, 1])

    # ---------- Panel A: paired distributions ----------
    x_base = np.arange(len(DESIGN_ORDER), dtype=float)
    offsets_a = {"D_slope_squared": -0.17, "R_direct": 0.17}

    # Schedule blocks make the 2 × 2 factorial structure explicit.
    ax_a.axvspan(-0.52, 1.50, color=COLORS["balanced_band"], zorder=0)
    ax_a.axvspan(1.50, 3.52, color=COLORS["irregular_band"], zorder=0)
    ax_a.axvline(1.50, color=COLORS["muted"], lw=0.9, alpha=0.75, zorder=1)

    for x, cell_id in zip(x_base, DESIGN_ORDER):
        group = true_zero.loc[true_zero["cell_id"].eq(cell_id)]
        for metric, column, _, color in metric_specs:
            draw_violin_with_numeric_summary(
                ax_a, group[column].to_numpy(float),
                x + offsets_a[metric], color,
            )

    # ax_a.axhline(
    #     SCORE_THRESHOLD, color=COLORS["muted"], linestyle="--", lw=1.0, zorder=2,
    # )
    # ax_a.text(
    #     1.53, SCORE_THRESHOLD + 0.015,
    #     "Illustrative .40 threshold",
    #     ha="left", va="center", rotation=90,
    #     fontsize=7.5, color=COLORS["muted"],
    #     bbox={"facecolor": COLORS["white"], "edgecolor": "none", "pad": 1.2, "alpha": 0.86},
    # )
    # ax_a.annotate(
    #     "Determinacy alone would have proceeded\n"
    #     "in 4,014/7,985 true-zero fits.",
    #     xy=(1.18, SCORE_THRESHOLD), xycoords="data",
    #     xytext=(0.50, 0.82), textcoords="axes fraction",
    #     ha="center", va="center", fontsize=8.3, color=COLORS["ink"],
    #     arrowprops={"arrowstyle": "-|>", "color": COLORS["full"], "lw": 1.0},
    #     bbox={
    #         "boxstyle": "round,pad=0.30", "facecolor": COLORS["white"],
    #         "edgecolor": COLORS["full"], "linewidth": 0.8, "alpha": 0.95,
    #     },
    #     zorder=8,
    # )

    ax_a.set_xlim(-0.52, 3.52)
    ax_a.set_ylim(0, 1.0)
    ax_a.set_xticks(x_base)
    ax_a.set_xticklabels([
        f"{row.variance_label}\n{q0_label(row.q0)} · n={int(SCORE_COUNTS.loc[row.cell_id]):,}"
        for row in design_manifest.itertuples(index=False)
    ])
    ax_a.set_ylabel("Fitted score quantity")
    ax_a.set_title("Paired true-zero fitted-score distributions", loc="left", pad=27)
    ax_a.text(
        0.25, 1.015, "Balanced 3-wave schedule",
        transform=ax_a.transAxes, ha="center", va="bottom",
        fontsize=9.0, fontweight="semibold", color=COLORS["ink"],
    )
    ax_a.text(
        0.75, 1.015, "Irregular AABC schedule",
        transform=ax_a.transAxes, ha="center", va="bottom",
        fontsize=9.0, fontweight="semibold", color=COLORS["ink"],
    )
    clean_axis(ax_a, "y")
    panel_label(ax_a, "A", x=-0.07, y=1.13)

    score_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=8.0,
            markerfacecolor=color, markeredgecolor=COLORS["white"],
            label=label,
        )
        for _, _, label, color in metric_specs
    ]
    ax_a.legend(
        handles=score_handles, loc="upper left", bbox_to_anchor=(0.0, -0.15),
        ncol=2, handletextpad=0.45, columnspacing=1.05,
    )

    # ---------- Panel A: visually separate heterogeneity-test strip ----------
    plot_test = heterogeneity.set_index("cell_id").reindex(DESIGN_ORDER)
    y = np.arange(len(DESIGN_ORDER))[::-1]
    x = plot_test["false_positive_rate"].to_numpy(float)
    low = plot_test["low"].to_numpy(float)
    high = plot_test["high"].to_numpy(float)

    ax_a_test.errorbar(
        x, y,
        xerr=np.vstack([x - low, high - x]),
        fmt="o", markersize=9.4,
        markerfacecolor=COLORS["heterogeneity"],
        markeredgecolor=COLORS["white"], markeredgewidth=0.9,
        color=COLORS["heterogeneity"], elinewidth=1.3, capsize=3.0, zorder=5,
    )
    ax_a_test.axvline(0.05, color=COLORS["muted"], linestyle="--", lw=1.0)
    for x_value, y_value, row in zip(x, y, plot_test.itertuples()):
        ax_a_test.text(
            min(0.088, row.high + 0.003), y_value,
            f"{int(row.k)}/{int(row.n)} ({100 * row.false_positive_rate:.1f}%)",
            ha="left", va="center", fontsize=7.9, color=COLORS["ink"],
        )

    ax_a_test.set_xlim(0, max(0.105, float(high.max()) + 0.025))
    ax_a_test.set_ylim(-0.55, len(DESIGN_ORDER) - 0.45)
    ax_a_test.set_yticks(y)
    ax_a_test.set_yticklabels([
        f"{'Balanced' if row.schedule == 'balanced_three_wave' else 'Irregular'} · "
        f"{'smaller' if math.isclose(float(row.q0), 0.5) else 'larger'}"
        for row in design_manifest.itertuples(index=False)
    ])
    ax_a_test.set_xlabel("False-positive rate (Wilson 95% interval)")
    ax_a_test.set_title("Bootstrap heterogeneity-test strip", loc="left", pad=9)
    ax_a_test.text(
        0.05, 0.50, "nominal .05",
        transform=ax_a_test.get_xaxis_transform(),
        ha="center", va="center", rotation=90,
        fontsize=7.6, color=COLORS["muted"],
        bbox={"facecolor": COLORS["white"], "edgecolor": "none", "pad": 1.1, "alpha": 0.88},
    )
    clean_axis(ax_a_test, "x")

    # ---------- Panels B and C ----------
    x_base_bc = np.arange(len(POSITIVE_REGIMES), dtype=float)
    offsets_bc = dict(zip(METHOD_ORDER, [-0.27, -0.09, 0.09, 0.27]))

    for method in METHOD_ORDER:
        subset = (
            performance.loc[performance["method"].eq(method)]
            .set_index("regime").reindex(POSITIVE_REGIMES).reset_index()
        )
        x_values = x_base_bc + offsets_bc[method]
        spec = METHODS[method]

        ax_b.errorbar(
            x_values, subset["mean_estimate_ratio"],
            yerr=np.maximum(0, np.vstack([
                subset["mean_estimate_ratio"] - subset["ratio_mc_low"],
                subset["ratio_mc_high"] - subset["mean_estimate_ratio"],
            ])),
            fmt=spec["marker"], markersize=9.4,
            markerfacecolor=spec["color"], markeredgecolor=COLORS["white"],
            markeredgewidth=0.85, color=spec["color"],
            elinewidth=1.15, capsize=2.8, linewidth=0, clip_on=False, zorder=5,
        )

        ax_c.errorbar(
            x_values, subset["coverage"],
            yerr=np.maximum(0, np.vstack([
                subset["coverage"] - subset["coverage_low"],
                subset["coverage_high"] - subset["coverage"],
            ])),
            fmt=spec["marker"], markersize=9.4,
            markerfacecolor=spec["color"], markeredgecolor=COLORS["white"],
            markeredgewidth=0.85, color=spec["color"],
            elinewidth=1.15, capsize=2.8, linewidth=0, clip_on=False, zorder=5,
        )

    ax_b.axhline(1.0, color=COLORS["muted"], linestyle="--", linewidth=1.0)
    ax_b.set_ylim(0, 1.17)
    ax_b.set_ylabel("Mean estimate / true effect")
    ax_b.set_title("Downstream effect recovery", loc="left", pad=9)
    panel_label(ax_b, "B")

    naive_plot = (
        performance.loc[performance["method"].eq("naive")]
        .set_index("regime").reindex(POSITIVE_REGIMES)
    )
    for x_value, attenuation, y_value in zip(
        x_base_bc + offsets_bc["naive"],
        naive_plot["attenuation"],
        naive_plot["mean_estimate_ratio"],
    ):
        ax_b.text(
            x_value, min(1.13, y_value + 0.045),
            f"{100 * attenuation:.0f}%\natten.",
            ha="center", va="bottom", fontsize=7.7,
            color=COLORS["naive"], fontweight="semibold",
        )

    ax_b.annotate(
        "Treat fitted slopes as error-free outcomes:\n"
        "attenuation and failed coverage",
        xy=(x_base_bc[-1] + offsets_bc["naive"], naive_plot.iloc[-1]["mean_estimate_ratio"]),
        xytext=(0.03, 0.14), textcoords="axes fraction",
        ha="left", va="center", fontsize=8.1, color=COLORS["naive"],
        arrowprops={"arrowstyle": "-|>", "color": COLORS["naive"], "lw": 1.0},
        bbox={
            "boxstyle": "round,pad=0.27", "facecolor": COLORS["white"],
            "edgecolor": COLORS["naive"], "linewidth": 0.75, "alpha": 0.94,
        },
    )

    ax_c.axhline(NOMINAL_COVERAGE, color=COLORS["muted"], linestyle="--", linewidth=1.0)
    ax_c.text(
        len(POSITIVE_REGIMES) - 0.42, NOMINAL_COVERAGE - 0.035,
        "nominal .95", ha="right", va="top", fontsize=8.0, color=COLORS["muted"],
    )
    ax_c.set_ylim(0, 1.04)
    ax_c.set_ylabel("95% interval coverage")
    ax_c.set_title("Interval calibration", loc="left", pad=9)
    panel_label(ax_c, "C")

    for x_value, successes, n, coverage in zip(
        x_base_bc + offsets_bc["naive"],
        naive_plot["coverage_successes"].astype(int),
        naive_plot["valid_n"].astype(int),
        naive_plot["coverage"],
    ):
        ax_c.text(
            x_value, max(0.035, coverage + 0.035), f"{successes}/{n}",
            ha="center", va="bottom", fontsize=7.7,
            color=COLORS["naive"], fontweight="semibold",
        )

    for axis in (ax_b, ax_c):
        axis.set_xlim(-0.52, len(POSITIVE_REGIMES) - 0.48)
        axis.set_xticks(x_base_bc)
        axis.set_xticklabels([REGIME_LABELS[regime] for regime in POSITIVE_REGIMES])
        axis.set_xlabel("Increasing reliance on other fitted response patterns", labelpad=8)
        clean_axis(axis, "y")

    method_handles = [
        Line2D(
            [0], [0], linestyle="none",
            marker=METHODS[method]["marker"], markersize=8.3,
            markerfacecolor=METHODS[method]["color"],
            markeredgecolor=COLORS["white"], markeredgewidth=0.8,
            label=METHODS[method]["label"],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=method_handles,
        loc="lower center", bbox_to_anchor=(0.52, 0.035),
        ncol=4, columnspacing=1.65, handletextpad=0.52,
        title="500 valid replicates per method and regime",
        title_fontsize=8.2,
    )

    pdf_path = OUTPUT_DIR / "figure_02_overall.pdf"
    png_path = OUTPUT_DIR / "figure_02_overall.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")
    plt.show()



    completion = {
        "source_files": {
            name: path.name
            for name, path in source_paths.items()
        },
        "source_sha256": SOURCE_SHA256,
        "strict_valid_true_zero_fits":
            int(len(true_zero)),
        "score_fit_denominators":
            SCORE_COUNTS.tolist(),
        "true_zero_Dslope2_ge_040":
            determinacy_proceeds,
        "true_zero_Rdirect_ge_040":
            direct_proceeds,
        "heterogeneity_test_k_over_n": [
            [int(row.k), int(row.n)]
            for row
            in heterogeneity.itertuples(index=False)
        ],
        "downstream_positive_regimes":
            int(performance["regime"].nunique()),
        "downstream_methods":
            int(performance["method"].nunique()),
        "all_downstream_denominators_500":
            bool(
                performance["valid_n"]
                .eq(500)
                .all()
            ),
        "operating_simulation_rerun": False,
        "mixed_model_refit": False,
        "figure_pdf": pdf_path.name,
        "figure_png": png_path.name,
        "pdf_exists": pdf_path.exists(),
        "png_exists": png_path.exists(),
    }

    manifest_path = (
        OUTPUT_DIR
        / "figure_02_overall_provenance_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            completion,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Figure 2 reproduction complete.")
    print(f"PDF: {pdf_path.name}")
    print(f"PNG: {png_path.name}")
    print(f"Provenance: {manifest_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce manuscript Figure 2 from "
            "explicit frozen analysis outputs."
        )
    )

    parser.add_argument(
        "--operating-csv",
        required=True,
        type=Path,
        help=(
            "Repaired S4 operating-study replicate CSV."
        ),
    )

    parser.add_argument(
        "--downstream-csv",
        required=True,
        type=Path,
        help=(
            "Frozen downstream simulation replicate CSV."
        ),
    )

    parser.add_argument(
        "--source-fits-csv",
        required=True,
        type=Path,
        help=(
            "Frozen observed source-fit audit CSV."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    make_figure_02(
        operating_csv=args.operating_csv,
        downstream_csv=args.downstream_csv,
        source_fits_csv=args.source_fits_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
