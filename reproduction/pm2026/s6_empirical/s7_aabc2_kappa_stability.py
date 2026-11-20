#!/usr/bin/env python3
r"""Evaluate AABC2 nuisance capacity on the original participant schedules (S6.4).

Uses saved bootstrap and likelihood-profile covariances; no models are refitted.
Inputs are supplied through --repo, --input-config, --exact-reliability,
and --bootstrap-geometry. Use --profile to select a saved profile CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


COGNITION = "fluid_cognition"
VENTRICLES = "lateral_ventricles"
THICKNESS = "global_mean_thickness"
LABELS = {COGNITION: "Fluid cognition", VENTRICLES: "Lateral ventricles",
          THICKNESS: "Cortical thickness"}
COLORS = {COGNITION: "#245b83", VENTRICLES: "#bc542b", THICKNESS: "#54865b"}
FIT_COLUMNS = [
    "random_intercept_variance", "random_slope_variance",
    "intercept_slope_covariance", "residual_variance",
]
SOURCE_METRICS = {
    "R_direct": "same_reference_fit_direct_GRR",
    "D2": "full_posterior_slope_predictability",
    "Delta_fit": "level_assisted_reliability_gain",
}
REPLAY_METRICS = (
    "median_direct_GRR", "median_full_predictability",
    "median_full_minus_direct_gap", "median_innovation_reliability",
)
REPLAY_DIFFS = [f"absolute_difference__{name}" for name in REPLAY_METRICS]
FIT_FLAGS = ["valid", "fit_succeeded", "converged"]
METRICS = ["median_sqrt_kappa", "mean_kappa", "mean_R_direct", "mean_Delta_fit", "mean_D2"]
NUMERICAL_RANK_TOL = 1e-10  # slopeaudit R/constants.R; diagnostic only
IDENTITY_TOL = np.sqrt(np.finfo(float).eps)  # slopeaudit R/constants.R
OBSERVED_TOL = 1e-6
REPLAY_TOL = 1e-5  # absolute difference on the unit scale; sensitivity only
PROFILE_PATH = "results/frozen_summaries/aabc2/anchored_d_profile.csv"
FIGURE4_DIR = "results/pm2026/empirical/s7_aabc2_aggregate"


def read_csv(path, required, **kwargs):
    frame = pd.read_csv(path, float_precision="round_trip", **kwargs)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    return frame


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}


def close(label, actual, expected, atol=IDENTITY_TOL, rtol=0.0):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        error = np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
        raise ValueError(f"{label}: comparison failed (max absolute difference {error:.6g})")


def schedule_moments(data):
    times = data.groupby("pid", sort=True)["t5_mid"]
    moments = times.agg(n="size", tbar="mean")
    centered = data["t5_mid"] - times.transform("mean")
    moments["I"] = centered.pow(2).groupby(data["pid"]).sum()
    if (moments.n < 2).any() or (moments.I <= 0).any():
        raise ValueError("An original-cohort schedule lacks repeated distinct times")
    if len(moments) == 0 or not np.isfinite(moments.to_numpy(float)).all():
        raise ValueError("Original-cohort schedules are empty or nonfinite")
    return moments


def evaluate(moments, G00, tau2, cov01, sigma2, *, profile_S=None):
    """Evaluate one midpoint-coded RI/RS fit using the package variance components.

    The observation-space solve reduces to the orthonormal person-mean and
    centered-time directions. q is Cov(y,s)/sqrt(tau2) in those coordinates.
    The remaining directions contribute zero to the slope weights.
    """
    if not np.isfinite([G00, tau2, cov01, sigma2]).all() or min(G00, tau2) < 0 or sigma2 <= 0:
        raise ValueError("invalid covariance or residual variance")

    g = np.array([[G00, cov01], [cov01, tau2]])
    if tau2 == 0:
        if cov01 != 0:
            raise ValueError("nonzero level–slope covariance at zero target variance")
        return "undefined_zero_target_variance", np.nan, None

    slope_sd = np.sqrt(tau2)
    level_loading = cov01 / slope_sd
    schur = float(np.longdouble(G00) -
                  np.longdouble(cov01) ** 2 / np.longdouble(tau2))
    if schur < -64 * np.finfo(float).eps * max(G00, level_loading ** 2):
        raise ValueError("non-PSD covariance")
    # The profile's stored d parameter preserves the determinant even at rank one.
    S = max(schur, 0.0) if profile_S is None else profile_S
    close("profile Schur complement", S, schur,
          atol=64 * np.finfo(float).eps * max(G00, level_loading ** 2))

    v0 = (G00 - cov01) + tau2 / 4
    v5 = (G00 + cov01) + tau2 / 4
    d = np.clip((S / v0) * (tau2 / v5), 0.0, 1.0) if min(v0, v5) > 0 else np.nan
    eigmax = np.linalg.eigvalsh(g)[-1]
    rank_ratio = (S / eigmax) * (tau2 / eigmax)
    quality = ("defined_rank_one" if S == 0 else
               "defined_numerical_near_rank_one" if rank_ratio <= NUMERICAL_RANK_TOL else
               "defined_interior")

    n, tbar, information = moments[["n", "tbar", "I"]].to_numpy(float).T
    mean_loading = np.sqrt(n) * (level_loading + tbar * slope_sd)
    direct_loading = np.sqrt(information) * slope_sd
    q = np.column_stack([mean_loading, direct_loading])
    # Solve V = q q' + diag(sigma2 + n*S, sigma2) without inverting G.
    v = np.column_stack([sigma2 + n * S, np.full(n.shape, sigma2)])
    precision_q = q / v
    information_total = np.einsum("ij,ij->i", q, precision_q)
    w = precision_q / (1.0 + information_total[:, None])
    D2 = np.einsum("ij,ij->i", q, w)
    direct_info = direct_loading ** 2 / sigma2
    R = direct_info / (1.0 + direct_info)
    # Algebraically D2 - R, evaluated without subtracting similar numbers.
    mean_info = mean_loading ** 2 / v[:, 0]
    Delta = (mean_info / (1.0 + information_total)) / (1.0 + direct_info)
    nuisance = n * S * w[:, 0] ** 2
    residual = sigma2 * np.einsum("ij,ij->i", w, w)
    fitted = D2 ** 2 + nuisance + residual
    if np.any(fitted <= 0) or not np.isfinite(fitted).all():
        raise ValueError("numerically unresolved fitted-slope variance")
    kappa = nuisance / fitted

    # Cross-check with the Figure 4 closed form.
    closed = Delta * (1 - D2) / (D2 * (1 - R)) * S / (S + sigma2 / n)
    close("component and closed-form kappa", kappa, closed)
    close("determinacy decomposition", D2, R + Delta)
    close("fitted-slope variance decomposition", D2, fitted)
    close("nuisance share composition", D2 + kappa + residual / fitted, 1)
    ceiling = np.minimum(1 - D2, Delta / D2)
    if np.any(kappa < -IDENTITY_TOL) or np.any(kappa - ceiling > IDENTITY_TOL):
        raise ValueError("kappa outside its theoretical bounds")

    return quality, float(d), pd.DataFrame(
        {"D2": D2, "R_direct": R, "Delta_fit": Delta, "kappa": kappa},
        index=moments.index,
    )


def aggregate(metrics):
    if metrics is None:
        return dict.fromkeys(METRICS, np.nan)
    return {
        "median_sqrt_kappa": np.sqrt(metrics.kappa).median(),
        "mean_kappa": metrics.kappa.mean(),
        "mean_R_direct": metrics.R_direct.mean(),
        "mean_Delta_fit": metrics.Delta_fit.mean(),
        "mean_D2": metrics.D2.mean(),
    }


def check_observed(outcome, moments, exact, published_dec, published_kappa):
    source = exact.loc[exact.outcome.eq(outcome)].set_index("participant_id", verify_integrity=True)
    if set(source.index) != set(moments.index):
        raise ValueError(f"{outcome}: original-cohort participant sets differ")
    source = source.reindex(moments.index)
    if not source[FIT_COLUMNS].nunique(dropna=False).eq(1).all():
        raise ValueError(f"{outcome}: expected one observed reference model")
    close(f"{outcome} visit counts", moments.n, source.scan_count, atol=0)
    close(f"{outcome} schedule information", moments.I, source.direct_schedule_information_I_i,
          atol=1e-10, rtol=1e-10)
    components = source[FIT_COLUMNS].iloc[0].to_numpy(float)
    status, d, metrics = evaluate(moments, *components)
    if metrics is None:
        raise ValueError(f"{outcome}: observed capacity is undefined")
    for key, column in SOURCE_METRICS.items():
        close(f"{outcome} observed {key}", metrics[key], source[column], atol=OBSERVED_TOL)

    result = {
        "outcome": outcome, "N": len(moments), "status": status, "d_0_5": d,
        **dict(zip(FIT_COLUMNS, components)), **aggregate(metrics),
    }
    if outcome != THICKNESS:
        dec = published_dec.loc[outcome]
        kap = published_kappa.loc[outcome]
        for key, expected in {
            "N": dec.N,
            **dict(zip(FIT_COLUMNS, dec[["G00", "tau2", "cov01", "sigma2"]])),
            "mean_D2": dec.mean_D2, "mean_R_direct": dec.mean_Rdirect,
            "mean_Delta_fit": dec.mean_Delta_fit, "mean_kappa": kap.mean_kappa,
            "median_sqrt_kappa": kap.sqrt_kappa_q50,
        }.items():
            close(f"Figure 4 {outcome} {key}", result[key], expected,
                  atol=OBSERVED_TOL, rtol=1e-8)
        for pct in (5, 25, 75, 95):
            close(f"Figure 4 {outcome} q{pct}", np.sqrt(metrics.kappa).quantile(pct / 100),
                  kap[f"sqrt_kappa_q{pct:02d}"], atol=OBSERVED_TOL)
    print(f"Observed {outcome}: N={len(moments)}, median sqrt(kappa)={result['median_sqrt_kappa']:.6f}")
    return result


def geometry_source(path, outcomes, expected_count):
    required = ["outcome", "replicate", *FIT_COLUMNS, *FIT_FLAGS,
                "reml_log_likelihood", "optimizer_v3", "optimizer_refit",
                "hessian_positive_definite", "warning_count", *REPLAY_DIFFS]
    frame = read_csv(path, required)
    frame = frame.loc[frame.outcome.isin(outcomes)].copy()
    for key in outcomes:
        ids = pd.to_numeric(frame.loc[frame.outcome.eq(key), "replicate"], errors="raise")
        n = len(ids) if expected_count is None else expected_count
        if n == 0 or sorted(ids) != list(range(1, n + 1)):
            raise ValueError(f"{path.name}: {key} must have one row per replicate 1 through {n}")
    frame["replicate"] = frame["replicate"].astype(int)
    return frame.sort_values(["outcome", "replicate"])


def evaluate_bootstrap(frame, schedules):
    rows = []
    for row in frame.to_dict("records"):
        out = row["outcome"]
        llf = float(row["reml_log_likelihood"])
        fitted_ok = is_true(row["fit_succeeded"]) and is_true(row["converged"]) and np.isfinite(llf)
        differences = np.array([row[name] for name in REPLAY_DIFFS], dtype=float)
        replay_ok = (is_true(row["valid"]) and np.isfinite(differences).all()
                     and np.all(differences <= REPLAY_TOL))
        if out in {COGNITION, THICKNESS}:
            replay_ok = replay_ok and row["optimizer_v3"] == row["optimizer_refit"]
            replay_status = "agrees" if replay_ok else "differs_or_unavailable"
        else:
            replay_status = "not_applicable"

        result = {
            "outcome": out, "replicate": row["replicate"], "N": len(schedules[out]),
            "status": "unresolved_fit", "reason": "", "d_0_5": np.nan,
            "reml_log_likelihood": llf,
            "original_fit_valid": is_true(row["valid"]),
            "fit_succeeded": is_true(row["fit_succeeded"]),
            "converged": is_true(row["converged"]),
            "optimizer_original": row["optimizer_v3"], "optimizer_selected": row["optimizer_refit"],
            "hessian_positive_definite": (is_true(row["hessian_positive_definite"])
                                          if pd.notna(row["hessian_positive_definite"]) else pd.NA),
            "warning_count": row["warning_count"], "historical_replay_status": replay_status,
            **{k: row[k] for k in [*FIT_COLUMNS, *REPLAY_DIFFS]}, **aggregate(None),
        }

        if fitted_ok:
            try:
                status, d, metrics = evaluate(schedules[out], *[float(row[k]) for k in FIT_COLUMNS])
                result.update(status=status, d_0_5=d, **aggregate(metrics))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                result.update(reason=str(exc))
        else:
            result.update(reason="no converged finite-likelihood fit in the saved diagnostics")
        rows.append(result)
    return pd.DataFrame(rows)


def stability_summary(observed, replicates):
    rows = []
    for original in observed:
        out = original["outcome"]
        full = replicates.loc[replicates.outcome.eq(out)]
        subsets = {"all_saved_fits": full}
        if out != VENTRICLES:
            subsets["historical_replay_agreement"] = full.loc[full.historical_replay_status.eq("agrees")]
        for analysis, group in subsets.items():
            defined = group.median_sqrt_kappa.notna()
            status = group.status
            row = {
                "outcome": out, "analysis": analysis, "N": original["N"],
                "B_requested": len(full), "B_in_subset": len(group), "B_defined": int(defined.sum()),
                "B_rank_one": int(status.eq("defined_rank_one").sum()),
                "B_numerical_near_rank_one": int(status.eq("defined_numerical_near_rank_one").sum()),
                "B_zero_target_variance": int(status.eq("undefined_zero_target_variance").sum()),
                "B_unresolved": int(status.eq("unresolved_fit").sum()),
                "B_replay_agrees": int(group.historical_replay_status.eq("agrees").sum()),
                "B_replay_differs_or_unavailable": int(group.historical_replay_status.eq("differs_or_unavailable").sum()),
            }
            for metric in METRICS:
                row[f"observed_{metric}"] = original[metric]
                values = group.loc[defined, metric]
                for name, q in [("q025", .025), ("q25", .25), ("median", .50),
                                ("q75", .75), ("q975", .975)]:
                    row[f"{metric}_{name}"] = values.quantile(q)
            rows.append(row)
    return pd.DataFrame(rows)


def report_bootstrap_status(summary):
    """Report each requested bootstrap and reject outcomes with no evaluable fits."""
    primary = summary.loc[summary.analysis.eq("all_saved_fits") & summary.B_requested.gt(0)]
    failed = []
    for row in primary.itertuples(index=False):
        label = LABELS[row.outcome]
        if row.B_unresolved == row.B_requested:
            failed.append(f"{label} ({row.B_unresolved}/{row.B_requested})")
        elif row.B_defined == 0:
            print(f"{label}: no defined bootstrap capacity. "
                  f"{row.B_zero_target_variance}/{row.B_requested} fits had zero target-slope "
                  f"variance (capacity undefined), and {row.B_unresolved} were unresolved. "
                  "Bootstrap percentiles are unavailable.")
        else:
            print(f"{label}: defined capacity for {row.B_defined}/{row.B_requested} fits.")
    if failed:
        raise SystemExit("Cannot evaluate bootstrap capacity: all saved fits were unresolved for "
                         + ", ".join(failed) + ". Check the saved fit flags, likelihoods, and covariance estimates.")


def load_profile(path):
    required = [
        "outcome", "dataset_role", "dataset_id", "path_direction", "d_0_5",
        "exact_PSD_boundary", "optimizer_agreement", "selected_gamma_sign",
        "selected_gamma_0_5", "optimized_log_slope_sd", "optimized_log_sd5_over_sd0",
        "optimized_log_residual_sd", "profile_log_likelihood", "reference_log_likelihood",
        "relative_deviance",
    ]
    source = read_csv(path, required)
    profile = source.loc[source.outcome.eq(COGNITION) & source.dataset_role.eq("observed")].copy()
    if (len(profile) != 43 or profile.d_0_5.nunique() != 43 or
            profile.dataset_id.nunique() != 1 or profile.path_direction.eq("selected").sum() != 1 or
            not profile.d_0_5.between(0, 1).all() or
            not profile.d_0_5.eq(0).equals(profile.exact_PSD_boundary.map(is_true)) or
            profile.d_0_5.eq(0).sum() != 1):
        raise ValueError("Expected the complete 43-point observed cognition profile with one PSD boundary")
    if not profile.optimizer_agreement.map(is_true).all():
        raise ValueError("Saved profile contains uncorroborated fits")
    return profile.sort_values("d_0_5")


def profile_components(row):
    """Recover midpoint G from the saved anchored profile parameterization."""
    d = row.d_0_5
    sign = row.selected_gamma_sign
    if sign not in {-1, 1}:
        raise ValueError("invalid saved endpoint-correlation sign")
    gamma = sign * np.sqrt(1.0 - d)
    close("profile endpoint correlation", gamma, row.selected_gamma_0_5)
    log_ratio = row.optimized_log_sd5_over_sd0
    ratio = np.exp(log_ratio)
    # Equivalent to 1 + ratio**2 - 2*gamma*ratio in the repository profile.
    one_minus_gamma = d / (1.0 + gamma) if sign > 0 else 1.0 - gamma
    slope_scale = np.expm1(log_ratio) ** 2 + 2 * ratio * one_minus_gamma
    if not np.isfinite(slope_scale) or slope_scale <= 1e-14:
        raise ValueError("degenerate saved endpoint parameterization")
    tau2 = np.exp(2 * row.optimized_log_slope_sd)
    v0 = tau2 / slope_scale
    v5 = v0 * ratio ** 2
    G00 = (v0 + v5 + 2 * gamma * ratio * v0) / 4
    cov01 = v0 * np.expm1(2 * log_ratio) / 2
    sigma2 = np.exp(2 * row.optimized_log_residual_sd)
    return G00, tau2, cov01, sigma2, (v0 / tau2) * v5 * d


def evaluate_profile(profile, moments, observed):
    close("saved profile deviance", profile.relative_deviance,
          2 * (profile.reference_log_likelihood - profile.profile_log_likelihood), atol=1e-7)
    results = []
    for row in profile.itertuples(index=False):
        *components, S = profile_components(row)
        status, d, metrics = evaluate(moments, *components, profile_S=S)
        close("reconstructed profile coordinate", d, row.d_0_5)
        if row.d_0_5 == 0:
            close("rank-one profile kappa", metrics.kappa, 0, atol=0)
        results.append({
            "outcome": COGNITION, "N": len(moments), "d_0_5": row.d_0_5,
            "relative_deviance": row.relative_deviance, "status": status,
            "exact_PSD_boundary": is_true(row.exact_PSD_boundary),
            "path_direction": row.path_direction, "optimizer_agreement": True,
            **dict(zip(FIT_COLUMNS, components)), **aggregate(metrics),
        })
    result = pd.DataFrame(results)
    selected = result.loc[result.path_direction.eq("selected")].iloc[0]
    close("profile fitted geometry vs. observed fit", selected.d_0_5, observed["d_0_5"], atol=OBSERVED_TOL)
    comparison = pd.DataFrame([
        {"outcome": COGNITION, "N": len(moments), "quantity": quantity,
         "original_fit": observed[quantity], "selected_profile": selected[quantity],
         "profile_minus_original": selected[quantity] - observed[quantity]}
        for quantity in ["d_0_5", *FIT_COLUMNS, *METRICS]
    ])
    difference = comparison.loc[comparison.quantity.eq("median_sqrt_kappa"), "profile_minus_original"].item()
    print(f"Profile selected minus Figure 4 median sqrt(kappa): {difference:+.6g}")
    return result, comparison


def plot_results(out_dir, observed, replicates, profile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.8, 3.7), constrained_layout=True)
    for row in observed:
        out = row["outcome"]
        group = replicates.loc[replicates.outcome.eq(out)]
        sample = np.sort(group.median_sqrt_kappa.dropna())
        if len(sample):
            left.step(np.r_[0, sample, 1], np.r_[0, np.arange(1, len(sample) + 1) / len(sample), 1],
                      where="post", color=COLORS[out], label=f"{LABELS[out]} ({len(sample)} fits)")
        else:
            label = "original fit" if group.empty else "no defined bootstrap capacity"
            left.plot([], [], linestyle=":", color=COLORS[out], label=f"{LABELS[out]} ({label})")
        left.axvline(row["median_sqrt_kappa"], linestyle=":", color=COLORS[out], alpha=.85)
    left.set(xlim=(0, 1), ylim=(0, 1.02), xlabel=r"Median participant $\sqrt{\kappa_i}$",
             ylabel="Cumulative fraction of defined fits", title="A  Participant bootstrap")
    left.legend(frameon=False, fontsize=8, loc="lower right")

    original = next(r for r in observed if r["outcome"] == COGNITION)
    cog = replicates.loc[replicates.outcome.eq(COGNITION) & replicates.median_sqrt_kappa.notna()]
    positive_d = cog.loc[cog.d_0_5.gt(0), "d_0_5"].tolist()
    if original["d_0_5"] > 0:
        positive_d.append(original["d_0_5"])
    zero_position = max((-np.log10(positive_d)).tolist(), default=0.0) + 1

    def xcoord(d):
        d = np.asarray(d, dtype=float)
        x = np.full(d.shape, np.nan)
        positive = d > 0
        x[positive] = -np.log10(d[positive])
        x[d == 0] = zero_position
        return x

    norm = Normalize(0, 1)
    points = right.scatter(xcoord(cog.d_0_5), cog.median_sqrt_kappa,
                           c=cog.mean_Delta_fit, vmin=norm.vmin, vmax=norm.vmax,
                           cmap="viridis", s=15, alpha=.75, linewidths=0)
    right.scatter(xcoord([original["d_0_5"]]), [original["median_sqrt_kappa"]],
                  c=[original["mean_Delta_fit"]], cmap="viridis", norm=norm,
                  marker="D", s=55, edgecolor="black", linewidth=.9,
                  zorder=3, label="Original cognition fit")
    if cog.d_0_5.eq(0).any():
        ticks = [tick for tick in right.get_xticks() if 0 <= tick < zero_position - .4]
        right.set_xticks([*ticks, zero_position], [*[f"{tick:g}" for tick in ticks],
                                                  r"$d_{0,5}=0$"])
        right.axvline(zero_position - .5, color=".85", linewidth=.8)
    right.set(ylim=(-.02, 1.02), xlabel=r"$-\log_{10}(d_{0,5})$",
              ylabel=r"Median participant $\sqrt{\kappa_i}$", title="B  Cognition covariance geometry")
    right.legend(frameon=False, fontsize=8, loc="upper right")
    if cog.empty:
        right.text(.5, .05, "No defined cognition bootstrap capacity", transform=right.transAxes,
                   ha="center", fontsize=8)
    fig.colorbar(points, ax=right, label=r"Mean $\Delta_{\mathrm{fit}}$")
    fig.savefig(out_dir / "aabc2_kappa_bootstrap.pdf")
    plt.close(fig)

    if profile is not None:
        plot_cognition_profile(profile, out_dir / "aabc2_kappa_cognition_profile.pdf")


def plot_cognition_profile(profile, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, StrMethodFormatter

    selected = profile.loc[profile.path_direction.eq("selected")]
    boundary = profile.loc[profile.exact_PSD_boundary]
    directions = [
        ("lower", "Toward rank one", COLORS[COGNITION], "-"),
        ("upper", "Away from rank one", "#8b5661", "--"),
    ]
    paths = [
        (pd.concat([selected, profile.loc[profile.path_direction.eq(direction)]])
         .sort_values("d_0_5"), label, color, style)
        for direction, label, color, style in directions
    ]

    def draw(ax, marker_size):
        for points, label, color, style in paths:
            ax.plot(points.relative_deviance, points.median_sqrt_kappa, label=label,
                    color=color, linestyle=style, linewidth=1.4, marker="o",
                    markersize=marker_size)
        ax.scatter(selected.relative_deviance, selected.median_sqrt_kappa,
                   label="Selected fit", marker="o", s=28, color="#212832", zorder=4)
        ax.scatter(boundary.relative_deviance, boundary.median_sqrt_kappa,
                   label="Rank-one boundary", marker="D", s=30, facecolor="white",
                   edgecolor=COLORS[COGNITION], zorder=4)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    draw(ax, 2.8)
    ax.set_xscale("symlog", linthresh=2)
    ax.set(xlabel="Relative profile deviance (symmetric log scale)",
           ylabel=r"Median participant $\sqrt{\kappa_i}$", ylim=(-.045, .86))
    ax.xaxis.set_major_locator(FixedLocator([0, 1, 10, 100]))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8)

    inset = ax.inset_axes([.55, .47, .40, .37])
    draw(inset, 2.3)
    inset.set(xlim=(-.07, 2), ylim=(-.055, .75), xticks=[0, 1, 2], yticks=[0, .3, .6])
    inset.set_title("Deviance 0–2", fontsize=8, loc="left", pad=3)
    inset.tick_params(labelsize=8)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("repo", "input-config", "exact-reliability", "bootstrap-geometry"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--profile", type=Path, help="Path to the observed-cognition profile CSV.")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--include-thickness", action="store_true")
    parser.add_argument("--ventricular-geometry", type=Path,
                        help="Optional saved ventricular whole-participant bootstrap fits in geometry-export schema. "
                             "Use only the same midpoint-coded observed reference model. "
                             "Set --ventricular-reps to the attempted count.")
    parser.add_argument("--ventricular-reps", type=int)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not args.preflight and args.out_dir is None:
        parser.error("--out-dir is required for a full run")
    if args.profile and args.skip_profile:
        parser.error("choose either --profile or --skip-profile")
    if bool(args.ventricular_geometry) != (args.ventricular_reps is not None):
        parser.error("--ventricular-geometry and --ventricular-reps are required together")
    if args.ventricular_reps is not None and args.ventricular_reps < 1:
        parser.error("--ventricular-reps must be positive")

    repo = args.repo.expanduser().resolve()
    common_path = repo / "reproduction/s7_s8_aabc2_restricted/aabc2_common.py"
    if not common_path.is_file():
        raise FileNotFoundError(f"Missing AABC2 cohort loader: {common_path}")
    sys.path.insert(0, str(common_path.parent))
    import aabc2_common as common

    geometry_path = args.bootstrap_geometry.expanduser()
    boot_outcomes = [COGNITION, THICKNESS] if args.include_thickness else [COGNITION]
    fits = geometry_source(geometry_path, boot_outcomes, 499)
    if args.ventricular_geometry:
        vent_path = args.ventricular_geometry.expanduser()
        vent = geometry_source(vent_path, [VENTRICLES], args.ventricular_reps)
        fits = pd.concat([fits, vent], ignore_index=True)
    profile = None
    if not args.skip_profile:
        profile_path = args.profile.expanduser() if args.profile else repo / PROFILE_PATH
        profile = load_profile(profile_path)

    exact_path = args.exact_reliability.expanduser()
    config_path = args.input_config.expanduser()
    dec_path = repo / FIGURE4_DIR / "s7_aabc2_decomposition.csv"
    kappa_path = repo / FIGURE4_DIR / "s7_aabc2_kappa_quantiles.csv"
    exact = read_csv(exact_path, ["outcome", "participant_id", "scan_count",
                                "direct_schedule_information_I_i", *FIT_COLUMNS, *SOURCE_METRICS.values()],
                     dtype={"participant_id": str})
    dec = pd.read_csv(dec_path, index_col="outcome", float_precision="round_trip")
    kappa = pd.read_csv(kappa_path, index_col="outcome", float_precision="round_trip")
    raw = common.build_raw_cohorts(config_path)
    specs = common.outcome_specs(raw)
    outcomes = [COGNITION, VENTRICLES, THICKNESS] if args.include_thickness else [COGNITION, VENTRICLES]
    schedules = {out: schedule_moments(common.prepare_outcome_data(out, specs)) for out in outcomes}
    observed = [check_observed(out, schedules[out], exact, dec, kappa) for out in outcomes]
    print("Saved bootstrap rows:", fits.groupby("outcome").size().to_dict())
    if not args.ventricular_geometry:
        print("Ventricles contribute the original fit only.")
    if profile is not None:
        profile, profile_comparison = evaluate_profile(profile, schedules[COGNITION], observed[0])
        print(f"Observed cognition profile points: {len(profile)}")
    replicates = evaluate_bootstrap(fits, schedules)
    summary = stability_summary(observed, replicates)
    print(summary[["outcome", "analysis", "B_requested", "B_defined",
                   "B_zero_target_variance", "B_unresolved",
                   "B_replay_differs_or_unavailable"]].to_string(index=False), flush=True)
    report_bootstrap_status(summary)
    if args.preflight:
        print("Preflight complete. Run without --preflight to write tables and figures.")
        return

    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {out_dir}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".kappa_stability_", dir=out_dir.parent) as staging:
        staged = Path(staging) / "results"
        staged.mkdir()
        pd.DataFrame(observed).to_csv(staged / "aabc2_kappa_observed.csv", index=False)
        replicates.to_csv(staged / "aabc2_kappa_bootstrap_replicates.csv", index=False)
        summary.to_csv(staged / "aabc2_kappa_bootstrap_summary.csv", index=False)
        if profile is not None:
            profile.to_csv(staged / "aabc2_kappa_cognition_profile.csv", index=False)
            profile_comparison.to_csv(staged / "aabc2_kappa_cognition_profile_comparison.csv", index=False)
        plot_results(staged, observed, replicates, profile)
        manifest = {
            "schedule_basis": "original outcome-specific cohort; midpoint time in five-year units",
            "historical_resampling": (
                "whole participants drawn with replacement from each outcome cohort; "
                "all visits retained; draws assigned fresh group IDs; no site stratification"
            ),
            "primary_quantity": "median_i sqrt(kappa_i) within each fit",
            "bootstrap_range": "central percentile range among fits with defined capacity",
            "replay_agreement_tolerance": REPLAY_TOL,
            "numerical_rank_tolerance_relative_to_largest_eigenvalue": NUMERICAL_RANK_TOL,
            "ventricular_bootstrap_included": bool(args.ventricular_geometry),
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(out_dir)
    print(f"Wrote aggregate tables and figures to {out_dir}")


if __name__ == "__main__":
    main()
