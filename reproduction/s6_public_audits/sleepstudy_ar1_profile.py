"""Reproduce the public sleepstudy AR(1) profile diagnostic."""

from __future__ import annotations

import argparse

import hashlib
import json
import math
import os
import platform
import sys
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import linalg, optimize, stats

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, desc=None, unit=None, total=None):
        if desc:
            print(desc)
        for j, item in enumerate(iterable, 1):
            if j == 1 or j % 25 == 0:
                print(f"  {j}/{total or '?'} {unit or ''}")
            yield item

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)
np.set_printoptions(precision=8, suppress=True)

def display(obj):
    """Notebook-compatible display shim for ordinary script execution."""
    print(obj)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the public sleepstudy AR(1) profile diagnostic."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which to write all analysis outputs.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a reduced nonreportable code-path QA check instead of "
            "the full analysis."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()
RESULTS_DIR = ARGS.output_dir.expanduser().resolve()
DATA_DIR = RESULTS_DIR / "data"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURE_DIR = RESULTS_DIR / "figures"
for directory in (RESULTS_DIR, DATA_DIR, CHECKPOINT_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ENGINE_VERSION = "sleepstudy_AR1_profile_v1_2026-08-04"
MASTER_SEED = 20260804
ALPHA = 0.05
PROVISIONAL_NULL_B = 999
FINAL_NULL_B = 4_999
PARTICIPANT_BOOT_B = 499
MIN_USABLE_RATE = 0.95
CHECKPOINT_EVERY = 10

NESTED_NLL_TOL = 1e-5
OPT_NLL_GAP_PER_OBS_TOL = 1e-6
OPT_SOURCE_ABS_GAP_TOL = 5e-3
SOURCE_SIGN_EPS = 1e-8
LR_RESCUE_THRESHOLD = 75.0
AR1_INTERIOR_LIMIT = 0.98

RUN_BOUNDARY_BOOTSTRAP = True
RUN_PARTICIPANT_BOOTSTRAP = True
RUN_LEAVE_ONE_OUT = True

# Use --smoke-test only for a quick code-path check.
# Smoke-test outputs are intentionally nonreportable.
SMOKE_TEST = ARGS.smoke_test
if SMOKE_TEST:
    PROVISIONAL_NULL_B = 7
    FINAL_NULL_B = 11
    PARTICIPANT_BOOT_B = 7
    print("SMOKE_TEST=True: inferential outputs are nonreportable code-path checks.")

FROZEN_CONFIG = {
    name: globals()[name] for name in [
        "ENGINE_VERSION", "MASTER_SEED", "ALPHA", "PROVISIONAL_NULL_B", "FINAL_NULL_B",
        "PARTICIPANT_BOOT_B", "MIN_USABLE_RATE", "CHECKPOINT_EVERY", "NESTED_NLL_TOL",
        "OPT_NLL_GAP_PER_OBS_TOL", "OPT_SOURCE_ABS_GAP_TOL", "SOURCE_SIGN_EPS",
        "LR_RESCUE_THRESHOLD", "AR1_INTERIOR_LIMIT", "SMOKE_TEST",
    ]
}
FROZEN_CONFIG.update({
    "dataset": "lme4::sleepstudy",
    "target_slope": "Days (reaction-time milliseconds per day)",
    "time_window": "Days 0--9",
    "primary_fixed_mean": "intercept + Days",
    "primary_random_effects": "correlated random intercept + Days slope by Subject",
    "primary_residual": "homogeneous independent Gaussian",
    "source_stability_quantity": "participant-mean R_direct - participant-mean Delta_fit",
    "source_stability_rule": "95% participant-cluster-bootstrap interval preserves the full-sample sign",
    "case_retention": "retain regardless of result",
})
(RESULTS_DIR / "00_frozen_analysis_contract.json").write_text(
    json.dumps(FROZEN_CONFIG, indent=2), encoding="utf-8"
)

print({
    "python": sys.version.split()[0], "platform": platform.platform(),
    "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
    "results_dir": str(RESULTS_DIR),
})

# This successor notebook does no inferential resampling.  Its purpose is to
# explain the prespecified AR(1) sensitivity by profiling rho_e while refitting G.
RUN_DAYS_2_9_POSTHOC = True
PROFILE_GRID_COARSE = np.linspace(-0.30, 0.95, 41)
PROFILE_FINE_HALF_WIDTH = 0.12
PROFILE_FINE_POINTS = 17
PROFILE_CI_CUTOFF = float(stats.chi2.ppf(0.95, 1))
PROFILE_CHECKPOINT_EVERY = 1
if SMOKE_TEST:
    PROFILE_GRID_COARSE = np.array([-0.20, 0.00, 0.30, 0.60, 0.85])
    PROFILE_FINE_POINTS = 3

PROFILE_CONTRACT = {
    "engine_version": ENGINE_VERSION,
    "primary_analysis": "Days 0--9; Reaction ~ Days + (1 + Days | Subject); homogeneous independent residuals",
    "diagnostic_target": "profile residual AR(1) rho_e while re-estimating fixed effects, G, and residual scale",
    "recorded_quantities": ["REML likelihood", "descriptive RI-vs-RI+RS LR", "R_direct", "P_full", "Delta_fit", "innovation reliability", "G boundary geometry"],
    "posthoc_subset": "Days 2--9, with Day 2 reset to analysis time 0; diagnostic only",
    "primary_freeze": "The Days 0--9 homogeneous-residual audit and its supported Question 1 result are not replaced by either diagnostic.",
}
(RESULTS_DIR / "00_profile_contract.json").write_text(json.dumps(PROFILE_CONTRACT, indent=2), encoding="utf-8")

SLEEPSTUDY_URLS = [
    "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv",
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/lme4/sleepstudy.csv",
]
CANONICAL_RDATSETS_SHA256 = "20922ddc87d538bf2344f73d49000eeb2d7e14934b6191a3e071fefc329942c4"
CANONICAL_SUBJECTS = {
    "308", "309", "310", "330", "331", "332", "333", "334", "335",
    "337", "349", "350", "351", "352", "369", "370", "371", "372",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def synthetic_sleepstudy_for_smoke() -> pd.DataFrame:
    rng = np.random.default_rng(20260804)
    subjects = sorted(CANONICAL_SUBJECTS)
    G = np.array([[24.7 ** 2, 9.6], [9.6, 5.92 ** 2]])
    rows = []
    for subject in subjects:
        b0, b1 = rng.multivariate_normal(np.zeros(2), G)
        for day in range(10):
            reaction = 251.4 + 10.47 * day + b0 + b1 * day + rng.normal(0, 25.6)
            rows.append({"Reaction": reaction, "Days": day, "Subject": subject})
    return pd.DataFrame(rows)


def load_sleepstudy() -> tuple[pd.DataFrame, dict]:
    if SMOKE_TEST:
        frame = synthetic_sleepstudy_for_smoke()
        return frame, {"source": "schema-faithful synthetic smoke fixture", "sha256": None}

    supplied = os.getenv("SLEEPSTUDY_CSV")
    cache_path = DATA_DIR / "sleepstudy.csv"
    if supplied:
        source_path = Path(supplied).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"SLEEPSTUDY_CSV does not exist: {source_path}")
        payload = source_path.read_bytes()
        cache_path.write_bytes(payload)
        source = "user-supplied file via SLEEPSTUDY_CSV"
    elif cache_path.exists():
        payload = cache_path.read_bytes()
        source = "verified local cache"
    else:
        payload, source, last_error = None, None, None
        for url in SLEEPSTUDY_URLS:
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    candidate = response.read()
                if len(candidate) < 1_000:
                    raise RuntimeError("downloaded object is unexpectedly small")
                payload, source = candidate, url
                cache_path.write_bytes(payload)
                break
            except Exception as exc:
                last_error = exc
        if payload is None:
            raise RuntimeError(
                "Could not download sleepstudy. Download the lme4 sleepstudy CSV, set "
                "SLEEPSTUDY_CSV to its path, and rerun."
            ) from last_error

    frame = pd.read_csv(cache_path)
    unnamed = [c for c in frame.columns if c.lower().startswith("unnamed") or c == "rownames"]
    frame = frame.drop(columns=unnamed, errors="ignore")
    return frame, {"source": source, "sha256": sha256_file(cache_path)}


sleep, provenance = load_sleepstudy()
sleep = sleep.rename(columns={c: c.strip() for c in sleep.columns})
required = {"Reaction", "Days", "Subject"}
if not required.issubset(sleep.columns):
    raise ValueError(f"Required columns absent: {sorted(required - set(sleep.columns))}")

sleep = sleep.loc[:, ["Reaction", "Days", "Subject"]].copy()
sleep["Reaction"] = pd.to_numeric(sleep["Reaction"], errors="raise")
sleep["Days"] = pd.to_numeric(sleep["Days"], errors="raise").astype(int)
sleep["Subject"] = sleep["Subject"].astype(str)
sleep = sleep.sort_values(["Subject", "Days"]).reset_index(drop=True)

counts = sleep.groupby("Subject").size()
checks = {
    "rows_180": len(sleep) == 180,
    "subjects_18": sleep["Subject"].nunique() == 18,
    "canonical_subject_ids": set(sleep["Subject"]) == CANONICAL_SUBJECTS,
    "ten_rows_each": counts.eq(10).all(),
    "days_0_to_9_each": all(set(d["Days"]) == set(range(10)) for _, d in sleep.groupby("Subject")),
    "no_missing": not sleep.isna().any().any(),
    "known_first_value": np.isclose(
        sleep.query("Subject == '308' and Days == 0")["Reaction"].iloc[0], 249.5600, atol=1e-4
    ) if not SMOKE_TEST else True,
    "canonical_response_range": (
        np.isclose(sleep["Reaction"].min(), 194.3322, atol=1e-3)
        and np.isclose(sleep["Reaction"].max(), 466.3535, atol=1e-3)
    ) if not SMOKE_TEST else True,
}
if not all(checks.values()):
    raise RuntimeError(f"sleepstudy identity/schedule verification failed: {checks}")

provenance.update({
    "checks": checks, "rows": len(sleep), "participants": sleep.Subject.nunique(),
    "canonical_RDatasets_SHA256": CANONICAL_RDATSETS_SHA256,
    "downloaded_bytes_match_canonical_RDatasets": provenance.get("sha256") == CANONICAL_RDATSETS_SHA256,
})
(RESULTS_DIR / "01_data_provenance.json").write_text(
    json.dumps(provenance, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)),
    encoding="utf-8",
)
sleep.to_csv(RESULTS_DIR / "02_sleepstudy_analysis_data.csv", index=False)
display(pd.DataFrame([checks]).T.rename(columns={0: "pass"}))
display(sleep.groupby("Days").agg(n=("Reaction", "size"), mean=("Reaction", "mean"), sd=("Reaction", "std")))

@dataclass(frozen=True)
class GroupData:
    subject: str
    y: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    days: np.ndarray


def fixed_design(days: np.ndarray, mean_model: str) -> tuple[np.ndarray, list[str]]:
    days = np.asarray(days, float)
    if mean_model == "linear":
        return np.column_stack([np.ones(len(days)), days]), ["Intercept", "Days"]
    if mean_model == "quadratic":
        day_c = days - 4.5
        return np.column_stack([np.ones(len(days)), days, day_c ** 2]), ["Intercept", "Days", "Days_c_squared"]
    raise ValueError(mean_model)


def build_groups(frame: pd.DataFrame, mean_model="linear") -> tuple[list[GroupData], list[str]]:
    groups, fixed_names = [], None
    for subject, d in frame.groupby("Subject", sort=True):
        d = d.sort_values("Days")
        days = d["Days"].to_numpy(float)
        X, names = fixed_design(days, mean_model)
        Z = np.column_stack([np.ones(len(days)), days])
        fixed_names = names
        groups.append(GroupData(str(subject), d["Reaction"].to_numpy(float), X, Z, days))
    stacked_x = np.vstack([g.X for g in groups])
    if np.linalg.matrix_rank(stacked_x) != stacked_x.shape[1]:
        raise RuntimeError("Fixed-effects design is rank deficient")
    return groups, fixed_names


def groups_with_new_y(template: Sequence[GroupData], ys: Sequence[np.ndarray], prefix="sim") -> list[GroupData]:
    return [
        GroupData(f"{prefix}_{j:03d}_{g.subject}", np.asarray(y, float), g.X, g.Z, g.days)
        for j, (g, y) in enumerate(zip(template, ys))
    ]


groups_linear, fixed_names_linear = build_groups(sleep, "linear")
groups_quadratic, fixed_names_quadratic = build_groups(sleep, "quadratic")
N_OBS = sum(len(g.y) for g in groups_linear)

def response_scale(groups: Sequence[GroupData]) -> float:
    return max(float(np.std(np.concatenate([g.y for g in groups]), ddof=1)), 1.0)


def parameter_bounds(groups: Sequence[GroupData], random_slope: bool, residual_model: str):
    scale = response_scale(groups)
    log_lo = math.log(1e-4)
    log_hi = math.log(max(5.0 * scale, 200.0))
    bounds = []
    if random_slope:
        bounds.extend([(log_lo, log_hi), (-2.0 * scale, 2.0 * scale), (log_lo, log_hi)])
    else:
        bounds.append((log_lo, log_hi))
    if residual_model == "hom":
        bounds.append((log_lo, log_hi))
    elif residual_model == "ar1":
        bounds.extend([(log_lo, log_hi), (-3.0, 3.0)])
    else:
        raise ValueError(residual_model)
    return bounds


def initial_theta(groups: Sequence[GroupData], random_slope: bool, residual_model: str) -> np.ndarray:
    frame_y = np.concatenate([g.y for g in groups])
    scale = max(float(np.std(frame_y, ddof=1)), 10.0)
    pars = [math.log(max(0.30 * scale, 5.0))]
    if random_slope:
        pars.extend([0.0, math.log(max(0.07 * scale, 1.0))])
    pars.append(math.log(max(0.30 * scale, 5.0)))
    if residual_model == "ar1":
        pars.append(np.arctanh(0.10))
    return np.asarray(pars, float)


def unpack_theta(theta: np.ndarray, random_slope: bool, residual_model: str):
    theta = np.asarray(theta, float)
    if random_slope:
        L = np.array([[np.exp(theta[0]), 0.0], [theta[1], np.exp(theta[2])]])
        G = L @ L.T
        k = 3
    else:
        G = np.array([[np.exp(2.0 * theta[0]), 0.0], [0.0, 0.0]])
        k = 1
    residual = {"sigma": float(np.exp(theta[k]))}
    if residual_model == "ar1":
        residual["rho"] = float(np.tanh(theta[k + 1]))
    return G, residual


def residual_covariance(group: GroupData, residual: dict, residual_model: str) -> np.ndarray:
    n = len(group.y)
    sigma2 = residual["sigma"] ** 2
    if residual_model == "hom":
        return sigma2 * np.eye(n)
    if residual_model == "ar1":
        lag = np.abs(group.days[:, None] - group.days[None, :]).astype(int)
        return sigma2 * np.power(residual["rho"], lag)
    raise ValueError(residual_model)


def balanced_design(groups: Sequence[GroupData]) -> bool:
    g0 = groups[0]
    return all(
        g.X.shape == g0.X.shape and g.Z.shape == g0.Z.shape
        and np.array_equal(g.X, g0.X) and np.array_equal(g.Z, g0.Z)
        and np.array_equal(g.days, g0.days)
        for g in groups[1:]
    )


def profiled_likelihood(theta, groups, random_slope=True, residual_model="hom", reml=True, return_all=False):
    try:
        G, residual = unpack_theta(theta, random_slope, residual_model)
        p = groups[0].X.shape[1]
        A, b = np.zeros((p, p)), np.zeros(p)
        y_vinv_y = logdet_v = 0.0
        n_total = sum(len(g.y) for g in groups)

        if balanced_design(groups):
            g0 = groups[0]
            R = residual_covariance(g0, residual, residual_model)
            V = g0.Z @ G @ g0.Z.T + R
            cf = linalg.cho_factor(V, lower=True, check_finite=False)
            ViX = linalg.cho_solve(cf, g0.X, check_finite=False)
            A = len(groups) * (g0.X.T @ ViX)
            logdet_one = 2.0 * np.log(np.diag(cf[0])).sum()
            logdet_v = len(groups) * logdet_one
            for group in groups:
                Viy = linalg.cho_solve(cf, group.y, check_finite=False)
                b += group.X.T @ Viy
                y_vinv_y += float(group.y @ Viy)
        else:
            for group in groups:
                R = residual_covariance(group, residual, residual_model)
                V = group.Z @ G @ group.Z.T + R
                cf = linalg.cho_factor(V, lower=True, check_finite=False)
                ViX = linalg.cho_solve(cf, group.X, check_finite=False)
                Viy = linalg.cho_solve(cf, group.y, check_finite=False)
                A += group.X.T @ ViX
                b += group.X.T @ Viy
                y_vinv_y += float(group.y @ Viy)
                logdet_v += 2.0 * np.log(np.diag(cf[0])).sum()

        sign_a, logdet_a = np.linalg.slogdet(A)
        if sign_a <= 0 or np.linalg.matrix_rank(A) < p:
            raise np.linalg.LinAlgError("fixed-effects information is singular")
        beta = np.linalg.solve(A, b)
        quad = max(y_vinv_y - float(b @ beta), 0.0)
        if reml:
            nll = 0.5 * (logdet_v + logdet_a + quad + (n_total - p) * np.log(2.0 * np.pi))
        else:
            nll = 0.5 * (logdet_v + quad + n_total * np.log(2.0 * np.pi))
        if not np.isfinite(nll):
            raise FloatingPointError("nonfinite likelihood")
        if return_all:
            return nll, beta, np.linalg.inv(A), G, residual, quad
        return nll
    except Exception:
        if return_all:
            raise
        return 1e100

OBSERVED_OPTIMIZERS = ("L-BFGS-B", "Powell", "Nelder-Mead")
BOOTSTRAP_OPTIMIZERS = ("L-BFGS-B", "Powell")


def fit_model(groups, random_slope, residual_model, optimizer, start, reml=True, maxiter=3000):
    bounds = parameter_bounds(groups, random_slope, residual_model)
    options = {"maxiter": maxiter}
    if optimizer == "L-BFGS-B":
        options.update({"ftol": 1e-12, "gtol": 1e-7, "maxls": 60})
    elif optimizer == "Powell":
        options.update({"xtol": 1e-7, "ftol": 1e-10})
    elif optimizer == "Nelder-Mead":
        options.update({"xatol": 1e-7, "fatol": 1e-8, "adaptive": True})
    result = optimize.minimize(
        profiled_likelihood, np.asarray(start, float),
        args=(groups, random_slope, residual_model, reml, False),
        method=optimizer, bounds=bounds, options=options,
    )
    try:
        nll, beta, cov_beta, G, residual, quad = profiled_likelihood(
            result.x, groups, random_slope, residual_model, reml, True
        )
        residual_ok = all(
            np.linalg.eigvalsh(residual_covariance(g, residual, residual_model)).min() > 0
            for g in groups
        )
        eig_g = np.linalg.eigvalsh(G)
        admissible = bool(eig_g.min() >= -1e-10 and G[0, 0] > 0 and residual_ok)
    except Exception:
        nll, beta, cov_beta, G, residual, quad = np.inf, None, None, None, None, np.nan
        admissible = False
    return {
        "success": bool(result.success), "admissible": admissible, "message": str(result.message),
        "optimizer": optimizer, "theta": np.asarray(result.x, float), "nll": float(nll),
        "loglik": float(-nll), "beta": beta, "cov_beta": cov_beta, "G": G,
        "residual": residual, "quad": quad, "nit": getattr(result, "nit", np.nan),
        "random_slope": random_slope, "residual_model": residual_model, "reml": reml,
    }


def fit_usable(fit: dict) -> bool:
    return bool(fit["success"] and fit["admissible"] and np.isfinite(fit["nll"]))


def residual_start_from_fit(source_fit: dict, target_model: str) -> list[float]:
    sigma = float(source_fit["residual"]["sigma"])
    out = [math.log(sigma)]
    if target_model == "ar1":
        rho = float(source_fit["residual"].get("rho", 0.0))
        out.append(np.arctanh(np.clip(rho, -0.98, 0.98)))
    return out


def mapped_start_from_fit(source_fit: dict, target_random_slope: bool, target_residual_model: str, slope_floor=1e-4):
    G = np.asarray(source_fit["G"], float)
    q0 = max(float(G[0, 0]), 1e-8)
    pars = []
    if target_random_slope:
        if source_fit["random_slope"] and G[1, 1] > slope_floor ** 2:
            candidate = G.copy()
        else:
            candidate = np.array([[q0, 0.0], [0.0, slope_floor ** 2]])
        jitter = 0.0
        for _ in range(10):
            try:
                L = np.linalg.cholesky(candidate + jitter * np.eye(2))
                break
            except np.linalg.LinAlgError:
                jitter = 1e-10 if jitter == 0 else 10.0 * jitter
        pars.extend([math.log(L[0, 0]), L[1, 0], math.log(max(L[1, 1], 1e-4))])
    else:
        pars.append(math.log(np.sqrt(q0)))
    pars.extend(residual_start_from_fit(source_fit, target_residual_model))
    return np.asarray(pars, float)


def unique_starts(groups, random_slope, residual_model, extras=()):
    specs = [("default", initial_theta(groups, random_slope, residual_model)), *list(extras)]
    bounds = parameter_bounds(groups, random_slope, residual_model)
    seen, out = set(), []
    for name, values in specs:
        values = np.asarray(values, float)
        if len(values) != len(bounds):
            raise ValueError(f"Start {name} has length {len(values)}; expected {len(bounds)}")
        clipped = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(values, bounds)], float)
        key = tuple(np.round(clipped, 10))
        if key not in seen:
            seen.add(key)
            out.append((name, clipped))
    return out


def fit_multistart(groups, random_slope, residual_model, extra_starts=(), reml=True, optimizers=OBSERVED_OPTIMIZERS):
    attempts = []
    for start_name, start in unique_starts(groups, random_slope, residual_model, extra_starts):
        anchor = start
        for optimizer_name in optimizers:
            fit = fit_model(groups, random_slope, residual_model, optimizer_name, anchor, reml=reml)
            fit["start_name"] = start_name if optimizer_name == optimizers[0] else f"{start_name} -> {optimizers[0]}"
            attempts.append(fit)
            if optimizer_name == optimizers[0] and fit["admissible"] and np.isfinite(fit["nll"]):
                anchor = fit["theta"]
    usable = [fit for fit in attempts if fit_usable(fit)]
    if not usable:
        raise RuntimeError(f"No usable {residual_model} {'RI+RS' if random_slope else 'RI'} fit")
    return {"best": min(usable, key=lambda x: x["nll"]), "attempts": attempts}


def fit_pair(groups, residual_model="hom", reml=True, seed_pair=None, optimizers=OBSERVED_OPTIMIZERS):
    null_extras = []
    if seed_pair is not None:
        null_extras.append(("mapped_seed_null", mapped_start_from_fit(seed_pair["null"], False, residual_model)))
    null_bundle = fit_multistart(groups, False, residual_model, null_extras, reml, optimizers)
    null_fit = null_bundle["best"]

    alt_extras = [
        ("nested_from_null_small", mapped_start_from_fit(null_fit, True, residual_model, 1e-4)),
        ("nested_from_null_moderate", mapped_start_from_fit(null_fit, True, residual_model, 0.50)),
    ]
    if seed_pair is not None:
        alt_extras.append(("mapped_seed_alt", mapped_start_from_fit(seed_pair["alt"], True, residual_model)))
    alt_bundle = fit_multistart(groups, True, residual_model, alt_extras, reml, optimizers)
    alt_fit = alt_bundle["best"]
    nested_ok = alt_fit["nll"] <= null_fit["nll"] + NESTED_NLL_TOL
    if not nested_ok:
        rescue = [
            ("nested_rescue_tiny", mapped_start_from_fit(null_fit, True, residual_model, 1e-4)),
            ("nested_rescue_small", mapped_start_from_fit(null_fit, True, residual_model, 0.05)),
            ("nested_rescue_large", mapped_start_from_fit(null_fit, True, residual_model, 2.0)),
        ]
        alt_bundle = fit_multistart(groups, True, residual_model, rescue, reml, OBSERVED_OPTIMIZERS)
        alt_fit = alt_bundle["best"]
        nested_ok = alt_fit["nll"] <= null_fit["nll"] + NESTED_NLL_TOL
    return {
        "null": null_fit, "alt": alt_fit, "null_bundle": null_bundle,
        "alt_bundle": alt_bundle, "nested_ok": nested_ok,
    }

def participant_source_table(groups: Sequence[GroupData], fit: dict) -> pd.DataFrame:
    G = np.asarray(fit["G"], float)
    q0, q01, q1 = float(G[0, 0]), float(G[0, 1]), float(G[1, 1])
    if q1 <= 0:
        raise ValueError("Slope variance must be positive before source attribution")
    q_innovation = max(q1 - q01 ** 2 / q0, 0.0)
    rows = []
    for group in groups:
        R = residual_covariance(group, fit["residual"], fit["residual_model"])
        info = group.Z.T @ np.linalg.solve(R, group.Z)
        J = max(float(info[1, 1] - info[1, 0] ** 2 / info[0, 0]), 0.0)
        effective_error = np.inf if J <= 1e-14 else 1.0 / J
        R_direct = q1 * J / (1.0 + q1 * J)
        V = group.Z @ G @ group.Z.T + R
        posterior = G - G @ group.Z.T @ np.linalg.solve(V, group.Z @ G)
        P_full = 1.0 - posterior[1, 1] / q1
        Delta_fit = P_full - R_direct
        R_innovation = q_innovation * J / (1.0 + q_innovation * J)
        rows.append({
            "subject": group.subject, "n_obs": len(group.y), "first_day": group.days.min(),
            "last_day": group.days.max(), "Sxx": float(np.sum((group.days - group.days.mean()) ** 2)),
            "direct_slope_information_J": J, "effective_OLS_slope_error_variance": effective_error,
            "R_direct": R_direct, "P_full": P_full, "Delta_fit": Delta_fit,
            "remaining_uncertainty": 1.0 - P_full, "innovation_variance_at_day0": q_innovation,
            "R_innovation": R_innovation,
        })
    out = pd.DataFrame(rows)
    if out["Delta_fit"].min() < -1e-7:
        raise AssertionError(f"Negative fitted assisted component: {out['Delta_fit'].min()}")
    if not np.allclose(out["P_full"], out["R_direct"] + out["Delta_fit"], atol=1e-11):
        raise AssertionError("P_full = R_direct + Delta_fit identity failed")
    return out


def source_summary(groups: Sequence[GroupData], fit: dict, label: str) -> dict:
    src = participant_source_table(groups, fit)
    G = np.asarray(fit["G"], float)
    rho = G[0, 1] / np.sqrt(G[0, 0] * G[1, 1])
    mean_r = float(src["R_direct"].mean())
    mean_p = float(src["P_full"].mean())
    mean_delta = mean_p - mean_r
    return {
        "model": label, "participants": len(src), "observations": sum(len(g.y) for g in groups),
        "mean_R_direct": mean_r, "mean_P_full": mean_p, "mean_Delta_fit": mean_delta,
        "mean_R_innovation": float(src["R_innovation"].mean()),
        "source_difference_d": mean_r - mean_delta,
        "direct_share": mean_r / mean_p if mean_p > 0 else np.nan,
        "var_intercept": G[0, 0], "cov_intercept_slope": G[0, 1], "var_slope": G[1, 1],
        "rho_intercept_slope_at_day0": rho, "one_minus_rho_squared": 1.0 - rho ** 2,
        "g_min_eigen_over_trace": np.linalg.eigvalsh(G).min() / np.trace(G),
        "residual_sd": fit["residual"]["sigma"], "nll": fit["nll"],
    }

primary_pair = fit_pair(groups_linear, "hom", reml=True)
primary_fit = primary_pair["alt"]
ar1_pair = fit_pair(groups_linear, "ar1", reml=True, seed_pair=primary_pair)
ar1_fit = ar1_pair["alt"]

if not primary_pair["nested_ok"] or not ar1_pair["nested_ok"]:
    raise RuntimeError("Observed RI+RS fit must be no worse than its RI null")
if ar1_fit["nll"] > primary_fit["nll"] + NESTED_NLL_TOL:
    raise RuntimeError("AR(1) model contains rho_e=0 and cannot fit worse than the homogeneous model")

primary_summary = source_summary(groups_linear, primary_fit, "Days0_9_homogeneous_primary")
ar1_summary = source_summary(groups_linear, ar1_fit, "Days0_9_AR1_unrestricted")
observed_summary = pd.DataFrame([primary_summary, ar1_summary])
observed_summary["residual_rho_e"] = [0.0, ar1_fit["residual"]["rho"]]
observed_summary["analysis_role"] = ["frozen primary", "prespecified sensitivity"]

if not SMOKE_TEST:
    guards = {
        "primary_nll": abs(primary_fit["nll"] - 871.814) <= 0.20,
        "ar1_nll": abs(ar1_fit["nll"] - 862.058) <= 0.20,
        "primary_D2": abs(primary_summary["mean_P_full"] - 0.8485) <= 0.01,
        "primary_Rdirect": abs(primary_summary["mean_R_direct"] - 0.8154) <= 0.01,
        "ar1_D2": abs(ar1_summary["mean_P_full"] - 0.8667) <= 0.015,
        "ar1_Rdirect": abs(ar1_summary["mean_R_direct"] - 0.5230) <= 0.02,
    }
    if not all(guards.values()):
        raise RuntimeError(f"Frozen sleepstudy benchmark guard failed: {guards}")

observed_summary.to_csv(RESULTS_DIR / "03_unrestricted_homogeneous_and_AR1_summary.csv", index=False)
display(observed_summary)

def _free_theta(full_theta: np.ndarray) -> np.ndarray:
    """Drop the final atanh(rho_e) coordinate from an AR(1) parameter vector."""
    return np.asarray(full_theta[:-1], float)


def _full_theta(free_theta: np.ndarray, rho_e: float) -> np.ndarray:
    if not (-0.999 < rho_e < 0.999):
        raise ValueError("rho_e must be strictly inside (-1, 1)")
    return np.r_[np.asarray(free_theta, float), np.arctanh(rho_e)]


def fixed_rho_objective(free_theta, groups, random_slope, rho_e, reml=True, return_all=False):
    return profiled_likelihood(
        _full_theta(free_theta, rho_e), groups, random_slope, "ar1", reml, return_all
    )


def fixed_rho_bounds(groups, random_slope):
    return parameter_bounds(groups, random_slope, "ar1")[:-1]


def fit_fixed_rho_once(groups, random_slope, rho_e, optimizer, start_free, reml=True, maxiter=3000):
    bounds = fixed_rho_bounds(groups, random_slope)
    options = {"maxiter": maxiter}
    if optimizer == "L-BFGS-B":
        options.update({"ftol": 1e-12, "gtol": 1e-7, "maxls": 60})
    elif optimizer == "Powell":
        options.update({"xtol": 1e-7, "ftol": 1e-10})
    result = optimize.minimize(
        fixed_rho_objective, np.asarray(start_free, float),
        args=(groups, random_slope, rho_e, reml, False),
        method=optimizer, bounds=bounds, options=options,
    )
    try:
        nll, beta, cov_beta, G, residual, quad = fixed_rho_objective(
            result.x, groups, random_slope, rho_e, reml, True
        )
        residual_ok = all(np.linalg.eigvalsh(residual_covariance(g, residual, "ar1")).min() > 0 for g in groups)
        admissible = bool(G[0, 0] > 0 and np.linalg.eigvalsh(G).min() >= -1e-10 and residual_ok)
    except Exception:
        nll, beta, cov_beta, G, residual, quad, admissible = np.inf, None, None, None, None, np.nan, False
    return {
        "success": bool(result.success), "admissible": admissible, "message": str(result.message),
        "optimizer": optimizer, "theta": _full_theta(result.x, rho_e), "free_theta": np.asarray(result.x, float),
        "nll": float(nll), "loglik": float(-nll), "beta": beta, "cov_beta": cov_beta,
        "G": G, "residual": residual, "quad": quad, "nit": getattr(result, "nit", np.nan),
        "random_slope": random_slope, "residual_model": "ar1", "reml": reml,
    }


def fit_fixed_rho_multistart(groups, random_slope, rho_e, seed_fits=()):
    starts = [("default", _free_theta(initial_theta(groups, random_slope, "ar1")))]
    for j, seed_fit in enumerate(seed_fits):
        starts.append((f"seed_{j}", _free_theta(mapped_start_from_fit(seed_fit, random_slope, "ar1"))))
    bounds = fixed_rho_bounds(groups, random_slope)
    seen, attempts = set(), []
    for start_name, start in starts:
        start = np.array([np.clip(x, lo, hi) for x, (lo, hi) in zip(start, bounds)], float)
        key = tuple(np.round(start, 9))
        if key in seen:
            continue
        seen.add(key)
        lb = fit_fixed_rho_once(groups, random_slope, rho_e, "L-BFGS-B", start)
        lb["start_name"] = start_name
        attempts.append(lb)
        anchor = lb["free_theta"] if lb["admissible"] and np.isfinite(lb["nll"]) else start
        pw = fit_fixed_rho_once(groups, random_slope, rho_e, "Powell", anchor)
        pw["start_name"] = start_name + " -> L-BFGS-B"
        attempts.append(pw)
    usable = [fit for fit in attempts if fit_usable(fit)]
    if not usable:
        raise RuntimeError(f"No usable fixed-rho fit at rho_e={rho_e:.5f}")
    return {"best": min(usable, key=lambda fit: fit["nll"]), "attempts": attempts}


def fixed_rho_profile_point(groups, rho_e, hom_pair, unrestricted_ar1_pair):
    null_bundle = fit_fixed_rho_multistart(
        groups, False, rho_e,
        seed_fits=(hom_pair["null"], unrestricted_ar1_pair["null"]),
    )
    null_fit = null_bundle["best"]
    alt_bundle = fit_fixed_rho_multistart(
        groups, True, rho_e,
        seed_fits=(hom_pair["alt"], unrestricted_ar1_pair["alt"], null_fit),
    )
    alt_fit = alt_bundle["best"]
    nested_ok = alt_fit["nll"] <= null_fit["nll"] + 2e-4
    if not nested_ok:
        # A final boundary-oriented start is mandatory before a point can be declared unusable.
        boundary_seed = dict(null_fit)
        boundary_seed["random_slope"] = False
        alt_bundle = fit_fixed_rho_multistart(
            groups, True, rho_e,
            seed_fits=(boundary_seed, hom_pair["alt"], unrestricted_ar1_pair["alt"]),
        )
        alt_fit = alt_bundle["best"]
        nested_ok = alt_fit["nll"] <= null_fit["nll"] + 2e-4

    src = source_summary(groups, alt_fit, f"fixed_rho_e_{rho_e:.6f}")
    by_optimizer = {}
    for optimizer_name in ("L-BFGS-B", "Powell"):
        candidates = [x for x in alt_bundle["attempts"] if x["optimizer"] == optimizer_name and fit_usable(x)]
        by_optimizer[optimizer_name] = min(x["nll"] for x in candidates) if candidates else np.nan
    optimizer_gap = abs(by_optimizer["L-BFGS-B"] - by_optimizer["Powell"])
    near_rank_one = bool(src["one_minus_rho_squared"] < 1e-4 or src["g_min_eigen_over_trace"] < 1e-4)
    return {
        "engine_version": ENGINE_VERSION, "rho_e": float(rho_e),
        "alt_nll_reml": alt_fit["nll"], "alt_loglik_reml": -alt_fit["nll"],
        "null_nll_reml": null_fit["nll"],
        "descriptive_RI_vs_RI_RS_LR": max(0.0, 2.0 * (null_fit["nll"] - alt_fit["nll"])) if nested_ok else np.nan,
        "nested_likelihood_ok": nested_ok, "optimizer_nll_gap": optimizer_gap,
        "residual_sd": alt_fit["residual"]["sigma"],
        **{k: v for k, v in src.items() if isinstance(v, (int, float, np.number))},
        "near_rank_one_G": near_rank_one,
        "boundary_status": "near-rank-one G" if near_rank_one else "interior G",
    }


def make_profile_grid(rho_hat):
    fine = np.linspace(
        max(-0.30, rho_hat - PROFILE_FINE_HALF_WIDTH),
        min(0.95, rho_hat + PROFILE_FINE_HALF_WIDTH),
        PROFILE_FINE_POINTS,
    )
    values = np.r_[PROFILE_GRID_COARSE, fine, 0.0, rho_hat]
    return np.array(sorted(set(np.round(values, 6))))


def run_profile(groups, label, hom_pair, unrestricted_ar1_pair, checkpoint_path):
    rho_hat = float(unrestricted_ar1_pair["alt"]["residual"]["rho"])
    grid = make_profile_grid(rho_hat)
    existing = pd.read_csv(checkpoint_path) if checkpoint_path.exists() else pd.DataFrame()
    if len(existing) and set(existing["engine_version"].astype(str)) != {ENGINE_VERSION}:
        raise RuntimeError(f"Checkpoint engine mismatch: {checkpoint_path}")
    completed = set(np.round(existing.get("rho_e", pd.Series(dtype=float)).astype(float), 6))
    rows = existing.to_dict("records")
    todo = [rho for rho in grid if round(float(rho), 6) not in completed]
    for j, rho_e in enumerate(tqdm(todo, desc=f"AR(1) profile: {label}", unit="rho"), 1):
        row = fixed_rho_profile_point(groups, float(rho_e), hom_pair, unrestricted_ar1_pair)
        row["analysis"] = label
        rows.append(row)
        if j % PROFILE_CHECKPOINT_EVERY == 0 or j == len(todo):
            pd.DataFrame(rows).sort_values("rho_e").to_csv(checkpoint_path, index=False)
    profile = pd.DataFrame(rows).query("rho_e in @grid").sort_values("rho_e").reset_index(drop=True)
    if len(profile) != len(grid):
        raise RuntimeError("Profile grid is incomplete")
    if not profile["nested_likelihood_ok"].astype(bool).all():
        raise RuntimeError("At least one fixed-rho point violates RI+RS <= RI likelihood ordering")
    min_nll = float(profile["alt_nll_reml"].min())
    profile["delta_nll"] = profile["alt_nll_reml"] - min_nll
    profile["profile_LR_from_min"] = 2.0 * profile["delta_nll"]
    profile["relative_likelihood"] = np.exp(-profile["delta_nll"])
    profile["inside_approx_95pct_profile_set"] = profile["profile_LR_from_min"] <= PROFILE_CI_CUTOFF
    return profile

PROFILE_DIR = RESULTS_DIR / "checkpoints_profile"
PROFILE_DIR.mkdir(exist_ok=True)

profile_0_9 = run_profile(
    groups_linear, "Days 0--9 (prespecified sample)", primary_pair, ar1_pair,
    PROFILE_DIR / "Days0_9_fixed_rho_profile.csv",
)

# rho_e=0 is exactly the homogeneous residual model; this is a hard implementation guard.
rho0 = profile_0_9.iloc[(profile_0_9["rho_e"] - 0.0).abs().argmin()]
if abs(rho0["rho_e"]) > 1e-9 or abs(rho0["alt_nll_reml"] - primary_fit["nll"]) > 2e-3:
    raise RuntimeError("rho_e=0 profile point failed to reproduce the homogeneous model")

inside = profile_0_9.query("inside_approx_95pct_profile_set")
profile_summary_rows = [{
    "analysis": "Days 0--9 (prespecified sample)",
    "unrestricted_rho_e": ar1_fit["residual"]["rho"],
    "unrestricted_AR1_nll": ar1_fit["nll"],
    "grid_min_rho_e": profile_0_9.loc[profile_0_9["alt_nll_reml"].idxmin(), "rho_e"],
    "grid_min_nll": profile_0_9["alt_nll_reml"].min(),
    "approx_profile_set_low": inside["rho_e"].min(),
    "approx_profile_set_high": inside["rho_e"].max(),
    "near_rank_one_inside_profile_set": bool(inside["near_rank_one_G"].astype(bool).any()),
    "R_direct_range_inside_profile_set": f"{inside['mean_R_direct'].min():.6g}--{inside['mean_R_direct'].max():.6g}",
    "Delta_fit_range_inside_profile_set": f"{inside['mean_Delta_fit'].min():.6g}--{inside['mean_Delta_fit'].max():.6g}",
    "analysis_role": "prespecified AR(1) diagnostic of frozen primary",
}]

if RUN_DAYS_2_9_POSTHOC:
    sleep_2_9 = sleep.loc[sleep["Days"] >= 2].copy()
    sleep_2_9["original_Days"] = sleep_2_9["Days"]
    sleep_2_9["Days"] = sleep_2_9["Days"] - 2  # Day-2 origin for interpretable G geometry.
    groups_2_9, _ = build_groups(sleep_2_9, "linear")
    hom_pair_2_9 = fit_pair(groups_2_9, "hom", reml=True)
    ar1_pair_2_9 = fit_pair(groups_2_9, "ar1", reml=True, seed_pair=hom_pair_2_9)
    profile_2_9 = run_profile(
        groups_2_9, "Days 2--9 (post hoc; Day 2 origin)", hom_pair_2_9, ar1_pair_2_9,
        PROFILE_DIR / "Days2_9_posthoc_fixed_rho_profile.csv",
    )
    inside_2_9 = profile_2_9.query("inside_approx_95pct_profile_set")
    profile_summary_rows.append({
        "analysis": "Days 2--9 (post hoc; Day 2 origin)",
        "unrestricted_rho_e": ar1_pair_2_9["alt"]["residual"]["rho"],
        "unrestricted_AR1_nll": ar1_pair_2_9["alt"]["nll"],
        "grid_min_rho_e": profile_2_9.loc[profile_2_9["alt_nll_reml"].idxmin(), "rho_e"],
        "grid_min_nll": profile_2_9["alt_nll_reml"].min(),
        "approx_profile_set_low": inside_2_9["rho_e"].min(),
        "approx_profile_set_high": inside_2_9["rho_e"].max(),
        "near_rank_one_inside_profile_set": bool(inside_2_9["near_rank_one_G"].astype(bool).any()),
        "R_direct_range_inside_profile_set": f"{inside_2_9['mean_R_direct'].min():.6g}--{inside_2_9['mean_R_direct'].max():.6g}",
        "Delta_fit_range_inside_profile_set": f"{inside_2_9['mean_Delta_fit'].min():.6g}--{inside_2_9['mean_Delta_fit'].max():.6g}",
        "analysis_role": "explicitly post hoc baseline-adaptation diagnostic; never replaces Days 0--9",
    })
else:
    profile_2_9 = pd.DataFrame()

profile_summary = pd.DataFrame(profile_summary_rows)
profile_0_9.to_csv(RESULTS_DIR / "04_Days0_9_AR1_profile_complete.csv", index=False)
if len(profile_2_9):
    profile_2_9.to_csv(RESULTS_DIR / "05_Days2_9_posthoc_AR1_profile_complete.csv", index=False)
profile_summary.to_csv(RESULTS_DIR / "06_AR1_profile_summary.csv", index=False)
display(profile_summary)

BLUE, PURPLE, AMBER, RED, GREY = "#2F6DAE", "#6C4AA1", "#D98C10", "#C43C39", "#5B6573"
fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)

for frame, label, style in [
    (profile_0_9, "Days 0--9", "-"),
    (profile_2_9, "Days 2--9 (post hoc)", "--"),
]:
    if frame is None or not len(frame):
        continue
    axes[0].plot(frame["rho_e"], frame["profile_LR_from_min"], style, lw=2.2, marker="o", ms=5.8, label=label)
axes[0].axhline(PROFILE_CI_CUTOFF, color=GREY, ls=":", lw=1.6)
axes[0].set(xlabel=r"Fixed residual AR(1) correlation $\rho_e$", ylabel=r"Profile $2\Delta$NLL", title="Likelihood profile")
axes[0].legend(frameon=False)

axes[1].plot(profile_0_9["rho_e"], profile_0_9["mean_R_direct"], color=BLUE, marker="o", ms=6.5, lw=2, label=r"$R_{\mathrm{direct}}$")
axes[1].plot(profile_0_9["rho_e"], profile_0_9["mean_P_full"], color=PURPLE, marker="s", ms=6.2, lw=2, label=r"$D_{\mathrm{slope}}^2=P_{\mathrm{full}}$")
axes[1].plot(profile_0_9["rho_e"], profile_0_9["mean_Delta_fit"], color=AMBER, marker="^", ms=6.7, lw=2, label=r"$\Delta_{\mathrm{fit}}$")
axes[1].set(xlabel=r"Fixed residual AR(1) correlation $\rho_e$", ylabel="Participant-mean component", title="Source decomposition", ylim=(-0.02, 1.02))
axes[1].legend(frameon=False)

axes[2].semilogy(profile_0_9["rho_e"], np.maximum(profile_0_9["one_minus_rho_squared"], 1e-12), color=RED, marker="o", ms=6.5, lw=2, label=r"$1-\rho_G^2$")
axes[2].semilogy(profile_0_9["rho_e"], np.maximum(profile_0_9["g_min_eigen_over_trace"], 1e-12), color=GREY, marker="s", ms=6.2, lw=2, label=r"$\lambda_{\min}(G)/\mathrm{tr}(G)$")
axes[2].axhline(1e-4, color=RED, ls=":", lw=1.5)
axes[2].set(xlabel=r"Fixed residual AR(1) correlation $\rho_e$", ylabel="Boundary diagnostic (log scale)", title="Random-effects covariance geometry")
axes[2].legend(frameon=False)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(FIGURE_DIR / "sleepstudy_AR1_profile_diagnostic.pdf", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "sleepstudy_AR1_profile_diagnostic.png", dpi=300, bbox_inches="tight")
plt.show()

full_profile_min = profile_0_9.loc[profile_0_9["alt_nll_reml"].idxmin()]
inside = profile_0_9.query("inside_approx_95pct_profile_set")
ordering_preserved = bool((inside["source_difference_d"] > 0).all())
boundary_is_profile_supported = bool(inside["near_rank_one_G"].astype(bool).any())

adjudication = {
    "frozen_primary_question1": "supported under homogeneous independent residuals",
    "frozen_primary_decomposition": f"{primary_summary['mean_P_full']:.6f} = {primary_summary['mean_R_direct']:.6f} + {primary_summary['mean_Delta_fit']:.6f}",
    "unrestricted_AR1_decomposition": f"{ar1_summary['mean_P_full']:.6f} = {ar1_summary['mean_R_direct']:.6f} + {ar1_summary['mean_Delta_fit']:.6f}",
    "direct_component_larger_through_approx_profile_set": ordering_preserved,
    "near_rank_one_G_occurs_inside_approx_profile_set": boundary_is_profile_supported,
    "interpretation": (
        "The AR(1) likelihood improvement is real, but source magnitudes co-move with rho_e and the fitted G enters a near-rank-one region; retain residual-model ranges and the boundary warning."
        if boundary_is_profile_supported else
        "The AR(1) profile does not place the supported rho_e region on a near-rank-one G branch; reassess whether the earlier boundary warning remains necessary."
    ),
    "Days2_9_role": "post hoc baseline-adaptation diagnostic only; it cannot replace the Days 0--9 primary analysis",
}
(RESULTS_DIR / "07_profile_adjudication.json").write_text(json.dumps(adjudication, indent=2), encoding="utf-8")
print(json.dumps(adjudication, indent=2))

required = [
    RESULTS_DIR / "03_unrestricted_homogeneous_and_AR1_summary.csv",
    RESULTS_DIR / "04_Days0_9_AR1_profile_complete.csv",
    RESULTS_DIR / "06_AR1_profile_summary.csv",
    RESULTS_DIR / "07_profile_adjudication.json",
    FIGURE_DIR / "sleepstudy_AR1_profile_diagnostic.pdf",
]
if RUN_DAYS_2_9_POSTHOC:
    required.append(RESULTS_DIR / "05_Days2_9_posthoc_AR1_profile_complete.csv")
assert all(path.exists() for path in required)
assert profile_0_9["nested_likelihood_ok"].astype(bool).all()
assert np.isfinite(profile_0_9[["alt_nll_reml", "mean_R_direct", "mean_P_full", "mean_Delta_fit"]]).all().all()
print(f"Complete outputs: {RESULTS_DIR.resolve()}")
