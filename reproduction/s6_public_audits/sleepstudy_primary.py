"""Reproduce the public sleepstudy primary random-slope audit."""

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
        description="Reproduce the public sleepstudy primary audit."
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

ENGINE_VERSION = "sleepstudy_primary_nested_safe_v1_2026-08-04"
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

primary_pair = fit_pair(groups_linear, "hom", reml=True)
primary_fit, primary_null = primary_pair["alt"], primary_pair["null"]
if not primary_pair["nested_ok"]:
    raise RuntimeError("Primary random-slope alternative has a worse objective than the RI null")


def attempt_row(component: str, fit: dict) -> dict:
    row = {
        "component": component, "optimizer": fit["optimizer"], "start_name": fit.get("start_name"),
        "success": fit["success"], "admissible": fit["admissible"], "nll": fit["nll"],
        "iterations": fit["nit"], "message": fit["message"],
    }
    if fit["G"] is not None:
        G = fit["G"]
        row.update({
            "var_intercept": G[0, 0], "cov_intercept_slope": G[0, 1], "var_slope": G[1, 1],
            "rho": G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]) if G[1, 1] > 0 else np.nan,
            "residual_sd": fit["residual"]["sigma"],
        })
    return row


attempts = pd.DataFrame(
    [attempt_row("RI null", x) for x in primary_pair["null_bundle"]["attempts"]]
    + [attempt_row("RI+RS alternative", x) for x in primary_pair["alt_bundle"]["attempts"]]
)
attempts.to_csv(RESULTS_DIR / "03_primary_optimizer_attempts.csv", index=False)
display(attempts.sort_values(["component", "nll"]))

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


primary_sources = participant_source_table(groups_linear, primary_fit)
primary_summary = source_summary(groups_linear, primary_fit, "primary_homogeneous_independent")
primary_sources.to_csv(RESULTS_DIR / "04_primary_participant_source_unrounded.csv", index=False)
pd.DataFrame([primary_summary]).to_csv(RESULTS_DIR / "05_primary_source_summary_unrounded.csv", index=False)

# Exact balanced-design OLS-slope anchor.
days = np.arange(10.0)
SXX = float(np.sum((days - days.mean()) ** 2))
ols_error_variance = primary_fit["residual"]["sigma"] ** 2 / SXX
ols_anchor = primary_fit["G"][1, 1] / (primary_fit["G"][1, 1] + ols_error_variance)
ols_anchor_check = pd.DataFrame([{
    "Sxx": SXX, "residual_variance": primary_fit["residual"]["sigma"] ** 2,
    "OLS_slope_error_variance": ols_error_variance, "OLS_special_case_R_direct": ols_anchor,
    "general_audit_R_direct": primary_summary["mean_R_direct"],
    "absolute_difference": abs(ols_anchor - primary_summary["mean_R_direct"]),
    "pass": abs(ols_anchor - primary_summary["mean_R_direct"]) <= 1e-11,
}])
ols_anchor_check.to_csv(RESULTS_DIR / "06_OLS_slope_anchor_check.csv", index=False)
display(pd.DataFrame([primary_summary]))
display(ols_anchor_check)

def best_attempt_by_optimizer(bundle: dict, optimizer_name: str) -> dict:
    candidates = [x for x in bundle["attempts"] if x["optimizer"] == optimizer_name and fit_usable(x)]
    if not candidates:
        raise RuntimeError(f"No usable {optimizer_name} attempt")
    return min(candidates, key=lambda x: x["nll"])


optimizer_fits = {
    method: best_attempt_by_optimizer(primary_pair["alt_bundle"], method)
    for method in OBSERVED_OPTIMIZERS
}
optimizer_sources = pd.DataFrame([
    source_summary(groups_linear, fit, method) for method, fit in optimizer_fits.items()
])
optimizer_nll_range_per_obs = (
    optimizer_sources["nll"].max() - optimizer_sources["nll"].min()
) / N_OBS
source_columns = ["mean_R_direct", "mean_P_full", "mean_Delta_fit", "mean_R_innovation"]
max_source_gap = max(
    float(optimizer_sources[col].max() - optimizer_sources[col].min()) for col in source_columns
)


def finite_difference_hessian(fun, x: np.ndarray, rel_step=2e-4) -> np.ndarray:
    x = np.asarray(x, float)
    n = len(x)
    h = rel_step * np.maximum(1.0, np.abs(x))
    H = np.zeros((n, n))
    f0 = float(fun(x))
    for i in range(n):
        ei = np.zeros(n); ei[i] = h[i]
        H[i, i] = (float(fun(x + ei)) - 2.0 * f0 + float(fun(x - ei))) / (h[i] ** 2)
        for j in range(i):
            ej = np.zeros(n); ej[j] = h[j]
            val = (
                float(fun(x + ei + ej)) - float(fun(x + ei - ej))
                - float(fun(x - ei + ej)) + float(fun(x - ei - ej))
            ) / (4.0 * h[i] * h[j])
            H[i, j] = H[j, i] = val
    return H


hessian = finite_difference_hessian(
    lambda th: profiled_likelihood(th, groups_linear, True, "hom", True, False), primary_fit["theta"]
)
hessian_eigenvalues = np.linalg.eigvalsh((hessian + hessian.T) / 2.0)
hessian_positive = bool(hessian_eigenvalues.min() > -1e-5)
hessian_table = pd.DataFrame({"eigenvalue": hessian_eigenvalues})
hessian_table.to_csv(RESULTS_DIR / "07_primary_variance_parameter_hessian.csv", index=False)

G = primary_fit["G"]
lme4_guard = pd.DataFrame([{
    "fixed_intercept": primary_fit["beta"][0], "fixed_Days": primary_fit["beta"][1],
    "sd_intercept": np.sqrt(G[0, 0]), "sd_Days": np.sqrt(G[1, 1]),
    "rho": G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]), "residual_sd": primary_fit["residual"]["sigma"],
    "fixed_intercept_match": abs(primary_fit["beta"][0] - 251.4051) <= 0.10,
    "fixed_Days_match": abs(primary_fit["beta"][1] - 10.4673) <= 0.05,
    "sd_intercept_match": abs(np.sqrt(G[0, 0]) - 24.7407) <= 0.20,
    "sd_Days_match": abs(np.sqrt(G[1, 1]) - 5.9221) <= 0.10,
    "rho_match": abs(G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]) - 0.0656) <= 0.03,
    "residual_sd_match": abs(primary_fit["residual"]["sigma"] - 25.5918) <= 0.10,
}])
lme4_guard["all_parameter_checks_pass"] = lme4_guard.filter(like="_match").all(axis=1)
if SMOKE_TEST:
    lme4_guard["all_parameter_checks_pass"] = True
lme4_guard.to_csv(RESULTS_DIR / "08_lme4_parameter_regression_guard.csv", index=False)

numerical_gate = pd.DataFrame([{
    "all_three_optimizers_usable": len(optimizer_sources) == 3,
    "optimizer_nll_range_per_observation": optimizer_nll_range_per_obs,
    "optimizer_nll_agreement_pass": optimizer_nll_range_per_obs <= OPT_NLL_GAP_PER_OBS_TOL,
    "max_source_metric_absolute_gap": max_source_gap,
    "optimizer_source_agreement_pass": max_source_gap <= OPT_SOURCE_ABS_GAP_TOL,
    "random_covariance_positive_definite": np.linalg.eigvalsh(G).min() > 0,
    "hessian_positive_semidefinite_with_tolerance": hessian_positive,
    "alternative_nll_le_null_nll": primary_pair["nested_ok"],
    "OLS_anchor_identity_pass": bool(ols_anchor_check["pass"].iloc[0]),
    "lme4_parameter_guard_pass": bool(lme4_guard["all_parameter_checks_pass"].iloc[0]),
}])
pass_columns = [c for c in numerical_gate if c.endswith("_pass") or c.startswith("all_three") or c.startswith("random_covariance") or c.startswith("hessian") or c.startswith("alternative")]
numerical_gate["gate_pass"] = numerical_gate[pass_columns].all(axis=1)
optimizer_sources.to_csv(RESULTS_DIR / "09_optimizer_source_agreement.csv", index=False)
numerical_gate.to_csv(RESULTS_DIR / "10_numerical_validity_gate.csv", index=False)
display(optimizer_sources)
display(lme4_guard)
display(numerical_gate)
if not bool(numerical_gate["gate_pass"].iloc[0]):
    raise RuntimeError("Primary numerical validity gate failed; inspect exported optimizer/Hessian tables")

def deterministic_rng(stream: int, replicate: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, stream, replicate]))


def simulate_under_null(template: Sequence[GroupData], null_fit: dict, replicate: int) -> list[GroupData]:
    rng = deterministic_rng(101, replicate)
    beta = null_fit["beta"]
    q0 = float(null_fit["G"][0, 0])
    ys = []
    for group in template:
        b0 = rng.normal(0.0, np.sqrt(q0))
        R = residual_covariance(group, null_fit["residual"], "hom")
        eps = rng.multivariate_normal(np.zeros(len(group.y)), R)
        ys.append(group.X @ beta + b0 + eps)
    return groups_with_new_y(template, ys, prefix=f"null_{replicate}")


def refit_boundary_replicate(sim_groups, replicate: int) -> dict:
    pair = fit_pair(
        sim_groups, "hom", reml=True, seed_pair=primary_pair,
        optimizers=BOOTSTRAP_OPTIMIZERS,
    )
    null_fit, alt_fit = pair["null"], pair["alt"]
    lr_raw = 2.0 * (null_fit["nll"] - alt_fit["nll"])
    rescue_triggered = lr_raw > LR_RESCUE_THRESHOLD or not pair["nested_ok"]
    if rescue_triggered:
        pair = fit_pair(
            sim_groups, "hom", reml=True, seed_pair=pair,
            optimizers=OBSERVED_OPTIMIZERS,
        )
        null_fit, alt_fit = pair["null"], pair["alt"]
        lr_raw = 2.0 * (null_fit["nll"] - alt_fit["nll"])
    usable = bool(fit_usable(null_fit) and fit_usable(alt_fit) and pair["nested_ok"])
    return {
        "engine_version": ENGINE_VERSION, "replicate": replicate, "usable": usable,
        "LR": max(0.0, lr_raw) if usable else np.nan, "nested_likelihood_ok": pair["nested_ok"],
        "rescue_triggered": rescue_triggered, "null_nll": null_fit["nll"], "alt_nll": alt_fit["nll"],
        "null_optimizer": null_fit["optimizer"], "alt_optimizer": alt_fit["optimizer"],
        "null_start": null_fit.get("start_name"), "alt_start": alt_fit.get("start_name"),
    }


BOUNDARY_PATH = CHECKPOINT_DIR / "primary_boundary_bootstrap.csv"


def run_boundary_bootstrap_to(target_b: int) -> pd.DataFrame:
    existing = pd.read_csv(BOUNDARY_PATH) if BOUNDARY_PATH.exists() else pd.DataFrame()
    if len(existing) and set(existing["engine_version"].astype(str)) != {ENGINE_VERSION}:
        raise RuntimeError("Boundary checkpoint engine mismatch; move the old checkpoint before rerunning")
    done = set(existing["replicate"].astype(int)) if len(existing) else set()
    rows = existing.to_dict("records")
    todo = [b for b in range(1, target_b + 1) if b not in done]
    for j, b in enumerate(tqdm(todo, desc=f"Boundary bootstrap to B={target_b}", unit="rep", total=len(todo)), 1):
        sim_groups = simulate_under_null(groups_linear, primary_null, b)
        rows.append(refit_boundary_replicate(sim_groups, b))
        if j % CHECKPOINT_EVERY == 0 or j == len(todo):
            pd.DataFrame(rows).sort_values("replicate").to_csv(BOUNDARY_PATH, index=False)
    return pd.DataFrame(rows).sort_values("replicate").query("replicate <= @target_b").reset_index(drop=True)


def exact_binomial_interval(k: int, n: int, alpha=0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def summarize_boundary(reps: pd.DataFrame) -> dict:
    usable = reps.loc[reps["usable"].astype(bool) & reps["nested_likelihood_ok"].astype(bool) & reps["LR"].notna()]
    observed_lr = max(0.0, 2.0 * (primary_null["nll"] - primary_fit["nll"]))
    k, n = int((usable["LR"] >= observed_lr - 1e-12).sum()), len(usable)
    lo, hi = exact_binomial_interval(k, n, ALPHA)
    p_plus_one = (k + 1) / (n + 1)
    return {
        "requested_replicates": len(reps), "usable_replicates": n, "usable_rate": n / len(reps),
        "nested_violations": int((~reps["nested_likelihood_ok"].astype(bool)).sum()),
        "rescue_refits": int(reps["rescue_triggered"].astype(bool).sum()),
        "observed_LR": observed_lr, "exceedances": k, "p_plus_one": p_plus_one,
        "mc_interval_low": lo, "mc_interval_high": hi, "mc_interval_crosses_0.05": lo <= ALPHA <= hi,
        "bootstrap_usability_gate_pass": n / len(reps) >= MIN_USABLE_RATE,
        "heterogeneity_status": (
            "borderline" if lo <= ALPHA <= hi else
            "supported" if p_plus_one < ALPHA else "unsupported by the present data"
        ),
    }


if RUN_BOUNDARY_BOOTSTRAP:
    boundary_reps = run_boundary_bootstrap_to(PROVISIONAL_NULL_B)
    boundary_summary = summarize_boundary(boundary_reps)
    if boundary_summary["mc_interval_crosses_0.05"] and FINAL_NULL_B > PROVISIONAL_NULL_B:
        print(f"Monte Carlo interval crosses .05; extending to B={FINAL_NULL_B}.")
        boundary_reps = run_boundary_bootstrap_to(FINAL_NULL_B)
        boundary_summary = summarize_boundary(boundary_reps)
else:
    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError("RUN_BOUNDARY_BOOTSTRAP=False but the boundary checkpoint is absent")
    boundary_reps = pd.read_csv(BOUNDARY_PATH)
    boundary_summary = summarize_boundary(boundary_reps)

boundary_reps.to_csv(RESULTS_DIR / "11_boundary_bootstrap_replicates.csv", index=False)
pd.DataFrame([boundary_summary]).to_csv(RESULTS_DIR / "12_boundary_bootstrap_summary.csv", index=False)
display(pd.DataFrame([boundary_summary]))

question_2_3 = pd.DataFrame([{
    "target": "participant Days slope (ms/day), Day-0 random-intercept origin",
    "slope_variance": primary_fit["G"][1, 1],
    "R_direct": primary_summary["mean_R_direct"],
    "P_full": primary_summary["mean_P_full"],
    "Delta_fit": primary_summary["mean_Delta_fit"],
    "additivity_error": primary_summary["mean_P_full"] - primary_summary["mean_R_direct"] - primary_summary["mean_Delta_fit"],
    "R_innovation": primary_summary["mean_R_innovation"],
    "rho_intercept_slope_at_day0": primary_summary["rho_intercept_slope_at_day0"],
    "source_difference_d": primary_summary["source_difference_d"],
    "direct_share": primary_summary["direct_share"],
}])
question_2_3.to_csv(RESULTS_DIR / "13_questions_2_and_3_unrounded.csv", index=False)
display(question_2_3)

linear_ml_pair = fit_pair(groups_linear, "hom", reml=False, seed_pair=primary_pair)
quadratic_ml_pair = fit_pair(groups_quadratic, "hom", reml=False, seed_pair=linear_ml_pair)
quadratic_reml_pair = fit_pair(groups_quadratic, "hom", reml=True, seed_pair=quadratic_ml_pair)
quadratic_fit = quadratic_reml_pair["alt"]

if quadratic_ml_pair["alt"]["nll"] > linear_ml_pair["alt"]["nll"] + NESTED_NLL_TOL:
    raise RuntimeError("Quadratic ML model has a worse likelihood than its nested linear fixed-mean model")

quadratic_lr = max(0.0, 2.0 * (linear_ml_pair["alt"]["nll"] - quadratic_ml_pair["alt"]["nll"]))
quadratic_p = float(stats.chi2.sf(quadratic_lr, 1))
quadratic_summary = source_summary(groups_quadratic, quadratic_fit, "quadratic_fixed_mean_REML")
quadratic_source_sign_preserved = (
    np.sign(quadratic_summary["source_difference_d"]) == np.sign(primary_summary["source_difference_d"])
)
quadratic_table = pd.DataFrame([{
    "linear_ML_nll": linear_ml_pair["alt"]["nll"],
    "quadratic_ML_nll": quadratic_ml_pair["alt"]["nll"],
    "quadratic_fixed_effect_LR_df1": quadratic_lr, "quadratic_fixed_effect_p": quadratic_p,
    "quadratic_coefficient_on_Days_minus_4p5_squared": quadratic_ml_pair["alt"]["beta"][2],
    **{f"quadratic_REML_{k}": v for k, v in quadratic_summary.items() if isinstance(v, (int, float, np.number))},
    "source_ordering_preserved": quadratic_source_sign_preserved,
    "numerically_valid": quadratic_reml_pair["nested_ok"] and fit_usable(quadratic_fit),
}])
quadratic_table.to_csv(RESULTS_DIR / "14_quadratic_fixed_trajectory_sensitivity.csv", index=False)
display(quadratic_table)

ar1_pair = fit_pair(groups_linear, "ar1", reml=True, seed_pair=primary_pair)
ar1_fit = ar1_pair["alt"]
if ar1_fit["nll"] > primary_fit["nll"] + NESTED_NLL_TOL:
    raise RuntimeError("AR(1) sensitivity violates nesting: optimized NLL is worse than homogeneous NLL")

ar1_summary = source_summary(groups_linear, ar1_fit, "AR1_residual_REML")
ar1_source_sign_preserved = np.sign(ar1_summary["source_difference_d"]) == np.sign(primary_summary["source_difference_d"])
ar1_optimizer_fits = {
    method: best_attempt_by_optimizer(ar1_pair["alt_bundle"], method)
    for method in OBSERVED_OPTIMIZERS
}
ar1_optimizer_sources = pd.DataFrame([
    source_summary(groups_linear, fit, method) for method, fit in ar1_optimizer_fits.items()
])
ar1_optimizer_nll_range_per_obs = (
    ar1_optimizer_sources["nll"].max() - ar1_optimizer_sources["nll"].min()
) / N_OBS
ar1_optimizer_source_gap = max(
    float(ar1_optimizer_sources[col].max() - ar1_optimizer_sources[col].min())
    for col in source_columns
)
ar1_optimizer_agreement = bool(
    ar1_optimizer_nll_range_per_obs <= OPT_NLL_GAP_PER_OBS_TOL
    and ar1_optimizer_source_gap <= OPT_SOURCE_ABS_GAP_TOL
)
ar1_random_covariance_near_singular = bool(
    ar1_summary["one_minus_rho_squared"] < 1e-4
    or ar1_summary["g_min_eigen_over_trace"] < 1e-4
)
ar1_valid_nested_fit = bool(
    ar1_pair["nested_ok"] and fit_usable(ar1_fit)
    and abs(ar1_fit["residual"]["rho"]) < AR1_INTERIOR_LIMIT
    and ar1_fit["nll"] <= primary_fit["nll"] + NESTED_NLL_TOL
    and ar1_optimizer_agreement
)
ar1_table = pd.DataFrame([{
    "homogeneous_alt_nll": primary_fit["nll"], "AR1_alt_nll": ar1_fit["nll"],
    "AR1_residual_rho": ar1_fit["residual"]["rho"],
    "AR1_null_nll": ar1_pair["null"]["nll"],
    "AR1_descriptive_heterogeneity_LR": max(0.0, 2.0 * (ar1_pair["null"]["nll"] - ar1_fit["nll"])),
    **{f"AR1_{k}": v for k, v in ar1_summary.items() if isinstance(v, (int, float, np.number))},
    "optimizer_nll_range_per_observation": ar1_optimizer_nll_range_per_obs,
    "optimizer_max_source_metric_gap": ar1_optimizer_source_gap,
    "optimizer_agreement_pass": ar1_optimizer_agreement,
    "random_covariance_near_singular_warning": ar1_random_covariance_near_singular,
    "source_ordering_preserved": ar1_source_sign_preserved,
    "valid_nested_fit": ar1_valid_nested_fit,
    "boundary_bootstrap_claim_made": False,
}])
ar1_attempts = pd.DataFrame(
    [attempt_row("AR1 RI null", x) for x in ar1_pair["null_bundle"]["attempts"]]
    + [attempt_row("AR1 RI+RS alternative", x) for x in ar1_pair["alt_bundle"]["attempts"]]
)
ar1_attempts.to_csv(RESULTS_DIR / "15a_AR1_optimizer_attempts.csv", index=False)
ar1_optimizer_sources.to_csv(RESULTS_DIR / "15b_AR1_optimizer_source_agreement.csv", index=False)
ar1_table.to_csv(RESULTS_DIR / "15_AR1_residual_sensitivity.csv", index=False)
display(ar1_optimizer_sources)
display(ar1_table)

def conditional_effects_and_residuals(groups: Sequence[GroupData], fit: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    G = fit["G"]
    effect_rows, residual_rows = [], []
    for group in groups:
        R = residual_covariance(group, fit["residual"], fit["residual_model"])
        V = group.Z @ G @ group.Z.T + R
        resid_marginal = group.y - group.X @ fit["beta"]
        b_hat = G @ group.Z.T @ np.linalg.solve(V, resid_marginal)
        cond = resid_marginal - group.Z @ b_hat
        effect_rows.append({"subject": group.subject, "b0_hat": b_hat[0], "bDays_hat": b_hat[1]})
        for day, y, marginal, conditional in zip(group.days, group.y, resid_marginal, cond):
            residual_rows.append({
                "subject": group.subject, "Days": day, "Reaction": y,
                "marginal_residual": marginal, "conditional_residual": conditional,
                "standardized_conditional_residual": conditional / fit["residual"]["sigma"],
            })
    return pd.DataFrame(effect_rows), pd.DataFrame(residual_rows)


eb_effects, residuals = conditional_effects_and_residuals(groups_linear, primary_fit)
residual_by_day = residuals.groupby("Days").agg(
    n=("conditional_residual", "size"), mean_conditional=("conditional_residual", "mean"),
    sd_conditional=("conditional_residual", "std"), mean_abs_standardized=("standardized_conditional_residual", lambda x: np.mean(np.abs(x))),
).reset_index()
lag1_rows = []
for subject, d in residuals.groupby("subject"):
    d = d.sort_values("Days")
    lag1_rows.append({"subject": subject, "conditional_residual_lag1_correlation": d["conditional_residual"].autocorr(1)})
lag1 = pd.DataFrame(lag1_rows)

observed_means = sleep.groupby("Days")["Reaction"].agg(["mean", "std", "count"]).reset_index()
observed_means["fitted_linear_mean"] = primary_fit["beta"][0] + primary_fit["beta"][1] * observed_means["Days"]
observed_means["observed_minus_fitted"] = observed_means["mean"] - observed_means["fitted_linear_mean"]

eb_effects.to_csv(RESULTS_DIR / "16_EB_effects_diagnostic_only.csv", index=False)
residuals.to_csv(RESULTS_DIR / "17_residual_diagnostics_long.csv", index=False)
residual_by_day.to_csv(RESULTS_DIR / "18_residual_diagnostics_by_day.csv", index=False)
lag1.to_csv(RESULTS_DIR / "19_residual_lag1_by_subject.csv", index=False)
observed_means.to_csv(RESULTS_DIR / "20_observed_vs_fitted_day_means.csv", index=False)
display(residual_by_day)
display(lag1.describe())
display(observed_means)

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
palette = plt.cm.viridis(np.linspace(0.08, 0.92, sleep.Subject.nunique()))
for color, (subject, d) in zip(palette, sleep.groupby("Subject")):
    axes[0, 0].plot(d["Days"], d["Reaction"], color=color, alpha=.75, lw=1.4)
axes[0, 0].plot(observed_means["Days"], observed_means["fitted_linear_mean"], color="black", lw=3, label="Fitted population mean")
axes[0, 0].set(xlabel="Day", ylabel="Reaction time (ms)", title="Observed participant trajectories")
axes[0, 0].legend(frameon=False, fontsize=9)

axes[0, 1].scatter(observed_means["Days"], observed_means["observed_minus_fitted"], s=75, color="#2C7FB8")
axes[0, 1].axhline(0, color="0.3", lw=1)
axes[0, 1].set(xlabel="Day", ylabel="Observed minus fitted mean (ms)", title="Mean-trajectory discrepancy")

fitted_conditional = residuals["Reaction"] - residuals["conditional_residual"]
axes[1, 0].scatter(fitted_conditional, residuals["standardized_conditional_residual"], s=34, alpha=.72, color="#2C7FB8")
axes[1, 0].axhline(0, color="0.3", lw=1)
axes[1, 0].set(xlabel="Conditional fitted value (ms)", ylabel="Standardized conditional residual", title="Residuals versus fitted values")

qq = stats.probplot(residuals["standardized_conditional_residual"], dist="norm")
axes[1, 1].scatter(qq[0][0], qq[0][1], s=30, alpha=.72, color="#2C7FB8")
axes[1, 1].plot(qq[0][0], qq[1][1] + qq[1][0] * qq[0][0], color="0.25", lw=1.5)
axes[1, 1].set(xlabel="Normal quantile", ylabel="Standardized conditional residual", title="Residual Q-Q diagnostic")

fig.tight_layout()
fig.savefig(FIGURE_DIR / "sleepstudy_model_diagnostics.pdf", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "sleepstudy_model_diagnostics.png", dpi=300, bbox_inches="tight")
plt.show()

LOO_PATH = CHECKPOINT_DIR / "leave_one_participant_out.csv"


def groups_without_subject(groups: Sequence[GroupData], subject: str) -> list[GroupData]:
    return [g for g in groups if g.subject != subject]


def run_leave_one_out() -> pd.DataFrame:
    existing = pd.read_csv(LOO_PATH) if LOO_PATH.exists() else pd.DataFrame()
    if len(existing) and set(existing["engine_version"].astype(str)) != {ENGINE_VERSION}:
        raise RuntimeError("LOO checkpoint engine mismatch")
    done = set(existing["omitted_subject"].astype(str)) if len(existing) else set()
    rows = existing.to_dict("records")
    subjects = [g.subject for g in groups_linear]
    for j, subject in enumerate(tqdm([s for s in subjects if s not in done], desc="Leave-one-participant-out", unit="fit"), 1):
        subset = groups_without_subject(groups_linear, subject)
        try:
            bundle = fit_multistart(
                subset, True, "hom",
                extra_starts=[("mapped_full", mapped_start_from_fit(primary_fit, True, "hom"))],
                reml=True, optimizers=BOOTSTRAP_OPTIMIZERS,
            )
            fit = bundle["best"]
            summary = source_summary(subset, fit, f"omit_{subject}")
            rows.append({"engine_version": ENGINE_VERSION, "omitted_subject": subject, "usable": True, **summary})
        except Exception as exc:
            rows.append({"engine_version": ENGINE_VERSION, "omitted_subject": subject, "usable": False, "error": repr(exc)})
        pd.DataFrame(rows).to_csv(LOO_PATH, index=False)
    return pd.DataFrame(rows).sort_values("omitted_subject").reset_index(drop=True)


if RUN_LEAVE_ONE_OUT:
    loo = run_leave_one_out()
else:
    if not LOO_PATH.exists():
        raise FileNotFoundError("RUN_LEAVE_ONE_OUT=False but checkpoint is absent")
    loo = pd.read_csv(LOO_PATH)

for metric in ["mean_R_direct", "mean_P_full", "mean_Delta_fit", "source_difference_d", "rho_intercept_slope_at_day0"]:
    loo[f"change_in_{metric}"] = loo[metric] - primary_summary[metric]
loo["max_absolute_source_change"] = loo[[
    "change_in_mean_R_direct", "change_in_mean_P_full", "change_in_mean_Delta_fit"
]].abs().max(axis=1)
loo = loo.sort_values("max_absolute_source_change", ascending=False).reset_index(drop=True)
loo.to_csv(RESULTS_DIR / "21_leave_one_participant_out_diagnostic.csv", index=False)
display(loo)

PARTICIPANT_BOOT_PATH = CHECKPOINT_DIR / "participant_cluster_bootstrap.csv"


def participant_bootstrap_groups(replicate: int) -> list[GroupData]:
    rng = deterministic_rng(201, replicate)
    draw = rng.integers(0, len(groups_linear), size=len(groups_linear))
    out = []
    for new_index, source_index in enumerate(draw):
        g = groups_linear[source_index]
        out.append(GroupData(f"boot{replicate:04d}_{new_index:02d}_{g.subject}", g.y.copy(), g.X, g.Z, g.days))
    return out


def refit_participant_bootstrap(replicate: int) -> dict:
    boot_groups = participant_bootstrap_groups(replicate)
    try:
        bundle = fit_multistart(
            boot_groups, True, "hom",
            extra_starts=[("mapped_observed", mapped_start_from_fit(primary_fit, True, "hom"))],
            reml=True, optimizers=BOOTSTRAP_OPTIMIZERS,
        )
        fit = bundle["best"]
        summary = source_summary(boot_groups, fit, f"participant_boot_{replicate}")
        return {"engine_version": ENGINE_VERSION, "replicate": replicate, "usable": True, **summary}
    except Exception as exc:
        return {"engine_version": ENGINE_VERSION, "replicate": replicate, "usable": False, "error": repr(exc)}


def run_participant_bootstrap(target_b: int) -> pd.DataFrame:
    existing = pd.read_csv(PARTICIPANT_BOOT_PATH) if PARTICIPANT_BOOT_PATH.exists() else pd.DataFrame()
    if len(existing) and set(existing["engine_version"].astype(str)) != {ENGINE_VERSION}:
        raise RuntimeError("Participant-bootstrap checkpoint engine mismatch")
    done = set(existing["replicate"].astype(int)) if len(existing) else set()
    rows = existing.to_dict("records")
    todo = [b for b in range(1, target_b + 1) if b not in done]
    for j, b in enumerate(tqdm(todo, desc=f"Participant bootstrap to B={target_b}", unit="rep", total=len(todo)), 1):
        rows.append(refit_participant_bootstrap(b))
        if j % CHECKPOINT_EVERY == 0 or j == len(todo):
            pd.DataFrame(rows).sort_values("replicate").to_csv(PARTICIPANT_BOOT_PATH, index=False)
    return pd.DataFrame(rows).sort_values("replicate").query("replicate <= @target_b").reset_index(drop=True)


if RUN_PARTICIPANT_BOOTSTRAP:
    participant_boot = run_participant_bootstrap(PARTICIPANT_BOOT_B)
else:
    if not PARTICIPANT_BOOT_PATH.exists():
        raise FileNotFoundError("RUN_PARTICIPANT_BOOTSTRAP=False but checkpoint is absent")
    participant_boot = pd.read_csv(PARTICIPANT_BOOT_PATH).query("replicate <= @PARTICIPANT_BOOT_B")

usable_boot = participant_boot.loc[participant_boot["usable"].astype(bool)].copy()
usable_rate = len(usable_boot) / len(participant_boot)
d_low, d_high = np.quantile(usable_boot["source_difference_d"], [0.025, 0.975]) if len(usable_boot) else (np.nan, np.nan)
full_d = primary_summary["source_difference_d"]
if full_d > SOURCE_SIGN_EPS:
    source_label = "directly informed"
    interval_preserves_sign = d_low > 0
elif full_d < -SOURCE_SIGN_EPS:
    source_label = "model-assisted"
    interval_preserves_sign = d_high < 0
else:
    source_label = "source-indeterminate"
    interval_preserves_sign = False

participant_stability = pd.DataFrame([{
    "requested_replicates": len(participant_boot), "usable_replicates": len(usable_boot),
    "usable_rate": usable_rate, "minimum_usable_rate": MIN_USABLE_RATE,
    "full_sample_source_difference_d": full_d, "bootstrap_d_q025": d_low,
    "bootstrap_d_median": usable_boot["source_difference_d"].median(), "bootstrap_d_q975": d_high,
    "full_sample_source_label": source_label,
    "interval_preserves_full_sample_sign": interval_preserves_sign,
    "usability_gate_pass": usable_rate >= MIN_USABLE_RATE,
    "source_stability_gate_pass": usable_rate >= MIN_USABLE_RATE and interval_preserves_sign,
}])
participant_boot.to_csv(RESULTS_DIR / "22_participant_cluster_bootstrap_replicates.csv", index=False)
participant_stability.to_csv(RESULTS_DIR / "23_participant_source_stability_gate.csv", index=False)
display(participant_stability)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
axes[0].hist(usable_boot["source_difference_d"], bins=25, color="#2C7FB8", alpha=.86, edgecolor="white")
axes[0].axvline(0, color="#C0392B", lw=1.5, ls="--")
axes[0].axvline(full_d, color="black", lw=2, label="Full sample")
axes[0].set(xlabel=r"$\overline{R}_{direct}-\overline{\Delta}_{fit}$", ylabel="Bootstrap fits", title="Participant-cluster stability")
axes[0].legend(frameon=False)

loo_plot = loo.sort_values("source_difference_d")
axes[1].scatter(loo_plot["source_difference_d"], np.arange(len(loo_plot)), s=62, color="#2C7FB8")
axes[1].axvline(full_d, color="black", lw=2, label="Full sample")
axes[1].axvline(0, color="#C0392B", lw=1.5, ls="--")
axes[1].set(yticks=np.arange(len(loo_plot)), yticklabels=loo_plot["omitted_subject"], xlabel=r"$\overline{R}_{direct}-\overline{\Delta}_{fit}$", ylabel="Omitted subject", title="Leave-one-out diagnostic")
axes[1].legend(frameon=False)

fig.tight_layout()
fig.savefig(FIGURE_DIR / "sleepstudy_participant_stability.pdf", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "sleepstudy_participant_stability.png", dpi=300, bbox_inches="tight")
plt.show()

heterogeneity_supported = boundary_summary["heterogeneity_status"] == "supported"
boundary_usable = boundary_summary["bootstrap_usability_gate_pass"]
numerical_valid = bool(numerical_gate["gate_pass"].iloc[0])
participant_stable = bool(participant_stability["source_stability_gate_pass"].iloc[0])
quadratic_valid_stable = bool(
    quadratic_table["numerically_valid"].iloc[0] and quadratic_table["source_ordering_preserved"].iloc[0]
)
ar1_valid_stable = bool(ar1_table["valid_nested_fit"].iloc[0] and ar1_table["source_ordering_preserved"].iloc[0])

all_mandatory_gates = bool(
    heterogeneity_supported and boundary_usable and numerical_valid and participant_stable
    and quadratic_valid_stable and ar1_valid_stable
)
action = "PROCEED—INDIVIDUAL-SLOPE CLAIM" if all_mandatory_gates else "STOP—INDIVIDUAL-SLOPE CLAIM"

questions = pd.DataFrame([
    {
        "question": "Q1. Do rates vary?", "primary_result": boundary_summary["heterogeneity_status"],
        "key_quantity": f"LR={boundary_summary['observed_LR']:.8g}; p={boundary_summary['p_plus_one']:.8g}",
        "gate_pass": heterogeneity_supported and boundary_usable,
    },
    {
        "question": "Q2. What did observed change measure?", "primary_result": "direct change information",
        "key_quantity": f"R_direct={primary_summary['mean_R_direct']:.8g}", "gate_pass": True,
    },
    {
        "question": "Q3. What did the complete model predict, and from where?",
        "primary_result": source_label,
        "key_quantity": (
            f"P_full={primary_summary['mean_P_full']:.8g}="
            f"{primary_summary['mean_R_direct']:.8g}+{primary_summary['mean_Delta_fit']:.8g}"
        ),
        "gate_pass": True,
    },
    {
        "question": "Q4. Is the attribution stable?",
        "primary_result": (
            "stable source ordering; AR(1) covariance-boundary warning"
            if participant_stable and ar1_random_covariance_near_singular else
            "stable" if participant_stable else "unstable"
        ),
        "key_quantity": f"95% cluster-bootstrap interval for d=[{d_low:.8g}, {d_high:.8g}]",
        "gate_pass": numerical_valid and participant_stable and quadratic_valid_stable and ar1_valid_stable,
    },
])

final_gate = pd.DataFrame([{
    "heterogeneity_supported": heterogeneity_supported,
    "boundary_bootstrap_usability_pass": boundary_usable,
    "numerical_validity_pass": numerical_valid,
    "participant_source_stability_pass": participant_stable,
    "quadratic_fixed_trajectory_valid_and_source_order_preserved": quadratic_valid_stable,
    "AR1_valid_and_source_order_preserved": ar1_valid_stable,
    "all_mandatory_gates_pass": all_mandatory_gates,
    "scoped_action": action,
    "case_retained_regardless_of_result": True,
}])
questions.to_csv(RESULTS_DIR / "24_questions_1_to_4.csv", index=False)
final_gate.to_csv(RESULTS_DIR / "25_final_scoped_action.csv", index=False)
display(questions)
display(final_gate)

compact_output = f"""sleepstudy slope-source audit (unrounded values are in the CSV exports)
Target: participant-specific linear Reaction-time deviation per Day, Days 0--9

Q1 heterogeneity: {boundary_summary['heterogeneity_status']}
  LR = {boundary_summary['observed_LR']:.6f}
  boundary-bootstrap p = {boundary_summary['p_plus_one']:.6f}
  usable = {boundary_summary['usable_replicates']}/{boundary_summary['requested_replicates']}

Q2 direct information:
  R_direct = {primary_summary['mean_R_direct']:.6f}
  OLS-slope special-case identity verified = {bool(ols_anchor_check['pass'].iloc[0])}

Q3 complete-model prediction and source:
  P_full = {primary_summary['mean_P_full']:.6f} = {primary_summary['mean_R_direct']:.6f} + {primary_summary['mean_Delta_fit']:.6f}
  R_innovation = {primary_summary['mean_R_innovation']:.6f}
  intercept--slope rho at Day 0 = {primary_summary['rho_intercept_slope_at_day0']:.6f}
  source description = {source_label}

Q4 stability:
  participant-bootstrap usable = {len(usable_boot)}/{len(participant_boot)}
  95% interval for R_direct - Delta_fit = [{d_low:.6f}, {d_high:.6f}]
  numerical gate = {'pass' if numerical_valid else 'fail'}
  quadratic fixed-trajectory source ordering preserved = {quadratic_source_sign_preserved}
  AR(1) source ordering preserved = {ar1_source_sign_preserved}
  AR(1) random-covariance near-singular warning = {ar1_random_covariance_near_singular}

ACTION: {action}
The action is scoped only to treating individual slopes as established person-level change traits.
The public case is retained regardless of this result.
"""
print(compact_output)
(RESULTS_DIR / "26_compact_expected_output.txt").write_text(compact_output, encoding="utf-8")

run_manifest = {
    "engine_version": ENGINE_VERSION,
    "smoke_test": SMOKE_TEST,
    "data_provenance": provenance,
    "primary_summary": primary_summary,
    "boundary_summary": boundary_summary,
    "participant_stability": participant_stability.iloc[0].to_dict(),
    "final_gate": final_gate.iloc[0].to_dict(),
    "versions": {
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
    },
}
(RESULTS_DIR / "27_run_manifest.json").write_text(
    json.dumps(run_manifest, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)),
    encoding="utf-8",
)

completion = pd.DataFrame([
    {"check": "canonical 18 x 10 schedule", "pass": all(checks.values())},
    {"check": "primary numerical gate", "pass": numerical_valid},
    {"check": "boundary requested denominator completed", "pass": len(boundary_reps) in {PROVISIONAL_NULL_B, FINAL_NULL_B}},
    {"check": "boundary usability at least 95%", "pass": boundary_usable},
    {"check": "participant bootstrap requested denominator completed", "pass": len(participant_boot) == PARTICIPANT_BOOT_B},
    {"check": "participant bootstrap usability at least 95%", "pass": usable_rate >= MIN_USABLE_RATE},
    {"check": "source additivity identity", "pass": abs(question_2_3["additivity_error"].iloc[0]) <= 1e-11},
    {"check": "quadratic sensitivity numerically valid", "pass": bool(quadratic_table["numerically_valid"].iloc[0])},
    {"check": "AR1 sensitivity is a valid nested optimum", "pass": ar1_valid_nested_fit},
    {"check": "case retained regardless of result", "pass": bool(final_gate["case_retained_regardless_of_result"].iloc[0])},
    {"check": "not a smoke-test publication freeze", "pass": not SMOKE_TEST},
])
completion.to_csv(RESULTS_DIR / "28_completion_checklist.csv", index=False)
display(completion)
if not SMOKE_TEST and not completion["pass"].all():
    raise RuntimeError("Publication completion checks failed; inspect 28_completion_checklist.csv")
