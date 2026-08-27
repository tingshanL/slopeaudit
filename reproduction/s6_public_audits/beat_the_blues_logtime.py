"""Reproduce the public Beat the Blues log-time estimand sensitivity analysis."""

from __future__ import annotations

import argparse

import hashlib
import io
import json
import math
import os
import platform
import sys
import tarfile
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import scipy
from scipy import linalg, optimize, stats
from tqdm.auto import tqdm

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)
np.set_printoptions(precision=7, suppress=True)

print({
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "pyreadr": getattr(pyreadr, "__version__", "unknown"),
})

def display(obj):
    """Notebook-compatible display shim for ordinary script execution."""
    print(obj)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the public Beat the Blues log-time sensitivity analysis."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which to write all analysis outputs.",
    )
    parser.add_argument(
        "--primary-results-dir",
        required=True,
        type=Path,
        help=(
            "Results directory produced by beat_the_blues_primary.py; "
            "must contain 13_boundary_bootstrap_summary.csv."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a reduced software check instead of the full analysis; "
            "outputs are not suitable for scientific reporting."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()
RESULTS_DIR = ARGS.output_dir.expanduser().resolve()
PRIMARY_RESULTS_DIR = ARGS.primary_results_dir.expanduser().resolve()

DATA_DIR = RESULTS_DIR / "data"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURE_DIR = RESULTS_DIR / "figures"
for directory in (RESULTS_DIR, DATA_DIR, CHECKPOINT_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ENGINE_VERSION = "btheb_logtime_estimand_v1_2026-08-04"

# Pinned official data artifact.
HSAUR3_VERSION = "1.0-15"
HSAUR3_ARCHIVE_SHA256 = "a17ac3ec9b0ec781512b26976a495e0c9abe95853bd71761791721962836cf9f"
BTHEB_RDA_SHA256 = "411af909549f229c71e4b835245fb32dff6ccc2d27dc84f89d98c4d7fb9d9abe"
HSAUR3_URLS = [
    f"https://cran.r-project.org/src/contrib/HSAUR3_{HSAUR3_VERSION}.tar.gz",
    f"https://cran.r-project.org/src/contrib/Archive/HSAUR3/HSAUR3_{HSAUR3_VERSION}.tar.gz",
]

MONTHS = np.array([0.0, 2.0, 3.0, 5.0, 8.0])
OCCASIONS = ["0", "2", "3", "5", "8"]
BDI_COLUMNS = ["bdi.pre", "bdi.2m", "bdi.3m", "bdi.5m", "bdi.8m"]
TREATMENTS = ["TAU", "BtheB"]
CENTER_MONTH = 4.0

MASTER_SEED = 20260803
PROVISIONAL_NULL_B = 999
FINAL_NULL_B = 4_999
PARTICIPANT_BOOT_B = 499
PREDICTIVE_CHECK_B = 499
ALPHA = 0.05
MIN_USABLE_RATE = 0.95
CHECKPOINT_EVERY = 10

NESTED_NLL_TOL = 1e-6
OPT_NLL_GAP_PER_OBS_TOL = 1e-6
OPT_SOURCE_ABS_GAP_TOL = 0.01
AR1_INTERIOR_LIMIT = 0.98
LR_RESCUE_THRESHOLD = 25.0
AUDIT_REPLICATES = {862}
DEFENSIBLE_RESIDUAL_MODEL = "hetero"

RUN_BOUNDARY_BOOTSTRAP = True
RUN_LEAVE_ONE_OUT = True
RUN_PARTICIPANT_BOOTSTRAP = True

# Smoke mode checks software execution, not scientific results.
SMOKE_TEST = ARGS.smoke_test
if SMOKE_TEST:
    PROVISIONAL_NULL_B = 7
    FINAL_NULL_B = 11
    PARTICIPANT_BOOT_B = 7
    PREDICTIVE_CHECK_B = 29
    print("SMOKE_TEST=True: outputs test software execution only.")

FROZEN_CONFIG = {
    name: globals()[name] for name in [
        "ENGINE_VERSION", "HSAUR3_VERSION", "HSAUR3_ARCHIVE_SHA256", "BTHEB_RDA_SHA256",
        "CENTER_MONTH", "MASTER_SEED", "PROVISIONAL_NULL_B", "FINAL_NULL_B",
        "PARTICIPANT_BOOT_B", "PREDICTIVE_CHECK_B", "ALPHA", "MIN_USABLE_RATE",
        "NESTED_NLL_TOL", "OPT_NLL_GAP_PER_OBS_TOL", "OPT_SOURCE_ABS_GAP_TOL",
        "AR1_INTERIOR_LIMIT", "LR_RESCUE_THRESHOLD", "DEFENSIBLE_RESIDUAL_MODEL", "SMOKE_TEST",
    ]
}
(RESULTS_DIR / "00_frozen_config.json").write_text(json.dumps(FROZEN_CONFIG, indent=2), encoding="utf-8")
FROZEN_CONFIG

# One focused post-hoc estimand: a centered, scaled log(1 + month) random slope.
# Scaling makes the transformed baseline-to-month-8 span equal to 8 units; centering
# at physical month 4 anchors G at the same physical reference used in the linear audit.
LOG_TIME_SCALE = 8.0 / np.log(9.0)
POSTHOC_RESIDUAL_MODEL = "hetero"  # planned occasion-specific residual variances
POSTHOC_BOOTSTRAP_STREAM = 211
RUN_POSTHOC_BOOTSTRAP = True

TIME_ESTIMAND_CONTRACT = {
    "engine_version": ENGINE_VERSION,
    "frozen_primary_estimand": "best linear individual deviation from treatment-by-occasion means over months 0--8",
    "posthoc_estimand": "random slope on 8*log(1+month)/log(9), centered at physical month 4",
    "fixed_mean": "unchanged saturated treatment-by-occasion cell means",
    "residual_model": "unchanged planned occasion-specific independent variances",
    "purpose": "determine whether the unsupported Question 1 result depends on the linear random-slope time basis",
    "hierarchy_lock": "The primary homogeneous and planned occasion-specific linear-time results remain frozen regardless of this post-hoc result.",
}
(RESULTS_DIR / "00_time_estimand_contract.json").write_text(json.dumps(TIME_ESTIMAND_CONTRACT, indent=2), encoding="utf-8")

def synthetic_btheb_wide_for_smoke() -> pd.DataFrame:
    rng = np.random.default_rng(20260803)
    n = 30
    treatment = np.where(np.arange(n) % 2 == 0, "TAU", "BtheB")
    drug = np.where(np.arange(n) % 3 == 0, "Yes", "No")
    length = np.where(np.arange(n) % 4 == 0, ">6m", "<6m")
    rows = []
    for i in range(n):
        b0, b1 = rng.multivariate_normal([0, 0], [[18, -0.25], [-0.25, 0.12]])
        values = []
        for m in MONTHS:
            mean = 27 - (4.5 if treatment[i] == "BtheB" else 3.0) * np.log1p(m)
            values.append(np.clip(mean + b0 + b1 * m + rng.normal(0, 5.5 - 0.18 * m), 0, 63))
        rows.append([drug[i], length[i], treatment[i], *values])
    return pd.DataFrame(rows, columns=["drug", "length", "treatment", *BDI_COLUMNS])

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_official_btheb() -> tuple[pd.DataFrame, dict]:
    archive_path = DATA_DIR / f"HSAUR3_{HSAUR3_VERSION}.tar.gz"
    if archive_path.exists() and sha256_bytes(archive_path.read_bytes()) == HSAUR3_ARCHIVE_SHA256:
        archive_bytes = archive_path.read_bytes()
        used_url = HSAUR3_URLS[0]
    else:
        archive_bytes, used_url, last_error = None, None, None
        for url in HSAUR3_URLS:
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    candidate = response.read()
                if sha256_bytes(candidate) != HSAUR3_ARCHIVE_SHA256:
                    raise RuntimeError(f"Archive checksum mismatch from {url}")
                archive_bytes, used_url = candidate, url
                archive_path.write_bytes(candidate)
                break
            except Exception as exc:
                last_error = exc
        if archive_bytes is None:
            raise RuntimeError("Could not download the pinned HSAUR3 source archive") from last_error

    member = "HSAUR3/data/BtheB.rda"
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
        handle = tf.extractfile(member)
        if handle is None:
            raise FileNotFoundError(member)
        rda_bytes = handle.read()
    if sha256_bytes(rda_bytes) != BTHEB_RDA_SHA256:
        raise RuntimeError("Extracted BtheB.rda checksum mismatch")

    rda_path = DATA_DIR / "BtheB.rda"
    rda_path.write_bytes(rda_bytes)
    objects = pyreadr.read_r(str(rda_path))
    if "BtheB" not in objects:
        raise KeyError(f"Expected object 'BtheB'; found {list(objects)}")
    wide = objects["BtheB"].copy()
    expected = ["drug", "length", "treatment", *BDI_COLUMNS]
    if list(wide.columns) != expected or (not SMOKE_TEST and wide.shape != (100, 8)):
        raise AssertionError(f"Unexpected HSAUR3::BtheB schema: {wide.shape}, {list(wide.columns)}")

    provenance = {
        "source": "CRAN HSAUR3 source archive",
        "package_version": HSAUR3_VERSION,
        "resolved_url": used_url,
        "archive_sha256": sha256_bytes(archive_bytes),
        "rda_sha256": sha256_bytes(rda_bytes),
        "rows": int(wide.shape[0]),
        "columns": list(wide.columns),
    }
    (RESULTS_DIR / "01_data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return wide, provenance


wide, provenance = (synthetic_btheb_wide_for_smoke(), {"source": "schema-faithful synthetic smoke fixture"}) if SMOKE_TEST else download_official_btheb()
wide = wide.copy()
wide.insert(0, "subject", wide.index.astype(str))
wide["drug_yes"] = (wide["drug"].astype(str) == "Yes").astype(float)
wide["episode_gt6m"] = (wide["length"].astype(str) == ">6m").astype(float)

parts = []
for col, month, occasion in zip(BDI_COLUMNS, MONTHS, OCCASIONS):
    keep = ["subject", "drug", "length", "drug_yes", "episode_gt6m", "treatment", col]
    part = wide[keep].rename(columns={col: "bdi"})
    part["month"] = month
    part["occasion"] = occasion
    parts.append(part)
long_all = pd.concat(parts, ignore_index=True)
long_all["observed"] = long_all["bdi"].notna()
long = long_all.loc[long_all["observed"]].copy()
long["month_c"] = long["month"] - CENTER_MONTH
long["occasion"] = pd.Categorical(long["occasion"], categories=OCCASIONS, ordered=True)
long["treatment"] = pd.Categorical(long["treatment"], categories=TREATMENTS)
long = long.sort_values(["subject", "month"]).reset_index(drop=True)

assert long["subject"].nunique() == (30 if SMOKE_TEST else 100)
assert long["bdi"].between(0, 63).all()
long.to_csv(DATA_DIR / "BtheB_long_observed.csv", index=False)
display(provenance)
display(wide.head())
print(f"{len(long)} observed BDI-II scores from {long['subject'].nunique()} participants")

@dataclass(frozen=True)
class GroupData:
    subject: str
    y: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    occasion_index: np.ndarray
    month: np.ndarray


CELL_COLUMNS = [f"{trt}__{occ}" for trt in TREATMENTS for occ in OCCASIONS]
CELL_TO_INDEX = {name: j for j, name in enumerate(CELL_COLUMNS)}


def build_groups(frame: pd.DataFrame, fixed_covariates: Sequence[str] = ()) -> tuple[list[GroupData], list[str]]:
    fixed_names = [*CELL_COLUMNS, *fixed_covariates]
    groups = []
    for subject, d in frame.groupby("subject", sort=True):
        d = d.sort_values("month")
        X = np.zeros((len(d), len(fixed_names)))
        for row_i, (trt, occ) in enumerate(zip(d["treatment"].astype(str), d["occasion"].astype(str))):
            X[row_i, CELL_TO_INDEX[f"{trt}__{occ}"]] = 1.0
        for j, name in enumerate(fixed_covariates, start=len(CELL_COLUMNS)):
            X[:, j] = d[name].to_numpy(float)
        Z = np.column_stack([np.ones(len(d)), d["month_c"].to_numpy(float)])
        occasion_index = np.array([OCCASIONS.index(x) for x in d["occasion"].astype(str)], dtype=int)
        groups.append(GroupData(
            subject=str(subject), y=d["bdi"].to_numpy(float), X=X, Z=Z,
            occasion_index=occasion_index, month=d["month"].to_numpy(float),
        ))
    stacked = np.vstack([g.X for g in groups])
    if np.linalg.matrix_rank(stacked) != len(fixed_names):
        raise RuntimeError(f"Fixed design is rank deficient: rank={np.linalg.matrix_rank(stacked)}, p={len(fixed_names)}")
    return groups, fixed_names


def groups_with_new_y(template: Sequence[GroupData], ys: Sequence[np.ndarray], prefix="sim") -> list[GroupData]:
    return [
        GroupData(f"{prefix}_{j:03d}_{g.subject}", np.asarray(y, float), g.X, g.Z, g.occasion_index, g.month)
        for j, (g, y) in enumerate(zip(template, ys))
    ]


groups, fixed_names = build_groups(long)
N_OBS = sum(len(g.y) for g in groups)
print({"participants": len(groups), "observations": N_OBS, "fixed_effects": len(fixed_names)})

def analysis_scale(groups: Sequence[GroupData]) -> float:
    return max(float(np.std(np.concatenate([g.y for g in groups]), ddof=1)), 1.0)


def parameter_bounds(groups: Sequence[GroupData], random_slope: bool, residual_model: str):
    scale = analysis_scale(groups)
    log_lo, log_hi = math.log(1e-3), math.log(max(10 * scale, 100.0))
    bounds = []
    if random_slope:
        bounds.extend([(log_lo, log_hi), (-10 * scale, 10 * scale), (log_lo, log_hi)])
    else:
        bounds.append((log_lo, log_hi))
    if residual_model == "hom":
        bounds.append((log_lo, log_hi))
    elif residual_model == "hetero":
        bounds.extend([(log_lo, log_hi)] * len(OCCASIONS))
    elif residual_model == "ar1":
        bounds.extend([(log_lo, log_hi), (-4.0, 4.0)])
    elif residual_model == "hetero_ar1":
        bounds.extend([(log_lo, log_hi)] * len(OCCASIONS))
        bounds.append((-4.0, 4.0))
    else:
        raise ValueError(residual_model)
    return bounds


def initial_theta(groups: Sequence[GroupData], random_slope: bool, residual_model: str):
    scale = analysis_scale(groups)
    pars = []
    if random_slope:
        pars.extend([math.log(0.55 * scale), 0.0, math.log(max(0.12 * scale, 0.25))])
    else:
        pars.append(math.log(0.55 * scale))
    if residual_model == "hom":
        pars.append(math.log(0.60 * scale))
    elif residual_model == "hetero":
        pars.extend([math.log(0.60 * scale)] * len(OCCASIONS))
    elif residual_model == "ar1":
        pars.extend([math.log(0.60 * scale), np.arctanh(0.15)])
    elif residual_model == "hetero_ar1":
        pars.extend([math.log(0.60 * scale)] * len(OCCASIONS))
        pars.append(np.arctanh(0.10))
    return np.asarray(pars, dtype=float)


def unpack_theta(theta: np.ndarray, random_slope: bool, residual_model: str):
    theta = np.asarray(theta, float)
    if random_slope:
        L = np.array([[np.exp(theta[0]), 0.0], [theta[1], np.exp(theta[2])]])
        G = L @ L.T
        k = 3
    else:
        G = np.array([[np.exp(2 * theta[0]), 0.0], [0.0, 0.0]])
        k = 1
    if residual_model == "hom":
        residual = {"sigma": float(np.exp(theta[k]))}
    elif residual_model == "hetero":
        residual = {"sigma_by_occasion": np.exp(theta[k:k + len(OCCASIONS)])}
    elif residual_model == "ar1":
        residual = {"sigma": float(np.exp(theta[k])), "rho": float(np.tanh(theta[k + 1]))}
    elif residual_model == "hetero_ar1":
        residual = {
            "sigma_by_occasion": np.exp(theta[k:k + len(OCCASIONS)]),
            "rho": float(np.tanh(theta[k + len(OCCASIONS)])),
        }
    else:
        raise ValueError(residual_model)
    return G, residual


def residual_covariance(group: GroupData, residual: dict, residual_model: str) -> np.ndarray:
    n = len(group.y)
    if residual_model == "hom":
        return residual["sigma"] ** 2 * np.eye(n)
    if residual_model == "hetero":
        sd = residual["sigma_by_occasion"][group.occasion_index]
        return np.diag(sd ** 2)
    lag = np.abs(group.occasion_index[:, None] - group.occasion_index[None, :])
    corr = np.power(residual["rho"], lag)
    if residual_model == "ar1":
        return residual["sigma"] ** 2 * corr
    if residual_model == "hetero_ar1":
        sd = residual["sigma_by_occasion"][group.occasion_index]
        return sd[:, None] * corr * sd[None, :]
    raise ValueError(residual_model)


def profiled_likelihood(theta, groups, random_slope=True, residual_model="hom", reml=True, return_all=False):
    try:
        G, residual = unpack_theta(theta, random_slope, residual_model)
        p = groups[0].X.shape[1]
        A, b = np.zeros((p, p)), np.zeros(p)
        y_vinv_y = logdet_v = 0.0
        n_total = 0
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
            n_total += len(group.y)
        sign_a, logdet_a = np.linalg.slogdet(A)
        if sign_a <= 0 or np.linalg.matrix_rank(A) < p:
            raise np.linalg.LinAlgError("fixed-effects information is singular")
        beta = np.linalg.solve(A, b)
        quad = max(y_vinv_y - float(b @ beta), 0.0)
        if reml:
            nll = 0.5 * (logdet_v + logdet_a + quad + (n_total - p) * np.log(2 * np.pi))
        else:
            nll = 0.5 * (logdet_v + quad + n_total * np.log(2 * np.pi))
        if not np.isfinite(nll):
            raise FloatingPointError("nonfinite likelihood")
        if return_all:
            return nll, beta, np.linalg.inv(A), G, residual, quad
        return nll
    except Exception:
        if return_all:
            raise
        return 1e100

def fit_model(
    groups: Sequence[GroupData], random_slope: bool, residual_model: str,
    optimizer: str, start: np.ndarray, reml=True, maxiter=2500,
):
    bounds = parameter_bounds(groups, random_slope, residual_model)
    options = {"maxiter": maxiter}
    if optimizer == "L-BFGS-B":
        options.update({"ftol": 1e-11, "gtol": 1e-7, "maxls": 50})
    elif optimizer == "Powell":
        options.update({"xtol": 1e-7, "ftol": 1e-9})
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
        admissible = bool(np.linalg.eigvalsh(G).min() >= -1e-10 and G[0, 0] > 0 and residual_ok)
    except Exception:
        nll, beta, cov_beta, G, residual, quad = np.inf, None, None, None, None, np.nan
        admissible = False
    return {
        "success": bool(result.success), "admissible": admissible,
        "message": str(result.message), "optimizer": optimizer,
        "theta": np.asarray(result.x, float), "nll": float(nll), "loglik": float(-nll),
        "beta": beta, "cov_beta": cov_beta, "G": G, "residual": residual,
        "quad": quad, "nit": getattr(result, "nit", np.nan),
        "random_slope": random_slope, "residual_model": residual_model, "reml": reml,
    }


def fit_usable(fit: dict) -> bool:
    return bool(fit["success"] and fit["admissible"] and np.isfinite(fit["nll"]))


def residual_components_for_target(source_fit: dict, target_model: str):
    src = source_fit["residual"]
    src_model = source_fit["residual_model"]
    if "sigma_by_occasion" in src:
        sds = np.asarray(src["sigma_by_occasion"], float)
        common = float(np.sqrt(np.mean(sds ** 2)))
    else:
        common = float(src["sigma"])
        sds = np.repeat(common, len(OCCASIONS))
    rho = float(src.get("rho", 0.0))
    if target_model == "hom":
        return [math.log(common)]
    if target_model == "hetero":
        return list(np.log(sds))
    if target_model == "ar1":
        return [math.log(common), np.arctanh(np.clip(rho, -.995, .995))]
    if target_model == "hetero_ar1":
        return [*np.log(sds), np.arctanh(np.clip(rho, -.995, .995))]
    raise ValueError(target_model)


def mapped_start_from_fit(
    source_fit: dict, target_random_slope: bool, target_residual_model: str,
    slope_sd_floor=1e-3,
) -> np.ndarray:
    G = np.asarray(source_fit["G"], float)
    q0 = max(float(G[0, 0]), 1e-6)
    pars = []
    if target_random_slope:
        if source_fit["random_slope"] and G[1, 1] > slope_sd_floor ** 2:
            candidate = G.copy()
        else:
            candidate = np.array([[q0, 0.0], [0.0, slope_sd_floor ** 2]])
        jitter = 0.0
        for _ in range(8):
            try:
                L = np.linalg.cholesky(candidate + jitter * np.eye(2))
                break
            except np.linalg.LinAlgError:
                jitter = 1e-10 if jitter == 0 else jitter * 10
        pars.extend([math.log(L[0, 0]), L[1, 0], math.log(max(L[1, 1], 1e-3))])
    else:
        pars.append(math.log(np.sqrt(q0)))
    pars.extend(residual_components_for_target(source_fit, target_residual_model))
    return np.asarray(pars, float)


def fixed_only_start(groups: Sequence[GroupData], random_slope: bool, residual_model: str) -> np.ndarray:
    theta = initial_theta(groups, random_slope, residual_model)
    theta[0] = math.log(1e-3)
    if random_slope:
        theta[1] = 0.0
        theta[2] = math.log(1e-3)
    return theta


def unique_start_specs(groups, random_slope, residual_model, extras=()):
    specs = [("default", initial_theta(groups, random_slope, residual_model)), *list(extras)]
    bounds = parameter_bounds(groups, random_slope, residual_model)
    seen, out = set(), []
    for name, values in specs:
        values = np.asarray(values, float)
        if len(values) != len(bounds):
            raise ValueError(f"Start {name!r} has length {len(values)}; expected {len(bounds)}")
        clipped = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(values, bounds)], float)
        key = tuple(np.round(clipped, 10))
        if key not in seen:
            seen.add(key)
            out.append((name, clipped))
    return out


def fit_multistart(groups, random_slope, residual_model, extra_starts=(), reml=True):
    attempts = []
    for start_name, start in unique_start_specs(groups, random_slope, residual_model, extra_starts):
        lb = fit_model(groups, random_slope, residual_model, "L-BFGS-B", start, reml=reml)
        lb["start_name"] = start_name
        attempts.append(lb)
        powell_start = lb["theta"] if lb["admissible"] and np.isfinite(lb["nll"]) else start
        pw = fit_model(groups, random_slope, residual_model, "Powell", powell_start, reml=reml)
        pw["start_name"] = start_name + " -> L-BFGS-B"
        attempts.append(pw)
    usable = [fit for fit in attempts if fit_usable(fit)]
    if not usable:
        raise RuntimeError(f"No usable {residual_model} {'alt' if random_slope else 'null'} fit")
    best = min(usable, key=lambda fit: fit["nll"])
    return {"best": best, "attempts": attempts}


def fit_pair(groups, residual_model, null_seed_fits=(), alt_seed_fits=()):
    null_starts = [
        (f"mapped_null_{j}", mapped_start_from_fit(fit, False, residual_model))
        for j, fit in enumerate(null_seed_fits)
    ]
    null_bundle = fit_multistart(groups, False, residual_model, null_starts)
    null_fit = null_bundle["best"]

    alt_starts = [("nested_from_fitted_null", mapped_start_from_fit(null_fit, True, residual_model))]
    alt_starts.extend(
        (f"mapped_alt_{j}", mapped_start_from_fit(fit, True, residual_model))
        for j, fit in enumerate(alt_seed_fits)
    )
    alt_bundle = fit_multistart(groups, True, residual_model, alt_starts)
    alt_fit = alt_bundle["best"]

    nested_ok = alt_fit["nll"] <= null_fit["nll"] + NESTED_NLL_TOL
    if not nested_ok:
        rescue = [
            ("nested_boundary_rescue", mapped_start_from_fit(null_fit, True, residual_model, slope_sd_floor=1e-3)),
            ("nested_moderate_rescue", mapped_start_from_fit(null_fit, True, residual_model, slope_sd_floor=.05)),
        ]
        alt_bundle = fit_multistart(groups, True, residual_model, rescue)
        alt_fit = alt_bundle["best"]
        nested_ok = alt_fit["nll"] <= null_fit["nll"] + NESTED_NLL_TOL
    return {"null": null_fit, "alt": alt_fit, "null_bundle": null_bundle, "alt_bundle": alt_bundle, "nested_ok": nested_ok}


def attempt_row(model_label: str, component: str, fit: dict) -> dict:
    row = {
        "model": model_label, "component": component, "optimizer": fit["optimizer"],
        "start_name": fit.get("start_name"), "success": fit["success"], "admissible": fit["admissible"],
        "nll_reml": fit["nll"], "iterations": fit["nit"], "message": fit["message"],
    }
    if fit["G"] is not None:
        G = fit["G"]
        row.update({"var_intercept": G[0, 0], "cov_intercept_slope": G[0, 1], "var_slope": G[1, 1]})
    if fit["residual"] is not None and "rho" in fit["residual"]:
        row["residual_rho"] = fit["residual"]["rho"]
    return row

def log_time_value(month):
    month = np.asarray(month, float)
    return LOG_TIME_SCALE * (np.log1p(month) - np.log1p(CENTER_MONTH))


def build_groups_for_basis(frame: pd.DataFrame, basis: str) -> tuple[list[GroupData], list[str]]:
    fixed_names = list(CELL_COLUMNS)
    groups_out = []
    for subject, d in frame.groupby("subject", sort=True):
        d = d.sort_values("month")
        X = np.zeros((len(d), len(fixed_names)))
        for row_i, (trt, occ) in enumerate(zip(d["treatment"].astype(str), d["occasion"].astype(str))):
            X[row_i, CELL_TO_INDEX[f"{trt}__{occ}"]] = 1.0
        if basis == "linear":
            time_score = d["month"].to_numpy(float) - CENTER_MONTH
        elif basis == "log1p":
            time_score = log_time_value(d["month"].to_numpy(float))
        else:
            raise ValueError(basis)
        Z = np.column_stack([np.ones(len(d)), time_score])
        occasion_index = np.array([OCCASIONS.index(x) for x in d["occasion"].astype(str)], dtype=int)
        groups_out.append(GroupData(
            subject=str(subject), y=d["bdi"].to_numpy(float), X=X, Z=Z,
            occasion_index=occasion_index, month=d["month"].to_numpy(float),
        ))
    stacked = np.vstack([g.X for g in groups_out])
    if np.linalg.matrix_rank(stacked) != len(fixed_names):
        raise RuntimeError("Treatment-by-occasion fixed design is rank deficient")
    return groups_out, fixed_names


groups_linear, _ = build_groups_for_basis(long, "linear")
groups_log, _ = build_groups_for_basis(long, "log1p")
assert all(np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y) for a, b in zip(groups_linear, groups_log))

time_basis_table = pd.DataFrame({
    "month": MONTHS,
    "linear_month_centered_at_4": MONTHS - CENTER_MONTH,
    "scaled_log1p_centered_at_month_4": log_time_value(MONTHS),
})
assert np.isclose(time_basis_table["scaled_log1p_centered_at_month_4"].iloc[-1] - time_basis_table["scaled_log1p_centered_at_month_4"].iloc[0], 8.0)
time_basis_table.to_csv(RESULTS_DIR / "02_time_basis_definition.csv", index=False)
display(time_basis_table)

def participant_source_table(groups: Sequence[GroupData], fit: dict) -> pd.DataFrame:
    G, residual, residual_model = fit["G"], fit["residual"], fit["residual_model"]
    q0, q01, q1 = G[0, 0], G[0, 1], G[1, 1]
    if q1 <= 0:
        raise ValueError("Slope variance must be positive for source attribution")
    q_innovation = max(q1 - q01 ** 2 / q0, 0.0)
    rows = []
    for group in groups:
        R = residual_covariance(group, residual, residual_model)
        info = group.Z.T @ np.linalg.solve(R, group.Z)
        slope_precision = max(float(info[1, 1] - info[1, 0] ** 2 / info[0, 0]), 0.0)
        effective_error = np.inf if slope_precision <= 1e-14 else 1.0 / slope_precision
        r_direct = q1 * slope_precision / (1.0 + q1 * slope_precision)
        V = group.Z @ G @ group.Z.T + R
        post = G - G @ group.Z.T @ np.linalg.solve(V, group.Z @ G)
        p_full = 1.0 - post[1, 1] / q1
        delta = p_full - r_direct
        r_innovation = q_innovation * slope_precision / (1.0 + q_innovation * slope_precision)
        rows.append({
            "subject": group.subject, "n_obs": len(group.y),
            "first_month": group.month.min(), "last_month": group.month.max(),
            "followup_months": group.month.max() - group.month.min(),
            "effective_slope_error_variance": effective_error,
            "R_direct": r_direct, "P_full": p_full, "Delta": delta,
            "remaining_uncertainty": 1.0 - p_full,
            "innovation_variance_at_month4_origin": q_innovation,
            "R_innovation": r_innovation,
        })
    out = pd.DataFrame(rows)
    if out["Delta"].min() < -1e-7:
        raise AssertionError(f"Negative assisted component: {out['Delta'].min()}")
    if not np.allclose(out["P_full"], out["R_direct"] + out["Delta"], atol=1e-11):
        raise AssertionError("P_full = R_direct + Delta identity failed")
    return out


def source_summary(groups: Sequence[GroupData], fit: dict, label: str) -> dict:
    src, G = participant_source_table(groups, fit), fit["G"]
    rho = G[0, 1] / np.sqrt(G[0, 0] * G[1, 1])
    row = {
        "model": label, "participants": len(src), "observations": sum(len(g.y) for g in groups),
        "mean_R_direct": src["R_direct"].mean(), "mean_P_full": src["P_full"].mean(),
        "mean_Delta": src["Delta"].mean(), "mean_R_innovation": src["R_innovation"].mean(),
        "median_R_direct": src["R_direct"].median(), "median_P_full": src["P_full"].median(),
        "median_Delta": src["Delta"].median(), "var_intercept": G[0, 0],
        "cov_intercept_slope": G[0, 1], "var_slope": G[1, 1], "rho": rho,
        "one_minus_rho_sq": 1.0 - rho ** 2,
        "g_min_eigen_over_trace": np.linalg.eigvalsh(G).min() / np.trace(G),
    }
    return row


def fit_parameter_row(label: str, fit: dict) -> dict:
    G = fit["G"]
    row = {
        "model": label, "optimizer": fit["optimizer"], "start_name": fit.get("start_name"),
        "nll_reml": fit["nll"], "var_intercept": G[0, 0], "cov_intercept_slope": G[0, 1],
        "var_slope": G[1, 1],
        "corr_intercept_slope": G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]) if G[1, 1] > 0 else np.nan,
    }
    if "sigma" in fit["residual"]:
        row["residual_sd"] = fit["residual"]["sigma"]
    if "sigma_by_occasion" in fit["residual"]:
        for occ, sd in zip(OCCASIONS, fit["residual"]["sigma_by_occasion"]):
            row[f"residual_sd_month_{occ}"] = sd
    if "rho" in fit["residual"]:
        row["residual_rho"] = fit["residual"]["rho"]
    return row

# Refit only the planned occasion-specific residual model so the time basis is the
# sole conceptual change.  The RI null is identical under the two time bases.
linear_pair = fit_pair(groups_linear, POSTHOC_RESIDUAL_MODEL)
log_pair = fit_pair(
    groups_log, POSTHOC_RESIDUAL_MODEL,
    null_seed_fits=(linear_pair["null"],), alt_seed_fits=(linear_pair["alt"],),
)
if not linear_pair["nested_ok"] or not log_pair["nested_ok"]:
    raise RuntimeError("Observed alternative must be no worse than its RI null")
if abs(linear_pair["null"]["nll"] - log_pair["null"]["nll"]) > 2e-5:
    raise RuntimeError("The RI null should be identical under linear and log-time slope bases")

linear_lr = max(0.0, 2.0 * (linear_pair["null"]["nll"] - linear_pair["alt"]["nll"]))
log_lr = max(0.0, 2.0 * (log_pair["null"]["nll"] - log_pair["alt"]["nll"]))
linear_source = source_summary(groups_linear, linear_pair["alt"], "linear_time_planned_hetero")
log_source = source_summary(groups_log, log_pair["alt"], "posthoc_log_time_planned_hetero")
if not SMOKE_TEST and abs(linear_lr - 2.51447) > 0.10:
    raise RuntimeError(
        f"Frozen planned-residual LR guard failed: expected approximately 2.51447, obtained {linear_lr:.6f}"
    )

def optimizer_min(bundle, optimizer_name):
    fits = [x for x in bundle["attempts"] if x["optimizer"] == optimizer_name and fit_usable(x)]
    return min((x["nll"] for x in fits), default=np.nan)


observed_comparison = pd.DataFrame([
    {
        "time_estimand": "linear month (frozen planned sensitivity)", "posthoc": False,
        "null_nll": linear_pair["null"]["nll"], "alt_nll": linear_pair["alt"]["nll"], "observed_LR": linear_lr,
        **linear_source,
        "optimizer_nll_gap": abs(optimizer_min(linear_pair["alt_bundle"], "L-BFGS-B") - optimizer_min(linear_pair["alt_bundle"], "Powell")),
    },
    {
        "time_estimand": "scaled log1p month (post hoc)", "posthoc": True,
        "null_nll": log_pair["null"]["nll"], "alt_nll": log_pair["alt"]["nll"], "observed_LR": log_lr,
        **log_source,
        "optimizer_nll_gap": abs(optimizer_min(log_pair["alt_bundle"], "L-BFGS-B") - optimizer_min(log_pair["alt_bundle"], "Powell")),
    },
])
observed_comparison["near_rank_one_G"] = (
    (observed_comparison["one_minus_rho_sq"] < 1e-4)
    | (observed_comparison["g_min_eigen_over_trace"] < 1e-4)
)
observed_comparison.to_csv(RESULTS_DIR / "03_observed_linear_vs_logtime_comparison.csv", index=False)

attempts = pd.DataFrame(
    [attempt_row("linear", "null", x) for x in linear_pair["null_bundle"]["attempts"]]
    + [attempt_row("linear", "alternative", x) for x in linear_pair["alt_bundle"]["attempts"]]
    + [attempt_row("log1p_posthoc", "null", x) for x in log_pair["null_bundle"]["attempts"]]
    + [attempt_row("log1p_posthoc", "alternative", x) for x in log_pair["alt_bundle"]["attempts"]]
)
attempts.to_csv(RESULTS_DIR / "04_observed_optimizer_attempts.csv", index=False)
display(observed_comparison)

def load_primary_hierarchy():
    path = PRIMARY_RESULTS_DIR / "13_boundary_bootstrap_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "Required primary Beat the Blues result is missing: "
            f"{path}. Run beat_the_blues_primary.py first and pass its "
            "output directory via --primary-results-dir."
        )

    frame = pd.read_csv(path)
    needed = {
        "residual_model",
        "observed_LR",
        "p_plus_one",
        "heterogeneity_status",
    }
    missing = sorted(needed.difference(frame.columns))
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {missing}"
        )

    out = frame.loc[
        frame["residual_model"].isin(["hom", "hetero"]),
        sorted(needed),
    ].copy()

    if set(out["residual_model"]) != {"hom", "hetero"}:
        raise RuntimeError(
            "Primary hierarchy must contain exactly the hom and hetero "
            f"branches; found {sorted(set(out['residual_model']))}"
        )

    out["frozen_source"] = "13_boundary_bootstrap_summary.csv"
    return out


def load_primary_run_mode():
    path = PRIMARY_RESULTS_DIR / "00_frozen_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Required primary Beat the Blues configuration is missing: "
            f"{path}"
        )

    config = json.loads(path.read_text(encoding="utf-8"))
    if "SMOKE_TEST" not in config:
        raise RuntimeError(
            f"{path} does not record the required SMOKE_TEST field"
        )

    return bool(config["SMOKE_TEST"]), path


frozen_primary = load_primary_hierarchy()
primary_smoke_test, primary_config_path = load_primary_run_mode()

if not set(frozen_primary["residual_model"]) == {"hom", "hetero"}:
    raise RuntimeError("Frozen primary hierarchy is incomplete")

if primary_smoke_test != SMOKE_TEST:
    raise RuntimeError(
        "Beat the Blues primary/log-time mode mismatch: "
        f"primary SMOKE_TEST={primary_smoke_test}, "
        f"log-time SMOKE_TEST={SMOKE_TEST}. "
        "Use smoke primary results only with --smoke-test, and full "
        "primary results only for the full log-time analysis."
    )

if (
    not SMOKE_TEST
    and frozen_primary["heterogeneity_status"]
        .str.lower()
        .str.startswith("supported")
        .any()
):
    raise RuntimeError(
        "Frozen hierarchy guard failed: both full-analysis primary "
        "branches must remain unsupported"
    )

frozen_primary.to_csv(
    RESULTS_DIR / "05_frozen_primary_hierarchy.csv",
    index=False,
)
display(frozen_primary)

def deterministic_rng(replicate: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, POSTHOC_BOOTSTRAP_STREAM, replicate]))


def groups_with_new_y_for_basis(template, ys, prefix):
    return [
        GroupData(f"{prefix}_{j:03d}_{g.subject}", np.asarray(y, float), g.X, g.Z, g.occasion_index, g.month)
        for j, (g, y) in enumerate(zip(template, ys))
    ]


def simulate_logtime_null(replicate):
    rng = deterministic_rng(replicate)
    null_fit = log_pair["null"]
    q0, beta = null_fit["G"][0, 0], null_fit["beta"]
    ys = []
    for group in groups_log:
        b0 = rng.normal(0.0, np.sqrt(q0))
        R = residual_covariance(group, null_fit["residual"], POSTHOC_RESIDUAL_MODEL)
        eps = rng.multivariate_normal(np.zeros(len(group.y)), R)
        ys.append(group.X @ beta + b0 + eps)
    return groups_with_new_y_for_basis(groups_log, ys, f"log_null_{replicate}")


def refit_logtime_bootstrap(sim_groups, replicate):
    null_bundle = fit_multistart(
        sim_groups, False, POSTHOC_RESIDUAL_MODEL,
        [("generating_null", mapped_start_from_fit(log_pair["null"], False, POSTHOC_RESIDUAL_MODEL))],
    )
    null_fit = null_bundle["best"]
    alt_bundle = fit_multistart(
        sim_groups, True, POSTHOC_RESIDUAL_MODEL,
        [
            ("nested_from_replicate_null", mapped_start_from_fit(null_fit, True, POSTHOC_RESIDUAL_MODEL)),
            ("observed_log_alternative", mapped_start_from_fit(log_pair["alt"], True, POSTHOC_RESIDUAL_MODEL)),
        ],
    )
    alt_fit = alt_bundle["best"]
    lr = 2.0 * (null_fit["nll"] - alt_fit["nll"])
    rescue = bool(lr < -NESTED_NLL_TOL or lr > LR_RESCUE_THRESHOLD)
    if rescue:
        null_bundle = fit_multistart(
            sim_groups, False, POSTHOC_RESIDUAL_MODEL,
            [
                ("generating_null", mapped_start_from_fit(log_pair["null"], False, POSTHOC_RESIDUAL_MODEL)),
                ("fixed_only_boundary", fixed_only_start(sim_groups, False, POSTHOC_RESIDUAL_MODEL)),
                ("alt_implied_null", mapped_start_from_fit(alt_fit, False, POSTHOC_RESIDUAL_MODEL)),
            ],
        )
        null_fit = null_bundle["best"]
        alt_bundle = fit_multistart(
            sim_groups, True, POSTHOC_RESIDUAL_MODEL,
            [
                ("nested_from_repaired_null", mapped_start_from_fit(null_fit, True, POSTHOC_RESIDUAL_MODEL)),
                ("observed_log_alternative", mapped_start_from_fit(log_pair["alt"], True, POSTHOC_RESIDUAL_MODEL)),
            ],
        )
        alt_fit = alt_bundle["best"]
        lr = 2.0 * (null_fit["nll"] - alt_fit["nll"])
    nested_ok = alt_fit["nll"] <= null_fit["nll"] + NESTED_NLL_TOL
    return {
        "engine_version": ENGINE_VERSION, "replicate": replicate,
        "usable": bool(fit_usable(null_fit) and fit_usable(alt_fit) and nested_ok),
        "LR": max(0.0, lr) if nested_ok else np.nan,
        "nested_likelihood_ok": nested_ok, "rescue_triggered": rescue,
        "null_nll": null_fit["nll"], "alt_nll": alt_fit["nll"],
        "null_optimizer": null_fit["optimizer"], "alt_optimizer": alt_fit["optimizer"],
    }


POSTHOC_BOOTSTRAP_PATH = CHECKPOINT_DIR / "logtime_null_bootstrap.csv"


def run_posthoc_bootstrap_to(target_b):
    existing = pd.read_csv(POSTHOC_BOOTSTRAP_PATH) if POSTHOC_BOOTSTRAP_PATH.exists() else pd.DataFrame()
    if len(existing) and set(existing["engine_version"].astype(str)) != {ENGINE_VERSION}:
        raise RuntimeError("Post-hoc bootstrap checkpoint belongs to a different engine version")
    done = set(existing.get("replicate", pd.Series(dtype=int)).astype(int))
    rows = existing.to_dict("records")
    todo = [b for b in range(1, target_b + 1) if b not in done]
    for j, b in enumerate(tqdm(todo, desc=f"Post-hoc log-time null bootstrap to B={target_b}", unit="rep"), 1):
        try:
            rows.append(refit_logtime_bootstrap(simulate_logtime_null(b), b))
        except Exception as exc:
            rows.append({
                "engine_version": ENGINE_VERSION, "replicate": b, "usable": False,
                "LR": np.nan, "nested_likelihood_ok": False, "rescue_triggered": True,
                "null_nll": np.nan, "alt_nll": np.nan,
                "null_optimizer": None, "alt_optimizer": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
        if j % CHECKPOINT_EVERY == 0 or j == len(todo):
            pd.DataFrame(rows).sort_values("replicate").to_csv(POSTHOC_BOOTSTRAP_PATH, index=False)
    return pd.DataFrame(rows).query("replicate <= @target_b").sort_values("replicate").reset_index(drop=True)


def exact_binomial_interval(k, n, alpha=ALPHA):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def summarize_posthoc_bootstrap(reps):
    usable = reps.query("usable and nested_likelihood_ok and LR == LR").copy()
    k, n = int((usable["LR"] >= log_lr - 1e-12).sum()), len(usable)
    if n == 0:
        raise RuntimeError("No usable post-hoc bootstrap replicates")
    lo, hi = exact_binomial_interval(k, n)
    p = (k + 1) / (n + 1)
    if lo <= ALPHA <= hi:
        status = "borderline Monte Carlo evidence in post-hoc sensitivity"
    elif p < ALPHA:
        status = "exploratory support under post-hoc log-time estimand"
    else:
        status = "not supported under post-hoc log-time estimand"
    return {
        "time_estimand": "scaled log1p month; centered at month 4",
        "residual_model": POSTHOC_RESIDUAL_MODEL,
        "observed_LR": log_lr, "requested_replicates": len(reps), "usable_replicates": n,
        "usable_rate": n / len(reps), "nested_violations": int((~reps["nested_likelihood_ok"].astype(bool)).sum()),
        "exceedances": k, "p_plus_one": p, "mc_interval_low": lo, "mc_interval_high": hi,
        "mc_interval_crosses_0.05": lo <= ALPHA <= hi, "posthoc_status": status,
    }


if RUN_POSTHOC_BOOTSTRAP:
    posthoc_reps = run_posthoc_bootstrap_to(PROVISIONAL_NULL_B)
    provisional = summarize_posthoc_bootstrap(posthoc_reps)
    if provisional["mc_interval_crosses_0.05"] and FINAL_NULL_B > PROVISIONAL_NULL_B:
        posthoc_reps = run_posthoc_bootstrap_to(FINAL_NULL_B)
else:
    if not POSTHOC_BOOTSTRAP_PATH.exists():
        raise FileNotFoundError("RUN_POSTHOC_BOOTSTRAP=False but checkpoint is absent")
    posthoc_reps = pd.read_csv(POSTHOC_BOOTSTRAP_PATH)

posthoc_summary = pd.DataFrame([summarize_posthoc_bootstrap(posthoc_reps)])
if posthoc_summary["usable_rate"].iloc[0] < MIN_USABLE_RATE:
    raise RuntimeError("Post-hoc bootstrap usability gate failed")
posthoc_reps.to_csv(RESULTS_DIR / "06_logtime_boundary_bootstrap_replicates.csv", index=False)
posthoc_summary.to_csv(RESULTS_DIR / "07_logtime_boundary_bootstrap_summary.csv", index=False)
display(posthoc_summary)

log_supported = "exploratory support" in posthoc_summary["posthoc_status"].iloc[0]
time_dependence = (
    "The Question 1 result is time-estimand-sensitive: the post-hoc log-time slope produced exploratory support, but it cannot overturn the frozen unsupported primary hierarchy."
    if log_supported else
    "The post-hoc log-time slope did not rescue Question 1; unsupported heterogeneity is not attributable solely to using the linear random-slope basis in this focused check."
)

adjudication = pd.DataFrame([{
    "frozen_primary_hierarchy": "individual-slope claim not supported by these data",
    "frozen_primary_unchanged": True,
    "posthoc_time_estimand": "8*log(1+month)/log(9), centered at month 4",
    "posthoc_residual_model": "occasion-specific independent residual variances",
    "posthoc_observed_LR": log_lr,
    "posthoc_bootstrap_p": posthoc_summary["p_plus_one"].iloc[0],
    "posthoc_mc_interval_low": posthoc_summary["mc_interval_low"].iloc[0],
    "posthoc_mc_interval_high": posthoc_summary["mc_interval_high"].iloc[0],
    "posthoc_status": posthoc_summary["posthoc_status"].iloc[0],
    "time_estimand_dependence": time_dependence,
    "question_2_3_scope": "conditional fitted-model descriptions only; no patient ranking or trait interpretation",
}])
adjudication.to_csv(RESULTS_DIR / "08_time_estimand_adjudication.csv", index=False)

report = (
    "The frozen Beat the Blues hierarchy remains unchanged: the a priori homogeneous model and the planned occasion-specific-variance sensitivity did not establish patient-specific slope heterogeneity. "
    "As an explicitly post-hoc estimand check, we replaced centered linear month only in the random-effects design with a centered and scaled log(1+month) score; treatment-by-occasion fixed means, observed schedules, and occasion-specific residual variances were unchanged. "
    f"The log-time RI-versus-RI+RS statistic was LR={log_lr:.6g}; the observed-schedule boundary bootstrap gave p={posthoc_summary['p_plus_one'].iloc[0]:.6g} "
    f"with Monte Carlo interval [{posthoc_summary['mc_interval_low'].iloc[0]:.6g}, {posthoc_summary['mc_interval_high'].iloc[0]:.6g}]. "
    + time_dependence + " Questions 2 and 3 remain conditional fitted-model descriptions rather than evidence for an established symptom-change trait."
)
(RESULTS_DIR / "09_reporting_paragraph.txt").write_text(report, encoding="utf-8")
display(adjudication)
print(report)

# Compact diagnostic: basis definition and null-bootstrap reference distribution.
BLUE, RED, GREY = "#2F6DAE", "#C43C39", "#5B6573"
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
axes[0].plot(MONTHS, time_basis_table["linear_month_centered_at_4"], marker="o", ms=8, lw=2, color=GREY, label="Linear month")
axes[0].plot(MONTHS, time_basis_table["scaled_log1p_centered_at_month_4"], marker="s", ms=8, lw=2, color=BLUE, label="Post-hoc log time")
axes[0].axhline(0, color="#BBBBBB", lw=1)
axes[0].set(xlabel="Physical month", ylabel="Random-slope time score", title="One changed estimand component")
axes[0].legend(frameon=False)
usable = posthoc_reps.query("usable and nested_likelihood_ok and LR == LR")
axes[1].hist(usable["LR"], bins=28, color=BLUE, alpha=.82, edgecolor="white")
axes[1].axvline(log_lr, color=RED, lw=2.5, label=f"Observed LR={log_lr:.2f}")
axes[1].set(xlabel="Null-bootstrap LR", ylabel="Replicates", title="Post-hoc Question 1 reference")
axes[1].legend(frameon=False)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(FIGURE_DIR / "Beat_the_Blues_logtime_estimand_diagnostic.pdf", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "Beat_the_Blues_logtime_estimand_diagnostic.png", dpi=300, bbox_inches="tight")
plt.show()

required = [
    RESULTS_DIR / "03_observed_linear_vs_logtime_comparison.csv",
    RESULTS_DIR / "05_frozen_primary_hierarchy.csv",
    RESULTS_DIR / "06_logtime_boundary_bootstrap_replicates.csv",
    RESULTS_DIR / "07_logtime_boundary_bootstrap_summary.csv",
    RESULTS_DIR / "08_time_estimand_adjudication.csv",
    RESULTS_DIR / "09_reporting_paragraph.txt",
]
assert all(path.exists() for path in required)
assert adjudication["frozen_primary_unchanged"].all()
assert log_pair["alt"]["nll"] <= log_pair["null"]["nll"] + NESTED_NLL_TOL
print(f"Complete outputs: {RESULTS_DIR.resolve()}")
