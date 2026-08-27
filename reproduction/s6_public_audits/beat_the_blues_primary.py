"""Reproduce the public Beat the Blues primary random-slope audit."""

from __future__ import annotations

import argparse

import hashlib
import io
import json
import math
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
        description="Reproduce the public Beat the Blues primary audit."
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
            "Run a reduced software check instead of the full analysis; "
            "outputs are not suitable for scientific reporting."
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

ENGINE_VERSION = "btheb_repaired_multistart_v2_2026-08-02"

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

def synthetic_btheb_wide_for_smoke() -> pd.DataFrame:
    rng = np.random.default_rng(20260803)
    n = 30
    treatment = np.where(np.arange(n) % 2 == 0, "TAU", "BtheB")
    drug = np.where(np.arange(n) % 3 == 0, "Yes", "No")
    length = np.where(np.arange(n) % 4 == 0, ">6m", "<6m")
    rows = []
    for i in range(n):
        b0, b1 = rng.multivariate_normal(
            [0, 0], [[18, -0.25], [-0.25, 0.12]]
        )
        values = []
        for month in MONTHS:
            mean = 27 - (
                4.5 if treatment[i] == "BtheB" else 3.0
            ) * np.log1p(month)
            values.append(
                np.clip(
                    mean
                    + b0
                    + b1 * month
                    + rng.normal(0, 5.5 - 0.18 * month),
                    0,
                    63,
                )
            )
        rows.append([drug[i], length[i], treatment[i], *values])
    return pd.DataFrame(
        rows,
        columns=["drug", "length", "treatment", *BDI_COLUMNS],
    )


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
    if list(wide.columns) != expected or wide.shape != (100, 8):
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


wide, provenance = (
    (
        synthetic_btheb_wide_for_smoke(),
        {"source": "schema-faithful synthetic smoke fixture"},
    )
    if SMOKE_TEST
    else download_official_btheb()
)
(RESULTS_DIR / "01_data_provenance.json").write_text(
    json.dumps(provenance, indent=2),
    encoding="utf-8",
)
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

attrition = (
    long_all.groupby(["treatment", "occasion"], observed=True)
    .agg(n_observed=("observed", "sum"))
    .reset_index()
)
arm_n = wide.groupby("treatment", observed=True)["subject"].nunique().rename("arm_n")
attrition = attrition.merge(arm_n, on="treatment", how="left")
attrition["n_missing"] = attrition["arm_n"] - attrition["n_observed"]
attrition["retained_pct"] = 100 * attrition["n_observed"] / attrition["arm_n"]
attrition["month"] = attrition["occasion"].astype(str).astype(float)
attrition = attrition[["treatment", "occasion", "month", "arm_n", "n_observed", "n_missing", "retained_pct"]]

schedule = (
    long.groupby("subject")
    .agg(
        treatment=("treatment", "first"), drug=("drug", "first"), episode_length=("length", "first"),
        n_obs=("bdi", "size"), first_month=("month", "min"), last_month=("month", "max"),
        schedule=("month", lambda x: "-".join(f"{v:g}" for v in x)),
    )
    .reset_index()
)
schedule["followup_months"] = schedule["last_month"] - schedule["first_month"]
schedule["at_least_three"] = schedule["n_obs"] >= 3
schedule["complete_five_wave"] = schedule["n_obs"] == 5

attrition.to_csv(RESULTS_DIR / "02_attrition_by_treatment_and_wave.csv", index=False)
schedule.to_csv(RESULTS_DIR / "03_participant_schedules.csv", index=False)
display(attrition)
display(schedule.groupby(["treatment", "schedule"], observed=True).size().rename("participants").reset_index())

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

observed_pairs = {}
observed_pairs["hom"] = fit_pair(groups, "hom")
observed_pairs["hetero"] = fit_pair(
    groups, "hetero",
    null_seed_fits=[observed_pairs["hom"]["null"]],
    alt_seed_fits=[observed_pairs["hom"]["alt"]],
)
observed_pairs["ar1"] = fit_pair(
    groups, "ar1",
    null_seed_fits=[observed_pairs["hom"]["null"]],
    alt_seed_fits=[observed_pairs["hom"]["alt"]],
)
observed_pairs["hetero_ar1"] = fit_pair(
    groups, "hetero_ar1",
    null_seed_fits=[observed_pairs["hetero"]["null"], observed_pairs["ar1"]["null"]],
    alt_seed_fits=[observed_pairs["hetero"]["alt"], observed_pairs["ar1"]["alt"]],
)

primary_pair = observed_pairs["hom"]
defensible_pair = observed_pairs[DEFENSIBLE_RESIDUAL_MODEL]
primary_fit, primary_null_fit = primary_pair["alt"], primary_pair["null"]
defensible_fit, defensible_null_fit = defensible_pair["alt"], defensible_pair["null"]

nested_rows = []
for model, pair in observed_pairs.items():
    nested_rows.append({
        "constraint": f"{model}: random-slope alternative <= random-intercept null",
        "child_nll": pair["alt"]["nll"], "parent_nll": pair["null"]["nll"],
        "difference_child_minus_parent": pair["alt"]["nll"] - pair["null"]["nll"],
        "pass": pair["nested_ok"],
    })
for child, parents in {"hetero": ["hom"], "ar1": ["hom"], "hetero_ar1": ["hetero", "ar1"]}.items():
    for component in ["null", "alt"]:
        for parent in parents:
            child_nll = observed_pairs[child][component]["nll"]
            parent_nll = observed_pairs[parent][component]["nll"]
            nested_rows.append({
                "constraint": f"{child} {component} <= {parent} {component}",
                "child_nll": child_nll, "parent_nll": parent_nll,
                "difference_child_minus_parent": child_nll - parent_nll,
                "pass": child_nll <= parent_nll + NESTED_NLL_TOL,
            })
nested_audit = pd.DataFrame(nested_rows)
if not nested_audit["pass"].all():
    display(nested_audit)
    raise RuntimeError("At least one required nested-likelihood inequality failed after robust refitting.")

attempt_rows = []
for model, pair in observed_pairs.items():
    attempt_rows.extend(attempt_row(model, "null", fit) for fit in pair["null_bundle"]["attempts"])
    attempt_rows.extend(attempt_row(model, "alternative", fit) for fit in pair["alt_bundle"]["attempts"])
observed_attempts = pd.DataFrame(attempt_rows)
observed_attempts.to_csv(RESULTS_DIR / "04_observed_fit_attempts.csv", index=False)
nested_audit.to_csv(RESULTS_DIR / "05_nested_likelihood_audit.csv", index=False)
display(observed_attempts.sort_values(["model", "component", "nll_reml"]))
display(nested_audit)

ar1_fit = observed_pairs["ar1"]["alt"]
ar1_check = pd.DataFrame([{
    "ar1_nll": ar1_fit["nll"],
    "homogeneous_nll": primary_fit["nll"],
    "ar1_residual_rho": ar1_fit["residual"]["rho"],
    "nll_nested_pass": ar1_fit["nll"] <= primary_fit["nll"] + NESTED_NLL_TOL,
    "real_data_regression_guard_applicable": not SMOKE_TEST,
    "expected_nll_neighborhood_pass": (
        np.nan
        if SMOKE_TEST
        else abs(ar1_fit["nll"] - 1313.016) <= 0.15
    ),
    "expected_rho_neighborhood_pass": (
        np.nan
        if SMOKE_TEST
        else abs(ar1_fit["residual"]["rho"] - 0.2065) <= 0.03
    ),
}])

if SMOKE_TEST:
    # Synthetic smoke data are not expected to reproduce the validated
    # HSAUR3 optimum. Retain the model-nesting validity check, but mark the
    # dataset-specific regression-neighborhood checks as not applicable.
    ar1_check["overall_pass"] = ar1_check["nll_nested_pass"].astype(bool)
else:
    ar1_check["overall_pass"] = ar1_check[
        [
            "nll_nested_pass",
            "expected_nll_neighborhood_pass",
            "expected_rho_neighborhood_pass",
        ]
    ].all(axis=1)

ar1_check.to_csv(
    RESULTS_DIR / "06_corrected_ar1_regression_guard.csv",
    index=False,
)
display(ar1_check)

if not bool(ar1_check["overall_pass"].iloc[0]):
    if SMOKE_TEST:
        raise RuntimeError(
            "Synthetic smoke AR(1) alternative failed the "
            "nested-likelihood validity guard."
        )
    raise RuntimeError(
        "Corrected AR(1) fit did not reproduce the validated "
        "optimum neighborhood."
    )

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


primary_sources = participant_source_table(groups, primary_fit)
defensible_sources = participant_source_table(groups, defensible_fit)
primary_summary = source_summary(groups, primary_fit, "homogeneous_independent")
defensible_summary = source_summary(groups, defensible_fit, "occasion_specific_independent")
primary_sources.to_csv(RESULTS_DIR / "07_participant_source_primary_unrounded.csv", index=False)
defensible_sources.to_csv(RESULTS_DIR / "08_participant_source_defensible_unrounded.csv", index=False)
display(pd.DataFrame([primary_summary, defensible_summary]))

def best_attempt_by_optimizer(bundle: dict, optimizer_name: str) -> dict:
    candidates = [x for x in bundle["attempts"] if x["optimizer"] == optimizer_name and fit_usable(x)]
    if not candidates:
        raise RuntimeError(f"No usable {optimizer_name} attempt")
    return min(candidates, key=lambda x: x["nll"])


primary_by_optimizer = {
    method: best_attempt_by_optimizer(primary_pair["alt_bundle"], method)
    for method in ["L-BFGS-B", "Powell"]
}
optimizer_source = pd.DataFrame([
    source_summary(groups, fit, method) for method, fit in primary_by_optimizer.items()
])
nll_gap_per_obs = abs(primary_by_optimizer["L-BFGS-B"]["nll"] - primary_by_optimizer["Powell"]["nll"]) / N_OBS
source_cols = ["mean_R_direct", "mean_P_full", "mean_Delta", "mean_R_innovation"]
max_source_gap = float(np.max(np.abs(
    optimizer_source.loc[0, source_cols].to_numpy(float) - optimizer_source.loc[1, source_cols].to_numpy(float)
)))
optimizer_gate_pass = bool(
    nll_gap_per_obs <= OPT_NLL_GAP_PER_OBS_TOL
    and max_source_gap <= OPT_SOURCE_ABS_GAP_TOL
    and nested_audit["pass"].all()
    and ar1_check["overall_pass"].iloc[0]
)
optimizer_gate = pd.DataFrame([{
    "nll_gap_per_observation": nll_gap_per_obs,
    "max_source_metric_abs_gap": max_source_gap,
    "all_nesting_constraints_pass": nested_audit["pass"].all(),
    "corrected_ar1_guard_pass": ar1_check["overall_pass"].iloc[0],
    "gate_pass": optimizer_gate_pass,
}])
optimizer_source.to_csv(RESULTS_DIR / "09_optimizer_source_agreement.csv", index=False)
optimizer_gate.to_csv(RESULTS_DIR / "10_optimizer_gate.csv", index=False)
display(optimizer_source)
display(optimizer_gate)

def optimizer_family_gap(pair: dict, component="alt") -> float:
    bundle = pair[f"{component}_bundle"]
    a = best_attempt_by_optimizer(bundle, "L-BFGS-B")
    b = best_attempt_by_optimizer(bundle, "Powell")
    return abs(a["nll"] - b["nll"]) / N_OBS


residual_rows, parameter_rows = [], []
for model, pair in observed_pairs.items():
    alt, null = pair["alt"], pair["null"]
    row = source_summary(groups, alt, model)
    row.update({
        "alt_nll_reml": alt["nll"], "null_nll_reml": null["nll"],
        "slope_heterogeneity_LR": max(0.0, 2 * (null["nll"] - alt["nll"])),
        "alt_le_null": alt["nll"] <= null["nll"] + NESTED_NLL_TOL,
        "optimizer_family_nll_gap_per_obs": optimizer_family_gap(pair),
    })
    rho_r = alt["residual"].get("rho", np.nan)
    row["residual_rho"] = rho_r
    row["cleanly_estimable"] = bool(
        pair["nested_ok"]
        and optimizer_family_gap(pair) <= OPT_NLL_GAP_PER_OBS_TOL
        and (not np.isfinite(rho_r) or abs(rho_r) < AR1_INTERIOR_LIMIT)
    )
    residual_rows.append(row)
    parameter_rows.extend([fit_parameter_row(f"{model}_alternative", alt), fit_parameter_row(f"{model}_null", null)])

residual_sensitivity = pd.DataFrame(residual_rows)
residual_parameters = pd.DataFrame(parameter_rows)
residual_sensitivity.to_csv(RESULTS_DIR / "11_residual_sensitivity_source_and_lr.csv", index=False)
residual_parameters.to_csv(RESULTS_DIR / "12_residual_sensitivity_parameters.csv", index=False)
display(residual_sensitivity)
display(residual_parameters)

def deterministic_rng(stream: int, replicate: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, stream, replicate]))


def simulate_under_null(template, null_fit, residual_model, replicate):
    stream = {"hom": 101, "hetero": 111}[residual_model]
    rng = deterministic_rng(stream, replicate)
    q0, beta = null_fit["G"][0, 0], null_fit["beta"]
    ys = []
    for group in template:
        b0 = rng.normal(0.0, np.sqrt(q0))
        R = residual_covariance(group, null_fit["residual"], residual_model)
        eps = rng.multivariate_normal(np.zeros(len(group.y)), R)
        ys.append(group.X @ beta + b0 + eps)
    return groups_with_new_y(template, ys, prefix=f"null_{residual_model}_{replicate}")


def refit_bootstrap_pair(sim_groups, residual_model, observed_pair, replicate):
    null_starts = [("generating_null", mapped_start_from_fit(observed_pair["null"], False, residual_model))]
    null_bundle = fit_multistart(sim_groups, False, residual_model, null_starts)
    null_fit = null_bundle["best"]
    alt_starts = [("nested_from_replicate_null", mapped_start_from_fit(null_fit, True, residual_model))]
    alt_bundle = fit_multistart(sim_groups, True, residual_model, alt_starts)
    alt_fit = alt_bundle["best"]
    lr = 2 * (null_fit["nll"] - alt_fit["nll"])
    rescue_triggered = replicate in AUDIT_REPLICATES or lr > LR_RESCUE_THRESHOLD or lr < -NESTED_NLL_TOL

    if rescue_triggered:
        null_rescue_starts = [
            ("generating_null", mapped_start_from_fit(observed_pair["null"], False, residual_model)),
            ("fixed_only_boundary", fixed_only_start(sim_groups, False, residual_model)),
            ("alt_implied_null", mapped_start_from_fit(alt_fit, False, residual_model)),
        ]
        null_bundle = fit_multistart(sim_groups, False, residual_model, null_rescue_starts)
        null_fit = null_bundle["best"]
        alt_rescue_starts = [
            ("nested_from_repaired_null", mapped_start_from_fit(null_fit, True, residual_model)),
            ("observed_alternative", mapped_start_from_fit(observed_pair["alt"], True, residual_model)),
        ]
        alt_bundle = fit_multistart(sim_groups, True, residual_model, alt_rescue_starts)
        alt_fit = alt_bundle["best"]
        lr = 2 * (null_fit["nll"] - alt_fit["nll"])

    nested_ok = alt_fit["nll"] <= null_fit["nll"] + NESTED_NLL_TOL
    if replicate in AUDIT_REPLICATES:
        audit = [attempt_row(residual_model, "null", f) for f in null_bundle["attempts"]]
        audit += [attempt_row(residual_model, "alternative", f) for f in alt_bundle["attempts"]]
        audit = pd.DataFrame(audit)
        audit.insert(0, "replicate", replicate)
        audit.to_csv(CHECKPOINT_DIR / f"replicate_{replicate}_{residual_model}_attempt_audit.csv", index=False)

    return {
        "engine_version": ENGINE_VERSION, "residual_model": residual_model, "replicate": replicate,
        "usable": bool(fit_usable(null_fit) and fit_usable(alt_fit) and nested_ok),
        "LR": max(0.0, lr) if nested_ok else np.nan,
        "nested_likelihood_ok": nested_ok, "rescue_triggered": rescue_triggered,
        "null_nll": null_fit["nll"], "alt_nll": alt_fit["nll"],
        "null_optimizer": null_fit["optimizer"], "alt_optimizer": alt_fit["optimizer"],
        "null_start": null_fit.get("start_name"), "alt_start": alt_fit.get("start_name"),
    }


def bootstrap_path(residual_model):
    return CHECKPOINT_DIR / f"heterogeneity_null_bootstrap_{residual_model}.csv"


def run_boundary_bootstrap_to(residual_model, target_b):
    path = bootstrap_path(residual_model)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(existing) and set(existing["engine_version"].astype(str)) != {ENGINE_VERSION}:
        raise RuntimeError(f"Checkpoint engine mismatch at {path}; move the old checkpoint before rerunning.")
    done = set(existing["replicate"].astype(int)) if len(existing) else set()
    rows = existing.to_dict("records")
    todo = [b for b in range(1, target_b + 1) if b not in done]
    observed_pair = observed_pairs[residual_model]
    for j, b in enumerate(tqdm(todo, desc=f"{residual_model} null bootstrap to B={target_b}", unit="rep"), start=1):
        sim_groups = simulate_under_null(groups, observed_pair["null"], residual_model, b)
        rows.append(refit_bootstrap_pair(sim_groups, residual_model, observed_pair, b))
        if j % CHECKPOINT_EVERY == 0 or j == len(todo):
            pd.DataFrame(rows).sort_values("replicate").to_csv(path, index=False)
    return pd.DataFrame(rows).sort_values("replicate").query("replicate <= @target_b").reset_index(drop=True)


def exact_binomial_interval(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def observed_lr(residual_model):
    pair = observed_pairs[residual_model]
    return max(0.0, 2 * (pair["null"]["nll"] - pair["alt"]["nll"]))


def summarize_boundary_bootstrap(reps, residual_model):
    usable = reps[reps["usable"].astype(bool) & reps["nested_likelihood_ok"].astype(bool) & reps["LR"].notna()]
    obs = observed_lr(residual_model)
    k, n = int((usable["LR"] >= obs - 1e-12).sum()), len(usable)
    lo, hi = exact_binomial_interval(k, n, ALPHA)
    return {
        "residual_model": residual_model, "requested_replicates": len(reps), "usable_replicates": n,
        "usable_rate": n / len(reps), "nested_violations": int((~reps["nested_likelihood_ok"].astype(bool)).sum()),
        "rescue_refits": int(reps["rescue_triggered"].astype(bool).sum()), "exceedances": k,
        "p_plus_one": (k + 1) / (n + 1), "mc_interval_low": lo, "mc_interval_high": hi,
        "mc_interval_crosses_0.05": lo <= ALPHA <= hi, "observed_LR": obs,
        "bootstrap_usability_gate_pass": n / len(reps) >= MIN_USABLE_RATE,
    }

boundary_replicates, boundary_summaries = {}, []
for residual_model in ["hom", "hetero"]:
    path = bootstrap_path(residual_model)
    if RUN_BOUNDARY_BOOTSTRAP:
        reps = run_boundary_bootstrap_to(residual_model, PROVISIONAL_NULL_B)
        provisional = summarize_boundary_bootstrap(reps, residual_model)
        if provisional["mc_interval_crosses_0.05"] and FINAL_NULL_B > PROVISIONAL_NULL_B:
            print(f"{residual_model}: repaired Monte Carlo interval crosses .05; extending to {FINAL_NULL_B}.")
            reps = run_boundary_bootstrap_to(residual_model, FINAL_NULL_B)
    else:
        if not path.exists():
            raise FileNotFoundError(f"RUN_BOUNDARY_BOOTSTRAP=False but {path} is absent")
        reps = pd.read_csv(path)
    summary = summarize_boundary_bootstrap(reps, residual_model)
    if summary["mc_interval_low"] <= ALPHA <= summary["mc_interval_high"]:
        status = "borderline"
    elif summary["p_plus_one"] < ALPHA:
        status = "supported"
    else:
        status = "unsupported by the present data"
    summary["heterogeneity_status"] = status
    boundary_replicates[residual_model] = reps
    boundary_summaries.append(summary)

boundary_summary = pd.DataFrame(boundary_summaries)
boundary_summary.to_csv(RESULTS_DIR / "13_boundary_bootstrap_summary.csv", index=False)
for model, reps in boundary_replicates.items():
    reps.to_csv(RESULTS_DIR / f"14_boundary_bootstrap_replicates_{model}.csv", index=False)
display(boundary_summary)

if not SMOKE_TEST:
    rep862 = boundary_replicates["hom"].query("replicate == 862")
    if len(rep862) != 1 or not bool(rep862["usable"].iloc[0]):
        raise RuntimeError("Repaired replicate 862 audit is missing or unusable.")
    replicate_862_guard = pd.DataFrame([{
        "replicate": 862,
        "archived_invalid_LR": 183.998917,
        "archived_invalid_null_nll": 1413.881313,
        "repaired_LR": rep862["LR"].iloc[0],
        "repaired_null_nll": rep862["null_nll"].iloc[0],
        "repaired_alt_nll": rep862["alt_nll"].iloc[0],
        "nested_likelihood_ok": rep862["nested_likelihood_ok"].iloc[0],
        "gross_outlier_repaired": bool(rep862["LR"].iloc[0] < LR_RESCUE_THRESHOLD),
    }])
    replicate_862_guard.to_csv(RESULTS_DIR / "14b_replicate_862_repair_guard.csv", index=False)
    if not bool(replicate_862_guard["gross_outlier_repaired"].iloc[0]):
        raise RuntimeError("Replicate 862 remains a gross LR outlier after exhaustive refitting.")
    display(replicate_862_guard)
    display(rep862)

def residual_diagnostic_table(groups: Sequence[GroupData], fit: dict, model_label: str) -> pd.DataFrame:
    rows = []
    for group in groups:
        marginal = group.y - group.X @ fit["beta"]
        R = residual_covariance(group, fit["residual"], fit["residual_model"])
        V = group.Z @ fit["G"] @ group.Z.T + R
        L = np.linalg.cholesky(V)
        whitened = linalg.solve_triangular(L, marginal, lower=True)
        b_hat = fit["G"] @ group.Z.T @ np.linalg.solve(V, marginal)
        conditional = marginal - group.Z @ b_hat
        conditional_std = conditional / np.sqrt(np.diag(R))
        marginal_std = marginal / np.sqrt(np.diag(V))
        fitted_conditional = group.X @ fit["beta"] + group.Z @ b_hat
        cell_indices = np.argmax(group.X[:, :len(CELL_COLUMNS)], axis=1)
        for j in range(len(group.y)):
            rows.append({
                "model": model_label, "subject": group.subject, "month": group.month[j],
                "occasion": OCCASIONS[group.occasion_index[j]], "cell": CELL_COLUMNS[cell_indices[j]],
                "observed_bdi": group.y[j], "fixed_mean": group.X[j] @ fit["beta"],
                "conditional_fitted": fitted_conditional[j], "marginal_residual": marginal[j],
                "marginal_standardized": marginal_std[j], "whitened_marginal_residual": whitened[j],
                "conditional_residual": conditional[j], "conditional_standardized": conditional_std[j],
                "eb_intercept": b_hat[0], "eb_slope": b_hat[1],
            })
    return pd.DataFrame(rows)


residual_observations = pd.concat([
    residual_diagnostic_table(groups, primary_fit, "hom"),
    residual_diagnostic_table(groups, defensible_fit, DEFENSIBLE_RESIDUAL_MODEL),
], ignore_index=True)

residual_global = (
    residual_observations.groupby("model")
    .agg(
        n=("whitened_marginal_residual", "size"),
        whitened_mean=("whitened_marginal_residual", "mean"),
        whitened_sd=("whitened_marginal_residual", "std"),
        whitened_skew=("whitened_marginal_residual", stats.skew),
        whitened_excess_kurtosis=("whitened_marginal_residual", stats.kurtosis),
        proportion_abs_whitened_gt2=("whitened_marginal_residual", lambda x: np.mean(np.abs(x) > 2)),
        proportion_abs_whitened_gt3=("whitened_marginal_residual", lambda x: np.mean(np.abs(x) > 3)),
    )
    .reset_index()
)
residual_by_occasion = (
    residual_observations.groupby(["model", "month"], observed=True)
    .agg(
        n=("conditional_residual", "size"),
        conditional_residual_mean=("conditional_residual", "mean"),
        conditional_residual_sd=("conditional_residual", "std"),
        conditional_standardized_mean=("conditional_standardized", "mean"),
        conditional_standardized_sd=("conditional_standardized", "std"),
        marginal_standardized_mean=("marginal_standardized", "mean"),
        marginal_standardized_sd=("marginal_standardized", "std"),
    )
    .reset_index()
)
residual_observations.to_csv(RESULTS_DIR / "15_observation_level_residual_diagnostics.csv", index=False)
residual_global.to_csv(RESULTS_DIR / "16_global_residual_diagnostics.csv", index=False)
residual_by_occasion.to_csv(RESULTS_DIR / "17_residual_diagnostics_by_occasion.csv", index=False)
display(residual_global)
display(residual_by_occasion)

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
for model, color in [("hom", "#5B6B7A"), (DEFENSIBLE_RESIDUAL_MODEL, "#2F6DAE")]:
    values = residual_observations.query("model == @model")["whitened_marginal_residual"].to_numpy()
    osm, osr = stats.probplot(values, dist="norm", fit=False)
    axes[0, 0].scatter(osm, osr, s=22, alpha=.65, color=color, label=model)
    d = residual_by_occasion.query("model == @model")
    axes[0, 1].plot(d["month"], d["conditional_residual_sd"], marker="o", ms=8, lw=2, color=color, label=model)
    axes[1, 0].plot(d["month"], d["conditional_standardized_sd"], marker="o", ms=8, lw=2, color=color, label=model)
axes[0, 0].set(xlabel="Normal quantile", ylabel="Whitened marginal residual", title="Marginal residual Q–Q")
axes[0, 1].set(xlabel="Month", ylabel="Conditional residual SD", title="Raw conditional residual scale", xticks=MONTHS)
axes[1, 0].axhline(1, color="#777777", lw=1, ls="--")
axes[1, 0].set(xlabel="Month", ylabel="Conditional standardized SD", title="Residual scale after standardization", xticks=MONTHS)

mean_fit_rows = []
for j, cell in enumerate(CELL_COLUMNS):
    trt, occ = cell.split("__")
    obs = long[(long["treatment"].astype(str) == trt) & (long["occasion"].astype(str) == occ)]["bdi"]
    mean_fit_rows.append({
        "cell": cell, "treatment": trt, "month": float(occ), "n": len(obs),
        "observed_mean": obs.mean(), "fitted_mean_hom": primary_fit["beta"][j],
        "fitted_mean_hetero": defensible_fit["beta"][j],
    })
mean_fit = pd.DataFrame(mean_fit_rows)
mean_fit["observed_minus_fitted_hom"] = mean_fit["observed_mean"] - mean_fit["fitted_mean_hom"]
mean_fit["observed_minus_fitted_hetero"] = mean_fit["observed_mean"] - mean_fit["fitted_mean_hetero"]
mean_fit.to_csv(RESULTS_DIR / "18_observed_vs_fitted_cell_means.csv", index=False)
for trt, color in [("TAU", "#5B6B7A"), ("BtheB", "#2F6DAE")]:
    d = mean_fit.query("treatment == @trt")
    axes[1, 1].scatter(d["month"], d["observed_mean"], s=65, facecolor="white", edgecolor=color, linewidth=2, label=f"{trt} observed")
    axes[1, 1].plot(d["month"], d["fitted_mean_hetero"], color=color, marker="o", ms=7, lw=2, label=f"{trt} fitted")
axes[1, 1].set(xlabel="Month", ylabel="BDI-II", title="Observed and fitted cell means", xticks=MONTHS)
for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "residual_and_cell_mean_diagnostics.png", dpi=300, bbox_inches="tight")
plt.show()
display(mean_fit)

def trajectory_shape_table(groups: Sequence[GroupData], ys: Sequence[np.ndarray] | None = None) -> pd.DataFrame:
    rows = []
    if ys is None:
        ys = [g.y for g in groups]
    for group, y in zip(groups, ys):
        y = np.asarray(y, float)
        if len(y) < 3:
            continue
        design = np.column_stack([np.ones(len(y)), group.month - CENTER_MONTH])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        deviation = y - design @ coef
        changes = np.diff(y)
        nz_sign = np.sign(changes[np.abs(changes) > 1e-12])
        direction_reversals = int(np.sum(nz_sign[1:] != nz_sign[:-1])) if len(nz_sign) > 1 else 0
        late_rebound = float(y[-1] - np.min(y[:-1])) if len(y) > 1 else np.nan
        rows.append({
            "subject": group.subject, "n_obs": len(y), "schedule": "-".join(f"{v:g}" for v in group.month),
            "best_linear_slope_per_month": coef[1], "linear_rmse": np.sqrt(np.mean(deviation ** 2)),
            "max_abs_linear_deviation": np.max(np.abs(deviation)), "direction_reversals": direction_reversals,
            "any_direction_reversal": direction_reversals > 0, "late_rebound_points": late_rebound,
            "late_rebound_ge5": late_rebound >= 5,
        })
    return pd.DataFrame(rows)


def predictive_statistics(groups: Sequence[GroupData], fit: dict, ys: Sequence[np.ndarray] | None = None) -> dict:
    if ys is None:
        ys = [g.y for g in groups]
    cell_values = {name: [] for name in CELL_COLUMNS}
    month_fixed_residuals = {float(m): [] for m in MONTHS}
    cell_gaps = []
    for group, y in zip(groups, ys):
        y = np.asarray(y, float)
        fixed = group.X @ fit["beta"]
        cell_indices = np.argmax(group.X[:, :len(CELL_COLUMNS)], axis=1)
        for j, value in enumerate(y):
            cell_values[CELL_COLUMNS[cell_indices[j]]].append(float(value))
            month_fixed_residuals[float(group.month[j])].append(float(value - fixed[j]))
    stats_out = {}
    for j, name in enumerate(CELL_COLUMNS):
        values = np.asarray(cell_values[name])
        stats_out[f"cell_mean__{name}"] = values.mean()
        stats_out[f"cell_sd__{name}"] = values.std(ddof=1)
        cell_gaps.append(abs(values.mean() - fit["beta"][j]))
    variances = []
    for month in MONTHS:
        values = np.asarray(month_fixed_residuals[float(month)])
        variance = values.var(ddof=1)
        stats_out[f"fixed_residual_variance__month_{month:g}"] = variance
        variances.append(variance)
    stats_out["core__max_abs_cell_mean_gap"] = max(cell_gaps)
    stats_out["core__fixed_residual_variance_linear_trend"] = np.polyfit(MONTHS, variances, 1)[0]
    shape = trajectory_shape_table(groups, ys)
    stats_out["core__trajectory_median_linear_rmse"] = shape["linear_rmse"].median()
    stats_out["core__trajectory_q90_linear_rmse"] = shape["linear_rmse"].quantile(.90)
    stats_out["core__trajectory_any_reversal_proportion"] = shape["any_direction_reversal"].mean()
    stats_out["core__trajectory_late_rebound_ge5_proportion"] = shape["late_rebound_ge5"].mean()
    return stats_out


def simulate_predictive_ys(groups, fit, replicate, model_stream):
    rng = deterministic_rng(model_stream, replicate)
    ys = []
    for group in groups:
        b = rng.multivariate_normal(np.zeros(2), fit["G"])
        R = residual_covariance(group, fit["residual"], fit["residual_model"])
        eps = rng.multivariate_normal(np.zeros(len(group.y)), R)
        ys.append(group.X @ fit["beta"] + group.Z @ b + eps)
    return ys


def run_predictive_check(groups, fit, model_label, B):
    observed = predictive_statistics(groups, fit)
    rows = []
    stream = {"hom": 301, "hetero": 311}[model_label]
    for b in tqdm(range(1, B + 1), desc=f"Observed-schedule predictive check: {model_label}", unit="draw"):
        ys = simulate_predictive_ys(groups, fit, b, stream)
        rows.append({"replicate": b, **predictive_statistics(groups, fit, ys)})
    draws = pd.DataFrame(rows)
    summary_rows = []
    for quantity, obs in observed.items():
        values = draws[quantity].to_numpy(float)
        lower_tail = (1 + np.sum(values <= obs)) / (len(values) + 1)
        upper_tail = (1 + np.sum(values >= obs)) / (len(values) + 1)
        qlo, qhi = np.quantile(values, [.025, .975])
        summary_rows.append({
            "model": model_label, "quantity": quantity, "observed": obs,
            "predictive_median": np.median(values), "predictive_2.5pct": qlo, "predictive_97.5pct": qhi,
            "two_sided_predictive_tail_probability": min(1.0, 2 * min(lower_tail, upper_tail)),
            "outside_95pct_predictive_interval": bool(obs < qlo or obs > qhi),
            "prespecified_core_diagnostic": quantity.startswith("core__"),
        })
    return draws, pd.DataFrame(summary_rows)


observed_trajectory_shapes = trajectory_shape_table(groups)
observed_trajectory_shapes.to_csv(RESULTS_DIR / "19_individual_trajectory_shape_diagnostics.csv", index=False)

ppc_draws, ppc_summaries = {}, []
for model_label, fit in [("hom", primary_fit), ("hetero", defensible_fit)]:
    draws, summary = run_predictive_check(groups, fit, model_label, PREDICTIVE_CHECK_B)
    ppc_draws[model_label] = draws
    ppc_summaries.append(summary)
    draws.to_csv(RESULTS_DIR / f"20_predictive_check_draws_{model_label}.csv", index=False)
predictive_check_summary = pd.concat(ppc_summaries, ignore_index=True)
predictive_check_summary.to_csv(RESULTS_DIR / "21_predictive_check_summary.csv", index=False)

defensible_core = predictive_check_summary.query(
    "model == @DEFENSIBLE_RESIDUAL_MODEL and prespecified_core_diagnostic"
)
adequacy_flag_count = int(defensible_core["outside_95pct_predictive_interval"].sum())
if adequacy_flag_count == 0:
    model_adequacy_status = "no prespecified core discrepancy outside the 95% predictive interval"
elif adequacy_flag_count == 1:
    model_adequacy_status = "1 prespecified core discrepancy outside the 95% predictive interval"
else:
    model_adequacy_status = (
        f"{adequacy_flag_count} prespecified core discrepancies outside the 95% predictive interval"
    )
model_adequacy_gate_pass = adequacy_flag_count == 0
flagged_defensible_core = defensible_core.loc[
    defensible_core["outside_95pct_predictive_interval"].astype(bool)
]
if len(flagged_defensible_core) != adequacy_flag_count:
    raise RuntimeError("Prespecified model-adequacy flag count is internally inconsistent.")


def format_core_diagnostic(row) -> str:
    label = str(row["quantity"]).removeprefix("core__").replace("_", " ")
    return (
        f"{label} (observed={row['observed']:.6g}, "
        f"predictive median={row['predictive_median']:.6g}, "
        "predictive 95% interval "
        f"[{row['predictive_2.5pct']:.6g}, {row['predictive_97.5pct']:.6g}], "
        "two-sided predictive tail probability="
        f"{row['two_sided_predictive_tail_probability']:.6g})"
    )


flagged_core_diagnostic_details = "; ".join(
    format_core_diagnostic(row) for _, row in flagged_defensible_core.iterrows()
)
model_adequacy_evidence = model_adequacy_status
if flagged_core_diagnostic_details:
    model_adequacy_evidence += f": {flagged_core_diagnostic_details}"
display(predictive_check_summary.query("prespecified_core_diagnostic"))
print("Observed-schedule adequacy status:", model_adequacy_status)

core_names = defensible_core["quantity"].tolist()
fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
for ax, quantity in zip(axes.flat, core_names):
    for model, color, y in [("hom", "#5B6B7A", 0), ("hetero", "#2F6DAE", 1)]:
        values = ppc_draws[model][quantity].to_numpy(float)
        observed = predictive_check_summary.query("model == @model and quantity == @quantity")["observed"].iloc[0]
        ax.violinplot(values, positions=[y], vert=False, widths=.7, showextrema=False)
        ax.scatter(observed, y, s=80, color=color, edgecolor="white", linewidth=1.2, zorder=3)
    ax.set(yticks=[0, 1], yticklabels=["hom", "hetero"], title=quantity.replace("core__", "").replace("_", " "))
    ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "observed_schedule_predictive_checks.png", dpi=300, bbox_inches="tight")
plt.show()

def sensitivity_case(label, frame, fixed_covariates=(), selected_sample_note=""):
    case_groups, case_fixed_names = build_groups(frame, fixed_covariates=fixed_covariates)
    if label == "full_sample_base":
        pair = defensible_pair
    else:
        pair = fit_pair(
            case_groups, DEFENSIBLE_RESIDUAL_MODEL,
            null_seed_fits=[defensible_pair["null"]], alt_seed_fits=[defensible_pair["alt"]],
        )
    summary = source_summary(case_groups, pair["alt"], label)
    summary.update({
        "fixed_covariates": ",".join(fixed_covariates) if fixed_covariates else "none",
        "selected_sample_note": selected_sample_note,
        "alt_nll_reml": pair["alt"]["nll"], "null_nll_reml": pair["null"]["nll"],
        "observed_LR_descriptive": max(0.0, 2 * (pair["null"]["nll"] - pair["alt"]["nll"])),
        "nested_likelihood_ok": pair["nested_ok"],
    })
    fixed = pd.DataFrame({
        "case": label, "fixed_effect": case_fixed_names, "estimate": pair["alt"]["beta"],
        "se": np.sqrt(np.diag(pair["alt"]["cov_beta"])),
    })
    return summary, fixed, pair


n_ge_3_ids = set(schedule.loc[schedule["at_least_three"], "subject"].astype(str))
complete_ids = set(schedule.loc[schedule["complete_five_wave"], "subject"].astype(str))

case_specs = [
    ("full_sample_base", long, (), "all available observations"),
    ("full_sample_prognostic_covariates", long, ("drug_yes", "episode_gt6m"),
     "baseline antidepressant use and episode length added as prognostic main effects"),
    ("at_least_three_measurements", long[long["subject"].astype(str).isin(n_ge_3_ids)].copy(), (),
     "participants with at least three observed occasions"),
    ("complete_five_wave_selected_sample", long[long["subject"].astype(str).isin(complete_ids)].copy(), (),
     "selected-sample evidence: all five scheduled BDI-II measurements observed"),
]
sensitivity_rows, sensitivity_fixed, sensitivity_pairs = [], [], {}
for label, frame, covariates, note in case_specs:
    row, fixed, pair = sensitivity_case(label, frame, covariates, note)
    sensitivity_rows.append(row)
    sensitivity_fixed.append(fixed)
    sensitivity_pairs[label] = pair
schedule_sensitivity = pd.DataFrame(sensitivity_rows)
schedule_sensitivity_fixed = pd.concat(sensitivity_fixed, ignore_index=True)
schedule_sensitivity.to_csv(RESULTS_DIR / "22_missingness_schedule_sensitivity.csv", index=False)
schedule_sensitivity_fixed.to_csv(RESULTS_DIR / "23_missingness_schedule_fixed_effects.csv", index=False)
display(schedule_sensitivity)
display(schedule_sensitivity_fixed.query("fixed_effect in ['drug_yes', 'episode_gt6m']"))

source_with_schedule = defensible_sources.merge(
    schedule[["subject", "treatment", "drug", "episode_length", "n_obs", "schedule", "complete_five_wave"]],
    on=["subject", "n_obs"], how="left", validate="one_to_one",
)
source_by_schedule = (
    source_with_schedule.groupby(["schedule", "n_obs"], observed=True)
    .agg(
        participants=("subject", "size"),
        mean_R_direct=("R_direct", "mean"), median_R_direct=("R_direct", "median"),
        mean_P_full=("P_full", "mean"), median_P_full=("P_full", "median"),
        mean_Delta=("Delta", "mean"), median_Delta=("Delta", "median"),
        mean_R_innovation=("R_innovation", "mean"), median_R_innovation=("R_innovation", "median"),
        mean_effective_slope_error_variance=("effective_slope_error_variance", "mean"),
    )
    .reset_index()
)
source_with_schedule.to_csv(RESULTS_DIR / "24_participant_source_with_schedule.csv", index=False)
source_by_schedule.to_csv(RESULTS_DIR / "25_source_quantities_by_observation_schedule.csv", index=False)
display(source_by_schedule)

INFLUENCE_PATH = CHECKPOINT_DIR / "leave_one_out_hetero.csv"
DEF_SOURCE_NAMES = ["mean_R_direct", "mean_P_full", "mean_Delta", "mean_R_innovation"]
DEF_FULL_VECTOR = np.array([defensible_summary[x] for x in DEF_SOURCE_NAMES])
DEF_OBS_LR = observed_lr(DEFENSIBLE_RESIDUAL_MODEL)


def run_one_influence(subject):
    reduced = [g for g in groups if g.subject != str(subject)]
    pair = fit_pair(
        reduced, DEFENSIBLE_RESIDUAL_MODEL,
        null_seed_fits=[defensible_pair["null"]], alt_seed_fits=[defensible_pair["alt"]],
    )
    fit = pair["alt"]
    summary = source_summary(reduced, fit, "leave_one_out")
    values = np.array([summary[name] for name in DEF_SOURCE_NAMES])
    shifts = values - DEF_FULL_VECTOR
    lr = max(0.0, 2 * (pair["null"]["nll"] - pair["alt"]["nll"]))
    row = {
        "engine_version": ENGINE_VERSION, "omitted_subject": str(subject), "usable": pair["nested_ok"],
        "optimizer": fit["optimizer"], "nll": fit["nll"], "observed_LR": lr,
        "shift_observed_LR": lr - DEF_OBS_LR,
    }
    row.update({name: summary[name] for name in DEF_SOURCE_NAMES})
    row.update({f"shift_{name}": shift for name, shift in zip(DEF_SOURCE_NAMES, shifts)})
    row["max_abs_source_shift"] = float(np.max(np.abs(shifts)))
    row["euclidean_source_shift"] = float(np.linalg.norm(shifts))
    return row


def run_leave_one_out():
    existing = pd.read_csv(INFLUENCE_PATH, dtype={"omitted_subject": str}) if INFLUENCE_PATH.exists() else pd.DataFrame()
    if len(existing) and set(existing["engine_version"].astype(str)) != {ENGINE_VERSION}:
        raise RuntimeError("Influence checkpoint belongs to another engine version.")
    done = set(existing["omitted_subject"].astype(str)) if len(existing) else set()
    requested = [g.subject for g in groups]
    if SMOKE_TEST:
        requested = sorted(set(requested[:6] + ["11", "83"]))
    todo = [subject for subject in requested if subject not in done]
    rows = existing.to_dict("records")
    for j, subject in enumerate(tqdm(todo, desc="Exact leave-one-out under occasion-specific residuals", unit="participant"), start=1):
        rows.append(run_one_influence(subject))
        if j % CHECKPOINT_EVERY == 0 or j == len(todo):
            pd.DataFrame(rows).to_csv(INFLUENCE_PATH, index=False)
    return pd.DataFrame(rows)


if RUN_LEAVE_ONE_OUT:
    influence = run_leave_one_out()
else:
    if not INFLUENCE_PATH.exists():
        raise FileNotFoundError(INFLUENCE_PATH)
    influence = pd.read_csv(INFLUENCE_PATH, dtype={"omitted_subject": str})

influence_usable = influence[influence["usable"].astype(bool)].copy()
top_two = influence_usable.nlargest(2, "euclidean_source_shift")["omitted_subject"].astype(str).tolist()
joint_groups = [g for g in groups if g.subject not in set(top_two)]
joint_pair = fit_pair(
    joint_groups, DEFENSIBLE_RESIDUAL_MODEL,
    null_seed_fits=[defensible_pair["null"]], alt_seed_fits=[defensible_pair["alt"]],
)
joint_summary = source_summary(joint_groups, joint_pair["alt"], "joint_top_two_deletion")
joint_values = np.array([joint_summary[name] for name in DEF_SOURCE_NAMES])
joint_shifts = joint_values - DEF_FULL_VECTOR
joint_lr = max(0.0, 2 * (joint_pair["null"]["nll"] - joint_pair["alt"]["nll"]))

influence_summary = pd.DataFrame([{
    "residual_model": DEFENSIBLE_RESIDUAL_MODEL,
    "requested_single_deletions": len(groups) if not SMOKE_TEST else len(influence_usable),
    "usable_single_deletions": len(influence_usable),
    "max_single_deletion_abs_source_shift": influence_usable["max_abs_source_shift"].max(),
    "max_single_deletion_abs_LR_shift": influence_usable["shift_observed_LR"].abs().max(),
    "joint_top_two_subjects": ",".join(top_two),
    "joint_top_two_max_abs_source_shift": np.max(np.abs(joint_shifts)),
    "joint_top_two_LR": joint_lr, "joint_top_two_LR_shift": joint_lr - DEF_OBS_LR,
    "participants_retained_in_primary_analysis": True,
}])
joint_row = {**joint_summary, **{f"shift_{name}": shift for name, shift in zip(DEF_SOURCE_NAMES, joint_shifts)}}
joint_row.update({"subjects_deleted_for_sensitivity_only": ",".join(top_two), "observed_LR": joint_lr, "shift_observed_LR": joint_lr - DEF_OBS_LR})

influence.to_csv(RESULTS_DIR / "26_leave_one_out_hetero.csv", index=False)
influence_summary.to_csv(RESULTS_DIR / "27_influence_summary_hetero.csv", index=False)
pd.DataFrame([joint_row]).to_csv(RESULTS_DIR / "28_joint_top_two_deletion_hetero.csv", index=False)
display(influence_summary)
display(influence_usable.nlargest(10, "euclidean_source_shift"))

highlight_subjects = sorted(set(top_two + ["11", "83"]))
highlight_data = long[long["subject"].astype(str).isin(highlight_subjects)].copy()
highlight_data.to_csv(RESULTS_DIR / "29_influential_participant_trajectories.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
palette = ["#2F6DAE", "#D9912B", "#6C5AA7", "#B64242"]
for ax, trt in zip(axes, TREATMENTS):
    background = long[long["treatment"].astype(str) == trt]
    for _, d in background.groupby("subject"):
        ax.plot(d["month"], d["bdi"], color="#C7CDD2", lw=.8, alpha=.35)
    for color, subject in zip(palette, highlight_subjects):
        d = highlight_data[(highlight_data["treatment"].astype(str) == trt) & (highlight_data["subject"].astype(str) == subject)]
        if len(d):
            ax.plot(d["month"], d["bdi"], color=color, marker="o", ms=7, lw=2.4, label=f"Participant {subject}")
    fitted = mean_fit.query("treatment == @trt")
    ax.plot(fitted["month"], fitted["fitted_mean_hetero"], color="#111111", lw=2.5, ls="--", label="Fitted treatment mean")
    ax.set(title=trt, xlabel="Month", xticks=MONTHS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("BDI-II")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "influential_participant_trajectories_retained.png", dpi=300, bbox_inches="tight")
plt.show()

PARTICIPANT_BOOT_PATH = CHECKPOINT_DIR / "participant_bootstrap_hetero.csv"


def run_one_participant_bootstrap(replicate):
    rng = deterministic_rng(402, replicate)
    draw = rng.integers(0, len(groups), size=len(groups))
    sampled = []
    for j, original_index in enumerate(draw):
        g = groups[int(original_index)]
        sampled.append(GroupData(f"draw{j:03d}_source{g.subject}", g.y, g.X, g.Z, g.occasion_index, g.month))
    bundle = fit_multistart(
        sampled, True, DEFENSIBLE_RESIDUAL_MODEL,
        [("full_sample_defensible", mapped_start_from_fit(defensible_fit, True, DEFENSIBLE_RESIDUAL_MODEL))],
    )
    fit = bundle["best"]
    summary = source_summary(sampled, fit, "participant_bootstrap")
    row = {
        "engine_version": ENGINE_VERSION, "replicate": replicate, "usable": fit_usable(fit),
        "optimizer": fit["optimizer"], "start_name": fit.get("start_name"),
    }
    row.update({name: summary[name] for name in DEF_SOURCE_NAMES})
    row.update({"rho": summary["rho"], "var_slope": summary["var_slope"]})
    return row


def run_participant_bootstrap(target_b):
    existing = pd.read_csv(PARTICIPANT_BOOT_PATH) if PARTICIPANT_BOOT_PATH.exists() else pd.DataFrame()
    if len(existing) and set(existing["engine_version"].astype(str)) != {ENGINE_VERSION}:
        raise RuntimeError("Participant-bootstrap checkpoint belongs to another engine version.")
    done = set(existing["replicate"].astype(int)) if len(existing) else set()
    rows = existing.to_dict("records")
    todo = [b for b in range(1, target_b + 1) if b not in done]
    for j, b in enumerate(tqdm(todo, desc=f"Participant bootstrap B={target_b}", unit="resample"), start=1):
        rows.append(run_one_participant_bootstrap(b))
        if j % CHECKPOINT_EVERY == 0 or j == len(todo):
            pd.DataFrame(rows).sort_values("replicate").to_csv(PARTICIPANT_BOOT_PATH, index=False)
    return pd.DataFrame(rows).sort_values("replicate").query("replicate <= @target_b").reset_index(drop=True)


if RUN_PARTICIPANT_BOOTSTRAP:
    participant_boot = run_participant_bootstrap(PARTICIPANT_BOOT_B)
else:
    if not PARTICIPANT_BOOT_PATH.exists():
        raise FileNotFoundError(PARTICIPANT_BOOT_PATH)
    participant_boot = pd.read_csv(PARTICIPANT_BOOT_PATH)

participant_boot_usable = participant_boot[participant_boot["usable"].astype(bool)].copy()
boot_summary_rows = []
for name in DEF_SOURCE_NAMES + ["rho", "var_slope"]:
    values = participant_boot_usable[name].dropna().to_numpy(float)
    full_value = defensible_summary.get(name, np.nan)
    boot_summary_rows.append({
        "quantity": name, "full_sample": full_value, "usable_replicates": len(values),
        "bootstrap_median": np.median(values), "bootstrap_2.5pct": np.quantile(values, .025),
        "bootstrap_97.5pct": np.quantile(values, .975),
    })
participant_boot_summary = pd.DataFrame(boot_summary_rows)
participant_boot.to_csv(RESULTS_DIR / "30_participant_bootstrap_hetero.csv", index=False)
participant_boot_summary.to_csv(RESULTS_DIR / "31_participant_bootstrap_intervals_hetero.csv", index=False)
display(participant_boot_summary)

bootstrap_usability_pass = bool(boundary_summary["bootstrap_usability_gate_pass"].all())
missingness_schedule_completed = bool(
    set(schedule_sensitivity["model"]) == {
        "full_sample_base", "full_sample_prognostic_covariates",
        "at_least_three_measurements", "complete_five_wave_selected_sample",
    }
    and len(source_by_schedule) >= 1
)
influence_completed = bool(
    len(influence_usable) == (len(groups) if not SMOKE_TEST else len(influence_usable))
    and bool(influence_summary["participants_retained_in_primary_analysis"].iloc[0])
)
participant_boot_completed = len(participant_boot_usable) / len(participant_boot) >= MIN_USABLE_RATE
reproduction_components_complete = bool(
    missingness_schedule_completed
    and influence_completed
    and participant_boot_completed
)

if SMOKE_TEST:
    reproduction_completion_status = (
        "SMOKE_COMPLETE_NONREPORTABLE"
        if reproduction_components_complete else
        "SMOKE_INCOMPLETE"
    )
else:
    reproduction_completion_status = (
        "COMPLETE" if reproduction_components_complete else "INCOMPLETE"
    )

gate = pd.DataFrame([
    {
        "category": "execution_validity",
        "status": "PASS" if optimizer_gate_pass else "FAIL",
        "release_blocker": not optimizer_gate_pass,
        "evidence": (
            f"NLL gap/obs={nll_gap_per_obs:.3g}; max source gap={max_source_gap:.3g}; "
            f"corrected AR1 NLL={ar1_fit['nll']:.6f}, "
            f"rho={ar1_fit['residual']['rho']:.6f}"
        ),
    },
    {
        "category": "bootstrap_usability",
        "status": "PASS" if bootstrap_usability_pass else "FAIL",
        "release_blocker": not bootstrap_usability_pass,
        "evidence": "; ".join(
            f"{r.residual_model}: {int(r.usable_replicates)}/"
            f"{int(r.requested_replicates)} usable "
            f"(rate={r.usable_rate:.6g}, threshold={MIN_USABLE_RATE:.6g}), "
            f"{int(r.nested_violations)} boundary-floor exclusion(s), "
            f"p={r.p_plus_one:.6g}"
            for r in boundary_summary.itertuples()
        ),
    },
    {
        "category": "scientific_model_adequacy",
        "status": "PASS" if model_adequacy_gate_pass else "WARNING",
        "release_blocker": False,
        "evidence": model_adequacy_evidence,
    },
    {
        "category": "reproduction_completion",
        "status": reproduction_completion_status,
        "release_blocker": not reproduction_components_complete,
        "evidence": (
            "missingness/schedule sensitivities completed="
            f"{missingness_schedule_completed}; influence diagnostics completed="
            f"{influence_completed}; participant bootstrap usable="
            f"{len(participant_boot_usable)}/{len(participant_boot)}; "
            "outcome-independent reporting generated"
        ),
    },
])

blocking_failure = bool(gate["release_blocker"].any())
if SMOKE_TEST:
    overall_reproduction_status = (
        "SMOKE_FAILED" if blocking_failure else "NONREPORTABLE_SMOKE_COMPLETE"
    )
elif blocking_failure:
    overall_reproduction_status = "FAIL"
elif model_adequacy_gate_pass:
    overall_reproduction_status = "PASS"
else:
    overall_reproduction_status = "PASS_WITH_SCIENTIFIC_WARNING"

gate["overall_reproduction_status"] = overall_reproduction_status
gate.to_csv(RESULTS_DIR / "32_repaired_validity_gate.csv", index=False)

hom_pb = boundary_summary.set_index("residual_model").loc["hom"]
hetero_pb = boundary_summary.set_index("residual_model").loc["hetero"]
sens = schedule_sensitivity.set_index("model")
report = (
    f"Beat the Blues contributed {len(groups)} participants and {N_OBS} observed BDI-II scores across months 0, 2, 3, 5, and 8. "
    f"The participant slope was defined as the best linear individual deviation from the treatment-specific mean trajectory over months 0–8, not as each participant's literal linear disease course or an individual causal treatment effect. "
    f"Under homogeneous independent residuals, the repaired heterogeneity statistic was LR={hom_pb.observed_LR:.6g} "
    f"(boundary-bootstrap p={hom_pb.p_plus_one:.6g}, Monte Carlo interval [{hom_pb.mc_interval_low:.6g}, {hom_pb.mc_interval_high:.6g}]). "
    f"With occasion-specific residual variances, LR={hetero_pb.observed_LR:.6g} "
    f"(boundary-bootstrap p={hetero_pb.p_plus_one:.6g}, interval [{hetero_pb.mc_interval_low:.6g}, {hetero_pb.mc_interval_high:.6g}]). "
    f"The corrected common-variance AR(1) fit had REML NLL={ar1_fit['nll']:.6g} and residual rho={ar1_fit['residual']['rho']:.6g}; "
    f"its objective was no worse than the homogeneous model as required by nesting. "
    f"Under the occasion-specific residual model, participant-mean P_full={defensible_summary['mean_P_full']:.6g}="
    f"{defensible_summary['mean_R_direct']:.6g}+{defensible_summary['mean_Delta']:.6g}, and mean innovation reliability={defensible_summary['mean_R_innovation']:.6g}. "
    f"Observed-schedule predictive checks yielded {model_adequacy_evidence}. "
    f"Sensitivity estimates were also reported after adding baseline antidepressant use and episode length, among participants with at least three measurements (n={len(n_ge_3_ids)}), "
    f"and among the complete five-wave selected sample (n={len(complete_ids)}). "
    f"Exact influence refits retained all participants in the analysis; the two largest source shifts were associated with participants {','.join(top_two)}, "
    f"and the joint deletion changed any source quantity by at most {np.max(np.abs(joint_shifts)):.6g}."
)
(RESULTS_DIR / "33_repaired_reporting_paragraph.txt").write_text(report, encoding="utf-8")
display(gate)
print("OVERALL REPRODUCTION STATUS:", overall_reproduction_status)
print("\n" + report)

manifest_rows = []
# Enumerate only files already written beneath the explicit output root.
for path in sorted(RESULTS_DIR.glob("**/*")):
    if path.is_file() and path.name != "34_output_manifest.csv":
        manifest_rows.append({
            "relative_path": str(path.relative_to(RESULTS_DIR)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(RESULTS_DIR / "34_output_manifest.csv", index=False)
display(manifest)
print(f"All repaired outputs written to: {RESULTS_DIR.resolve()}")
