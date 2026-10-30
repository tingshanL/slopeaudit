#!/usr/bin/env python3
"""Downstream simulation (Supplement S3): fitted slopes analyzed as data at matched determinacy.

Simulation design
-----------------
Random-intercept, random-slope Gaussian model with independent equal-variance residuals.
A person-level variable x is related to the part of the level unrelated to the slope, not
to the slope itself:

    s  ~ N(0, tau2)
    b0 = (G0s / tau2) s + sqrt(S) (c x + sqrt(1 - c^2) z),   S = G00 - G0s^2 / tau2
    y_ij = beta0 + beta_t t_ij + b0_i + s_i t_ij + e_ij,       e_ij ~ N(0, sigma2)

with x, z independent standard normal. This preserves the marginal covariance G for every
c, so D^2, R_direct, Delta_fit and kappa are fixed within a construction while c controls how
much of the nuisance capacity is realized. The true association between x and the slope is
zero in every cell.

Population predictions (known parameters, schedules independent of x):
    common schedule:  Corr(x, s_hat) = c * sign(w'1) * sqrt(kappa)
    pooled schedules: Corr(x, s_hat) = gamma * E(w_i'1) / sqrt(tau2 * E(D_i^2)),  gamma = c sqrt(S)
    EB-outcome population coefficient on x: beta_pop = gamma * E(w_i'1)

Analyses of each data set
    known parameters : EB omitting x (marginal G); x-adjusted EB (G | x, mean includes gamma x)
    estimated (REML) : EB omitting x                        (y ~ t,           re 1 + t)
                       EB adjusting for x, no x:t          (y ~ t + x,       re 1 + t)
                       one-stage x:t                       (y ~ t + x + x:t, re 1 + t)
                       uncertainty-aware two-stage (unshrunk slopes, REML meta-regression)

Fit selection (Supplement S5.1)
    Every prespecified statsmodels optimizer is fitted, plus an independent multistart REML
    maximizer (reml_tools.py) and the random-intercept solution embedded at zero slope
    variance. All candidates are evaluated under one REML objective; the highest-likelihood
    admissible candidate is always selected, and the .01 log-likelihood tolerance is used only
    to assess corroboration. Uncorroborated fits trigger up to two restarts; fits that remain
    uncorroborated are flagged and excluded ("fit_ok" is False).
    Fixed effects, model-based standard errors and EB slopes are computed from the selected
    (G, sigma2) by GLS. The validity-based rule of S3.2 is emulated on the same candidates and
    its likelihood shortfall recorded. Validity flags, the unshrunk two-stage routine and that
    emulation use the attenuation-study core in reproduction/s5_downstream_simulation.

Usage (from the repository root)
    python3 reproduction/pm2026/s3_downstream/downstream_simulation.py --stage fixtures
    python3 reproduction/pm2026/s3_downstream/downstream_simulation.py --stage common \
        --target-d2 0.8 --reps 200 --workers 8 --output-dir runs/common
    python3 reproduction/pm2026/s3_downstream/downstream_simulation.py --stage irregular \
        --target-d2 0.8 --reps 200 --n 200 --workers 8 --output-dir runs/irregular
    python3 reproduction/pm2026/s3_downstream/downstream_simulation.py --stage common --smoke
The exact commands for the reported runs are in reproduction/pm2026/README.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm
import statsmodels.formula.api as smf

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parents[2]
S5_CORE_RELATIVE = Path("reproduction/s5_downstream_simulation/simulation_core.py")

# ---------------------------------------------------------------------------
# Fixed design constants
# ---------------------------------------------------------------------------
TAU2 = 1.0                      # target slope variance
BETA0, BETA_T = 0.0, 0.5        # fixed effects (outcome units)
COMMON_TIMES = np.array([0.0, 1.0, 2.0, 3.0])
COMMON_CENTER = 1.5             # time coding: t = visit time - 1.5
IRREGULAR_NOMINAL = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
IRREGULAR_CENTER = 2.0
IRREGULAR_JITTER = 0.2
IRREGULAR_WAVES = (3, 4, 5)
DEFAULT_TARGET_D2 = 0.80        # matches the manuscript's .80 = .80 + .00 / .20 + .60
CONSTRUCTIONS = {               # direct share R_direct / D^2 at the common schedule
    "A": 1.000,                 # D2 = .80: .80 + .00
    "M": 0.625,                 # D2 = .80: .50 + .30
    "B": 0.250,                 # D2 = .80: .20 + .60
}
MAX_ABS_CORRELATION = 0.95      # keep generating G comfortably nonsingular
POOLED_PREDICTION_DRAWS = 50_000
Z_CRIT = norm.ppf(0.975)


# ---------------------------------------------------------------------------
# Attenuation-study core import (explicit path: S4 also has a module named simulation_core)
# ---------------------------------------------------------------------------
def load_s5_core(repo_root: Path):
    path = repo_root / S5_CORE_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"S5 core not found at {path}; pass --repo-root")
    spec = importlib.util.spec_from_file_location("s5_simulation_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Known-parameter source quantities (matrix route; target coordinate last)
# ---------------------------------------------------------------------------
def source_quantities(Z_a, z_s, G, R):
    """R_direct, D^2, Delta_fit, kappa and posterior weights for one participant.

    Z_a : (n, p) nuisance random-effect design;  z_s : (n,) target design
    G   : (p+1, p+1) covariance of (a, s), target last;  R : (n, n) residual covariance
    """
    Z_a = np.atleast_2d(np.asarray(Z_a, float))
    if Z_a.shape[0] != len(z_s):
        Z_a = Z_a.T
    z_s = np.asarray(z_s, float)
    G = np.asarray(G, float)
    R = np.asarray(R, float)
    Z = np.column_stack([Z_a, z_s])
    tau2 = G[-1, -1]
    if not tau2 > 0:
        raise ValueError("target slope variance must be positive")
    V = Z @ G @ Z.T + R
    g_col = Z @ G[:, -1]                       # Cov(y, s)
    w = np.linalg.solve(V, g_col)              # s_hat = w'(y - mu)
    D2 = float(w @ g_col) / tau2
    # Direct information J (Eq. 49): residual-precision-weighted projection off C(Z_a)
    W = np.linalg.inv(R)
    M = W - W @ Z_a @ np.linalg.pinv(Z_a.T @ W @ Z_a) @ Z_a.T @ W
    J = float(z_s @ M @ z_s)
    R_direct = tau2 * J / (1 + tau2 * J) if J > 1e-12 else 0.0
    g_as = G[:-1, -1]
    S = G[:-1, :-1] - np.outer(g_as, g_as) / tau2   # nuisance covariance net of slope
    loading = Z_a.T @ w                               # Z_a' w
    if D2 > 0:
        kappa = float(loading @ S @ loading) / (tau2 * D2)
        kappa_identity = 1 - D2 - float(w @ R @ w) / (tau2 * D2)
    else:
        kappa = kappa_identity = np.nan
    return {
        "R_direct": R_direct, "D2": D2, "Delta_fit": D2 - R_direct,
        "kappa": kappa, "kappa_identity": kappa_identity,
        "w": w, "w_sum_loading": loading, "J": J,
    }


def ri_rs_quantities(times_centered, G2, sigma2):
    t = np.asarray(times_centered, float)
    return source_quantities(np.ones((len(t), 1)), t, G2, sigma2 * np.eye(len(t)))


def check_ceilings(q, label, tol=1e-10):
    k, D2, Delta = q["kappa"], q["D2"], q["Delta_fit"]
    assert abs(k - q["kappa_identity"]) < 1e-9, f"{label}: kappa identity mismatch"
    assert -tol <= k <= 1 - D2 + tol, f"{label}: kappa exceeds 1 - D2"
    assert k <= Delta / D2 + tol, f"{label}: kappa exceeds Delta/D2"


# ---------------------------------------------------------------------------
# Analytic fixtures
# ---------------------------------------------------------------------------
def run_fixtures(verbose=True):
    rows = []
    # Construction A (S1.9): t = (-.5, .5), G = I, R = I/8
    tA = np.array([-0.5, 0.5])
    qA = source_quantities(np.ones((2, 1)), tA, np.eye(2), np.eye(2) / 8)
    # Construction B (S1.9): G = [[18, sqrt15], [sqrt15, 1]], R = 2I
    GB = np.array([[18.0, np.sqrt(15)], [np.sqrt(15), 1.0]])
    qB = source_quantities(np.ones((2, 1)), tA, GB, 2 * np.eye(2))
    # Diagonal-G counterexample: y0 = a + e0, y1 = a + s + e1, all standard normal
    qC = source_quantities(np.ones((2, 1)), np.array([0.0, 1.0]), np.eye(2), np.eye(2))
    expected = {
        "A": (qA, {"R_direct": .80, "D2": .80, "Delta_fit": 0.0, "kappa": 0.0}),
        "B": (qB, {"R_direct": .20, "D2": .80, "Delta_fit": .60, "kappa": .140625}),
        "diagonal_G": (qC, {"D2": .40, "kappa": .10}),
    }
    for label, (q, targets) in expected.items():
        for key, value in targets.items():
            assert abs(q[key] - value) < 1e-10, f"fixture {label}: {key}={q[key]} != {value}"
        check_ceilings(q, f"fixture {label}")
        rows.append({"fixture": label, **{k: q[k] for k in ("R_direct", "D2", "Delta_fit", "kappa")},
                     "max_abs_corr": np.sqrt(q["kappa"])})
    # Construction B weights: s_hat = .05 d + (sqrt15/20) l  ->  w = (.05*(-1) + l/2, .05 + l/2)
    lw = np.sqrt(15) / 20
    assert np.allclose(qB["w"], [-0.05 + lw / 2, 0.05 + lw / 2]), "fixture B: weights"
    assert abs(np.sqrt(qB["kappa"]) - 0.375) < 1e-10
    assert abs(np.sqrt(qC["kappa"]) - np.sqrt(.10)) < 1e-12
    table = pd.DataFrame(rows)
    if verbose:
        print("Analytic fixtures passed:")
        print(table.round(6).to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# Generating settings: matched D^2 = .80 at the common schedule
# ---------------------------------------------------------------------------
def build_settings(g00, target_d2=DEFAULT_TARGET_D2):
    """Matched-D^2 constructions at the common schedule.

    Reaching a large model-assisted share at high D^2 requires a strong intercept-slope
    correlation: |rho|^2 >= Delta / ((1 - R)^2 + Delta R) for any G00. At D^2 = .80,
    construction B needs |rho| >= .89; at D^2 = .50 it needs only about .68.
    """
    t = COMMON_TIMES - COMMON_CENTER
    n = len(t)
    sst = float(np.sum((t - t.mean()) ** 2))
    settings = {}
    for name, direct_share in CONSTRUCTIONS.items():
        r_target = target_d2 * direct_share
        delta_target = target_d2 - r_target
        J = r_target / ((1 - r_target) * TAU2)
        sigma2 = sst / J
        # k = 0 for a centered common schedule, so delta = G0s and
        # Delta = delta^2 (1-R)^2 / (tau2 (G00 + sigma2/n - delta^2 R / tau2))
        delta2 = delta_target * TAU2 * (g00 + sigma2 / n) / ((1 - r_target) ** 2 + delta_target * r_target)
        g0s = float(np.sqrt(delta2))
        G = np.array([[g00, g0s], [g0s, TAU2]])
        rho = g0s / np.sqrt(g00 * TAU2)
        if abs(rho) >= MAX_ABS_CORRELATION:
            raise ValueError(f"construction {name}: |rho|={rho:.3f}; increase --g00")
        q = ri_rs_quantities(t, G, sigma2)
        check_ceilings(q, f"setting {name}")
        assert abs(q["D2"] - target_d2) < 1e-9 and abs(q["R_direct"] - r_target) < 1e-9, name
        S = g00 - g0s ** 2 / TAU2
        settings[name] = {
            "construction": name, "G": G, "sigma2": sigma2, "S": S, "rho": rho,
            "common_R_direct": q["R_direct"], "common_D2": q["D2"],
            "common_Delta_fit": q["Delta_fit"], "common_kappa": q["kappa"],
            "common_w_sum": float(q["w_sum_loading"][0]),
        }
    return settings


def predictions(setting, c, schedule_kind, rng_seed=20261015):
    """Population Corr(x, s_hat) and EB-outcome coefficient under known parameters."""
    gamma = c * np.sqrt(setting["S"])
    if schedule_kind == "common":
        w_sum, D2 = setting["common_w_sum"], setting["common_D2"]
        out = {"E_w_sum": w_sum, "E_D2": D2, "E_R_direct": setting["common_R_direct"],
               "E_Delta_fit": setting["common_Delta_fit"], "mean_kappa_i": setting["common_kappa"]}
    else:
        rng = np.random.default_rng(rng_seed)
        cache, ws, d2s, rs, ks = {}, [], [], [], []
        for _ in range(POOLED_PREDICTION_DRAWS):
            t = draw_irregular_schedule(rng)
            key = tuple(np.round(t, 12))
            if key not in cache:
                q = ri_rs_quantities(t, setting["G"], setting["sigma2"])
                cache[key] = (float(q["w_sum_loading"][0]), q["D2"], q["R_direct"], q["kappa"])
            w_sum_i, d2_i, r_i, k_i = cache[key]
            ws.append(w_sum_i); d2s.append(d2_i); rs.append(r_i); ks.append(k_i)
        out = {"E_w_sum": float(np.mean(ws)), "E_D2": float(np.mean(d2s)),
               "E_R_direct": float(np.mean(rs)), "E_Delta_fit": float(np.mean(d2s) - np.mean(rs)),
               "mean_kappa_i": float(np.mean(ks))}
    out["gamma"] = gamma
    out["beta_pop_eb_omit"] = gamma * out["E_w_sum"]
    out["corr_pop_eb_omit"] = gamma * out["E_w_sum"] / np.sqrt(TAU2 * out["E_D2"])
    return out


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def draw_irregular_schedule(rng):
    m = int(rng.choice(IRREGULAR_WAVES))
    later = np.sort(rng.choice(IRREGULAR_NOMINAL[1:], size=m - 1, replace=False))
    later = later + rng.uniform(-IRREGULAR_JITTER, IRREGULAR_JITTER, size=m - 1)
    return np.concatenate([[0.0], later]) - IRREGULAR_CENTER


def simulate_dataset(setting, c, schedule_kind, n, seed):
    rng = np.random.default_rng(int(seed))
    G, sigma2, S = setting["G"], setting["sigma2"], setting["S"]
    g0s = G[0, 1]
    s = rng.normal(0, np.sqrt(TAU2), n)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    b0 = (g0s / TAU2) * s + np.sqrt(S) * (c * x + np.sqrt(1 - c ** 2) * z)
    if schedule_kind == "common":
        schedules = [COMMON_TIMES - COMMON_CENTER] * n
    else:
        # schedules drawn independently of x, random effects and residuals
        sched_rng = np.random.default_rng([int(seed), 7])
        schedules = [draw_irregular_schedule(sched_rng) for _ in range(n)]
    frames = []
    for i in range(n):
        t = schedules[i]
        y = BETA0 + BETA_T * t + b0[i] + s[i] * t + rng.normal(0, np.sqrt(sigma2), len(t))
        frames.append(pd.DataFrame({"pid": f"p{i:04d}", "t": t, "y": y, "x": x[i], "true_b1": s[i]}))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def ols_on_x(values, x):
    fit = smf.ols("v ~ x", data=pd.DataFrame({"v": values, "x": x})).fit(cov_type="HC3")
    beta, se = float(fit.params["x"]), float(fit.bse["x"])
    return beta, se, 2 * norm.sf(abs(beta / se)) if se > 0 else np.nan


def record(row, prefix, beta, se, p, beta_pop=None, corr=None):
    row[f"{prefix}_beta"], row[f"{prefix}_SE"], row[f"{prefix}_p"] = beta, se, p
    ok = np.isfinite(beta) and np.isfinite(se) and se > 0
    row[f"{prefix}_valid"] = bool(ok)
    row[f"{prefix}_covers_zero"] = bool(ok and abs(beta) <= Z_CRIT * se)
    if beta_pop is not None:
        row[f"{prefix}_covers_beta_pop"] = bool(ok and abs(beta - beta_pop) <= Z_CRIT * se)
    if corr is not None:
        row[f"{prefix}_corr"] = corr


def known_parameter_analyses(data, setting, c, row, beta_pop):
    G, sigma2 = setting["G"], setting["sigma2"]
    gamma = c * np.sqrt(setting["S"])
    G_x = np.array([[G[0, 0] - gamma ** 2, G[0, 1]], [G[0, 1], TAU2]])
    cache, omit, adj, xs = {}, [], [], []
    for _, grp in data.groupby("pid", sort=False):
        t = grp.t.to_numpy(float)
        key = tuple(np.round(t, 12))
        if key not in cache:
            cache[key] = (ri_rs_quantities(t, G, sigma2)["w"], ri_rs_quantities(t, G_x, sigma2)["w"])
        w_m, w_x = cache[key]
        resid = grp.y.to_numpy(float) - BETA0 - BETA_T * t
        xi = float(grp.x.iloc[0])
        omit.append(float(w_m @ resid))
        adj.append(float(w_x @ (resid - gamma * xi)))
        xs.append(xi)
    xs, omit, adj = map(np.asarray, (xs, omit, adj))
    record(row, "known_EB_omit", *ols_on_x(omit, xs), beta_pop=beta_pop,
           corr=float(np.corrcoef(xs, omit)[0, 1]))
    record(row, "known_EB_adjusted", *ols_on_x(adj, xs), beta_pop=0.0,
           corr=float(np.corrcoef(xs, adj)[0, 1]))


def load_reml_tools():
    path = SCRIPT_DIR / "reml_tools.py"
    spec = importlib.util.spec_from_file_location("s5b_reml_tools", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plug_in_sources(G, sigma2, data):
    G = np.asarray(G, float)
    if G.shape != (2, 2) or not (G[1, 1] > 0 and sigma2 > 0):
        return {}
    vals, cache = [], {}
    for _, grp in data.groupby("pid", sort=False):
        key = tuple(np.round(grp.t.to_numpy(float), 12))
        if key not in cache:
            q = ri_rs_quantities(np.asarray(key), G, sigma2)
            cache[key] = (q["R_direct"], q["D2"], q["Delta_fit"], q["kappa"])
        vals.append(cache[key])
    v = np.asarray(vals)
    return {"plugin_mean_R_direct": v[:, 0].mean(), "plugin_mean_D2": v[:, 1].mean(),
            "plugin_mean_Delta_fit": v[:, 2].mean(), "plugin_mean_kappa": v[:, 3].mean()}


FIT_MODELS = {   # key: (formula, role); all use random effects 1 + t
    "EB_omit": ("y ~ t", "reference_omitting_x"),
    "EB_adjusted": ("y ~ t + x", "x_adjusted_no_interaction"),
    "one_stage": ("y ~ t + x + x:t", "one_stage"),
}


def estimated_analyses(core, rt, data, row, beta_pop, optimizers, n_starts, seed):
    """Likelihood-based selection for each model; downstream quantities from the selected fit."""
    fits, cand_rows = {}, []
    for key, (formula, role) in FIT_MODELS.items():
        rec, cands = rt.fit_best_reml(data, formula, "1+t", core, model_role=role,
                                      optimizers=optimizers, n_starts=n_starts, seed=seed)
        fits[key] = rec
        for c in cands:
            cand_rows.append({"model": key, **c})
        row[f"{key}_fit_status"] = rec["fit_status"]
        row[f"{key}_fit_ok"] = bool(rec["fit_ok"])
        for k in ("llf", "best_source", "n_agree", "restarts", "improved_by_restart", "best_rho", "best_boundary",
                  "best_slope_variance_boundary", "best_min_eig_fraction",
                  "frozen_source", "frozen_minus_best_llf", "frozen_rho"):
            row[f"{key}_{k}"] = rec.get(k, np.nan)

    for key in ("EB_omit", "EB_adjusted"):
        rec = fits[key]
        target = beta_pop if key == "EB_omit" else 0.0
        row[f"est_{key}_fit_ok"] = bool(rec["fit_ok"])
        if "eb" not in rec:
            record(row, f"est_{key}", np.nan, np.nan, np.nan, beta_pop=target)
            continue
        # neither model contains x:t, so the EB total slope is beta_t + EB slope deviation
        slope = rec["beta"]["t"] + rec["eb"]["t"]
        xs = data.groupby("pid", sort=False).x.first().reindex(slope.index).to_numpy(float)
        eb = slope.to_numpy(float)
        record(row, f"est_{key}", *ols_on_x(eb, xs), beta_pop=target, corr=float(np.corrcoef(xs, eb)[0, 1]))
    if "G" in fits["EB_omit"]:
        row.update(plug_in_sources(fits["EB_omit"]["G"], fits["EB_omit"]["s2"], data))

    one = fits["one_stage"]
    row["est_one_stage_fit_ok"] = bool(one["fit_ok"])
    row["est_two_stage_unshrunk_fit_ok"] = bool(one["fit_ok"])  # uses the one-stage residual variance
    if "beta" in one:
        beta = float(one["beta"]["x:t"])
        se = float(np.sqrt(one["cov_beta"].loc["x:t", "x:t"]))
        record(row, "est_one_stage", beta, se, 2 * norm.sf(abs(beta / se)) if se > 0 else np.nan)
        raw = core.uncertainty_aware_slopes(data, float(one["s2"]))
        response, design = patsy.dmatrices("unshrunk_slope ~ x", raw, return_type="dataframe")
        meta_beta, meta_se, _tau2 = core.reml_meta_regression(
            response.iloc[:, 0], raw.loc[response.index, "slope_sampling_variance"], design, design.columns)
        beta, se = float(meta_beta["x"]), float(meta_se["x"])
        record(row, "est_two_stage_unshrunk", beta, se, 2 * norm.sf(abs(beta / se)) if se > 0 else np.nan,
               corr=float(np.corrcoef(raw.x, raw.unshrunk_slope)[0, 1]))
    else:
        record(row, "est_one_stage", np.nan, np.nan, np.nan)
        record(row, "est_two_stage_unshrunk", np.nan, np.nan, np.nan)
    return cand_rows


def run_task(task):
    core = load_s5_core(Path(task["repo_root"]))
    rt = load_reml_tools()
    setting = task["setting"]
    pred = task["prediction"]
    data = simulate_dataset(setting, task["c"], task["schedule"], task["n"], task["seed"])
    row = {k: task[k] for k in ("construction", "c", "schedule", "n", "replicate", "seed")}
    row.update({f"pred_{k}": v for k, v in pred.items()})
    row["sample_corr_x_true_slope"] = float(np.corrcoef(
        data.groupby("pid").x.first(), data.groupby("pid").true_b1.first())[0, 1])
    t0 = time.time()
    cand_rows = []
    try:
        known_parameter_analyses(data, setting, task["c"], row, pred["beta_pop_eb_omit"])
        for m in ("known_EB_omit", "known_EB_adjusted"):
            row[f"{m}_fit_ok"] = True  # no fitting involved
        cand_rows = estimated_analyses(core, rt, data, row, pred["beta_pop_eb_omit"],
                                       task["optimizers"], task["n_starts"], task["seed"])
        row["error"] = ""
    except Exception as exc:  # keep the replicate in the denominator
        row["error"] = repr(exc)
    row["seconds"] = time.time() - t0
    for c in cand_rows:
        c.update(construction=task["construction"], c=task["c"], replicate=task["replicate"], seed=task["seed"])
    return row, cand_rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
METHODS = ["known_EB_omit", "known_EB_adjusted", "est_EB_omit", "est_EB_adjusted",
           "est_one_stage", "est_two_stage_unshrunk"]
CELL = ["construction", "c", "schedule"]


def _mcse_prop(p, n):
    return float(np.sqrt(p * (1 - p) / n)) if n else np.nan


def summarize(rep):
    """Method performance. The estimand of every coefficient is the true slope-x association (0).
    Only EB-omit has a nonzero workflow target (the population EB-outcome coefficient)."""
    out = []
    for key, g in rep.groupby(CELL):
        for m in METHODS:
            fit_ok = g.get(f"{m}_fit_ok", pd.Series(True, index=g.index)).fillna(False).astype(bool)
            comp = g.get(f"{m}_valid", pd.Series(False, index=g.index)).fillna(False).astype(bool)
            ok = g[fit_ok & comp]
            n_ok = len(ok)
            row = dict(zip(CELL, key), method=m, requested=len(g), fit_ok=int(fit_ok.sum()),
                       computable=int(comp.sum()), usable=n_ok, estimand_beta=0.0,
                       workflow_target_beta=g.pred_beta_pop_eb_omit.iloc[0] if "EB_omit" in m else 0.0,
                       workflow_target_corr=g.pred_corr_pop_eb_omit.iloc[0] if "EB_omit" in m else 0.0)
            if n_ok:
                b = ok[f"{m}_beta"]
                rej = float((ok[f"{m}_p"] < .05).mean())
                row.update(mean_beta=b.mean(), bias_vs_estimand=b.mean(),
                           mc_se_mean_beta=b.std(ddof=1) / np.sqrt(n_ok) if n_ok > 1 else np.nan,
                           empirical_sd_beta=b.std(ddof=1) if n_ok > 1 else np.nan,
                           mean_model_SE=ok[f"{m}_SE"].mean(),
                           se_ratio=ok[f"{m}_SE"].mean() / b.std(ddof=1) if n_ok > 1 else np.nan,
                           reject_zero_05=rej, mc_se_reject=_mcse_prop(rej, n_ok),
                           cover_zero=ok[f"{m}_covers_zero"].mean())
                if f"{m}_covers_beta_pop" in ok and "EB_omit" in m:
                    row["cover_workflow_target"] = ok[f"{m}_covers_beta_pop"].mean()
                if f"{m}_corr" in ok:
                    row["mean_corr_x"] = ok[f"{m}_corr"].mean()
                    row["mc_se_mean_corr"] = ok[f"{m}_corr"].std(ddof=1) / np.sqrt(n_ok) if n_ok > 1 else np.nan
            out.append(row)
    return pd.DataFrame(out)


def paired_summary(rep):
    """Paired (common-random-numbers) contrasts; supplements, does not replace, bias vs zero."""
    pairs = [("oracle_workflow_gap", "known_EB_omit", "known_EB_adjusted", "pred_beta_pop_eb_omit"),
             ("estimation_effect_EB_omit", "est_EB_omit", "known_EB_omit", None),
             ("estimation_effect_EB_adjusted", "est_EB_adjusted", "known_EB_adjusted", None)]
    out = []
    for key, g in rep.groupby(CELL):
        for name, a, b, pred in pairs:
            ok = g[g[f"{a}_valid"].fillna(False).astype(bool) & g[f"{b}_valid"].fillna(False).astype(bool)
                   & g.get(f"{a}_fit_ok", pd.Series(True, index=g.index)).fillna(False).astype(bool)]
            d = ok[f"{a}_beta"] - ok[f"{b}_beta"]
            out.append(dict(zip(CELL, key), contrast=name, n=len(d), mean_difference=d.mean(),
                            mc_se=d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                            predicted=g[pred].iloc[0] if pred else np.nan))
    return pd.DataFrame(out)


def fit_quality_summary(rep):
    out = []
    for key, g in rep.groupby(CELL):
        for m in FIT_MODELS:
            st = g[f"{m}_fit_status"].fillna("missing")
            gap = g[f"{m}_frozen_minus_best_llf"]
            out.append(dict(zip(CELL, key), model=m, n=len(g),
                            agreed=int((st == "agreed").sum()),
                            confirmed_on_restart=int((st == "confirmed_on_restart").sum()),
                            improved_by_restart=int(g[f"{m}_improved_by_restart"].fillna(False).astype(bool).sum()),
                            unresolved=int(st.isin(["unresolved_single_source", "no_admissible_fit", "missing"]).sum()),
                            best_is_random_intercept_embedded=int((g[f"{m}_best_source"] == "random_intercept_embedded").sum()),
                            boundary_rate=g[f"{m}_best_boundary"].astype(float).mean(),
                            mean_rho=g[f"{m}_best_rho"].mean(),
                            frozen_rule_below_best_gt_001=int((gap < -0.01).sum()),
                            frozen_rule_below_best_gt_1=int((gap < -1).sum()),
                            frozen_gap_min=gap.min(), frozen_gap_median=gap.median()))
    q = pd.DataFrame(out)
    plug = rep.groupby(CELL).agg(
        pred_mean_kappa_i=("pred_mean_kappa_i", "first"), pred_E_D2=("pred_E_D2", "first"),
        plugin_mean_kappa=("plugin_mean_kappa", "mean"), plugin_mean_D2=("plugin_mean_D2", "mean"),
        plugin_mean_R_direct=("plugin_mean_R_direct", "mean"),
        errors=("error", lambda e: int((e.fillna("") != "").sum()))).reset_index()
    return q, plug


def boundary_strata(rep):
    """EB-omit results by boundary status of the selected reference fit (descriptive only)."""
    g = rep[rep["est_EB_omit_valid"].fillna(False).astype(bool) & rep["EB_omit_fit_ok"].fillna(False).astype(bool)]
    return g.groupby(CELL + ["EB_omit_best_boundary"]).agg(
        n=("seed", "size"), mean_rho=("EB_omit_best_rho", "mean"),
        plugin_kappa=("plugin_mean_kappa", "mean"), plugin_D2=("plugin_mean_D2", "mean"),
        mean_beta=("est_EB_omit_beta", "mean"), mean_known_beta=("known_EB_omit_beta", "mean"),
        reject=("est_EB_omit_p", lambda p: float((p < .05).mean()))).reset_index()


def compare_with_previous(rep, old_path):
    """Same seeds: an earlier validity-rule run versus this run."""
    old = pd.read_csv(old_path)
    keep = ["seed", "est_EB_omit_beta", "est_EB_omit_p", "EB_omit_fit_rho", "known_EB_omit_beta"]
    m = rep.merge(old[[c for c in keep if c in old]], on="seed", suffixes=("", "_old"), validate="one_to_one")
    regen = (m["known_EB_omit_beta"] - m["known_EB_omit_beta_old"]).abs().max()
    emul = (m["EB_omit_frozen_rho"] - m["EB_omit_fit_rho"]).abs()
    out = []
    for key, g in m.groupby(CELL):
        out.append(dict(zip(CELL, key), n=len(g),
                        old_mean_beta=g.est_EB_omit_beta_old.mean(), new_mean_beta=g.est_EB_omit_beta.mean(),
                        old_reject=float((g.est_EB_omit_p_old < .05).mean()), new_reject=float((g.est_EB_omit_p < .05).mean()),
                        old_mean_rho=g.EB_omit_fit_rho.mean(), new_mean_rho=g.EB_omit_best_rho.mean(),
                        frozen_emulation_matches_old_rho=int((emul.loc[g.index] < 1e-4).sum())))
    return pd.DataFrame(out), float(regen)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["fixtures", "common", "irregular"], default="fixtures")
    ap.add_argument("--constructions", nargs="+", default=list(CONSTRUCTIONS), choices=list(CONSTRUCTIONS))
    ap.add_argument("--c-values", nargs="+", type=float, default=[0.0, 0.5])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--g00", type=float, default=100.0, help="random-intercept variance (tau2 = 1)")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=Path("contamination_pilot_output"))
    ap.add_argument("--smoke", action="store_true", help="2 replicates per cell; not for reporting")
    ap.add_argument("--target-d2", type=float, default=DEFAULT_TARGET_D2,
                    help="matched total determinacy at the common schedule (default .80)")
    ap.add_argument("--optimizers", nargs="*", default=["lbfgs", "powell", "bfgs"],
                    help="statsmodels candidates (all are fitted; none stops the sequence)")
    ap.add_argument("--n-starts", type=int, default=8, help="starts for the independent REML maximizer")
    ap.add_argument("--compare-with", type=Path, default=None,
                    help="replicates.csv from an earlier run with the same seeds")
    args = ap.parse_args(argv)

    run_fixtures()
    settings = build_settings(args.g00, args.target_d2)
    print(f"\nGenerating settings, matched D2 = {args.target_d2} (common schedule t = 0..3 centered at 1.5):")
    print(pd.DataFrame([{k: v for k, v in s.items() if k != "G"} | {"G00": s["G"][0, 0], "G0s": s["G"][0, 1]}
                        for s in settings.values()]).round(4).to_string(index=False))
    if args.stage == "fixtures":
        return 0
    for c in args.c_values:
        if not 0 <= c <= 1:
            raise ValueError("c must lie in [0, 1]")

    reps = 2 if args.smoke else args.reps
    out_dir = args.output_dir / (f"{args.stage}_D2_{args.target_d2:.2f}_likelihood_selection"
                                 + ("_smoke" if args.smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    load_s5_core(args.repo_root)  # fail fast if the S5 core is missing
    load_reml_tools()

    import platform, scipy, statsmodels
    tasks, manifest = [], {"stage": args.stage, "n": args.n, "reps": reps, "g00": args.g00,
                           "target_d2": args.target_d2, "tau2": TAU2, "seed": args.seed, "smoke": args.smoke,
                           "selection_rule": "highest REML likelihood among admissible candidates (statsmodels optimizers, "
                                             "independent multistart, embedded random-intercept fit); corroboration "
                                             "tolerance 0.01 log-lik; up to 2 restarts; uncorroborated fits not fit_ok",
                           "optimizers": args.optimizers, "n_starts": args.n_starts,
                           "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                        "scipy": scipy.__version__, "pandas": pd.__version__,
                                        "statsmodels": statsmodels.__version__},
                           "cells": []}
    # Seeds follow the construction and c-value order, as in the earlier validity-rule run.
    for ci, cons in enumerate(args.constructions):
        setting = settings[cons]
        for cj, c in enumerate(args.c_values):
            pred = predictions(setting, c, args.stage)
            manifest["cells"].append({"construction": cons, "c": c,
                                      **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                         for k, v in setting.items()}, **pred})
            print(f"cell {cons} c={c}: oracle EB-omit Corr(x, s_hat) = {pred['corr_pop_eb_omit']:+.4f}, "
                  f"E D2 = {pred['E_D2']:.4f}, E R_direct = {pred['E_R_direct']:.4f}")
            for r in range(reps):
                tasks.append({"repo_root": str(args.repo_root), "setting": setting, "prediction": pred,
                              "construction": cons, "c": c, "schedule": args.stage, "n": args.n,
                              "replicate": r, "seed": args.seed + 100_000 * (10 * ci + cj) + r,
                              "optimizers": tuple(args.optimizers), "n_starts": args.n_starts})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))

    rows, cands, t0 = [], [], time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_task, task) for task in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                row, cr = fut.result()
                rows.append(row); cands.extend(cr)
                if i % 25 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} replicates ({time.time() - t0:.0f}s)", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            row, cr = run_task(task)
            rows.append(row); cands.extend(cr)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} replicates ({time.time() - t0:.0f}s)", flush=True)

    rep = pd.DataFrame(rows).sort_values(["construction", "c", "replicate"]).reset_index(drop=True)
    rep.to_csv(out_dir / "replicates.csv", index=False)
    pd.DataFrame(cands).to_csv(out_dir / "model_candidates.csv", index=False)
    summary = summarize(rep)
    paired = paired_summary(rep)
    quality, plug = fit_quality_summary(rep)
    strata = boundary_strata(rep)
    summary.to_csv(out_dir / "summary_by_method.csv", index=False)
    paired.to_csv(out_dir / "summary_paired_contrasts.csv", index=False)
    quality.to_csv(out_dir / "summary_fit_quality.csv", index=False)
    plug.to_csv(out_dir / "summary_plugin.csv", index=False)
    strata.to_csv(out_dir / "summary_EB_omit_by_boundary.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["construction", "c", "method", "usable", "workflow_target_corr", "mean_corr_x", "workflow_target_beta",
            "mean_beta", "mc_se_mean_beta", "se_ratio", "reject_zero_05", "mc_se_reject", "cover_zero"]
    print("\nMethods (estimand = 0 for every method; workflow target shown only for EB-omit):")
    print(summary[[c for c in cols if c in summary]].round(4).to_string(index=False))
    print("\nPaired contrasts:\n" + paired.round(5).to_string(index=False))
    print("\nFit quality (frozen-rule shortfall = frozen-rule likelihood minus selected likelihood):")
    print(quality.round(4).to_string(index=False))
    print("\n" + plug.round(4).to_string(index=False))
    print("\nEB-omit by boundary status of the selected reference fit:\n" + strata.round(4).to_string(index=False))
    if args.compare_with is not None:
        comp, regen = compare_with_previous(rep, args.compare_with)
        comp.to_csv(out_dir / "comparison_with_previous_run.csv", index=False)
        print(f"\nComparison with {args.compare_with} (max |known-parameter beta difference| = {regen:.2e}; "
              "should be ~1e-15 if datasets regenerated exactly):")
        print(comp.round(4).to_string(index=False))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
