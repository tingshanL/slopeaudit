#!/usr/bin/env python3
"""Exact REML objective and likelihood-based fit selection for y ~ X, (1 | pid) or (1 + t | pid).

Model: y_i = X_i beta + Z_i b_i + e_i,  b_i ~ N(0, G),  e_i ~ N(0, s2 I).

`REMLProblem.loglik(G, s2)` evaluates the REML log-likelihood with beta profiled out,
using the same constant as statsmodels MixedLM (-(n - p)/2 log 2 pi), so values are
directly comparable with `MixedLMResults.llf` for identical data and formulas.

`fit_best_reml` implements the selection rule:
    fit every prespecified statsmodels optimizer (no early stopping) plus an
    independent multistart maximizer; evaluate every candidate under this one
    objective; keep the highest-likelihood admissible solution (singular G allowed);
    restart from the best solution when no second source agrees; flag unresolved cases.
It also emulates the frozen S5 rule (first strictly valid optimizer, else first basic
valid, else first fitted) so the old selection's likelihood shortfall can be recorded.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import patsy
from scipy.optimize import minimize
import statsmodels.formula.api as smf

LOG2PI = np.log(2 * np.pi)
AGREE_TOL = 0.01          # log-likelihood units: candidates within this of the best "agree"
PSD_TOL = 1e-8            # relative tolerance for admissible (PSD) covariance
BOUNDARY_EIG_FRACTION = 1e-6
DEFAULT_OPTIMIZERS = ("lbfgs", "powell", "bfgs")
SM_MAXITER = 1200


class REMLProblem:
    def __init__(self, data: pd.DataFrame, formula: str, re_formula: str):
        y, X = patsy.dmatrices(formula, data, return_type="dataframe")
        self.fe_names = list(X.columns)
        self.p = X.shape[1]
        self.n = len(y)
        if re_formula.replace(" ", "") == "1":
            self.q = 1
        elif re_formula.replace(" ", "") == "1+t":
            self.q = 2
        else:
            raise ValueError("re_formula must be '1' or '1+t'")
        yv = y.to_numpy(float).ravel()
        Xv = X.to_numpy(float)
        tv = data.loc[y.index, "t"].to_numpy(float)
        pid = data.loc[y.index, "pid"].to_numpy()
        self.pids = list(pd.unique(pid))
        index_of = {p: i for i, p in enumerate(self.pids)}
        rows_by_pid = [[] for _ in self.pids]
        for r, p in enumerate(pid):
            rows_by_pid[index_of[p]].append(r)
        self.batches = []  # (person indices, Y (B,m), X (B,m,p), Z (B,m,q))
        sizes = np.array([len(r) for r in rows_by_pid])
        for m in np.unique(sizes):
            people = np.flatnonzero(sizes == m)
            idx = np.array([rows_by_pid[i] for i in people])
            Z = np.ones((len(people), m, self.q))
            if self.q == 2:
                Z[:, :, 1] = tv[idx]
            self.batches.append((people, yv[idx], Xv[idx], Z))
        self.const = 0.5 * (self.n - self.p) * LOG2PI

    # ----- objective -------------------------------------------------------
    def _accumulate(self, G, s2):
        XtViX = np.zeros((self.p, self.p))
        XtViy = np.zeros(self.p)
        ytViy = 0.0
        logdet = 0.0
        cache = []
        for people, Y, X, Z in self.batches:
            m = Y.shape[1]
            V = np.einsum("bik,kl,bjl->bij", Z, G, Z) + s2 * np.eye(m)
            L = np.linalg.cholesky(V)  # raises if not PD
            logdet += 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum()
            Vi = np.linalg.inv(V)
            ViX = Vi @ X
            Viy = np.einsum("bij,bj->bi", Vi, Y)
            XtViX += np.einsum("bik,bil->kl", X, ViX)
            XtViy += np.einsum("bik,bi->k", X, Viy)
            ytViy += np.einsum("bi,bi->", Y, Viy)
            cache.append((people, Y, X, Z, Vi))
        return XtViX, XtViy, ytViy, logdet, cache

    def loglik(self, G, s2):
        G = np.asarray(G, float).reshape(self.q, self.q)
        if not (np.isfinite(G).all() and np.isfinite(s2) and s2 > 0):
            return -np.inf
        try:
            XtViX, XtViy, ytViy, logdet, _ = self._accumulate(G, s2)
            sign, ld_x = np.linalg.slogdet(XtViX)
            if sign <= 0:
                return -np.inf
            beta = np.linalg.solve(XtViX, XtViy)
            quad = ytViy - beta @ XtViy
            return float(-0.5 * (logdet + ld_x + quad) - self.const)
        except np.linalg.LinAlgError:
            return -np.inf

    def gls(self, G, s2):
        """beta, model-based Cov(beta) = (X'V^-1X)^-1, and EB random effects (by pid)."""
        G = np.asarray(G, float).reshape(self.q, self.q)
        XtViX, XtViy, _, _, cache = self._accumulate(G, s2)
        cov_beta = np.linalg.inv(XtViX)
        beta = cov_beta @ XtViy
        eb = np.full((len(self.pids), self.q), np.nan)
        for people, Y, X, Z, Vi in cache:
            resid = Y - X @ beta
            eb[people] = np.einsum("kl,bil,bij,bj->bk", G, Z, Vi, resid)
        return (pd.Series(beta, index=self.fe_names),
                pd.DataFrame(cov_beta, index=self.fe_names, columns=self.fe_names),
                pd.DataFrame(eb, index=self.pids, columns=["intercept", "t"][: self.q]))

    # ----- parameterization and independent maximizer -----------------------
    def unpack(self, theta):
        if self.q == 1:
            return np.array([[np.exp(2 * theta[0])]]), float(np.exp(theta[1]))
        L = np.array([[np.exp(theta[0]), 0.0], [theta[1], np.exp(theta[2])]])
        return L @ L.T, float(np.exp(theta[3]))

    def pack(self, G, s2, floor=1e-10):
        G = np.asarray(G, float).reshape(self.q, self.q)
        s2 = max(float(s2), floor)
        if self.q == 1:
            return np.array([0.5 * np.log(max(G[0, 0], floor)), np.log(s2)])
        g00 = max(G[0, 0], floor)
        l11 = np.sqrt(g00)
        l21 = G[0, 1] / l11
        l22sq = max(G[1, 1] - l21 ** 2, floor * max(1.0, G[1, 1]))
        return np.array([np.log(l11), l21, 0.5 * np.log(l22sq), np.log(s2)])

    def _negll_theta(self, theta):
        G, s2 = self.unpack(theta)
        ll = self.loglik(G, s2)
        return 1e300 if not np.isfinite(ll) else -ll

    def starting_values(self, rng, n_starts):
        """Moment-based scales from per-person OLS, then log-scale jitter."""
        means, slopes, resid = [], [], []
        for people, Y, X, Z in self.batches:
            for b in range(Y.shape[0]):
                if self.q == 2 and np.ptp(Z[b, :, 1]) > 0:
                    coef, res, *_ = np.linalg.lstsq(Z[b], Y[b], rcond=None)
                    means.append(coef[0]); slopes.append(coef[1])
                    if Y.shape[1] > 2:
                        resid.append(np.sum((Y[b] - Z[b] @ coef) ** 2) / (Y.shape[1] - 2))
                else:
                    means.append(Y[b].mean())
                    if Y.shape[1] > 1:
                        resid.append(Y[b].var(ddof=1))
        s2_0 = max(np.mean(resid) if resid else np.var(means), 1e-6)
        g00_0 = max(np.var(means), 1e-3 * s2_0)
        starts = []
        if self.q == 1:
            for k in range(n_starts):
                f = np.exp(rng.normal(0, 1)) if k else 1.0
                starts.append(self.pack([[g00_0 * f]], s2_0 * np.exp(rng.normal(0, .5)) if k else s2_0))
            return starts
        t2_0 = max(np.var(slopes) * 0.5, 1e-3 * s2_0 / max(1.0, np.var(means)))
        c_0 = np.cov(means, slopes)[0, 1] if len(slopes) > 2 else 0.0
        rho_grid = [np.clip(c_0 / np.sqrt(g00_0 * t2_0), -.9, .9), 0.0, 0.8, -0.8, 0.98, -0.98, 0.5, -0.5]
        for k in range(n_starts):
            rho = rho_grid[k % len(rho_grid)]
            fg, ft, fs = (1.0, 1.0, 1.0) if k < len(rho_grid) else np.exp(rng.normal(0, 1, 3))
            g00, t2 = g00_0 * fg, t2_0 * ft
            G = np.array([[g00, rho * np.sqrt(g00 * t2)], [rho * np.sqrt(g00 * t2), t2]])
            starts.append(self.pack(G, s2_0 * fs))
        return starts

    def maximize(self, n_starts=8, seed=0, extra_starts=()):
        """Multistart L-BFGS-B followed by Nelder-Mead polish.

        Returns (G, s2, loglik, info). `info` records termination of the winning start and how
        many starts reached the same optimum (within AGREE_TOL), so the result is not assumed
        to be a successful optimization merely because it was returned."""
        rng = np.random.default_rng(seed)
        results = []
        for theta0 in list(extra_starts) + self.starting_values(rng, n_starts):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r1 = minimize(self._negll_theta, theta0, method="L-BFGS-B")
                r2 = minimize(self._negll_theta, r1.x, method="Nelder-Mead",
                              options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-9})
            results.append((r1, r2))
        r1, best = min(results, key=lambda rr: rr[1].fun)
        finite = [rr[1].fun for rr in results if np.isfinite(rr[1].fun) and rr[1].fun < 1e299]
        info = {"opt_starts": len(results),
                "opt_starts_within_tol": int(sum(f - best.fun <= AGREE_TOL for f in finite)),
                "opt_lbfgsb_success": bool(r1.success), "opt_lbfgsb_message": str(r1.message)[:80],
                "opt_nm_success": bool(best.success), "opt_nm_message": str(best.message)[:80],
                "opt_nm_iterations": int(getattr(best, "nit", -1))}
        G, s2 = self.unpack(best.x)
        return G, s2, -best.fun, info


# ---------------------------------------------------------------------------
# Candidate assessment
# ---------------------------------------------------------------------------
def admissible_covariance(G, s2):
    G = np.asarray(G, float)
    if not (np.isfinite(G).all() and np.isfinite(s2) and s2 > 0):
        return False, G
    G = (G + G.T) / 2
    w, U = np.linalg.eigh(G)
    scale = max(1.0, float(np.max(np.abs(w))), abs(float(s2)))
    if w.min() < -PSD_TOL * scale:
        return False, G
    return True, (U * np.clip(w, 0, None)) @ U.T   # clip rounding-level negatives only


def boundary_status(G, s2):
    G = np.asarray(G, float)
    w = np.linalg.eigvalsh(G)
    tr = float(np.trace(G))
    out = {"min_eig_fraction": w.min() / tr if tr > 0 else np.nan}
    if G.shape == (2, 2):
        out["rho"] = G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]) if G[0, 0] > 0 and G[1, 1] > 0 else np.nan
        out["slope_variance_boundary"] = bool(G[1, 1] <= 1e-8 * max(1.0, s2))
    out["boundary"] = bool(tr <= 0 or out["min_eig_fraction"] <= BOUNDARY_EIG_FRACTION
                           or out.get("slope_variance_boundary", False))
    return out


MAX_RESTARTS = 2
OK_STATUSES = ("agreed", "confirmed_on_restart")


def fit_best_reml(data, formula, re_formula, core, model_role="",
                  optimizers=DEFAULT_OPTIMIZERS, n_starts=8, seed=0):
    """Return (record, candidate_rows). record['G'], record['s2'] hold the selected fit.

    Selection: the highest-likelihood admissible candidate is always selected. The agreement
    tolerance is used only to assess corroboration: the selected solution is "agreed" when at
    least one other candidate lies within AGREE_TOL of it. Otherwise the independent maximizer
    is restarted from the current best (up to MAX_RESTARTS times); after each restart the best
    is re-selected and agreement reassessed against it. Remaining single-source solutions are
    flagged "unresolved_single_source" and are not fit_ok.

    Candidates: every statsmodels optimizer; an independent multistart maximizer; and, for
    random-slope models, the fitted random-intercept-only solution embedded with zero slope
    variance and covariance (the exact boundary the Cholesky parameterization only approaches).
    """
    prob = REMLProblem(data, formula, re_formula)
    has_slope = prob.q == 2
    cands = []
    for opt in optimizers:
        c = {"model_role": model_role, "source": f"statsmodels_{opt}"}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm(formula, data=data, groups=data["pid"], re_formula=re_formula).fit(
                    reml=True, method=opt, maxiter=SM_MAXITER, disp=False)
            geo = core.covariance_geometry(res, has_slope)
            G = np.asarray(res.cov_re, float)
            s2 = float(res.scale)
            ok, Gc = admissible_covariance(G, s2)
            c.update(fitted=True, sm_llf=float(res.llf), converged=bool(res.converged),
                     strict=bool(geo["strict_numerical_validity"]), basic=bool(geo["basic_numerical_validity"]),
                     admissible=ok, G=Gc, s2=s2, llf=prob.loglik(Gc, s2) if ok else -np.inf)
        except Exception as exc:
            c.update(fitted=False, admissible=False, llf=-np.inf, error=repr(exc)[:120])
        cands.append(c)

    G, s2, ll, info = prob.maximize(n_starts=n_starts, seed=seed)
    cands.append({"model_role": model_role, "source": "own_multistart", "fitted": True, "admissible": True,
                  "G": G, "s2": s2, "llf": ll, **info})

    if has_slope:
        ri = REMLProblem(data, formula, "1")
        g1, s21, ll1, info1 = ri.maximize(n_starts=max(4, n_starts // 2), seed=seed)
        G_emb = np.array([[g1[0, 0], 0.0], [0.0, 0.0]])
        ll_emb = prob.loglik(G_emb, s21)
        cands.append({"model_role": model_role, "source": "random_intercept_embedded", "fitted": True,
                      "admissible": True, "G": G_emb, "s2": s21, "llf": ll_emb,
                      "embedding_check": ll_emb - ll1, **info1})   # should be ~0: same marginal model

    def best_of(cs):
        ok = [c for c in cs if c.get("admissible") and np.isfinite(c["llf"])]
        return max(ok, key=lambda c: c["llf"]) if ok else None

    def n_agree(b):
        return int(sum(c.get("admissible", False) and np.isfinite(c["llf"]) and b["llf"] - c["llf"] <= AGREE_TOL
                       for c in cands))

    best = best_of(cands)
    pre_restart_llf = best["llf"] if best is not None else np.nan
    restarts = 0
    status = "no_admissible_fit"
    while best is not None:
        if n_agree(best) >= 2:
            status = "agreed" if restarts == 0 else "confirmed_on_restart"
            break
        if restarts >= MAX_RESTARTS:
            status = "unresolved_single_source"
            break
        restarts += 1
        Gr, s2r, llr, infor = prob.maximize(n_starts=4, seed=seed + restarts,
                                            extra_starts=[prob.pack(best["G"], best["s2"])])
        cands.append({"model_role": model_role, "source": f"restart_{restarts}", "fitted": True,
                      "admissible": True, "G": Gr, "s2": s2r, "llf": llr, **infor})
        best = best_of(cands)          # always the highest likelihood
    improved_by_restart = bool(best is not None and best["source"].startswith("restart_")
                               and best["llf"] > pre_restart_llf + AGREE_TOL)

    # frozen S5 rule, emulated on the statsmodels candidates
    sm = [c for c in cands if c["source"].startswith("statsmodels_") and c.get("fitted")]
    frozen = next((c for c in sm if c["strict"]), None) or next((c for c in sm if c["basic"]), None) or (sm[0] if sm else None)

    rec = {"model_role": model_role, "fit_status": status, "fit_ok": status in OK_STATUSES,
           "restarts": restarts, "improved_by_restart": improved_by_restart}
    if best is not None:
        rec.update(G=best["G"], s2=best["s2"], llf=best["llf"], best_source=best["source"], n_agree=n_agree(best),
                   **{f"best_{k}": v for k, v in boundary_status(best["G"], best["s2"]).items()})
        beta, cov_beta, eb = prob.gls(best["G"], best["s2"])
        rec.update(beta=beta, cov_beta=cov_beta, eb=eb)
    if frozen is not None and best is not None:
        rec["frozen_source"] = frozen["source"]
        rec["frozen_minus_best_llf"] = (frozen["llf"] - best["llf"]) if frozen.get("admissible") else -np.inf
        if has_slope and frozen.get("admissible"):
            rec["frozen_rho"] = boundary_status(frozen["G"], frozen["s2"]).get("rho", np.nan)
    rows = []
    for c in cands:
        row = {k: v for k, v in c.items() if k not in ("G",)}
        if "G" in c and c.get("admissible"):
            Gc = np.asarray(c["G"])
            row.update(G00=Gc[0, 0], **({"G0s": Gc[0, 1], "Gss": Gc[1, 1]} if Gc.shape == (2, 2) else {}))
            row["gap_to_best"] = c["llf"] - rec.get("llf", np.nan)
        row["selected"] = c is best
        row["frozen_rule_selected"] = c is frozen
        rows.append(row)
    return rec, rows
