#!/usr/bin/env python3
"""Numerical checks of the Supplement S2 identities (no data; random configurations only).

  1. Proposition 2(a) composition and 2(c) bounds: 20,000 random configurations
     (1-2 nuisance effects, 3-6 occasions, diagonal and correlated R).
  2. Corollary S1 closed form for kappa: 5,000 random irregular random-intercept/random-slope schedules.
  3. Invariance of D2 and kappa under affine recoding of time: 3,000 random cases.
  4. Proposition 3 finite-epsilon expression and limit: heteroscedastic four-occasion design.
  5. Remark S1 singular-G example (tau2 = 1): J = 1/2, R_direct = 1/3, D2 = 1/2, Delta = 1/6.
Writes s2_numerical_checks.json with the recorded maxima. Usage: python3 s2_numerical_checks.py [out.json]
"""
import json
import sys

import numpy as np


def general(Za, zs, G, R):
    Z = np.column_stack([Za, zs]); V = Z @ G @ Z.T + R; tau2 = G[-1, -1]
    w = np.linalg.solve(V, Z @ G[:, -1]); D2 = (w @ Z @ G[:, -1]) / tau2
    W = np.linalg.inv(R); M = W - W @ Za @ np.linalg.pinv(Za.T @ W @ Za) @ Za.T @ W
    J = zs @ M @ zs; Rd = tau2 * J / (1 + tau2 * J)
    S = G[:-1, :-1] - np.outer(G[:-1, -1], G[:-1, -1]) / tau2
    kappa = (w @ Za @ S @ Za.T @ w) / (tau2 * D2)
    resid = (w @ R @ w) / (tau2 * D2)
    return D2, Rd, D2 - Rd, kappa, resid, w, S


def rirs(t, G, s2):
    n = len(t); Z = np.column_stack([np.ones(n), t]); V = Z @ G @ Z.T + s2 * np.eye(n)
    w = np.linalg.solve(V, Z @ G[:, 1]); tau2 = G[1, 1]; D2 = w @ Z @ G[:, 1] / tau2
    S = G[0, 0] - G[0, 1] ** 2 / tau2
    return D2, (w.sum() ** 2) * S / (tau2 * D2), w


def main(out):
    rec = {}
    rng = np.random.default_rng(1)
    worst_comp, worst_bound, worst_nonneg, n_viol, nonfinite = 0.0, -np.inf, np.inf, 0, 0
    for _ in range(20000):
        n = rng.integers(3, 7); p = rng.integers(1, 3)
        t = np.sort(rng.uniform(0, 5, n)); zs = t - t.mean() * rng.uniform(0, 1.5)
        Za = np.column_stack([np.ones(n)] + [(t - 2) ** 2] * (p - 1)) if p > 1 else np.ones((n, 1))
        A = rng.normal(size=(p + 1, p + 1)); G = A @ A.T * rng.uniform(.1, 3)
        B = rng.normal(size=(n, n)) * .3
        R = np.diag(rng.uniform(.2, 3, n)) + (B @ B.T if rng.random() < .5 else 0)
        D2, Rd, Df, k, res, _, _ = general(Za, zs, G, R)
        if not np.all(np.isfinite([D2, Rd, Df, k, res])):
            nonfinite += 1
            continue
        comp = abs(D2 + k + res - 1)
        gap = k - min(1 - D2, Df / D2)
        worst_comp = max(worst_comp, comp); worst_bound = max(worst_bound, gap); worst_nonneg = min(worst_nonneg, k, Df)
        n_viol += int(comp > 1e-8 or gap > 1e-9 or k < -1e-10 or Df < -1e-10)
    rec["prop2_random_configs"] = dict(n=20000, violations=n_viol, max_abs_composition_error=worst_comp,
                                       max_kappa_minus_ceiling=worst_bound, min_kappa_or_Delta=worst_nonneg,
                                       nonfinite=nonfinite)

    rng = np.random.default_rng(2); err = 0.0; nonfinite_cf = 0; skipped_cf = 0; checked_cf = 0
    for _ in range(5000):
        n = rng.integers(2, 7); t = np.sort(rng.uniform(0, 5, n)) - rng.uniform(0, 4)
        g00 = rng.uniform(.1, 5); tau2 = rng.uniform(.05, 3); rho = rng.uniform(-.99, .99)
        g0s = rho * np.sqrt(g00 * tau2); s2 = rng.uniform(.1, 4)
        G = np.array([[g00, g0s], [g0s, tau2]]); D2, k, _ = rirs(t, G, s2)
        tc = t - t.mean(); J = tc @ tc / s2; Rd = tau2 * J / (1 + tau2 * J); S = g00 - g0s ** 2 / tau2
        if not np.all(np.isfinite([D2, Rd, k, S])):
            nonfinite_cf += 1
            continue
        if not (Rd < 1 - 1e-9 and D2 > 1e-9):
            skipped_cf += 1  # outside the closed form's domain (legitimately skipped)
            continue
        cf = (D2 - Rd) * (1 - D2) / (D2 * (1 - Rd)) * S / (S + s2 / n)
        if not np.isfinite(cf):
            nonfinite_cf += 1
            continue
        checked_cf += 1
        err = max(err, abs(cf - k))
    rec["corollary_S1_closed_form"] = dict(n=5000, checked=checked_cf, skipped_outside_domain=skipped_cf,
                                           max_abs_error=err, nonfinite=nonfinite_cf)

    rng = np.random.default_rng(3); err = 0.0; nonfinite_af = 0
    for _ in range(3000):
        n = rng.integers(2, 7); t = np.sort(rng.uniform(0, 5, n)); g00 = rng.uniform(.1, 5); tau2 = rng.uniform(.05, 3)
        rho = rng.uniform(-.99, .99); g0s = rho * np.sqrt(g00 * tau2); s2 = rng.uniform(.1, 3)
        G = np.array([[g00, g0s], [g0s, tau2]]); D2, k, _ = rirs(t, G, s2)
        t0 = rng.uniform(-3, 3); b = rng.uniform(.2, 4) * rng.choice([-1, 1])
        T = np.array([[1, t0], [0, 1 / b]]); D2b, kb, _ = rirs(b * (t - t0), T @ G @ T.T, s2)
        if not np.all(np.isfinite([D2, D2b, k, kb])):
            nonfinite_af += 1
            continue
        err = max(err, abs(D2 - D2b), abs(k - kb))
    rec["affine_invariance"] = dict(n=3000, max_abs_difference=err, nonfinite=nonfinite_af)

    t = np.array([0., 1, 2, 3.5]); zs = t - 1.0; n = 4; R = np.diag([1, .5, 2, 1.5]); sa2 = 2.0; rows = []
    for eps in (.5, .1, 1e-3):
        G = sa2 * np.array([[1, eps], [eps, eps ** 2]]); Z = np.column_stack([np.ones(n), zs]); V = Z @ G @ Z.T + R
        w = np.linalg.solve(V, Z @ G[:, 1]); D2 = w @ Z @ G[:, 1] / G[1, 1]
        v = 1 + eps * zs; f = sa2 * v @ np.linalg.solve(R, v)
        rows.append(dict(eps=eps, D2_direct=D2, D2_formula=f / (1 + f)))
    o = np.ones(n); f0 = sa2 * o @ np.linalg.solve(R, o)
    rec["prop3_rank_one"] = dict(rows=rows, limit=f0 / (1 + f0),
                                 balanced_three_wave_limits={"q0=.5": .5 * 3 / (1 + .5 * 3), "q0=4": 4 * 3 / (1 + 4 * 3)})

    G = np.array([[0., 0.], [0., 1.]]); D2, Rd, Df, *_ = general(np.ones((2, 1)), np.array([0., 1.]), G, np.eye(2))
    rec["remark_S1_singular_example"] = dict(D2=D2, R_direct=Rd, Delta_fit=Df)

    # ---- pass/fail with explicit tolerances (NaN-safe: every value must be finite)
    TOL = 1e-10
    p2 = rec["prop2_random_configs"]
    rows3 = rec["prop3_rank_one"]["rows"]
    sing = rec["remark_S1_singular_example"]
    vals = [p2["max_abs_composition_error"], p2["max_kappa_minus_ceiling"], p2["min_kappa_or_Delta"],
            rec["corollary_S1_closed_form"]["max_abs_error"], rec["affine_invariance"]["max_abs_difference"],
            rec["prop3_rank_one"]["limit"], *[r["D2_direct"] for r in rows3], *[r["D2_formula"] for r in rows3],
            sing["D2"], sing["R_direct"], sing["Delta_fit"]]
    failures = []
    if not np.all(np.isfinite(vals)):
        failures.append("non-finite value in recorded checks")
    for key in ("prop2_random_configs", "corollary_S1_closed_form", "affine_invariance"):
        if rec[key]["nonfinite"] != 0:
            failures.append(f"{key}: {rec[key]['nonfinite']} non-finite results")
    if p2["violations"] != 0:
        failures.append(f"Proposition 2: {p2['violations']} violations")
    if rec["corollary_S1_closed_form"]["checked"] == 0:
        failures.append("Corollary S1: no cases checked")
    if rec["corollary_S1_closed_form"]["max_abs_error"] > TOL:
        failures.append("Corollary S1 closed form error above tolerance")
    if rec["affine_invariance"]["max_abs_difference"] > TOL:
        failures.append("affine invariance difference above tolerance")
    if any(abs(r["D2_direct"] - r["D2_formula"]) > TOL for r in rows3):
        failures.append("Proposition 3 finite-epsilon formula mismatch")
    if abs(rec["prop3_rank_one"]["limit"] - rows3[-1]["D2_direct"]) > 1e-3:
        failures.append("Proposition 3 limit not approached at eps = 1e-3")
    if max(abs(sing["D2"] - .5), abs(sing["R_direct"] - 1 / 3), abs(sing["Delta_fit"] - 1 / 6)) > TOL:
        failures.append("Remark S1 example mismatch")
    rec["tolerance"] = TOL
    rec["status"] = "PASS" if not failures else "FAIL"
    rec["failures"] = failures
    json.dump(rec, open(out, "w"), indent=2, default=float)
    print(json.dumps(rec, indent=2, default=float))
    if failures:
        sys.exit("S2 numerical checks FAILED: " + "; ".join(failures))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "s2_numerical_checks.json")
