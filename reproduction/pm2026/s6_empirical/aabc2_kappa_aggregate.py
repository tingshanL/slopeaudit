#!/usr/bin/env python3
"""Supplement S6.3: source decomposition and nuisance capacity for AABC2 outcomes, from local
restricted participant-level inputs. Writes aggregate summaries only (no participant
identifiers and no participant-level rows):

  s7_aabc2_decomposition.csv   per outcome: N, fitted variance components, equal-participant means
                               of D2, R_direct and Delta_fit (additive), visit-count distribution
                               (cells with fewer than 10 participants masked)
  s7_aabc2_kappa_quantiles.csv per outcome: mean kappa_i and the 5, 10, 25, 50, 75, 90 and 95th
                               percentiles of kappa_i and sqrt(kappa_i); shares with sqrt(kappa_i)
                               >= .1, .2 and .3
  s7_aabc2_checks.csv          internal consistency checks (below)
(The s7_ prefix of the output names is kept from the original run.) Whether an output may be
released is decided by the reviewed release manifest, not by this script.

kappa_i uses the closed form for a random-intercept, random-slope model with independent
equal-variance residuals (Corollary S1 in Supplement S2):
    kappa_i = Delta_i (1 - D2_i) / (D2_i (1 - R_i)) * S / (S + sigma2 / n_i),  S = G00 - g01^2 / tau2,
with n_i = scan_count. Inputs must be finite, with positive slope and residual variances,
nonnegative S, positive visit counts, 0 < D2 < 1 and 0 <= R_direct < 1; outcomes outside this
domain are flagged, not evaluated, and nonfinite kappa or ceilings are flagged.
Checks: (1) R_direct recomputed from direct_schedule_information_I_i as tau2*J/(1+tau2*J),
J = I_i/sigma2, must match the file; (2) Delta = D2 - R_direct; (3) the implied person-mean
precision C_i = (D2(A+B) - B)/(1 - D2) must be >= 0; if check (1) fails for an outcome, kappa is
not reported; (4) raw kappa is checked against 0 and both ceilings, 1 - D2 and Delta/D2, before
any clipping. These checks establish algebraic consistency with the random-intercept,
random-slope model with equal-variance residuals; they do not verify that the source model used
that residual structure.

Usage (on the data holder's machine):
  python3 reproduction/pm2026/s6_empirical/aabc2_kappa_aggregate.py \
      --input <restricted participant-level reliability CSV> \
      --outcomes lateral_ventricles fluid_cognition --out-dir <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

Q = [.05, .10, .25, .50, .75, .90, .95]
MASK = 10  # mask visit-count cells with fewer participants than this


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outcomes", nargs="+", default=["lateral_ventricles", "fluid_cognition"])
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.input, float_precision="round_trip")
    need = ["outcome", "scan_count", "random_intercept_variance", "random_slope_variance",
            "intercept_slope_covariance", "intercept_slope_correlation", "residual_variance",
            "direct_schedule_information_I_i", "same_reference_fit_direct_GRR",
            "full_posterior_slope_predictability", "level_assisted_reliability_gain"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise SystemExit(f"missing columns: {miss}")
    available = sorted(d.outcome.unique())
    print("outcomes in file:", available)

    dec, kq, chk = [], [], []
    for out in a.outcomes:
        g = d[d.outcome == out]
        if g.empty:
            raise SystemExit(f"outcome {out!r} not in file; available: {available}")
        for c in ["random_intercept_variance", "random_slope_variance", "intercept_slope_covariance", "residual_variance"]:
            if g[c].nunique() != 1:
                raise SystemExit(f"{out}: {c} varies across rows; expected one fitted model")
        G00, tau2 = g.random_intercept_variance.iloc[0], g.random_slope_variance.iloc[0]
        g01, s2 = g.intercept_slope_covariance.iloc[0], g.residual_variance.iloc[0]
        if not (np.isfinite([G00, tau2, g01, s2]).all() and tau2 > 0 and s2 > 0):
            raise SystemExit(f"{out}: non-finite or non-positive variance components")
        if (G00 - g01 ** 2 / tau2) < -1e-10 * max(1.0, G00):
            raise SystemExit(f"{out}: negative slope-orthogonal level variance S")
        num = g[["scan_count", "full_posterior_slope_predictability", "same_reference_fit_direct_GRR",
                 "level_assisted_reliability_gain", "direct_schedule_information_I_i"]].to_numpy(float)
        if not np.isfinite(num).all():
            raise SystemExit(f"{out}: non-finite participant-level inputs")
        n = g.scan_count.to_numpy(float)
        D2 = g.full_posterior_slope_predictability.to_numpy(float)
        R = g.same_reference_fit_direct_GRR.to_numpy(float)
        Df = g.level_assisted_reliability_gain.to_numpy(float)
        J = g.direct_schedule_information_I_i.to_numpy(float) / s2
        domain = dict(n_visits_positive=bool((n > 0).all()), D2_in_open_unit_interval=bool(((D2 > 0) & (D2 < 1)).all()),
                      Rdirect_in_0_1=bool(((R >= 0) & (R < 1)).all()), I_nonnegative=bool((J >= 0).all()))
        if not all(domain.values()):
            chk.append(dict(outcome=out, N=len(g), homoscedastic_RIRS_closed_form_valid=False, **domain))
            print(f"WARNING {out}: inputs outside the closed form's domain {domain}; nothing computed for this outcome")
            continue
        R_re = tau2 * J / (1 + tau2 * J)
        A, B = 1 / tau2, J
        C = (D2 * (A + B) - B) / (1 - D2)
        S = G00 - g01 ** 2 / tau2
        c1 = float(np.max(np.abs(R_re - R)))
        c2 = float(np.max(np.abs(Df - (D2 - R))))
        c3 = float(np.min(C))
        ok = c1 < 1e-6 and c2 < 1e-8 and c3 > -1e-8
        chk.append(dict(outcome=out, N=len(g), max_abs_Rdirect_recompute_error=c1,
                        max_abs_Delta_identity_error=c2, min_implied_person_mean_precision=c3,
                        homoscedastic_RIRS_closed_form_valid=ok))

        counts = pd.Series(n.astype(int)).value_counts().sort_index()
        vc = "; ".join(f"{k} visits: {v if v >= MASK else f'<{MASK}'}" for k, v in counts.items())
        dec.append(dict(outcome=out, N=len(g), G00=G00, tau2=tau2, cov01=g01,
                        rho=g.intercept_slope_correlation.iloc[0], sigma2=s2, S=S,
                        mean_D2=D2.mean(), mean_Rdirect=R.mean(), mean_Delta_fit=Df.mean(),
                        median_D2=np.median(D2), median_Rdirect=np.median(R), median_Delta_fit=np.median(Df),
                        mean_visits=n.mean(), visit_counts=vc))
        if not ok:
            print(f"WARNING {out}: closed-form checks failed; kappa not reported")
            continue
        S = max(S, 0.0)
        kappa = Df * (1 - D2) / (D2 * (1 - R)) * S / (S + s2 / n)
        ceiling = np.minimum(1 - D2, Df / D2)
        if not (np.isfinite(kappa).all() and np.isfinite(ceiling).all()):
            chk[-1]["homoscedastic_RIRS_closed_form_valid"] = False
            print(f"WARNING {out}: non-finite kappa or ceiling; kappa not reported")
            continue
        tol = 1e-9
        chk[-1].update(min_raw_kappa=float(kappa.min()),
                       max_kappa_minus_ceiling=float(np.max(kappa - ceiling)))
        if kappa.min() < -tol or np.max(kappa - ceiling) > tol:
            chk[-1]["homoscedastic_RIRS_closed_form_valid"] = False
            print(f"WARNING {out}: raw kappa outside [0, min(1-D2, Delta/D2)]; kappa not reported")
            continue
        kappa = np.clip(kappa, 0, None)  # removes only rounding-level negatives (checked above)
        row = dict(outcome=out, N=len(g), mean_kappa=kappa.mean(), mean_sqrt_kappa=np.sqrt(kappa).mean())
        for q in Q:
            row[f"kappa_q{int(q * 100):02d}"] = float(np.quantile(kappa, q))
            row[f"sqrt_kappa_q{int(q * 100):02d}"] = float(np.quantile(np.sqrt(kappa), q))
        for t in (.1, .2, .3):
            row[f"share_sqrt_kappa_ge_{t:.1f}"] = float(np.mean(np.sqrt(kappa) >= t))
        kq.append(row)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    for name, rows in (("checks", chk), ("decomposition", dec), ("kappa_quantiles", kq)):
        df = pd.DataFrame(rows)
        df.to_csv(a.out_dir / f"s7_aabc2_{name}.csv", index=False)
        print(f"\n== {name} ==\n" + df.round(5).to_string(index=False))
    print(f"\nwrote aggregate files to {a.out_dir} (no participant-level rows)")


if __name__ == "__main__":
    main()
