#!/usr/bin/env python3
"""Operating-study audit, sample audit of balanced three-wave fits (Supplement S5.3).

Question: are the original near-rank-one fits (|rho_hat| near 1, small tau2_hat) REML maxima,
or approximations to a zero-slope-variance solution that the original parameterization cannot
reach exactly?

The original fitter (simulation_core.fit_specialized_reml in reproduction/s4_operating_study)
uses one start and bounded L-BFGS-B on log-Cholesky parameters (log-diagonals >= -12), with
Powell only if L-BFGS-B fails. Because tau2 = l21^2 + exp(2*l22), tau2 = 0 is not representable
at finite parameters, and the correlation approaches +/-1 when exp(2*l22) becomes negligible
relative to l21^2.

Part A (original output only): restricted LR by fitted-covariance class.
Part B (refits a stratified sample; the cells are synthetic, so no restricted inputs are needed):
  1. regenerate y from the stored seed with the released S4 generator;
  2. rerun the original fitter (random-slope and random-intercept models) and compare with the
     stored tau2_hat, rho_hat and restricted LR (environment check);
  3. evaluate under one REML objective (reml_tools.REMLProblem, same constant as the S4 fitter)
     the original solution, the original-procedure null solution, an independent multistart
     maximum and the random-intercept solution embedded at zero slope variance;
  4. record likelihood differences and the changes in tau2, rho and D2 between solutions
     (D2 is undefined at tau2 = 0 and is reported as missing there);
  5. with --ranges only: exploratory D2 extrema within delta of the best likelihood found.
     These local searches are not certified ranges, and delta = 1.92 is not a calibrated cutoff.
A random-intercept solution that ties within tolerance shows weak distinction; it does not by
itself show that the unique maximum has zero slope variance.

Usage (from the repository root):
    python3 reproduction/pm2026/s4_operating_audit/s4_step3_fit_audit.py --repo-root . \
        --s4-dir <original S4 output dir> --reml-tools reproduction/pm2026/s3_downstream/reml_tools.py \
        --per-class 10 --out s4_step3_audit.csv
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CELLS = ["true_zero__balanced_three_wave__q0_0.5", "true_zero__balanced_three_wave__q0_4",
         "core__q1_0.05__q0_0.5__rho_0", "core__q1_0.05__q0_4__rho_0"]
TIMES = np.array([-0.5, 0.0, 0.5])
N_PEOPLE = 600
SING_TOL, NEAR_RHO = 1e-4, 0.95
TIE_TOL = 0.01
DELTAS = (0.01, 1.92)


def load(name, path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def classify(row):
    """Same rule as the step-2 script."""
    tau, rho = row.random_slope_variance, row.intercept_slope_correlation
    if not np.isfinite(tau) or tau < 0 or (np.isfinite(rho) and abs(rho) > 1 + 1e-8):
        return "invalid_parameter"
    if tau == 0:
        return "exact_zero_slope"
    if not np.isfinite(rho):
        return "unknown_correlation"
    if bool(row.boundary_flag) or 1 - rho ** 2 <= SING_TOL:
        return "boundary_or_singular_proxy"
    if abs(rho) >= NEAR_RHO:
        return "near_correlation"
    return "interior_correlation"


def regenerate(seed, q0, q1, rho):
    """Copy of S4 generate_simulation_outcome for balanced_three_wave, misspecification 'none'."""
    rng = np.random.default_rng(int(seed))
    ref_info = float(np.sum((TIMES - TIMES.mean()) ** 2))  # median information = 0.5
    slope_var = q1 / ref_info if q1 > 0 else 0.0
    cov = rho * np.sqrt(q0 * slope_var) if slope_var > 0 else 0.0
    G = np.array([[q0, cov], [cov, slope_var]])
    re = rng.multivariate_normal(np.zeros(2), G, size=N_PEOPLE, check_valid="ignore")
    t = np.tile(TIMES, N_PEOPLE)
    gi = np.repeat(np.arange(N_PEOPLE), len(TIMES))
    y = 0.10 * t + (re[gi, 0] + re[gi, 1] * t) + rng.normal(0, np.ones(len(t)))
    return y, t, gi


def generator_reference(path, design, template):
    """Read only the pure generator and its seed constant from the local S4 source.

    This avoids executing the generator's orchestration or restricted-data setup.
    """
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "generate_simulation_outcome")
    seed_nodes = [n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "SEED" for t in n.targets)]
    if len(seed_nodes) != 1:
        raise ValueError("Cannot identify the S4 generator's SEED constant")
    seed = ast.literal_eval(seed_nodes[0].value)
    ns = {"np": np, "SEED": seed, "simulation_manifest": design,
          "SIMULATION_CELL_LOOKUP": design.set_index("cell_id").to_dict("index"),
          "SCHEDULE_TEMPLATES": {"balanced_three_wave": template}}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["generate_simulation_outcome"]


def s4_template():
    pt = [TIMES.copy() for _ in range(N_PEOPLE)]
    positions = np.arange(N_PEOPLE * 3).reshape(N_PEOPLE, 3)
    times = np.tile(TIMES, (N_PEOPLE, 1))
    design = np.stack([np.ones_like(times), times], axis=2)
    return {"participant_times": pt, "flat_times": np.tile(TIMES, N_PEOPLE),
            "group_index": np.repeat(np.arange(N_PEOPLE), 3), "participants": N_PEOPLE,
            "observations": N_PEOPLE * 3, "by_length": {3: {"positions": positions, "times": times, "design": design}},
            "schedule_information": np.full(N_PEOPLE, 0.5), "reference_information": 0.5}


def d2_common(G, s2):
    """Full posterior slope determinacy for the common schedule; NaN at tau2 = 0."""
    G = np.asarray(G, float)
    tau2 = G[1, 1]
    if not tau2 > 1e-300:
        return np.nan
    Z = np.column_stack([np.ones(3), TIMES])
    V = Z @ G @ Z.T + s2 * np.eye(3)
    g = Z @ G[:, 1]
    return float(g @ np.linalg.solve(V, g) / tau2)


def rho_of(G):
    return G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]) if G[0, 0] > 0 and G[1, 1] > 0 else np.nan


def d2_range(prob, ll_best, delta, starts):
    """Exploratory feasible extrema, not certified global bounds or confidence limits.

    Only successful, feasible SLSQP results count. Parameter boxes restrict the
    search; zero slope is excluded because D2 is undefined there.
    """
    def d2_theta(th):
        G, s2 = prob.unpack(th)
        v = d2_common(G, s2)
        return v if np.isfinite(v) else np.nan

    def cons(th):
        G, s2 = prob.unpack(th)
        return prob.loglik(G, s2) - (ll_best - delta)

    out, successes = [], []
    for sign in (+1, -1):
        best = None
        successful = 0
        for th0 in starts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = minimize(lambda th: sign * d2_theta(th), th0, method="SLSQP",
                             constraints=[{"type": "ineq", "fun": cons}],
                             bounds=[(-12, 6), (-50, 50), (-20, 6), (-12, 6)],
                             options={"maxiter": 500, "ftol": 1e-10})
            if r.success and np.isfinite(r.fun) and cons(r.x) >= -1e-6:
                successful += 1
                val = d2_theta(r.x)
                if best is None or sign * val < sign * best:
                    best = val
        out.append(best if best is not None else np.nan)
        successes.append(successful)
    if np.all(np.isfinite(out)) and out[0] > out[1] + 1e-6:
        raise RuntimeError("Inconsistent exploratory D2 extrema")
    return out[0], out[1], successes[0], successes[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--s4-dir", type=Path, required=True)
    ap.add_argument("--reml-tools", type=Path,
                    help="Explicit path to the likelihood-selection reml_tools.py")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--ranges", action="store_true",
                    help="Optional exploratory constrained D2 searches; omit for first audit")
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("s4_step3_audit.csv"))
    a = ap.parse_args()

    if a.per_class < 1:
        ap.error("--per-class must be positive")
    rt_path = a.reml_tools or a.repo_root / "reproduction/s5b_contamination_pilot/reml_tools.py"
    core_path = a.repo_root / "reproduction/s4_operating_study/simulation_core.py"
    gen_path = a.repo_root / "reproduction/s4_operating_study/generate_operating_study.py"
    rt = load("reml_tools", rt_path)
    s4 = load("s4_core", core_path)
    design = pd.read_csv(a.s4_dir / "operating_study_design.csv")
    res = pd.read_csv(a.s4_dir / "operating_study_replicate_results.csv", low_memory=False)
    before = pd.read_csv(a.s4_dir / "operating_study_replicate_results_before_targeted_refit.csv", low_memory=False)
    keys = ["cell_id", "replicate"]
    if res.duplicated(keys).any() or before.duplicated(keys).any():
        raise ValueError("Duplicate replicate keys")
    before = before.set_index(keys)
    res = res[res.cell_id.isin(CELLS)].copy()
    if set(res.cell_id) != set(CELLS):
        raise ValueError("One or more requested balanced cells are missing")
    if not res.schedule.eq("balanced_three_wave").all() or not res.misspecification.eq("none").all():
        raise ValueError("This audit supports only the four specified balanced, correctly specified cells")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res["sing"] = res.apply(classify, axis=1)
    pd.set_option("display.width", 250, "display.max_rows", 200)

    print("=== Part A: archived restricted LR by cell and class (all fits; no refitting) ===")
    A = res.groupby(["cell_id", "sing"]).agg(
        n=("replicate", "size"), LR_median=("restricted_LR", "median"),
        LR_le_1e4=("restricted_LR", lambda s: float((s <= 1e-4).mean())),
        LR_le_001=("restricted_LR", lambda s: float((s <= 0.01).mean())),
        LR_gt_271=("restricted_LR", lambda s: float((s > 2.71).mean())),
        tau2_hat_median=("random_slope_variance", "median"),
        abs_rho_median=("absolute_intercept_slope_correlation", "median"),
        D2_hat_median=("mean_Pfull", "median"))
    print(A.round(5).to_string())
    A.to_csv(a.out.with_name(a.out.stem + "_archive_summary.csv"))

    pick = pd.concat([g.sample(min(a.per_class, len(g)), random_state=a.sample_seed)
                      for _, g in res.groupby(["cell_id", "sing"])])
    print(f"\n=== Part B: refitting {len(pick)} sampled replicates ===", flush=True)
    pick.to_csv(a.out.with_name(a.out.stem + "_sample.csv"), index=False)
    template = s4_template()
    reference = generator_reference(gen_path, design, template)
    metadata = {"per_class": a.per_class, "sample_seed": a.sample_seed, "ranges": a.ranges,
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), rt_path, core_path, gen_path]},
                "note": "Stratified diagnostic sample, not a population failure-rate estimate. "
                        "Optional D2 extrema are exploratory and 1.92 is not a calibrated cutoff here."}
    a.out.with_name(a.out.stem + "_manifest.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    for k, (_, r) in enumerate(pick.iterrows(), 1):
        y, t, gi = regenerate(r.seed, r.q0, r.q1, r.rho)
        y_reference, seed_reference = reference(r.cell_id, int(r.replicate))
        regeneration_diff = float(np.max(np.abs(y-y_reference)))
        if int(seed_reference) != int(r.seed) or regeneration_diff > 1e-12:
            raise RuntimeError("Dataset regeneration differs from the local S4 generator")
        old = before.loc[(r.cell_id, r.replicate)]
        if int(old.seed) != int(r.seed):
            raise RuntimeError("Before/after archive seed mismatch")
        alt = s4.fit_specialized_reml(template, y, alternative=True)
        nul = s4.fit_specialized_reml(template, y, alternative=False)
        if not alt.get("fit_valid", False) or not nul.get("fit_valid", False):
            raise RuntimeError("Archived-procedure rerun failed; previous completed rows are checkpointed")
        lr_rerun = max(0.0, 2 * (alt["log_likelihood"] - nul["log_likelihood"]))
        row = {"cell_id": r.cell_id, "replicate": int(r.replicate), "seed": int(r.seed), "sing": r.sing,
               "strict_fit_valid": bool(r.strict_fit_valid), "repair_attempted": bool(r.repair_attempted),
               "generator_max_absdiff": regeneration_diff,
               "regen_tau2_absdiff": abs(alt["random_slope_variance"] - old.random_slope_variance),
               "regen_rho_absdiff": abs(alt["intercept_slope_correlation"] - old.intercept_slope_correlation),
               "regen_LR_absdiff": abs(lr_rerun - old.restricted_LR)}
        data = pd.DataFrame({"pid": gi, "t": t, "y": y})
        prob = rt.REMLProblem(data, "y ~ t", "1+t")
        G_arch = np.array([[r.random_intercept_variance, r.intercept_slope_covariance],
                           [r.intercept_slope_covariance, r.random_slope_variance]])
        ll_arch = prob.loglik(G_arch, r.residual_variance)
        ll_arch_s4 = s4.vectorized_reml_loglike(template, y, G_arch, r.residual_variance)[0]
        row["objective_check"] = ll_arch - ll_arch_s4  # IDENTICAL parameters and data
        G_own, s2_own, ll_own, info = prob.maximize(n_starts=8, seed=int(r.seed) % 2**31,
                                                extra_starts=[prob.pack(G_arch, r.residual_variance)])
        ri = rt.REMLProblem(data, "y ~ t", "1")
        g1, s21, ll_ri_q1, info_ri = ri.maximize(n_starts=4, seed=1)
        G_emb = np.array([[g1[0, 0], 0.0], [0.0, 0.0]])
        ll_emb = prob.loglik(G_emb, s21)
        ll_null_s4 = prob.loglik(np.array([[nul["random_intercept_variance"], 0], [0, 0]]), nul["residual_variance"])
        if not np.all(np.isfinite([ll_arch, ll_own, ll_emb])):
            raise RuntimeError("Nonfinite candidate likelihood; inspect the last checkpoint")
        ll_best = max(ll_arch, ll_own, ll_emb)
        objective_ok = np.isfinite(ll_arch) and abs(row["objective_check"]) <= 1e-5
        embedding_ok = np.isfinite(ll_emb) and abs(ll_emb-ll_ri_q1) <= 1e-5
        own_ok = bool(info.get("opt_nm_success", False) and info["opt_starts_within_tol"] >= 2
                      and np.isfinite(ll_own) and ll_best-ll_own <= TIE_TOL)
        ri_ok = bool(info_ri.get("opt_nm_success", False) and info_ri["opt_starts_within_tol"] >= 2
                     and np.isfinite(ll_ri_q1))
        row.update(objective_ok=objective_ok, embedding_ok=embedding_ok,
                   own_fit_corroborated=own_ok, null_fit_corroborated=ri_ok,
                   audit_ok=bool(objective_ok and embedding_ok and own_ok and ri_ok),
                   own_opt_success=bool(info.get("opt_nm_success", False)),
                   null_opt_success=bool(info_ri.get("opt_nm_success", False)),
                   null_opt_starts_within_tol=info_ri["opt_starts_within_tol"])
        row.update(
            ll_archived_minus_best=ll_arch - ll_best, ll_own_minus_best=ll_own - ll_best,
            ll_zero_slope_minus_best=ll_emb - ll_best, ll_s4null_minus_zero_slope=ll_null_s4 - ll_emb,
            embedding_check=ll_emb - ll_ri_q1,
            # archived LR versus the LR implied by the archived alternative and the TRUE zero-slope maximum;
            # a positive gap means the archived null fit fell short (LR inflated)
            LR_archived=r.restricted_LR, LR_vs_zero_slope_max=max(0.0, 2 * (ll_arch - ll_emb)),
            LR_vs_zero_slope_raw=2 * (ll_arch - ll_emb),
            LR_archived_minus_implied=r.restricted_LR - max(0.0, 2 * (ll_arch - ll_emb)),
            best_is=("zero_slope" if ll_emb >= ll_best - 1e-9 else "own" if ll_own >= ll_best - 1e-9 else "archived"),
            zero_slope_ties_best=bool(ll_best - ll_emb <= TIE_TOL),
            own_opt_starts_within_tol=info["opt_starts_within_tol"],
            tau2_archived=r.random_slope_variance, tau2_own=G_own[1, 1],
            rho_archived=r.intercept_slope_correlation, rho_own=rho_of(G_own),
            D2_archived=d2_common(G_arch, r.residual_variance), D2_archived_reported=r.mean_Pfull,
            D2_own=d2_common(G_own, s2_own))
        row["D2_best"] = (np.nan if row["best_is"] == "zero_slope" else
                          row["D2_own"] if row["best_is"] == "own" else row["D2_archived"])
        theta_own = prob.pack(G_own, s2_own)
        starts = [theta_own, prob.pack(G_arch, r.residual_variance),
                  prob.pack(np.array([[g1[0, 0], 0.0], [0.0, 1e-6]]), s21),
                  prob.pack(np.array([[g1[0, 0], 0.999 * np.sqrt(g1[0, 0] * 1e-3)], [0.999 * np.sqrt(g1[0, 0] * 1e-3), 1e-3]]), s21)]
        for delta in DELTAS:
            lo, hi, nlo, nhi = (np.nan, np.nan, 0, 0)
            if a.ranges and row["audit_ok"]:
                lo, hi, nlo, nhi = d2_range(prob, ll_best, delta, starts)
            row[f"D2_min_within_{delta}"], row[f"D2_max_within_{delta}"] = lo, hi
            row[f"D2_min_successes_{delta}"], row[f"D2_max_successes_{delta}"] = nlo, nhi
        rows.append(row)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"  {k}/{len(pick)} audit_ok={row['audit_ok']}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print("\nChecks: max |tau2 rerun - archive| %.2e; max |LR rerun - archive| %.2e; max |objective - S4 loglik| %.2e; "
          "max |embedding check| %.2e" % (out.regen_tau2_absdiff.max(), out.regen_LR_absdiff.max(),
                                          out.objective_check.abs().max(), out.embedding_check.abs().max()))
    print("  Original-procedure LR mismatches versus BEFORE-refit archive (> 1e-3): %d of %d. "
          "Dataset regeneration is checked separately against the local generator. Rerun fit differences "
          "can reflect environment or optimization and are not automatically regeneration failures."
          % (int((out.regen_LR_absdiff > 1e-3).sum()), len(out)))
    summ = out.groupby(["cell_id", "sing", "audit_ok"]).agg(
        n=("seed", "size"),
        audit_ok_n=("audit_ok", "sum"),
        archived_below_best_gt_tol=("ll_archived_minus_best", lambda s: int((s < -TIE_TOL).sum())),
        archived_gap_min=("ll_archived_minus_best", "min"),
        zero_slope_ties_best=("zero_slope_ties_best", "sum"),
        s4_null_rerun_below_zero_slope_gt_tol=("ll_s4null_minus_zero_slope", lambda s: int((s < -TIE_TOL).sum())),
        regen_LR_mismatch=("regen_LR_absdiff", lambda s: int((s > 1e-3).sum())),
        archived_LR_minus_implied_max=("LR_archived_minus_implied", lambda s: s.abs().max()),
        best_is_zero_slope=("best_is", lambda s: int((s == "zero_slope").sum())),
        tau2_archived_median=("tau2_archived", "median"), tau2_own_median=("tau2_own", "median"),
        D2_archived_median=("D2_archived", "median"), D2_own_median=("D2_own", "median"),
        D2_min_001=("D2_min_within_0.01", "median"), D2_max_001=("D2_max_within_0.01", "median"),
        D2_min_192=("D2_min_within_1.92", "median"), D2_max_192=("D2_max_within_1.92", "median"))
    print("\nPer cell x class (gaps are log-likelihood units; D2 ranges are medians across replicates):")
    print(summ.round(4).reset_index().to_string(index=False))
    summ.to_csv(a.out.with_name(a.out.stem + "_summary.csv"))
    print("Unresolved audit rows:", int((~out.audit_ok).sum()))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
