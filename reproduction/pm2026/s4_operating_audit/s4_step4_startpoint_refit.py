#!/usr/bin/env python3
"""Operating-study audit, refit of selected data sets (Supplement S5.1, S5.3).

Used twice: for the 245 fits that remained at their starting values (first screen), and, with
--extra-keys, for the consolidated refit of all 6,906 screened data sets. --repo-root is the
repository root (for the released S4 generator); --reml-tools is the likelihood helper.
Empirical-schedule cells need the restricted schedule JSON (see restricted/README.md). The seed
and the original starting variance are checked against the regenerated data, and the manifest
records the hashes of all inputs and sources; the variance check alone cannot certify
participant ordering.

The random-intercept solution is fit first and embedded among the random-slope candidates.
Distinct starts are optimized and their success flags kept. A refit is accepted only when the
optimum is reached by at least two successful distinct starts within .01 log-likelihood units,
the candidates are finite and admissible, the objectives agree with the S4 fitter, the
embedding is exact, the likelihoods are nested and the determinacy summaries are finite. These
are numerical checks, not a guarantee of a global maximum.

The audit CSV is checkpointed after every candidate. Exceptions and unresolved fits are marked
not strictly valid in the patched copy and cause a nonzero exit; the original inputs are not
changed. --smoke refits one row per family and schedule and is only a software check.

After a full run, summarize the patched table with reproduction/s4_operating_study/
summarize_operating_study.py. No Hessian is evaluated for refit rows, so their assessed Hessian
count is reported separately. At zero slope variance the original zero-summary convention is
kept; slope determinacy itself is undefined there, and such selections are flagged.

Usage: see reproduction/pm2026/README.md for the exact commands.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TIE_TOL = 0.01
SLOPE_BOUNDARY_REL_TOL = 1e-8
SING_EIG_REL = 1e-10
METHOD = "startpoint_screen_likelihood_refit_v2"
G = {}  # per-process globals


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def init_worker(repo_root, s4_dir, schedule_json, reml_tools, needed_schedules):
    s4_code = Path(repo_root) / "reproduction" / "s4_operating_study"
    sys.path.insert(0, str(s4_code))
    import simulation_core as s4core          # S4 core, as the generator imports it
    import generate_operating_study as gen
    rt = load_by_path("reml_tools", Path(reml_tools))
    design = pd.read_csv(Path(s4_dir) / "operating_study_design.csv")
    # Without the restricted schedule JSON only the balanced template can be built (smoke runs);
    # the full design is still used for the seed formula and the cell lookup.
    # Build only schedules required by the selected tasks; keep the FULL design
    # below for the historical seed formula and cell lookup.
    template_design = design[design.schedule.isin(needed_schedules)]
    templates = gen.load_schedule_templates(Path(schedule_json) if schedule_json else None, template_design, "full")
    gen.SCHEDULE_TEMPLATES = templates
    gen.SIMULATION_CELL_LOOKUP = design.set_index("cell_id").to_dict("index")
    gen.simulation_manifest = design
    G.update(s4=s4core, gen=gen, rt=rt, templates=templates, design=design)


def admissible(Gm, s2):
    return (np.isfinite(Gm).all() and np.isfinite(s2) and s2 > 0
            and np.linalg.eigvalsh(Gm).min() >= -1e-10 * max(1., np.linalg.norm(Gm)))


def maximize_checked(prob, n_starts, seed, extra_starts=()):
    """Keep the best finite candidate; count successful, distinct starts separately."""
    starts = []
    for x in list(extra_starts) + prob.starting_values(np.random.default_rng(seed), n_starts):
        x = np.asarray(x, float)
        if np.isfinite(x).all() and not any(np.allclose(x, y, rtol=1e-10, atol=1e-12) for y in starts):
            starts.append(x)
    results = []
    for x in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = minimize(prob._negll_theta, x, method="L-BFGS-B")
            final = minimize(prob._negll_theta, first.x, method="Nelder-Mead",
                             options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-9})
        gm, s2 = prob.unpack(final.x)
        ll = prob.loglik(gm, s2)
        if admissible(gm, s2) and np.isfinite(ll):
            results.append((gm, s2, ll, bool(final.success)))
    if not results:
        raise RuntimeError("No finite admissible multistart candidate")
    best = max(results, key=lambda r: r[2])
    return (*best[:3], {"opt_nm_success": best[3], "opt_starts": len(starts),
            "opt_starts_within_tol": sum(r[3] and best[2]-r[2] <= TIE_TOL for r in results)})


def select_candidate(prob, candidates):
    lls = {k: prob.loglik(gm, s2) for k, (gm, s2) in candidates.items()
           if admissible(gm, s2)}
    lls = {k: v for k, v in lls.items() if np.isfinite(v)}
    if not lls:
        raise RuntimeError("No finite admissible candidate")
    return max(lls, key=lls.get), lls


def stuck_signature(d):
    s2 = d.residual_variance
    return (np.isclose(d.random_intercept_variance, s2, rtol=1e-8, atol=0)
            & (d.intercept_slope_covariance.abs() <= 1e-10 * s2)
            & np.isclose(d.random_slope_variance, 0.1 * s2, rtol=1e-8, atol=0))


def robust_metrics(Gm, s2, template):
    """mean R_direct and mean D2 for PSD (possibly singular) G; R_innovation not defined here."""
    tau2 = Gm[1, 1]
    direct, full = [], []
    for _, block in template["by_length"].items():
        times = block["times"]
        info = np.sum((times - times.mean(axis=1, keepdims=True)) ** 2, axis=1)
        direct.append(tau2 / (tau2 + s2 / info))
        Z = block["design"]                                   # (people, m, 2)
        V = np.einsum("pki,ij,plj->pkl", Z, Gm, Z) + s2 * np.eye(times.shape[1])[None]
        g = np.einsum("pki,i->pk", Z, Gm[:, 1])
        full.append(np.einsum("pk,pk->p", g, np.linalg.solve(V, g[..., None])[..., 0]) / tau2)
    direct, full = np.concatenate(direct), np.clip(np.concatenate(full), 0, 1)
    return {"mean_Rdirect": float(direct.mean()), "mean_Pfull": float(full.mean()),
            "mean_Rinnovation": np.nan, "mean_Delta": float((full - direct).mean())}


def covariance_fields(Gm, s2):
    ev = np.linalg.eigvalsh(Gm)
    g00, g01, tau2 = Gm[0, 0], Gm[0, 1], Gm[1, 1]
    rho = g01 / np.sqrt(g00 * tau2) if g00 > 0 and tau2 > 0 else np.nan
    innov = tau2 - g01 ** 2 / g00 if g00 > 0 else np.nan
    tr = np.trace(Gm)
    out = {"random_intercept_variance": g00, "random_slope_variance": tau2, "intercept_slope_covariance": g01,
           "intercept_slope_correlation": rho, "absolute_intercept_slope_correlation": abs(rho) if np.isfinite(rho) else np.nan,
           "covariance_minimum_eigenvalue": float(ev.min()),
           "one_minus_rho_squared": 1 - rho ** 2 if np.isfinite(rho) else np.nan,
           "normalized_determinant": float(np.linalg.det(Gm) / tr ** 2) if tr > 0 else np.nan,
           "innovation_variance": innov, "innovation_fraction": innov / tau2 if tau2 > 0 and np.isfinite(innov) else np.nan,
           "residual_variance": s2,
           "boundary_flag": bool(tau2 <= SLOPE_BOUNDARY_REL_TOL * max(1.0, s2)
                                 or ev.min() <= SLOPE_BOUNDARY_REL_TOL * max(1.0, tr))}
    return out


def refit_one(task):
    s4, gen, rt = G["s4"], G["gen"], G["rt"]
    cell, rep = task["cell_id"], int(task["replicate"])
    template = G["templates"][task["schedule"]]
    t0 = time.time()
    y, seed = gen.generate_simulation_outcome(cell, rep)
    out = {"cell_id": cell, "replicate": rep, "seed_archive": int(task["seed"]), "seed_regenerated": int(seed)}
    if int(seed) != int(task["seed"]):
        out["error"] = "seed mismatch"
        return out, None
    if task.get("starting_signature", False):
        expected_s2 = .5 * max(float(np.var(y, ddof=1)), 1e-4)
        out["starting_variance_absdiff"] = abs(expected_s2 - task["residual_variance"])
        if not np.isclose(expected_s2, task["residual_variance"], rtol=1e-8, atol=1e-10):
            raise RuntimeError("Regenerated data do not match archived starting variance; verify schedules/order")
    alt_r = s4.fit_specialized_reml(template, y, alternative=True)
    nul_r = s4.fit_specialized_reml(template, y, alternative=False)
    out["rerun_tau2_absdiff"] = abs(alt_r.get("random_slope_variance", np.nan) - task["random_slope_variance"])
    data = pd.DataFrame({"pid": template["group_index"], "t": template["flat_times"], "y": y})
    pa = rt.REMLProblem(data, "y ~ t", "1+t")
    pn = rt.REMLProblem(data, "y ~ t", "1")

    G_arch = np.array([[task["random_intercept_variance"], task["intercept_slope_covariance"]],
                       [task["intercept_slope_covariance"], task["random_slope_variance"]]])
    cands = {"archived": (G_arch, task["residual_variance"])}
    if alt_r.get("fit_valid"):
        Gr = np.array([[alt_r["random_intercept_variance"], alt_r["intercept_slope_covariance"]],
                       [alt_r["intercept_slope_covariance"], alt_r["random_slope_variance"]]])
        cands["s4_rerun"] = (Gr, alt_r["residual_variance"])
    extra = [pa.pack(Gc, sc) for Gc, sc in cands.values()]
    Go, so, llo, info = maximize_checked(pa, n_starts=8, seed=int(seed) % 2 ** 31, extra_starts=extra)
    cands["independent"] = (Go, so)
    g1, s1, ll1, info1 = maximize_checked(pn, n_starts=6, seed=1)
    ncands = {"independent": (g1, s1)}
    if nul_r.get("fit_valid"):
        ncands["s4_rerun"] = (np.array([[nul_r["random_intercept_variance"]]]), nul_r["residual_variance"])
        g2, s2b, _, info2 = maximize_checked(pn, n_starts=4, seed=2,
                                            extra_starts=[pn.pack(*ncands["s4_rerun"])])
        ncands["independent_from_s4"] = (g2, s2b)
    else:
        info2 = None
    nbest, nll = select_candidate(pn, ncands)
    # Embed the FINAL selected null, so nesting holds by construction.
    gn, sn = ncands[nbest]
    cands["zero_slope_embedded"] = (np.array([[gn[0, 0], 0.], [0., 0.]]), sn)
    best, lls = select_candidate(pa, cands)
    alt_corr = bool(info["opt_nm_success"] and info["opt_starts_within_tol"] >= 2
                    and lls[best] - lls["independent"] <= TIE_TOL)
    null_corr = any(inf is not None and inf["opt_nm_success"]
                    and inf["opt_starts_within_tol"] >= 2
                    and nll[nbest] - nll[key] <= TIE_TOL
                    for key, inf in [("independent", info1), ("independent_from_s4", info2)])
    objective_diff = lls["archived"] - s4.vectorized_reml_loglike(
        template, y, G_arch, task["residual_variance"])[0]
    embedding_diff = lls["zero_slope_embedded"] - nll[nbest]
    objective_ok = bool(np.isfinite(objective_diff) and abs(objective_diff) <= 1e-5)
    embedding_ok = bool(np.isfinite(embedding_diff) and abs(embedding_diff) <= 1e-5)

    G_best, s2_best = cands[best]
    lr_raw = 2 * (lls[best] - nll[nbest])
    out.update({f"ll_alt_{k}": v for k, v in lls.items()})
    out.update({f"ll_null_{k}": v for k, v in nll.items()})
    out.update(alt_selected=best, null_selected=nbest, alt_corroborated=alt_corr, null_corroborated=null_corr,
               alt_gain_over_archived=lls[best] - lls["archived"],
               null_gain_over_s4_rerun=(nll[nbest] - nll["s4_rerun"]) if "s4_rerun" in nll else np.nan,
               LR_archived=task["restricted_LR"], LR_refit=max(0.0, lr_raw), LR_refit_raw=lr_raw,
               own_alt_starts_within_tol=info["opt_starts_within_tol"], own_null_starts_within_tol=info1["opt_starts_within_tol"],
               embedding_check=embedding_diff, objective_check=objective_diff,
               objective_ok=objective_ok, embedding_ok=embedding_ok,
               alt_optimizer_success=info["opt_nm_success"],
               null_optimizer_success=bool(null_corr))

    fields = covariance_fields(G_best, s2_best)
    fields.update(s4.anchored_endpoint_geometry(G_best))
    fields["cov_L0_L5"] = fields.get("g_0_5", np.nan)
    tau2 = G_best[1, 1]
    ev = np.linalg.eigvalsh(G_best)
    if tau2 <= 0:
        metrics, how = s4.schedule_predictability_summary(G_best, s2_best, template), "zero_slope_archive_convention"
    elif ev.min() > SING_EIG_REL * max(1.0, np.trace(G_best)):
        metrics, how = s4.schedule_predictability_summary(G_best, s2_best, template), "s4_helper"
    else:
        metrics, how = robust_metrics(G_best, s2_best, template), "v_based_singular"
    fields.update(metrics)
    fields["estimated_anchored_near"] = bool(np.isfinite(fields["anchored_endpoint_minimum_eigenvalue_fraction"])
                                             and fields["anchored_endpoint_minimum_eigenvalue_fraction"] <= 0.01)
    strict = bool(alt_corr and null_corr and objective_ok and embedding_ok
                  and np.isfinite(lr_raw) and lr_raw >= -1e-5
                  and all(np.isfinite(fields[k]) for k in ("mean_Rdirect", "mean_Pfull", "mean_Delta")))
    if "d_0_5" in task:
        fields["d_0_5"] = (1-fields["gamma_0_5"]**2 if np.isfinite(fields["gamma_0_5"]) else np.nan)
    fields.update(fit_valid=bool(objective_ok and embedding_ok), strict_fit_valid=strict,
                  alternative_converged=bool(info["opt_nm_success"]),
                  null_converged=bool(null_corr), alternative_optimizers_attempted=info["opt_starts"], restricted_LR=max(0.0, lr_raw),
                  hessian_positive_definite=np.nan, hessian_minimum_information_eigenvalue=np.nan,
                  alternative_optimizer="independent_multistart+candidates", null_optimizer="independent_multistart+candidates",
                  optimizer_agreement=alt_corr, null_optimizer_agreement=null_corr,
                  repair_attempted=True, repair_succeeded=strict, repair_method=METHOD, error="")
    out.update({f"new_{k}": v for k, v in fields.items() if k in
                ("random_slope_variance", "intercept_slope_correlation", "residual_variance", "mean_Pfull",
                 "mean_Rdirect", "boundary_flag", "strict_fit_valid")})
    out.update(old_mean_Pfull=task["mean_Pfull"], old_mean_Rdirect=task["mean_Rdirect"],
               metric_method=how, zero_slope_selected=bool(best == "zero_slope_embedded"), seconds=time.time() - t0, error="")
    out["unresolved"] = not strict
    return out, fields


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--reml-tools", type=Path, required=True,
                    help="Explicit path to the reviewed reml_tools.py")
    ap.add_argument("--smoke", action="store_true",
                    help="Select one candidate per cell family and schedule")
    ap.add_argument("--s4-dir", type=Path, required=True)
    ap.add_argument("--schedule-json", type=Path, default=None, help="restricted S4 schedule JSON (needed for empirical cells)")
    ap.add_argument("--extra-keys", type=Path, default=None, help="optional CSV with cell_id, replicate to refit as well")
    ap.add_argument("--families", nargs="*", default=None, help="restrict candidates to these cell families")
    ap.add_argument("--limit", type=int, default=None, help="smoke: refit only the first N candidates")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for path in [a.reml_tools, a.repo_root / "reproduction/s4_operating_study/simulation_core.py",
                 a.repo_root / "reproduction/s4_operating_study/generate_operating_study.py"]:
        if not path.is_file():
            ap.error("Missing source file: " + str(path))
    if (a.out_dir / "startpoint_refit_audit.csv").exists():
        ap.error("Output already exists; choose a new --out-dir to preserve the previous run")

    arch = pd.read_csv(a.s4_dir / "operating_study_replicate_results.csv", low_memory=False, float_precision="round_trip")
    keys = ["cell_id", "replicate"]
    if arch.duplicated(keys).any():
        raise ValueError("duplicate keys in archive")
    arch["_stuck"] = stuck_signature(arch)
    print("stuck-at-start signature by family:", arch.groupby("cell_family")._stuck.sum().astype(int).to_dict(),
          "total", int(arch._stuck.sum()))
    sel = arch._stuck.copy()
    if a.extra_keys:
        ek = pd.read_csv(a.extra_keys)[keys]
        sel |= arch.set_index(keys).index.isin(ek.set_index(keys).index)
    if a.families:
        sel &= arch.cell_family.isin(a.families)
    cand = arch[sel]
    if a.smoke:
        cand = cand.groupby(["cell_family", "schedule"], sort=True).head(1)
    if a.limit:
        cand = cand.head(a.limit)
    if (~cand.schedule.eq("balanced_three_wave")).any() and a.schedule_json is None:
        raise SystemExit("empirical-schedule candidates need --schedule-json (or use --families to exclude them)")
    print(f"refitting {len(cand)} candidates", flush=True)
    cols = ["cell_id", "replicate", "seed", "schedule", "random_intercept_variance", "intercept_slope_covariance",
            "random_slope_variance", "residual_variance", "restricted_LR", "mean_Pfull", "mean_Rdirect"]
    if "d_0_5" in cand:
        cols.append("d_0_5")
    cand = cand.copy()
    cand["starting_signature"] = cand._stuck
    tasks = cand[cols + ["starting_signature"]].to_dict("records")
    if not tasks:
        raise SystemExit("No candidates selected")

    rows, patches = [], {}
    init = (str(a.repo_root), str(a.s4_dir), str(a.schedule_json) if a.schedule_json else None,
            str(a.reml_tools), sorted(cand.schedule.unique()))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, a.workers), initializer=init_worker, initargs=init) as pool:
        futs = {pool.submit(refit_one, t): (t["cell_id"], t["replicate"]) for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                o, fields = f.result()
            except Exception as exc:
                o, fields = {"cell_id": futs[f][0], "replicate": futs[f][1], "error": repr(exc)[:200]}, None
            rows.append(o)
            pd.DataFrame(rows).to_csv(a.out_dir / "startpoint_refit_audit.csv", index=False)
            if fields is not None:
                patches[futs[f]] = fields
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)

    audit = pd.DataFrame(rows).merge(arch[keys + ["cell_family"]], on=keys, how="left")
    audit.to_csv(a.out_dir / "startpoint_refit_audit.csv", index=False)

    patched = arch.drop(columns="_stuck").copy()
    patched["startpoint_refit"] = False
    patched["startpoint_refit_metric_method"] = ""
    idx = patched.set_index(keys).index
    touched = sorted({c for f in patches.values() for c in f if c in patched.columns})
    for c in touched:                      # allow values of a different dtype (e.g., NaN Hessian flags)
        patched[c] = patched[c].astype(object)
    # A failed candidate must not remain eligible under its old strict-valid flag.
    for key in [(t["cell_id"], t["replicate"]) for t in tasks]:
        pos = np.flatnonzero(idx == key)[0]
        patched.loc[patched.index[pos], "strict_fit_valid"] = False
    for key, fields in patches.items():
        pos = np.flatnonzero(idx == key)[0]
        for col, val in fields.items():
            if col in patched.columns:
                patched.iat[pos, patched.columns.get_loc(col)] = val
        patched.iat[pos, patched.columns.get_loc("startpoint_refit")] = True
        row = audit[(audit.cell_id == key[0]) & (audit.replicate == key[1])]
        patched.iat[pos, patched.columns.get_loc("startpoint_refit_metric_method")] = row.metric_method.iloc[0]
    patched["startpoint_refit_attempted"] = idx.isin([(t["cell_id"], t["replicate"]) for t in tasks])
    patched["startpoint_refit_unresolved"] = (patched.startpoint_refit_attempted
                                             & ~patched.strict_fit_valid.astype(bool))
    unresolved = int(patched.startpoint_refit_unresolved.sum())
    patched = patched.infer_objects()
    patched.to_csv(a.out_dir / "operating_study_replicate_results_startpoint_patched.csv", index=False)

    files = [Path(__file__), a.reml_tools, a.s4_dir / "operating_study_design.csv",
             a.repo_root / "reproduction/s4_operating_study/simulation_core.py",
             a.repo_root / "reproduction/s4_operating_study/generate_operating_study.py",
             a.s4_dir / "operating_study_replicate_results.csv"]
    if a.schedule_json:
        files.append(a.schedule_json)
    (a.out_dir / "manifest.json").write_text(json.dumps({
        "method": METHOD, "smoke": a.smoke, "unresolved_rows": unresolved, "candidates": len(tasks), "patched_rows": len(patches), "tie_tol": TIE_TOL,
        "signature": "G00 == sigma2 (rtol 1e-8), |G01| <= 1e-10 sigma2, tau2 == 0.1 sigma2 (rtol 1e-8)",
        "strict_rule_for_refit_rows": "successful winning optimization corroborated by >=2 distinct successful starts within tol; objective, embedding, nesting and finite metrics checks pass",
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}}, indent=2))

    ok = audit[audit.error.fillna("") == ""]
    pd.set_option("display.width", 250)
    print("\nChecks: errors %d; seed mismatches %d; max |archived-procedure rerun tau2 - archive| %.2e; max |embedding| %.2e"
          % (int((audit.error.fillna("") != "").sum()), int((ok.seed_regenerated != ok.seed_archive).sum()),
             ok.rerun_tau2_absdiff.max(), ok.embedding_check.abs().max()))
    if ok.empty:
        raise SystemExit("No completed error-free candidates; inspect checkpoint and patched table")
    s = ok.groupby("cell_family").agg(
        n=("replicate", "size"),
        alt_gain_gt_001=("alt_gain_over_archived", lambda x: int((x > TIE_TOL).sum())),
        alt_gain_gt_1=("alt_gain_over_archived", lambda x: int((x > 1).sum())),
        alt_gain_median=("alt_gain_over_archived", "median"),
        null_gain_gt_001=("null_gain_over_s4_rerun", lambda x: int((x > TIE_TOL).sum())),
        zero_slope_selected=("zero_slope_selected", "sum"),
        uncorroborated=("new_strict_fit_valid", lambda x: int((~x.astype(bool)).sum())),
        LR_archived_zero=("LR_archived", lambda x: int((x == 0).sum())),
        LR_refit_zero=("LR_refit", lambda x: int((x == 0).sum())),
        LR_refit_median=("LR_refit", "median"),
        D2_old_median=("old_mean_Pfull", "median"), D2_new_median=("new_mean_Pfull", "median"))
    print(s.round(4).to_string())
    print("by cell (empirical):")
    e = ok[ok.cell_family == "empirical_AABC"]
    if len(e):
        print(e.groupby("cell_id").agg(n=("replicate", "size"), alt_gain_median=("alt_gain_over_archived", "median"),
                                       LR_refit_median=("LR_refit", "median"), D2_old=("old_mean_Pfull", "median"),
                                       D2_new=("new_mean_Pfull", "median")).round(4).to_string())
    print(f"\nwrote {a.out_dir}; unresolved candidates: {unresolved}")
    if unresolved:
        raise SystemExit("Unresolved candidates remain: inspect the audit; do not summarize this run as final")
    (a.out_dir / "READY_FOR_SUMMARY.txt").write_text(
        "All selected candidates passed repair checks. "
        + ("SMOKE RUN ONLY: run the full candidate set before final summaries.\n" if a.smoke or a.limit or a.families
           else "Full starting-signature candidate set processed.\n"))


if __name__ == "__main__":
    main()
