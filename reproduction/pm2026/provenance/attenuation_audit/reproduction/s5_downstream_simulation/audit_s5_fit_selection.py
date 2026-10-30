#!/usr/bin/env python3
"""Audit of mixed-model fit selection in the completed S5 downstream simulation.

S5 selected, for each model, the first strictly valid optimizer in the frozen order
(L-BFGS -> Powell -> BFGS), without comparing likelihoods. This script checks how often
that selection is not the REML maximum and whether reported outputs change.

Part A (seconds; existing outputs only, no refitting)
  A1  Nested-likelihood check. The random-slope primary model nests the random-intercept
      null model (same fixed effects), so at the REML maxima llf(primary) >= llf(null).
      A selected primary fit with llf(primary) < llf(null) - tol is certainly not at its
      maximum; its restricted LR was clamped to zero.
  A2  Attempts check. Where more than one optimizer was attempted, the selected attempt's
      likelihood is compared with the best attempted likelihood.

Part B (refits a stratified sample of tasks; requires the authorized anchor manifest)
  For each sampled task: regenerate the dataset from its recorded seed; rerun the frozen
  rule and confirm it reproduces the recorded selected likelihoods (dataset/environment
  check); select the highest-likelihood admissible fit among all statsmodels optimizers
  plus an independent multistart maximizer (reml_tools.py); record the frozen rule's
  shortfall; recompute the reported outputs from the selected fits and compare.

Usage (repository root)
  python3 reproduction/s5_downstream_simulation/audit_s5_fit_selection.py \
      --s5-output <S5 generator output dir> --part A --output-dir s5_fit_audit
  python3 reproduction/s5_downstream_simulation/audit_s5_fit_selection.py \
      --s5-output <S5 generator output dir> --anchor-manifest <source_anchor_manifest.json> \
      --part B --per-stratum 10 --include-flagged 40 --workers 4 --output-dir s5_fit_audit
"""
from __future__ import annotations

import argparse
import importlib.util
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

HERE = Path(__file__).resolve().parent
REML_TOOLS = HERE.parent / "s5b_contamination_pilot" / "reml_tools.py"
NESTED_TOL = 1e-6
REGEN_TOL = 1e-6

MODELS_EVAL = {"random_slope_primary": ("y ~ t + x + x:t", "1+t"),
               "random_intercept_null": ("y ~ t + x + x:t", "1"),
               "random_slope_predictor_omitted": ("y ~ t + x", "1+t")}
MODELS_LR = {k: MODELS_EVAL[k] for k in ("random_slope_primary", "random_intercept_null")}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
def part_a(s5_dir: Path, out_dir: Path):
    diag = pd.read_csv(s5_dir / "model_fit_diagnostics.csv")
    raw = pd.read_csv(s5_dir / "simulation_replicate_results_raw.csv")
    sel = diag[diag.get("selected", True).astype(bool)] if "selected" in diag else diag
    w = sel.pivot_table(index="task_id", columns="model_role", values="reml_log_likelihood", aggfunc="first")
    meta = sel.groupby("task_id")[["row_type", "regime", "effect_condition", "replicate", "seed"]].first()
    w = w.join(meta)
    w["primary_minus_null"] = w["random_slope_primary"] - w["random_intercept_null"]
    w["nested_violation"] = w["primary_minus_null"] < -NESTED_TOL
    st = sel.pivot_table(index="task_id", columns="model_role", values="selection_status", aggfunc="first")
    w = w.join(st.add_suffix("_status"))
    lr = raw.set_index("task_id")["restricted_LR"]
    w = w.join(lr)
    w.reset_index().to_csv(out_dir / "partA_nested_check_by_task.csv", index=False)
    a1 = w.groupby(["row_type", "regime"]).agg(
        tasks=("nested_violation", "size"), nested_violations=("nested_violation", "sum"),
        min_primary_minus_null=("primary_minus_null", "min"),
        LR_zero=("restricted_LR", lambda s: int((s == 0).sum()))).reset_index()

    att = pd.read_csv(s5_dir / "optimizer_attempts.csv")
    att = att[att.fit_succeeded.astype(bool)]
    rows = []
    for (task, role), g in att.groupby(["task_id", "model_role"]):
        if len(g) < 2:
            continue
        valid = g[g.basic_numerical_validity.astype(bool)]
        selrow = g[g.selected.astype(bool)] if "selected" in g else g.iloc[:0]
        if valid.empty or selrow.empty:
            continue
        rows.append({"task_id": task, "model_role": role, "row_type": g.row_type.iloc[0], "regime": g.regime.iloc[0],
                     "attempts": len(g), "selected_optimizer": selrow.optimizer.iloc[0],
                     "selected_minus_best_attempted": float(selrow.reml_log_likelihood.iloc[0] - valid.reml_log_likelihood.max())})
    a2 = pd.DataFrame(rows)
    if not a2.empty:
        a2.to_csv(out_dir / "partA_multi_attempt_by_model.csv", index=False)
        a2s = a2.groupby(["row_type", "regime", "model_role"]).agg(
            models_with_multiple_attempts=("attempts", "size"),
            selected_below_best_attempt_gt_001=("selected_minus_best_attempted", lambda s: int((s < -0.01).sum())),
            min_gap=("selected_minus_best_attempted", "min")).reset_index()
    else:
        a2s = pd.DataFrame()
    a1.to_csv(out_dir / "partA_nested_check_summary.csv", index=False)
    a2s.to_csv(out_dir / "partA_multi_attempt_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("A1 nested-likelihood check (primary below null => primary fit not at its REML maximum):")
    print(a1.round(4).to_string(index=False))
    print(f"   total violations: {int(w.nested_violation.sum())} of {len(w)} tasks")
    print("\nA2 models with >1 optimizer attempted (only where the first attempt failed strict validity):")
    print(a2s.round(4).to_string(index=False) if not a2s.empty else "   none")
    return w


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
def ols_x(values, x):
    fit = smf.ols("v ~ x", data=pd.DataFrame({"v": values, "x": x})).fit(cov_type="HC3")
    return float(fit.params["x"]), float(fit.bse["x"]), float(fit.pvalues["x"])


def predictability(G, s2, data):
    """Mean direct and full posterior slope predictability (valid for singular G)."""
    G = np.asarray(G, float)
    tau2 = G[1, 1]
    if not tau2 > 0:
        return np.nan, np.nan
    d, f = [], []
    for _, grp in data.groupby("pid", sort=False):
        t = grp.t.to_numpy(float)
        sst = float(np.sum((t - t.mean()) ** 2))
        d.append(tau2 / (tau2 + s2 / sst) if sst > 0 else np.nan)
        Z = np.column_stack([np.ones_like(t), t])
        V = Z @ G @ Z.T + s2 * np.eye(len(t))
        g = Z @ G[:, 1]
        f.append(float(g @ np.linalg.solve(V, g)) / tau2)
    return float(np.nanmean(d)), float(np.mean(f))


def run_task(task):
    sys.path.insert(0, str(HERE))
    import simulation_core as core
    gen = load("s5_generator", HERE / "generate_downstream_simulation.py")
    rt = load("reml_tools", REML_TOOLS)
    sp, _, _ = gen.load_source_parameters(Path(task["anchor_manifest"]), [task["regime"]])
    is_eval = task["row_type"] == "evaluation"
    data, *_ = core.simulated_dataset(sp, task["regime"], task["effect_condition"], int(task["seed"]),
                                      force_zero_slope=not is_eval)
    out = {k: task[k] for k in ("task_id", "row_type", "regime", "effect_condition", "replicate", "seed")}
    models = MODELS_EVAL if is_eval else MODELS_LR
    best, cand_rows = {}, []
    t0 = time.time()
    for role, (formula, re) in models.items():
        res, dg, _ = core.fit_optimizer_sequence(data, formula, re, role)
        rec_ll = task["recorded_llf"].get(role, np.nan)
        out[f"{role}_recorded_llf"] = rec_ll
        out[f"{role}_frozen_rerun_llf"] = float(res.llf) if res is not None else np.nan
        out[f"{role}_regenerated"] = bool(res is not None and abs(float(res.llf) - rec_ll) < REGEN_TOL)
        rec, cands = rt.fit_best_reml(data, formula, re, core, model_role=role, seed=int(task["seed"]) % 2**31)
        best[role] = rec
        for c in cands:
            cand_rows.append({"task_id": task["task_id"], **c})
        out[f"{role}_fit_status"] = rec["fit_status"]
        out[f"{role}_best_llf"] = rec.get("llf", np.nan)
        out[f"{role}_best_source"] = rec.get("best_source", "")
        out[f"{role}_recorded_minus_best"] = rec_ll - rec.get("llf", np.nan)
        out[f"{role}_best_boundary"] = rec.get("best_boundary", np.nan)
        out[f"{role}_best_rho"] = rec.get("best_rho", np.nan)
    p, n = best["random_slope_primary"], best["random_intercept_null"]
    out["LR_recorded"] = task["recorded"].get("restricted_LR", np.nan)
    out["LR_best"] = max(0.0, 2 * (p["llf"] - n["llf"])) if "llf" in p and "llf" in n else np.nan
    if is_eval and "beta" in p:
        x_by = data.groupby("pid", sort=False).x.first()
        beta = float(p["beta"]["x:t"]); se = float(np.sqrt(p["cov_beta"].loc["x:t", "x:t"]))
        out.update(one_stage_LMM_beta_best=beta, one_stage_LMM_p_best=2 * norm.sf(abs(beta / se)))
        tot = p["beta"]["t"] + p["eb"]["t"] + p["beta"]["x:t"] * x_by.reindex(p["eb"].index)
        b, s, pv = ols_x(tot.to_numpy(float), x_by.reindex(p["eb"].index).to_numpy(float))
        out.update(predictor_double_use_overconfidence_beta_best=b, predictor_double_use_overconfidence_p_best=pv)
        o = best["random_slope_predictor_omitted"]
        if "eb" in o:
            tot = o["beta"]["t"] + o["eb"]["t"]
            b, s, pv = ols_x(tot.to_numpy(float), x_by.reindex(o["eb"].index).to_numpy(float))
            out.update(BLUP_as_outcome_attenuation_beta_best=b, BLUP_as_outcome_attenuation_p_best=pv)
        raw = core.uncertainty_aware_slopes(data, float(p["s2"]))
        resp, des = patsy.dmatrices("unshrunk_slope ~ x", raw, return_type="dataframe")
        mb, ms, _ = core.reml_meta_regression(resp.iloc[:, 0], raw.loc[resp.index, "slope_sampling_variance"], des, des.columns)
        out.update(uncertainty_aware_two_stage_beta_best=float(mb["x"]),
                   uncertainty_aware_two_stage_p_best=2 * norm.sf(abs(float(mb["x"]) / float(ms["x"]))))
        dpr, fpr = predictability(p["G"], p["s2"], data)
        out.update(primary_mean_direct_predictability_best=dpr, primary_mean_full_posterior_predictability_best=fpr)
        for k, v in task["recorded"].items():
            out[f"{k}_recorded"] = v
    out["seconds"] = time.time() - t0
    return out, cand_rows


def part_b(args, s5_dir: Path, out_dir: Path, partA):
    tm = pd.read_csv(s5_dir / "task_manifest.csv")
    raw = pd.read_csv(s5_dir / "simulation_replicate_results_raw.csv").set_index("task_id")
    diag = pd.read_csv(s5_dir / "model_fit_diagnostics.csv")
    sel = diag[diag["selected"].astype(bool)] if "selected" in diag else diag
    recorded_llf = sel.pivot_table(index="task_id", columns="model_role", values="reml_log_likelihood", aggfunc="first")
    strata = ["row_type", "regime", "effect_condition"]
    idx = [g.sample(min(args.per_stratum, len(g)), random_state=args.sample_seed).index
           for _, g in tm.groupby(strata)]
    pick = tm.loc[np.concatenate(idx)]
    if args.include_flagged and partA is not None:
        flagged = partA.index[partA.nested_violation.astype(bool)]
        extra = tm[tm.task_id.isin(flagged) & ~tm.task_id.isin(pick.task_id)]
        pick = pd.concat([pick, extra.sample(min(args.include_flagged, len(extra)), random_state=args.sample_seed)])
    keep = ["restricted_LR", "one_stage_LMM_beta", "one_stage_LMM_p", "BLUP_as_outcome_attenuation_beta",
            "BLUP_as_outcome_attenuation_p", "predictor_double_use_overconfidence_beta",
            "predictor_double_use_overconfidence_p", "uncertainty_aware_two_stage_beta", "uncertainty_aware_two_stage_p",
            "primary_mean_direct_predictability", "primary_mean_full_posterior_predictability"]
    tasks = []
    for _, r in pick.iterrows():
        rec = raw.loc[r.task_id] if r.task_id in raw.index else pd.Series(dtype=float)
        tasks.append({**r.to_dict(), "anchor_manifest": str(args.anchor_manifest),
                      "recorded_llf": recorded_llf.loc[r.task_id].dropna().to_dict() if r.task_id in recorded_llf.index else {},
                      "recorded": {k: rec.get(k, np.nan) for k in keep}})
    print(f"\nPart B: refitting {len(tasks)} tasks ...", flush=True)
    rows, cands, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(run_task, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            o, c = f.result()
            rows.append(o); cands.extend(c)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "partB_by_task.csv", index=False)
    pd.DataFrame(cands).to_csv(out_dir / "partB_candidates.csv", index=False)

    regen_cols = [c for c in res if c.endswith("_regenerated")]
    print("\nRegeneration/environment check (frozen rerun reproduces recorded likelihood): %d / %d model fits"
          % (int(res[regen_cols].sum().sum()), int(res[regen_cols].notna().sum().sum())))
    fit_rows = []
    for role in MODELS_EVAL:
        col = f"{role}_recorded_minus_best"
        if col not in res:
            continue
        for key, g in res.groupby(["row_type", "regime"]):
            s = g[col].dropna()
            fit_rows.append({"row_type": key[0], "regime": key[1], "model_role": role, "n": len(s),
                             "recorded_below_best_gt_001": int((s < -0.01).sum()),
                             "recorded_below_best_gt_1": int((s < -1).sum()),
                             "min_gap": s.min(), "median_gap": s.median(),
                             "best_boundary_rate": g[f"{role}_best_boundary"].astype(float).mean(),
                             "unresolved": int((g[f"{role}_fit_status"] == "unresolved_single_source").sum())})
    fq = pd.DataFrame(fit_rows)
    fq.to_csv(out_dir / "partB_fit_quality_summary.csv", index=False)
    out_rows = []
    for key, g in res.groupby(["row_type", "regime"]):
        r = {"row_type": key[0], "regime": key[1], "n": len(g),
             "LR_mean_recorded": g.LR_recorded.mean(), "LR_mean_best": g.LR_best.mean(),
             "LR_changed_gt_001": int(((g.LR_best - g.LR_recorded).abs() > 0.01).sum()),
             "LR_zero_recorded": int((g.LR_recorded == 0).sum()), "LR_zero_best": int((g.LR_best == 0).sum())}
        for m in ("one_stage_LMM", "BLUP_as_outcome_attenuation", "predictor_double_use_overconfidence",
                  "uncertainty_aware_two_stage"):
            if f"{m}_beta_best" in g:
                d = (g[f"{m}_beta_best"] - g[f"{m}_beta_recorded"]).abs()
                r[f"{m}_max_abs_beta_change"] = d.max()
                r[f"{m}_p05_decision_flips"] = int(((g[f"{m}_p_best"] < .05) != (g[f"{m}_p_recorded"] < .05)).sum())
        if "primary_mean_full_posterior_predictability_best" in g:
            r["full_predictability_max_abs_change"] = (g.primary_mean_full_posterior_predictability_best
                                                       - g.primary_mean_full_posterior_predictability_recorded).abs().max()
        out_rows.append(r)
    oc = pd.DataFrame(out_rows)
    oc.to_csv(out_dir / "partB_output_changes_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nFit quality (recorded frozen-rule likelihood minus best likelihood; negative = frozen rule not at maximum):")
    print(fq.round(4).to_string(index=False))
    print("\nChanges in reported outputs when recomputed from the best fits:")
    print(oc.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s5-output", type=Path, required=True, help="directory with the S5 generator CSVs")
    ap.add_argument("--anchor-manifest", type=Path, default=None, help="required for part B")
    ap.add_argument("--part", choices=["A", "B", "AB"], default="A")
    ap.add_argument("--per-stratum", type=int, default=10)
    ap.add_argument("--include-flagged", type=int, default=40, help="also refit up to N tasks flagged in part A1")
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, default=Path("s5_fit_audit"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partA = part_a(args.s5_output, args.output_dir)
    if "B" in args.part:
        if args.anchor_manifest is None:
            raise SystemExit("--anchor-manifest is required for part B")
        part_b(args, args.s5_output, args.output_dir, partA)
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
