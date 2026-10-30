#!/usr/bin/env python3
"""S4 null screen v2: evaluate the CURRENT patched table, without changing it.

A: stored alternative below a feasible null candidate (a witness of suboptimality).
B: for positive, unclamped LR, a better null than the one implied by stored LR.
C: optional feasible alternative improvement, not a proof about local optimality.

Uses Step 4's checked multistart routine. Objective/embedding agreement, finite
admissible fits and successful distinct-start corroboration are acceptance gates.
Negative inferred-null shortfalls are unresolved, not silently called clean.
Zero/clamped LR cannot identify the old null; B is explicitly unassessable there.
Numerical corroboration does not certify a global maximum.

--per-cell is a validation sample, not a prevalence estimate. It samples across
repair status and zero/positive LR, then fills randomly. Omit for a full screen
of the explicitly selected families/schedules. Missing schedules fail BEFORE
workers launch. No automatic omission of unavailable schedules is allowed.

Outputs step5_screen.csv, step5_flagged_keys.csv, step5_unresolved_keys.csv,
checkpoint and manifest. Unresolved rows cause exit status 2. Never interpret
an incomplete/failed screen as evidence of absence. No source file is patched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 0.01
CHECK_TOL = 1e-5
KEYS = ["cell_id", "replicate"]
STATE = {}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def require_bool(s, name):
    if s.isna().any() or not s.isin([True, False]).all():
        raise ValueError(f"{name}: expected complete Boolean values")
    return s.astype(bool)


def sample_cells(frame, per_cell, seed):
    if per_cell is None:
        return frame.copy()
    rng = np.random.default_rng(seed)
    picked = []
    for _, block in frame.groupby("cell_id", sort=True):
        count = min(per_cell, len(block))
        strata = block.assign(lr_positive=block.restricted_LR.gt(0)).groupby(
            ["step4_repaired", "lr_positive"], sort=True)
        initial = [int(rng.choice(b.index.to_numpy())) for _, b in strata]
        if len(initial) > count:
            initial = list(rng.choice(initial, count, replace=False))
        rest = block.index.difference(initial).to_numpy()
        picked.extend(initial + list(rng.choice(rest, count - len(initial), replace=False)))
    return frame.loc[picked].sort_values(KEYS)


def available_schedules(path):
    names = {"balanced_three_wave"}
    if path:
        content = json.loads(path.read_text())
        if not isinstance(content.get("schedules"), dict):
            raise ValueError("schedule JSON must contain a 'schedules' object")
        names.update(content["schedules"])
    return names


def init_worker(repo, s4_dir, schedule_json, reml_tools, step4_script, needed):
    code = Path(repo) / "reproduction" / "s4_operating_study"
    sys.path.insert(0, str(code))
    import simulation_core as core
    import generate_operating_study as gen
    rt = load("step5_reml_tools", reml_tools)
    repair = load("step5_checked_fitter", step4_script)
    # Same parser as the original generator and Step 4; preserve full design
    # order for its seed formula, even when only some schedules are screened.
    design = pd.read_csv(Path(s4_dir) / "operating_study_design.csv")
    templates = gen.load_schedule_templates(Path(schedule_json) if schedule_json else None,
                                             design[design.schedule.isin(needed)], "full")
    gen.SCHEDULE_TEMPLATES = templates
    gen.SIMULATION_CELL_LOOKUP = design.set_index("cell_id").to_dict("index")
    gen.simulation_manifest = design
    STATE.update(core=core, gen=gen, rt=rt, repair=repair, templates=templates)


def screen_one(task):
    row = {k: task[k] for k in KEYS}
    row.update(error="", resolved=False, flag_A_alt_below_null=False,
               flag_B_null_shortfall=False, flag_C_alt_polish=False,
               B_assessable=False, negative_null_shortfall=False)
    try:
        core, gen, rt, repair = (STATE[k] for k in ["core", "gen", "rt", "repair"])
        template = STATE["templates"][task["schedule"]]
        y, seed = gen.generate_simulation_outcome(task["cell_id"], int(task["replicate"]))
        row["seed_match"] = int(seed) == int(task["seed"])
        if not row["seed_match"]:
            raise ValueError("seed mismatch")
        data = pd.DataFrame({"pid": template["group_index"], "t": template["flat_times"], "y": y})
        pa, pn = rt.REMLProblem(data, "y ~ t", "1+t"), rt.REMLProblem(data, "y ~ t", "1")
        ga = np.array([[task["random_intercept_variance"], task["intercept_slope_covariance"]],
                       [task["intercept_slope_covariance"], task["random_slope_variance"]]])
        sa = float(task["residual_variance"])
        if not repair.admissible(ga, sa):
            raise ValueError("stored alternative parameters not finite/admissible")
        ll_alt = pa.loglik(ga, sa)
        ll_s4 = core.vectorized_reml_loglike(template, y, ga, sa)[0]
        row.update(ll_alt_input=ll_alt, objective_check=ll_alt - ll_s4,
                   LR_input=float(task["restricted_LR"]))
        if not np.isfinite([ll_alt, ll_s4, row["LR_input"]]).all():
            raise ValueError("nonfinite stored likelihood or LR")
        if abs(row["objective_check"]) > CHECK_TOL:
            raise ValueError("REML objective mismatch")
        if row["LR_input"] < 0:
            raise ValueError("negative stored LR: check archive convention")

        # Include the exact G00=0 endpoint, not just a small Cholesky value.
        X = np.column_stack([np.ones(len(y)), data.t.to_numpy()])
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank != 2 or len(y) <= rank:
            raise ValueError("fixed-effects design is rank deficient")
        s0 = float(np.sum((y - X @ beta) ** 2) / (len(y) - rank))
        g0 = np.zeros((1, 1))
        ll0 = pn.loglik(g0, s0)
        gn, sn, ln, info = repair.maximize_checked(pn, n_starts=3, seed=1)
        numerical_ll = ln
        if ll0 > ln:
            gn, sn, ln = g0, s0, ll0
        row.update(ll_null_best=ln, null_endpoint_selected=bool(gn[0, 0] == 0),
                   null_optimizer_success=bool(info["opt_nm_success"]),
                   null_successful_starts_within_tol=int(info["opt_starts_within_tol"]))
        # Corroboration refers to the numerical fit, so also check that its
        # likelihood agrees with the finally selected (possibly endpoint) fit.
        # Do not call a better endpoint corroborated just because inferior
        # numerical starts agreed with each other.
        endpoint_agrees = abs(ln - numerical_ll) <= TOL
        if not (info["opt_nm_success"] and info["opt_starts_within_tol"] >= 2 and endpoint_agrees):
            raise ValueError("null not corroborated by successful distinct starts")
        if not repair.admissible(gn, sn) or not np.isfinite(ln):
            raise ValueError("null candidate not finite/admissible")
        embedded = np.array([[gn[0, 0], 0.], [0., 0.]])
        row["embedding_check"] = pa.loglik(embedded, sn) - ln
        if not np.isfinite(row["embedding_check"]) or abs(row["embedding_check"]) > CHECK_TOL:
            raise ValueError("null embedding mismatch")
        row["flag_A_alt_below_null"] = bool(ll_alt < ln - TOL)
        row["B_assessable"] = row["LR_input"] > 0
        row["null_shortfall"] = ln - (ll_alt - row["LR_input"] / 2) if row["B_assessable"] else np.nan
        row["flag_B_null_shortfall"] = bool(row["null_shortfall"] > TOL)
        row["negative_null_shortfall"] = bool(row["null_shortfall"] < -TOL)
        if row["negative_null_shortfall"]:
            raise ValueError("implied old null exceeds best new null; inspect reconstruction/optimization")
        row["LR_null_corrected_screen_only"] = max(0., 2 * (ll_alt - ln))
        row["alt_polish_gain"] = np.nan
        if task["polish"]:
            gp, sp, lp, ip = repair.maximize_checked(pa, n_starts=0, seed=0,
                                                    extra_starts=[pa.pack(ga, sa)])
            row.update(alt_polish_gain=lp - ll_alt, polish_success=bool(ip["opt_nm_success"]))
            # A finite feasible improvement is already a witness; however,
            # failure without improvement cannot establish a negative screen.
            row["flag_C_alt_polish"] = bool(lp - ll_alt > TOL)
            if not ip["opt_nm_success"] and not row["flag_C_alt_polish"]:
                raise ValueError("optional alternative polish did not converge")
        row["resolved"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:400]
    return row


def screen_batch(tasks):
    return [screen_one(t) for t in tasks]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--reml-tools", type=Path, required=True)
    ap.add_argument("--step4-script", type=Path, required=True)
    ap.add_argument("--s4-dir", type=Path, required=True)
    ap.add_argument("--replicates-csv", type=Path, required=True, help="CURRENT Step-4-patched table")
    ap.add_argument("--step4-audit", type=Path, required=True)
    ap.add_argument("--schedule-json", type=Path)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--schedules", nargs="+", default=None)
    ap.add_argument("--polish-families", nargs="*", default=[], help="optional; disabled by default")
    ap.add_argument("--per-cell", type=int, help="validation sample size per selected cell")
    ap.add_argument("--sample-seed", type=int, default=20261019)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    if min(a.workers, a.batch_size) < 1 or (a.per_cell is not None and a.per_cell < 1):
        ap.error("workers, batch-size and per-cell must be positive")
    code = a.repo_root / "reproduction" / "s4_operating_study"
    sources = [Path(__file__), a.reml_tools, a.step4_script, a.replicates_csv, a.step4_audit,
               a.s4_dir / "operating_study_design.csv", code / "simulation_core.py",
               code / "generate_operating_study.py"] + ([a.schedule_json] if a.schedule_json else [])
    for p in sources:
        if not p.is_file():
            ap.error(f"missing file: {p}")
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        ap.error("output folder is not empty; choose a new --out-dir")
    d = pd.read_csv(a.replicates_csv, low_memory=False, float_precision="round_trip")
    audit = pd.read_csv(a.step4_audit, float_precision="round_trip")
    if d.duplicated(KEYS).any() or audit.duplicated(KEYS).any():
        ap.error("duplicate participant-free simulation keys")
    if audit.error.fillna("").ne("").any() or require_bool(audit.unresolved, "audit.unresolved").any():
        ap.error("Step 4 audit contains unresolved rows")
    if not require_bool(audit.new_strict_fit_valid, "audit.new_strict_fit_valid").all():
        ap.error("Step 4 audit contains unaccepted repairs")
    d["strict_fit_valid"] = require_bool(d.strict_fit_valid, "strict_fit_valid")
    index4 = pd.MultiIndex.from_frame(audit[KEYS])
    d["step4_repaired"] = pd.MultiIndex.from_frame(d[KEYS]).isin(index4)
    current = d.set_index(KEYS)
    if not index4.isin(current.index).all():
        ap.error("patched table is missing Step 4 keys")
    for old, new in [("LR_refit", "restricted_LR"), ("new_random_slope_variance", "random_slope_variance"),
                     ("new_residual_variance", "residual_variance"), ("new_mean_Pfull", "mean_Pfull")]:
        if not np.allclose(audit[old], current.loc[index4, new], rtol=1e-9, atol=1e-10, equal_nan=True):
            ap.error(f"input is not the Step-4-patched table ({new} mismatch)")
    sel = d.copy()
    for arg, column in [(a.families, "cell_family"), (a.schedules, "schedule")]:
        if arg:
            unknown = set(arg) - set(d[column])
            if unknown:
                ap.error(f"unknown {column}: {sorted(unknown)}")
            sel = sel[sel[column].isin(arg)]
    if sel.empty:
        ap.error("no rows selected")
    missing = set(sel.schedule) - available_schedules(a.schedule_json)
    if missing:
        ap.error(f"schedule JSON lacks {sorted(missing)}; explicitly select available --schedules or reconstruct missing schedules")
    sel = sample_cells(sel, a.per_cell, a.sample_seed)
    sel = sel.assign(polish=sel.cell_family.isin(a.polish_families))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"method": "s4_null_screen_v2", "rows": len(sel), "tol": TOL,
                "check_tol": CHECK_TOL, "per_cell": a.per_cell, "sample_seed": a.sample_seed,
                "families": a.families, "schedules": sorted(sel.schedule.unique()),
                "polish_families": a.polish_families,
                "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    cols = KEYS + ["seed", "schedule", "random_intercept_variance", "intercept_slope_covariance",
                   "random_slope_variance", "residual_variance", "restricted_LR", "polish"]
    tasks = sel[cols].to_dict("records")
    batches = [tasks[i:i+a.batch_size] for i in range(0, len(tasks), a.batch_size)]
    print(f"Screening {len(tasks)} rows; optional polish: {int(sel.polish.sum())}; validation sample: {a.per_cell is not None}", flush=True)
    init = (str(a.repo_root), str(a.s4_dir), str(a.schedule_json) if a.schedule_json else None,
            str(a.reml_tools), str(a.step4_script), sorted(sel.schedule.unique()))
    rows, start = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=init_worker, initargs=init) as pool:
        futures = {pool.submit(screen_batch, b): b for b in batches}
        for f in as_completed(futures):
            try:
                rows.extend(f.result())
            except Exception as exc:
                rows.extend({**{k: t[k] for k in KEYS}, "resolved": False,
                             "error": f"worker failure: {exc}"[:400]} for t in futures[f])
            pd.DataFrame(rows).to_csv(a.out_dir / "step5_screen_checkpoint.csv", index=False)
            elapsed = time.time() - start
            print(f"  {len(rows)}/{len(tasks)} ({elapsed:.0f}s; estimated remaining {elapsed/len(rows)*(len(tasks)-len(rows))/60:.1f} min)", flush=True)
    scr = pd.DataFrame(rows).merge(sel[KEYS + ["cell_family", "schedule", "step4_repaired", "strict_fit_valid"]],
                                    on=KEYS, validate="one_to_one").sort_values(KEYS)
    scr.to_csv(a.out_dir / "step5_screen.csv", index=False)
    unresolved = scr[~scr.resolved]
    ok = scr[scr.resolved].copy()
    flags = ["flag_A_alt_below_null", "flag_B_null_shortfall", "flag_C_alt_polish"]
    flagged = ok[ok.reindex(columns=flags, fill_value=False).fillna(False).any(axis=1)]
    flagged[KEYS].to_csv(a.out_dir / "step5_flagged_keys.csv", index=False)
    unresolved[KEYS + ["error"]].to_csv(a.out_dir / "step5_unresolved_keys.csv", index=False)
    print(f"\nChecks: rows {len(scr)}; unresolved {len(unresolved)}; flagged for refit {len(flagged)}")
    if len(ok):
        print(f"max |objective check| {ok.objective_check.abs().max():.2e}; max |embedding| {ok.embedding_check.abs().max():.2e}; min successful null starts {ok.null_successful_starts_within_tol.min()}")
        for group in ["cell_family", "cell_id"]:
            summary = ok.groupby(group).agg(rows=("replicate", "size"),
                A=(flags[0], "sum"), B_assessable=("B_assessable", "sum"), B=(flags[1], "sum"),
                C=(flags[2], "sum"), step4_repaired=("step4_repaired", "sum"))
            print("\n" + summary.to_string())
    if len(unresolved):
        print("\nUnresolved reasons:\n" + unresolved.error.value_counts().to_string())
        raise SystemExit(2)
    print(f"\nComplete. Inspect individual flagged rows before extending the screen. Wrote {a.out_dir}")


if __name__ == "__main__":
    main()
