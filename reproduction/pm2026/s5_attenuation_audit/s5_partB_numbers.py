#!/usr/bin/env python3
"""Numbers in Supplement S5.4 (attenuation-study audit) from partB_by_task.csv alone. No refitting.

Works with the reported audit run, whose output predates the standard-error and coverage columns.
The CSV is parsed with float_precision="round_trip" (correctly rounded). pandas' default parser gave
different last bits on x86-64 Linux and Apple Silicon for the largest rerun discrepancy.
Usage: python3 reproduction/pm2026/s5_attenuation_audit/s5_partB_numbers.py <partB_by_task.csv>
"""
import json
import sys

import numpy as np
import pandas as pd

ROLES = ["random_slope_primary", "random_intercept_null", "random_slope_predictor_omitted"]
METHODS = ["one_stage_LMM", "BLUP_as_outcome_attenuation", "predictor_double_use_overconfidence",
           "uncertainty_aware_two_stage"]
REPORTED = ["high_slope_information", "intermediate", "near_boundary"]

d = pd.read_csv(sys.argv[1], float_precision="round_trip")
regen = [c for c in d if c.endswith("_regenerated")]
gap_cols = [f"{r}_recorded_minus_best" for r in ROLES]
rerun_diff = pd.concat([(d[f"{r}_frozen_rerun_llf"] - d[f"{r}_recorded_llf"]).abs() for r in ROLES]).dropna()
status = pd.concat([d[f"{r}_fit_status"] for r in ROLES]).dropna()


def gaps(g):
    s = pd.concat([g[c] for c in gap_cols]).dropna()
    if not np.all(np.isfinite(s)):
        raise SystemExit("nonfinite likelihood gap")
    return int(len(s)), int((s < -0.01).sum()), float(s.min())


def changes(g):
    beta = max(float((g[f"{m}_beta_best"] - g[f"{m}_beta_recorded"]).abs().max()) for m in METHODS)
    rel = max(float(((g[f"{m}_beta_best"] - g[f"{m}_beta_recorded"]).abs() / g[f"{m}_beta_recorded"].abs()).max())
              for m in METHODS)
    flips = sum(int(((g[f"{m}_p_best"] < .05) != (g[f"{m}_p_recorded"] < .05)).sum()) for m in METHODS)
    summ = max(float((g[f"{k}_best"] - g[f"{k}_recorded"]).abs().max())
               for k in ("primary_mean_direct_predictability", "primary_mean_full_posterior_predictability"))
    return dict(max_abs_beta_change=beta, max_rel_beta_change=rel, p05_flips=f"{flips} of {len(g) * len(METHODS)}",
                max_abs_D2_or_Rdirect_change=summ)


ev = d[d.row_type == "evaluation"]
rep = ev[ev.regime.isin(REPORTED)]
tz = ev[~ev.regime.isin(REPORTED)]
out = {"datasets": int(len(d)), "model_fits": int(len(status)),
       "regeneration_reproduced": f"{int(d[regen].sum().sum())} / {int(d[regen].notna().sum().sum())}",
       "max_frozen_rerun_llf_discrepancy": float(rerun_diff.max()),
       "corroborated_refits": f"{int(status.isin(['agreed', 'confirmed_on_restart']).sum())} / {len(status)}",
       "reported_evaluation_datasets": int(len(rep)),
       "reported": dict(zip(["fits", "below_best_gt_001", "min_gap"], gaps(rep)), **changes(rep)),
       "true_zero_evaluation": dict(zip(["fits", "below_best_gt_001", "min_gap"], gaps(tz)), **changes(tz)),
       "auxiliary_LR": {rt: dict(zip(["fits", "below_best_gt_001", "min_gap"], gaps(g)))
                        for rt, g in d[d.row_type != "evaluation"].groupby("row_type")}}
print(json.dumps(out, indent=2))
