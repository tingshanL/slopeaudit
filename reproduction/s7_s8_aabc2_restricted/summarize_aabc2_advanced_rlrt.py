#!/usr/bin/env python3
"""Recompute the final AABC2 RLRT denominator, Monte Carlo, and Holm summaries.

This step performs no model refits.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
ALPHA=.05; PB_INITIAL_B=999; PB_MAX_B=4999; RLRT_USABLE_RATE_TARGET=.95
PB_OUTCOMES=("lateral_ventricles","hippocampus","global_mean_thickness","global_cortical_cbf","global_cortical_att","fluid_cognition")
def coerce_boolean(s):
    if s.dtype==bool: return s.fillna(False)
    return s.astype(str).str.strip().str.lower().map({"true":True,"false":False,"1":True,"0":False}).fillna(False).astype(bool)
def exact_binomial_interval(successes,trials,confidence=.95):
    if trials<=0:return np.nan,np.nan
    tail=(1-confidence)/2; lo=0.0 if successes==0 else stats.beta.ppf(tail,successes,trials-successes+1); hi=1.0 if successes==trials else stats.beta.ppf(1-tail,successes+1,trials-successes); return float(lo),float(hi)
p=argparse.ArgumentParser(); p.add_argument("--observed-tests",required=True); p.add_argument("--repaired-replicates",required=True); p.add_argument("--ventricular-repairs",required=True); p.add_argument("--output-dir",required=True); a=p.parse_args()
out=Path(a.output_dir).expanduser().resolve();
if out.exists(): raise FileExistsError(out)
out.mkdir(parents=True)
obs=pd.read_csv(a.observed_tests,low_memory=False); reps=pd.read_csv(a.repaired_replicates,low_memory=False); vent=pd.read_csv(a.ventricular_repairs,low_memory=False)
for c in ("basic_pair_valid","strict_pair_valid"):
    reps[c]=coerce_boolean(reps[c]); vent[c]=coerce_boolean(vent[c])
reps["replicate"]=pd.to_numeric(reps["replicate"],errors="raise").astype(int); vent["replicate"]=pd.to_numeric(vent["replicate"],errors="raise").astype(int)
if reps.duplicated(["outcome","replicate"]).any(): raise RuntimeError("Duplicate final RLRT replicate keys")
if len(reps)!=9994: raise RuntimeError(f"Expected 9994 final RLRT rows; found {len(reps)}")
def summarize(outcome,requested_B):
    observed=obs.set_index("outcome").loc[outcome]; rows=reps.loc[reps.outcome.eq(outcome)&reps.replicate.le(int(requested_B))].drop_duplicates("replicate",keep="last").copy(); rows["basic_pair_valid"]=coerce_boolean(rows["basic_pair_valid"]); rows["strict_pair_valid"]=coerce_boolean(rows["strict_pair_valid"]); completed=int(rows.replicate.nunique()); finite=np.isfinite(pd.to_numeric(rows.restricted_LR,errors="coerce")); usable=rows.loc[rows.basic_pair_valid&finite]; hp=rows.loc[rows.strict_pair_valid&finite]; usable_B=int(usable.replicate.nunique()); hp_B=int(hp.replicate.nunique()); exceed=int((usable.restricted_LR>=float(observed.observed_restricted_LR)).sum()); pval=(exceed+1)/(usable_B+1) if usable_B else np.nan; lo,hi=exact_binomial_interval(exceed,usable_B); usable_rate=usable_B/completed if completed else np.nan; hp_rate=hp_B/completed if completed else np.nan; obvalid=bool(observed.null_basic_panel_valid and observed.alternative_basic_panel_valid)
    return {"outcome":outcome,"test_specification":"REML parametric bootstrap; RI null versus unstructured RI+t5_mid slope; observed schedules/covariates; a bootstrap replicate is usable when both restricted likelihoods are finite and multistart optimizers agree. Positive-Hessian status is reported diagnostically and is not an exclusion rule.","requested_B":int(requested_B),"completed_B":completed,"usable_optimizer_agreeing_denominator":usable_B,"unusable_denominator":completed-usable_B,"usable_optimizer_agreeing_rate":usable_rate,"Hessian_positive_denominator":hp_B,"Hessian_positive_rate":hp_rate,"observed_restricted_LR":float(observed.observed_restricted_LR),"tail_exceedances":exceed,"bootstrap_p_value_plus_one":pval,"MC_interval_low":lo,"MC_interval_high":hi,"MC_interval_overlaps_0p05":bool(np.isfinite(lo) and lo<=ALPHA<=hi),"extension_requested":bool(int(requested_B)>PB_INITIAL_B),"extension_completed":bool(completed>=int(requested_B)),"observed_basic_optimizer_panel_valid":obvalid,"p_value_licensed":bool(completed>=int(requested_B) and obvalid and np.isfinite(usable_rate) and usable_rate>=RLRT_USABLE_RATE_TARGET)}
requested={o:(PB_MAX_B if o=="global_mean_thickness" else PB_INITIAL_B) for o in PB_OUTCOMES}; summary=pd.DataFrame([summarize(o,requested[o]) for o in PB_OUTCOMES]); raw=summary.bootstrap_p_value_plus_one.to_numpy(float); adj=np.full(len(raw),np.nan); finite=np.flatnonzero(np.isfinite(raw))
if len(finite):
    order=finite[np.argsort(raw[finite],kind="mergesort")]; vals=raw[order]; adj[order]=np.minimum(np.maximum.accumulate((len(vals)-np.arange(len(vals)))*vals),1.0)
summary["Holm_adjusted_p_across_six"]=adj; summary["reporting_interpretation"]=summary.apply(lambda r:("not licensed: incomplete requested B, invalid observed panel, or <95% finite optimizer agreement" if not r.p_value_licensed else "supported after Holm adjustment" if r.Holm_adjusted_p_across_six<ALPHA else "borderline: raw p<.05 but not Holm-significant" if r.bootstrap_p_value_plus_one<ALPHA else "not supported at raw .05"),axis=1)
summary.to_csv(out/"05_parametric_RLRT_summary_repaired.csv",index=False); pd.DataFrame([summarize("lateral_ventricles",PB_INITIAL_B)]).to_csv(out/"05b_ventricular_RLRT_adjudication.csv",index=False)
final_vent=reps.loc[reps.outcome.eq("lateral_ventricles")]; overlap=[c for c in vent.columns if c in final_vent.columns]; merged=vent[overlap].merge(final_vent[overlap],on=["outcome","replicate"],how="inner",suffixes=("__repair","__final"),validate="one_to_one")
if len(merged)!=len(vent): raise RuntimeError("Not every ventricular repair is folded into final 06")
print(summary[["outcome","requested_B","completed_B","usable_optimizer_agreeing_denominator","Hessian_positive_denominator","tail_exceedances","bootstrap_p_value_plus_one","Holm_adjusted_p_across_six","p_value_licensed","reporting_interpretation"]].to_string(index=False)); print("VENTRICULAR REPAIR OVERLAY ROWS:",len(merged)); print("REFITS PERFORMED: 0")
