#!/usr/bin/env python3
"""Construct final anchored geometry and deterministic profile selections.

Consumes the validated 499 x 2 reference-time bootstrap geometry and writes
only aggregate/resample-manifest products.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from _aabc2_advanced_rlrt_core import (coerce_boolean, midpoint_to_anchor_covariance, anchored_endpoint_geometry)

TARGET_GEOMETRY_OUTCOMES=("global_mean_thickness","fluid_cognition")
GAP_METRIC="median_full_minus_direct_gap"
HIGH_GAP_THRESHOLD=0.10

p=argparse.ArgumentParser()
p.add_argument("--reference-time-geometry",required=True)
p.add_argument("--output-dir",required=True)
a=p.parse_args()
source=Path(a.reference_time_geometry).expanduser().resolve()
out=Path(a.output_dir).expanduser().resolve()
if not source.is_file(): raise FileNotFoundError(source)
if out.exists(): raise FileExistsError(out)
out.mkdir(parents=True)
full=pd.read_csv(source,low_memory=False)
for c in ("valid","geometry_eligible"): full[c]=coerce_boolean(full[c])
full["replicate"]=pd.to_numeric(full["replicate"],errors="raise").astype(int)
full=full.loc[full.outcome.isin(TARGET_GEOMETRY_OUTCOMES)].copy()
assert full.groupby("outcome").replicate.nunique().eq(499).all()
assert not full.duplicated(["outcome","replicate"]).any()
rows=[]
for r in full.itertuples(index=False):
    G=np.array([[r.random_intercept_variance,r.intercept_slope_covariance],[r.intercept_slope_covariance,r.random_slope_variance]])
    rows.append(anchored_endpoint_geometry(G))
anch=pd.concat([full.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
repaired=anch.copy()
if "cov_L0_L5" not in repaired.columns: repaired=repaired.rename(columns={"g_0_5":"cov_L0_L5"})
repaired["d_0_5_normalized_determinant"]=repaired["one_minus_gamma_0_5_squared"]
ratios=[]
for r in repaired.itertuples(index=False):
    E=np.array([[r.anchor_random_level_variance,r.cov_L0_L5],[r.cov_L0_L5,r.five_year_random_level_variance]])
    den=r.anchor_random_level_variance*r.five_year_random_level_variance
    ratios.append(np.linalg.det(E)/den if den>0 else np.nan)
repaired["det_endpoint_covariance_over_variance_product"]=ratios
err=float(np.nanmax(np.abs(repaired["d_0_5_normalized_determinant"]-repaired["det_endpoint_covariance_over_variance_product"])))
if not err<1e-7: raise RuntimeError(f"Anchored determinant identity failed: {err}")
if "g_0_5" in repaired.columns: raise RuntimeError("Historical g_0_5 name survived final repair")

def select_gap_regimes(group,n=10):
    group=group.loc[group.valid & group.geometry_eligible & np.isfinite(group[GAP_METRIC])].sort_values([GAP_METRIC,"replicate"],kind="mergesort").copy()
    if len(group)<3*n: raise ValueError(f"{group.outcome.iloc[0]} has {len(group)} eligible rows; need {3*n}.")
    low=group.head(n).copy(); high=group.tail(n).copy(); excluded=set(low.replicate)|set(high.replicate)
    midpool=group.loc[~group.replicate.isin(excluded)].copy(); med=float(group[GAP_METRIC].median())
    midpool["distance_to_median"]=(midpool[GAP_METRIC]-med).abs(); mid=midpool.sort_values(["distance_to_median","replicate"],kind="mergesort").head(n).copy()
    low["gap_regime"]="lowest"; mid["gap_regime"]="median"; high["gap_regime"]="highest"
    z=pd.concat([low,mid,high],ignore_index=True); z["outcome_median_gap"]=med; return z
selected=pd.concat([select_gap_regimes(g,10) for _,g in anch.groupby("outcome")],ignore_index=True)
assert len(selected)==60 and selected.groupby(["outcome","gap_regime"]).size().eq(10).all()
profile_rows=[]
for outcome,g in anch.groupby("outcome"):
    high=g.loc[g.valid & g.geometry_eligible & g[GAP_METRIC].ge(HIGH_GAP_THRESHOLD)].copy(); definition=f"gap >= {HIGH_GAP_THRESHOLD:.2f}"
    if high.empty:
        high=g.loc[g.valid & g.geometry_eligible].nlargest(max(10,int(np.ceil(.10*len(g)))),GAP_METRIC); definition="fallback: top 10% because no row met the frozen threshold"
    med=float(high[GAP_METRIC].median()); high["distance_to_high_regime_median"]=(high[GAP_METRIC]-med).abs()
    chosen=high.sort_values(["distance_to_high_regime_median","replicate"],kind="mergesort").iloc[0].to_dict(); chosen["high_regime_definition"]=definition; chosen["high_regime_median_gap"]=med; profile_rows.append(chosen)
profile=pd.DataFrame(profile_rows)
repaired.to_csv(out/"01_full499_anchored_geometry_repaired.csv",index=False)
selected.to_csv(out/"02_selected60_manifest.csv",index=False)
profile.to_csv(out/"03_profile_resample_manifest.csv",index=False)
print("anchored rows:",len(repaired)); print("selected rows:",len(selected)); print("profile rows:",len(profile)); print("GEOMETRY IDENTITY MAX ERROR:",err)
