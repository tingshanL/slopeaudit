#!/usr/bin/env python3
"""Export the original ATT, CBF and cognition schedules needed for S4's 245 repairs.

Reads the reviewed notebook's data-preparation code only (no model fitting or
simulation study). Uses four restricted input files locally. Exports ordered time
arrays without IDs or outcomes. Verifies every empirical-schedule starting-point
candidate against its archived starting variance and the local S4 generator.
No schedule JSON is written unless all reconstruction checks pass.
"""
from __future__ import annotations
import argparse
import ast
import fnmatch
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE_SHA = '0ec11c65dedd522d906f76daae6a745cc25bc81b92e26ab7c0953fa3359fb4c5'
INPUTS = {
 'subjects': ('*AABC2_subjects*.csv', 'c40b78fd0198ff0835d4e33d8aa65c08e5fde9b03679cc1507d7fd6ac58f2705'),
 'asl_audit': ('AABC_ASL_metabolic_perfusion_audit*.xlsx', 'd8f42b58d7590f07e1134e81366e47a854762351afbe7f2ba8b32f2397f64d96'),
 'asl_zip': ('AABC_Release2_ASLIDPs*.zip', None),
 'info_zip': ('AABC_Release2_IDP_info*.zip', None),
}
OUTCOMES = ['global_cortical_cbf', 'global_cortical_att', 'fluid_cognition']
EXPECTED = {'global_cortical_cbf': (802, 2047), 'global_cortical_att': (802, 2047), 'fluid_cognition': (691, 1805)}


def sha(path):
 h = hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
 return h.hexdigest()


def read_zip_csv(path, suffix, **kw):
 with zipfile.ZipFile(path) as z:
  matches = [n for n in z.namelist() if n.endswith(suffix) and not n.startswith('__MACOSX/')]
  if len(matches) != 1: raise ValueError(f'{path}: expected one {suffix}, found {len(matches)}')
  with z.open(matches[0]) as f: return pd.read_csv(f, **kw)


def split_cells(text):
 pieces = re.split(r'^# CELL (\d+)\s*$', text, flags=re.M)
 return {int(pieces[i]): pieces[i+1] for i in range(1,len(pieces),2)}


def extract_function(texts, name, namespace):
 for text in texts:
  if not re.search(r'^def '+re.escape(name)+r'\(', text, flags=re.M): continue
  tree = ast.parse(text)
  nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
  if len(nodes) != 1: raise ValueError('Ambiguous function: '+name)
  exec(compile(ast.Module(body=nodes,type_ignores=[]), name, 'exec'), namespace)
  return namespace[name]
 raise ValueError('Missing function: '+name)


def locate_inputs(args):
 found = {k: [] for k in INPUTS}
 for root in dict.fromkeys(Path(p).expanduser().resolve() for p in args.search_root):
  if not root.is_dir(): continue
  print('Searching:',root,flush=True)
  for parent, dirs, files in os.walk(root):
   dirs[:] = [d for d in dirs if d not in {'.git','node_modules','__pycache__','.venv'}]
   for name in files:
    for key,(pattern,_) in INPUTS.items():
     if fnmatch.fnmatch(name,pattern): found[key].append(Path(parent)/name)
 chosen = {}
 for key,(_,expected_hash) in INPUTS.items():
  explicit = getattr(args,key)
  candidates = [explicit.expanduser().resolve()] if explicit else sorted(set(found[key]),key=str)
  hashes = {}
  for p in candidates:
   if p.is_file(): hashes[p] = sha(p)
  if expected_hash:
   candidates = [p for p,h in hashes.items() if h == expected_hash]
  else:
   candidates = list(hashes)
   if len(set(hashes.values())) > 1:
    raise ValueError(f'Different versions of {key} found. Specify --{key.replace("_","-")}:\n'+'\n'.join(map(str,candidates)))
  if not candidates:
   raise ValueError(f'No matching original {key} input. Searched candidates:\n'+'\n'.join(map(str,hashes)))
  chosen[key] = candidates[0]
  print(key+':',chosen[key],flush=True)
 return chosen


def build_schedules(source, paths):
 cells = split_cells(source)
 ns = dict(np=np,pd=pd,Path=Path,zipfile=zipfile,read_zip_csv=read_zip_csv,
           FIVE_YEAR_DAYS=365.25*5,AGE_REFERENCE_YEARS=60.,AGE_SCALE_YEARS=10.,
           ASL_AUDIT=paths['asl_audit'],ASL_ZIP=paths['asl_zip'],INFO_ZIP=paths['info_zip'],
           display=lambda *a,**k:None)
 # Exact subject metadata construction from notebook cell 4; no structural data.
 subject_cols = ['id_event','study','days_from_V1','age_open','sex','site','scanner','tlbx_cfc_uncorrected_standard_score']
 subject_events = pd.read_csv(paths['subjects'],usecols=subject_cols,low_memory=False)
 subject_events['id_event'] = subject_events['id_event'].astype(str).str.strip()
 subject_events['id'] = subject_events['id_event'].str.replace(r'_[^_]+$','',regex=True)
 if not subject_events.id_event.is_unique: raise ValueError('Duplicate subject events')
 ns['subject_events'] = subject_events
 # Omit its final display, which also mentions the unused structural cohort.
 cell5 = ast.parse(cells[5])
 cell5.body = [n for n in cell5.body if not (isinstance(n,ast.Expr)
                and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
                and n.value.func.id=='display')]
 exec(compile(cell5,'reviewed notebook cell 5','exec'),ns)
 ns['OUTCOMES'] = {
  k:dict(modality='cognition' if k=='fluid_cognition' else 'ASL',
         source=ns['cognition'] if k=='fluid_cognition' else ns['asl_visits'],
         source_col=k,id_col='id',icv=False) for k in OUTCOMES}
 prepare = extract_function([cells[6]],'prepare_outcome_data',ns)
 schedules = {}
 for k in OUTCOMES:
  frame = prepare(k)
  times = [g.t5_mid.to_numpy(float).tolist() for _,g in frame.groupby('pid',sort=False)]
  size = len(times),sum(map(len,times))
  if size != EXPECTED[k]: raise ValueError(f'{k}: expected {EXPECTED[k]}, got {size}')
  if any(len(t)<2 or not np.isfinite(t).all() or np.ptp(t)<=0 for t in times):
   raise ValueError('Invalid schedule in '+k)
  schedules['empirical__'+k] = dict(participants=size[0],observations=size[1],participant_times=times)
  print(k, 'participants/observations:',size,flush=True)
 return schedules


def validate(schedules, source, repo, s4dir):
 design = pd.read_csv(s4dir/'operating_study_design.csv',float_precision='round_trip')
 archive = pd.read_csv(s4dir/'operating_study_replicate_results.csv',low_memory=False,float_precision='round_trip')
 s2 = archive.residual_variance
 signature = (np.isclose(archive.random_intercept_variance,s2,rtol=1e-8,atol=0)
              & (archive.intercept_slope_covariance.abs() <= 1e-10*s2)
              & np.isclose(archive.random_slope_variance,.1*s2,rtol=1e-8,atol=0))
 candidates = archive[signature & archive.schedule.isin(schedules)]
 nonbalanced = archive[signature & ~archive.schedule.eq('balanced_three_wave')]
 if len(candidates)!=len(nonbalanced): raise ValueError('Export does not cover all empirical candidate schedules')
 cells = split_cells(source)
 ns = dict(np=np,pd=pd,SEED=20260718,simulation_manifest=design,
           SIMULATION_CELL_LOOKUP=design.set_index('cell_id').to_dict('index'))
 build = extract_function(cells.values(),'build_schedule_template',ns)
 ns['SCHEDULE_TEMPLATES'] = {k:build(k,v['participant_times']) for k,v in schedules.items()}
 original_generate = extract_function(cells.values(),'generate_simulation_outcome',ns)
 other = dict(ns)
 current = (repo/'reproduction/s4_operating_study/generate_operating_study.py').read_text()
 local_generate = extract_function([current],'generate_simulation_outcome',other)
 rows = []
 for r in candidates.itertuples(index=False):
  y,seed = original_generate(r.cell_id,int(r.replicate))
  y2,seed2 = local_generate(r.cell_id,int(r.replicate))
  diff = float(np.max(np.abs(y-y2)))
  v = .5*max(float(np.var(y,ddof=1)),1e-4)
  ok = (int(seed)==int(r.seed)==int(seed2) and diff<=1e-12
        and np.isclose(v,r.residual_variance,rtol=1e-8,atol=1e-10))
  rows.append(dict(cell_id=r.cell_id,replicate=int(r.replicate),seed=int(seed),
                   generator_max_absdiff=diff,starting_variance_absdiff=abs(v-r.residual_variance),check_ok=bool(ok)))
 check = pd.DataFrame(rows)
 if check.empty or not check.check_ok.all():
  bad = check[~check.check_ok] if not check.empty else check
  raise ValueError('Schedule reconstruction checks failed; no JSON written.\n'+bad.head(12).to_string(index=False))
 print('Reconstruction checks:',len(check),'passed;', 'maximum starting-variance difference:',check.starting_variance_absdiff.max(),flush=True)
 return check


def main():
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--source-code',type=Path,required=True)
 ap.add_argument('--repo-root',type=Path,required=True)
 ap.add_argument('--s4-dir',type=Path,required=True)
 ap.add_argument('--search-root',type=Path,action='append',default=[])
 ap.add_argument('--out-dir',type=Path,required=True)
 for key in INPUTS: ap.add_argument('--'+key.replace('_','-'),type=Path)
 args = ap.parse_args()
 source = args.source_code.read_text()
 if sha(args.source_code)!=SOURCE_SHA: ap.error('Source differs from the reviewed notebook extract')
 if not args.search_root:
  args.search_root = [Path.home()/'Desktop/2026Spring/AD/exposome',
                      Path.home()/'Desktop/AMPPS_release_work',Path.home()/'Downloads']
 output = args.out_dir/'s4_refit_schedules.json'
 if output.exists(): ap.error('Output already exists; choose another --out-dir')
 paths = locate_inputs(args)
 schedules = build_schedules(source,paths)
 check = validate(schedules,source,args.repo_root,args.s4_dir)
 args.out_dir.mkdir(parents=True,exist_ok=True)
 output.write_text(json.dumps(dict(schedules=schedules),allow_nan=False))
 check.to_csv(args.out_dir/'schedule_reconstruction_checks.csv',index=False)
 files = [Path(__file__),args.source_code,args.s4_dir/'operating_study_design.csv',
          args.s4_dir/'operating_study_replicate_results.csv',
          args.repo_root/'reproduction/s4_operating_study/generate_operating_study.py',*paths.values(),output]
 (args.out_dir/'schedule_export_manifest.json').write_text(json.dumps(dict(
    source_sha256=SOURCE_SHA,checks_passed=len(check),
    note='Ordered time arrays only; no participant IDs or outcomes. Original raw-input hashes checked for subjects and audit workbook.',
    sha256={str(p):sha(p) for p in files}),indent=2))
 print('SUCCESS:',output)
 print('Keep this restricted-derived file locally. Next: Step 4 --smoke with this --schedule-json.')

if __name__=='__main__':
 main()
