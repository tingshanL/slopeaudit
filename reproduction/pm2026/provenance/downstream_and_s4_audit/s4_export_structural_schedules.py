#!/usr/bin/env python3
"""Add the three original structural-MRI schedules to the verified S4 schedule JSON.

Runs only reviewed notebook data preparation (cell 4 and prepare_outcome_data).
No model fitting or operating-study simulation is run. As in the notebook, each
structural CSV is read from --struct-dir when present, otherwise --struct-zip.
The selected input files/members and their hashes are recorded.

Checks: pinned notebook/subjects/existing-schedule hashes; original cohort counts;
finite ordered schedules; oracle covariance and reliability summaries; archived
fit reliability summaries for three interior rows per outcome; and original vs
released generator agreement and archived seeds for those nine rows. Aggregate
metric agreement is not a proof of historical participant order: that order is
reconstructed by the original preparation function and its lexical pid/time sort.

Output: a NEW six-schedule s4_refit_schedules.json, checks CSV and manifest. Only
ordered time arrays and counts are exported, without participant IDs or outcomes.
Keep the resulting restricted-derived JSON locally. Original files are unchanged.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_SHA = "0ec11c65dedd522d906f76daae6a745cc25bc81b92e26ab7c0953fa3359fb4c5"
SUBJECTS_SHA = "c40b78fd0198ff0835d4e33d8aa65c08e5fde9b03679cc1507d7fd6ac58f2705"
EXISTING_SHA = "38e2570253453f7fed9250e57a27d1f63e4f8d583465f0ed124874fd136316f6"
STRUCTURAL = ("lateral_ventricles", "hippocampus", "global_mean_thickness")
EXISTING = {"empirical__global_cortical_cbf": (802, 2047),
            "empirical__global_cortical_att": (802, 2047),
            "empirical__fluid_cognition": (691, 1805)}
EXPECTED_STRUCTURAL = (622, 1629)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_cells(text):
    parts = re.split(r"^# CELL (\d+)\s*$", text, flags=re.M)
    return {int(parts[i]): parts[i+1] for i in range(1, len(parts), 2)}


def extract_function(texts, name, ns):
    matches = []
    for text in texts:
        if not re.search(r"^def " + re.escape(name) + r"\(", text, flags=re.M):
            continue
        matches.extend(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    if len(matches) != 1:
        raise ValueError(f"Expected one reviewed definition of {name}; found {len(matches)}")
    exec(compile(ast.Module(body=matches, type_ignores=[]), name, "exec"), ns)
    return ns[name]


def checked_times(name, spec, expected):
    times = spec["participant_times"]
    counts = (len(times), sum(len(t) for t in times))
    if counts != expected or (spec["participants"], spec["observations"]) != counts:
        raise ValueError(f"{name}: expected {expected}, got {counts}")
    for t in times:
        t = np.asarray(t, dtype=float)
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.ptp(t) <= 0 or (np.diff(t) < 0).any():
            raise ValueError(f"Invalid or unordered times in {name}")


def build_structural(source, subjects, roster, struct_dir, struct_zip, final622_ids=None):
    cells = split_cells(source)
    selected = []

    def read_structural_csv(filename, **kwargs):
        path = struct_dir / filename if struct_dir else None
        if path is not None and path.is_file():
            selected.append({"filename": filename, "source": str(path.resolve()), "sha256": sha(path)})
            return pd.read_csv(path, **kwargs)
        if struct_zip is None or not struct_zip.is_file():
            raise FileNotFoundError(f"{filename} absent from structural directory and no usable ZIP supplied")
        with zipfile.ZipFile(struct_zip) as z:
            members = [m for m in z.namelist() if m.endswith(filename) and not m.startswith("__MACOSX/")]
            if len(members) != 1:
                raise ValueError(f"Expected one ZIP member ending {filename}, found {len(members)}")
            content = z.read(members[0])
        selected.append({"filename": filename, "source": str(struct_zip.resolve()),
                         "member": members[0], "sha256": hashlib.sha256(content).hexdigest()})
        return pd.read_csv(io.BytesIO(content), **kwargs)

    ns = dict(np=np, pd=pd, STRUCT_ROSTER=roster, FINAL622_IDS=final622_ids,
              SUBJECTS_CSV=subjects, read_structural_csv=read_structural_csv,
              FIVE_YEAR_DAYS=365.25*5, AGE_REFERENCE_YEARS=60., AGE_SCALE_YEARS=10.,
              MIN_THICKNESS_COMPLETENESS=.95, display=lambda *args, **kwargs: None)
    extract_function([cells[3]], "canonical_event_id", ns)
    # Execute the pinned data-preparation cell verbatim. No fitting code is in it.
    exec(compile(cells[4], "reviewed structural preparation cell 4", "exec"), ns)
    visits = ns["struct_visits"]
    if (visits.participant_id.nunique(), len(visits)) != EXPECTED_STRUCTURAL:
        raise ValueError("Frozen structural cohort count mismatch")
    ns["OUTCOMES"] = {
        k: dict(modality="structural MRI", source=visits, source_col=k,
                id_col="participant_id", icv=k in ("lateral_ventricles", "hippocampus"))
        for k in STRUCTURAL
    }
    prepare = extract_function([cells[6]], "prepare_outcome_data", ns)
    schedules = {}
    for k in STRUCTURAL:
        frame = prepare(k)
        times = [g.t5_mid.to_numpy(float).tolist() for _, g in frame.groupby("pid", sort=False)]
        name = "empirical__" + k
        spec = {"participants": len(times), "observations": sum(map(len, times)), "participant_times": times}
        checked_times(name, spec, EXPECTED_STRUCTURAL)
        schedules[name] = spec
        print(f"{k}: participants/observations = {spec['participants']}/{spec['observations']}", flush=True)
    return schedules, selected


def validate(schedules, source, repo, s4dir):
    design = pd.read_csv(s4dir / "operating_study_design.csv", float_precision="round_trip")
    archive = pd.read_csv(s4dir / "operating_study_replicate_results.csv", low_memory=False, float_precision="round_trip")
    if design.cell_id.duplicated().any() or archive.duplicated(["cell_id", "replicate"]).any():
        raise ValueError("Duplicate design/replicate keys")
    cells = split_cells(source)
    ns = dict(np=np, pd=pd, SEED=20260718, simulation_manifest=design,
              SIMULATION_CELL_LOOKUP=design.set_index("cell_id").to_dict("index"))
    build = extract_function(cells.values(), "build_schedule_template", ns)
    summary = extract_function(cells.values(), "schedule_predictability_summary", ns)
    ns["SCHEDULE_TEMPLATES"] = {k: build(k, v["participant_times"]) for k, v in schedules.items()}
    original_generate = extract_function(cells.values(), "generate_simulation_outcome", ns)
    local_ns = dict(ns)
    local_code = (repo / "reproduction/s4_operating_study/generate_operating_study.py").read_text()
    local_generate = extract_function([local_code], "generate_simulation_outcome", local_ns)
    checks = []

    def check(kind, cell, field, expected, actual, replicate=None, atol=1e-8, rtol=1e-8):
        ok = bool(np.isfinite([expected, actual]).all() and np.isclose(actual, expected, atol=atol, rtol=rtol))
        checks.append(dict(kind=kind, cell_id=cell, replicate=replicate, field=field,
                           expected=float(expected), actual=float(actual), absdiff=abs(float(actual)-float(expected)), check_ok=ok))

    fields = ["mean_Rdirect", "mean_Pfull", "mean_Rinnovation", "mean_Delta"]
    for outcome in STRUCTURAL:
        cell = "empirical__" + outcome
        selected = design.loc[design.cell_id.eq(cell)]
        if len(selected) != 1:
            raise ValueError(f"Missing/ambiguous design row: {cell}")
        spec = selected.iloc[0]
        template = ns["SCHEDULE_TEMPLATES"][cell]
        tau2 = spec.q1 / template["reference_information"]
        covariance = spec.rho * np.sqrt(spec.q0 * tau2)
        gm = np.array([[spec.q0, covariance], [covariance, tau2]])
        check("oracle", cell, "slope_variance", spec.oracle_slope_variance, tau2)
        check("oracle", cell, "covariance", spec.oracle_covariance, covariance)
        metrics = summary(gm, 1., template)
        for field in fields:
            check("oracle", cell, field, spec["oracle_" + field], metrics[field])

        block = archive.loc[archive.cell_id.eq(cell)].copy().sort_values("replicate")
        if len(block) != int(spec.requested_reps):
            raise ValueError(f"Archive row count mismatch for {cell}")
        covs = np.array([[[r.random_intercept_variance, r.intercept_slope_covariance],
                          [r.intercept_slope_covariance, r.random_slope_variance]]
                         for r in block.itertuples()])
        finite = np.isfinite(covs).all(axis=(1, 2))
        interior = np.zeros(len(block), dtype=bool)
        eig = np.linalg.eigvalsh(covs[finite])
        interior[finite] = (eig[:, 0] > 1e-6 * np.maximum(1., eig[:, 1]))
        good = block.loc[interior & block.strict_fit_valid.eq(True).to_numpy()
                         & np.isfinite(block[fields]).all(axis=1).to_numpy()]
        if len(good) < 3:
            raise ValueError(f"Fewer than three interior archived fits for {cell}")
        # Prespecified first/middle/last interior rows; no likelihood selection.
        for r in good.iloc[[0, len(good)//2, len(good)-1]].itertuples(index=False):
            Gfit = np.array([[r.random_intercept_variance, r.intercept_slope_covariance],
                             [r.intercept_slope_covariance, r.random_slope_variance]])
            fit_metrics = summary(Gfit, r.residual_variance, template)
            for field in fields:
                check("archived_fit", cell, field, getattr(r, field), fit_metrics[field], int(r.replicate))
            y, seed = original_generate(cell, int(r.replicate))
            y2, seed2 = local_generate(cell, int(r.replicate))
            if y.shape != y2.shape:
                raise ValueError(f"Generator output shape mismatch: {cell}")
            check("generator", cell, "archived_seed", int(r.seed), int(seed), int(r.replicate), atol=0, rtol=0)
            check("generator", cell, "released_seed", int(seed), int(seed2), int(r.replicate), atol=0, rtol=0)
            check("generator", cell, "max_abs_response_difference", 0., np.max(np.abs(y-y2)), int(r.replicate), atol=1e-12, rtol=0)
    return pd.DataFrame(checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-code", type=Path, required=True)
    ap.add_argument("--subjects", type=Path, required=True)
    ap.add_argument("--struct-roster", type=Path, required=True)
    ap.add_argument("--struct-dir", type=Path)
    ap.add_argument("--struct-zip", type=Path)
    ap.add_argument("--final622-ids", type=Path)
    ap.add_argument("--existing-schedules", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--s4-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        ap.error("Output folder is not empty; choose a new --out-dir")
    if args.struct_dir is None and args.struct_zip is None:
        ap.error("Supply --struct-dir and/or --struct-zip")
    if args.struct_dir is not None and not args.struct_dir.is_dir():
        ap.error(f"Missing directory: {args.struct_dir}")
    files = [Path(__file__), args.source_code, args.subjects, args.struct_roster, args.existing_schedules,
             args.s4_dir / "operating_study_design.csv", args.s4_dir / "operating_study_replicate_results.csv",
             args.repo_root / "reproduction/s4_operating_study/generate_operating_study.py"]
    if args.final622_ids is not None:
        files.append(args.final622_ids)
    for path in files + ([args.struct_zip] if args.struct_zip else []):
        if not path.is_file():
            ap.error(f"Missing file: {path}")
    for path, expected in [(args.source_code, SOURCE_SHA), (args.subjects, SUBJECTS_SHA),
                           (args.existing_schedules, EXISTING_SHA)]:
        if sha(path) != expected:
            ap.error(f"Input differs from the reviewed version: {path}")
    old = json.loads(args.existing_schedules.read_text())
    if not isinstance(old.get("schedules"), dict) or set(old["schedules"]) != set(EXISTING):
        ap.error("Expected the original three-schedule JSON")
    for name, expected in EXISTING.items():
        checked_times(name, old["schedules"][name], expected)
    source = args.source_code.read_text()
    structural, selected_files = build_structural(source, args.subjects, args.struct_roster,
                                                  args.struct_dir, args.struct_zip, args.final622_ids)
    combined = {**old, "schedules": {**old["schedules"], **structural}}
    checks = validate(combined["schedules"], source, args.repo_root, args.s4_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checks.to_csv(args.out_dir / "structural_schedule_checks.csv", index=False)
    if not checks.check_ok.all():
        print(checks.loc[~checks.check_ok].to_string(index=False))
        raise SystemExit("Structural schedule checks failed; no schedule JSON written.")
    for name in EXISTING:
        if combined["schedules"][name] != old["schedules"][name]:
            raise RuntimeError("An existing schedule was changed")
    output = args.out_dir / "s4_refit_schedules.json"
    output.write_text(json.dumps(combined, allow_nan=False))
    manifest = dict(method="structural_schedule_export_v1", checks_passed=len(checks),
                    existing_schedules_preserved=True, schedules=list(combined["schedules"]),
                    structural_sources=selected_files,
                    sha256={str(p): sha(p) for p in files + [output]},
                    note="Restricted-derived ordered times only. Aggregate checks validate schedule metrics; original data-preparation code determines order.")
    (args.out_dir / "schedule_export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Checks: {len(checks)}/{len(checks)} passed; max metric difference "
          f"{checks.loc[checks.kind.ne('generator'), 'absdiff'].max():.2e}; existing schedules preserved: 3/3")
    print(f"SUCCESS: {output}")
    print("Six schedules available. Keep this restricted-derived JSON locally; next screen only the three structural cells.")


if __name__ == "__main__":
    main()
