#!/usr/bin/env python3
"""Assemble the public result files for the 2026 manuscript from a local project folder.

Two outputs, both built only from an explicit allowlist:
  --github-out DIR   small summaries for results/pm2026/ in the repository
  --zenodo-out DIR   zip archives with the full run outputs, a file manifest and SHA256SUMS

The project folder (--root) is the local working folder that holds runs/ and inputs/.
JSON manifests are copied with machine-specific absolute paths rewritten to
repository-relative ones; every rewritten file is listed in path_rewrites.csv with its
original and public SHA-256, so the private originals remain verifiable. The script stops
if any selected file is a restricted input, contains an AABC2 participant identifier, or
still contains an absolute home-directory path after rewriting. It never reads restricted
inputs, schedule exports or notebook extracts.

Usage:
  python3 reproduction/pm2026/release/prepare_release_files.py --root <project folder> \
      --github-out results/pm2026 [--zenodo-out <dir>] [--extra <dir>=<label> ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

# Absolute-path prefixes rewritten in manifests: (pattern, replacement).
PATH_REWRITES = [
    (re.compile(r"(?:[A-Za-z]:)?[/\\][^\"']*?[/\\]slopeaudit_PM[/\\]"), ""),
    (re.compile(r"(?:[A-Za-z]:)?[/\\][^\"']*?[/\\]AMPPS_release_work[/\\]slopeaudit[/\\]"), "slopeaudit-v0.1.0/"),
]
ABSOLUTE_HOME = re.compile(r"(/Users/|/home/|[A-Za-z]:\\\\Users\\\\)")
PARTICIPANT_ID = re.compile(r"\bHC[AD]\d{6,}")
RESTRICTED_NAMES = re.compile(r"s4_refit_schedules|s4_source_code|source_anchor|anchor_manifest|restricted\.csv$",
                              re.IGNORECASE)

DOWNSTREAM_RUNS = {
    "pilot_v2/common_D2_0.80_likelihood_selection": "common_80",
    "pilot_v2/common_D2_0.50_likelihood_selection": "common_50",
    "contam_irregular_full/irregular_D2_0.80_likelihood_selection": "irregular_80",
    "contam_irregular_full/irregular_D2_0.50_likelihood_selection": "irregular_50",
    "pilot_v2_A_leakage/common_D2_0.50_likelihood_selection": "dose_A_50",
    "pilot_v2_A_leakage_N600/common_D2_0.50_likelihood_selection": "n600_A_50",
}
DOWNSTREAM_SMALL = ["manifest.json", "summary_by_method.csv", "summary_paired_contrasts.csv",
                    "summary_fit_quality.csv", "summary_plugin.csv", "summary_EB_omit_by_boundary.csv",
                    "comparison_with_previous_run.csv"]
DOWNSTREAM_LARGE = ["replicates.csv", "model_candidates.csv"]

S4_SMALL = [
    "s4_step1/s4_step1_report.csv",
    "s4_step2/README.txt", "s4_step2/cell_denominators.csv", "s4_step2/repair_class_transitions.csv",
    *[f"s4_step2/{d}/{t}" for d in ("all_fits", "strict_valid")
      for t in ("t1_true_zero_by_cell_sing.csv", "t2_true_zero_by_cell_q_sing.csv",
                "t3_positive_by_cell_sing.csv", "t4_pooled_family_sing.csv")],
    *[f"s4_step3_full/{f}" for f in ("s4_step3_audit.csv", "s4_step3_audit_archive_summary.csv",
                                      "s4_step3_audit_manifest.json", "s4_step3_audit_sample.csv",
                                      "s4_step3_audit_summary.csv")],
    "s4_step4_full_v2/manifest.json", "s4_step4_full_v2/startpoint_refit_audit.csv",
    *[f"{d}/{f}" for d in ("s4_step5_balanced_full_quiet2", "s4_step5_empirical_full", "s4_step5_structural_full")
      for f in ("manifest.json", "step5_flagged_keys.csv", "step5_unresolved_keys.csv")],
    "s4_step5_all_flagged_keys.csv",
    "s4_step6_consolidated_refit/manifest.json",
    *[f"{d}/{f}" for d in ("s4_original_summary", "s4_step6_consolidated_refit/summary")
      for f in ("likelihood_ratio_calibration.csv", "operating_characteristics.csv",
                "operating_metric_quantiles.csv", "simulation_cell_validity.csv",
                "true_zero_predictability_diagnostics.csv")],
    *[f"s4_step7_reporting/{f}" for f in ("t2_true_zero_table.csv", "t2b_boundary_breakdown.csv",
                                           "t2c_repaired_true_zero.csv", "oc_changes_long.csv",
                                           "lr_calibration_compare.csv", "decision_flips_by_cell.csv",
                                           "covariance_class_definitions.json", "figure_02_contract.json")],
]
S4_LARGE = [
    "s4_step2/replicate_strata.csv",
    "s4_step4_full_v2/operating_study_replicate_results_startpoint_patched.csv",
    *[f"{d}/step5_screen.csv" for d in ("s4_step5_balanced_full_quiet2", "s4_step5_empirical_full",
                                         "s4_step5_structural_full")],
    "s4_step6_consolidated_refit/startpoint_refit_audit.csv",
    "s4_step6_consolidated_refit/summary/operating_study_replicate_results.csv",
    "s4_original_summary/operating_study_replicate_results.csv",
]
S4_INPUTS_SMALL = ["inputs/s4_completion_validation_v1/operating_study_design.csv"]

ATTENUATION = "s5_fit_audit_AB_Cp3Lb1"
ATTENUATION_SMALL = ["partA_nested_check_summary.csv", "partA_multi_attempt_summary.csv",
                     "partA_multi_attempt_by_model.csv", "partB_by_task.csv",
                     "partB_fit_quality_summary.csv", "partB_output_changes_summary.csv"]
ATTENUATION_LARGE = ["partA_nested_check_by_task.csv", "partB_candidates.csv"]

EMPIRICAL_SMALL = ["s7_aabc2_aggregate/s7_aabc2_checks.csv", "s7_aabc2_aggregate/s7_aabc2_decomposition.csv",
                   "s7_aabc2_aggregate/s7_aabc2_kappa_quantiles.csv",
                   "s7_sleepstudy/s7_sleepstudy_per_subject.csv", "s7_sleepstudy/s7_sleepstudy_summary.csv"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rewrite_paths(obj):
    if isinstance(obj, dict):
        return {rewrite_paths(k): rewrite_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_paths(v) for v in obj]
    if isinstance(obj, str):
        for pattern, repl in PATH_REWRITES:
            obj = pattern.sub(repl, obj)
    return obj


def check_text(path: Path, data: bytes):
    if RESTRICTED_NAMES.search(path.name):
        sys.exit(f"STOP: restricted file selected: {path}")
    text = data.decode("utf-8", errors="ignore")
    if PARTICIPANT_ID.search(text):
        sys.exit(f"STOP: participant identifier found in {path}")
    if ABSOLUTE_HOME.search(text):
        sys.exit(f"STOP: absolute home-directory path remains in {path}")


def public_bytes(src: Path, rewrites: list, public_name: str) -> bytes:
    data = src.read_bytes()
    if src.suffix == ".json":
        original = json.loads(data)
        rewritten = rewrite_paths(original)
        if rewritten != original:
            data = (json.dumps(rewritten, indent=2, sort_keys=False) + "\n").encode()
            rewrites.append({"public_path": public_name, "original_sha256": sha256(src),
                             "public_sha256": hashlib.sha256(data).hexdigest(),
                             "change": "absolute paths rewritten to repository-relative paths"})
    check_text(src, data)
    return data


def selection(root: Path, include_large: bool):
    runs = root / "runs"
    items = []  # (source path, public relative path)
    for run, label in DOWNSTREAM_RUNS.items():
        for f in DOWNSTREAM_SMALL + (DOWNSTREAM_LARGE if include_large else []):
            p = runs / run / f
            if p.exists() or f not in ("comparison_with_previous_run.csv",):
                items.append((p, f"downstream/{label}/{f}"))
    for f in S4_SMALL + (S4_LARGE if include_large else []):
        items.append((runs / f, f"operating_audit/{f}"))
    for f in S4_INPUTS_SMALL:
        items.append((root / f, f"operating_audit/{Path(f).name}"))
    for f in ATTENUATION_SMALL + (ATTENUATION_LARGE if include_large else []):
        items.append((runs / ATTENUATION / f, f"attenuation_audit/{f}"))
    for f in EMPIRICAL_SMALL:
        items.append((runs / f, f"empirical/{f}"))
    missing = [str(s) for s, _ in items if not s.exists()]
    if missing:
        sys.exit("STOP: expected files are missing:\n  " + "\n  ".join(missing))
    return items


COMPACT_COLUMNS = ["cell_id", "replicate", "cell_family", "repair_method", "strict_fit_valid",
                   "mean_Pfull", "mean_Rdirect", "LR_evaluation_replicate", "slope_heterogeneity_supported"]


def write_compact_replicates(root: Path, out: Path):
    """Columns of the refit replicate table needed by make_S4_tables.py and make_figure_03.py."""
    import pandas as pd
    src = root / "runs/s4_step6_consolidated_refit/summary/operating_study_replicate_results.csv"
    d = pd.read_csv(src, usecols=COMPACT_COLUMNS, low_memory=False, float_precision="round_trip")
    rel = "operating_audit/s4_step6_consolidated_refit/summary/operating_study_replicate_results_compact.csv.gz"
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    d[COMPACT_COLUMNS].to_csv(dst, index=False, compression={"method": "gzip", "mtime": 0})
    return rel, dst, sha256(src)


def write_tree(items, out: Path, extra, root: Path):
    out.mkdir(parents=True, exist_ok=True)
    rewrites, rows = [], []
    for src, rel in items:
        data = public_bytes(src, rewrites, rel)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        rows.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    for d, label in extra:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                rel = f"{label}/{p.relative_to(d).as_posix()}"
                data = public_bytes(p, rewrites, rel)
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                (out / rel).write_bytes(data)
                rows.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    rel, dst, src_hash = write_compact_replicates(root, out)
    rows.append({"path": rel, "bytes": dst.stat().st_size, "sha256": sha256(dst)})
    rewrites.append({"public_path": rel, "original_sha256": src_hash, "public_sha256": sha256(dst),
                     "change": "column subset of operating_study_replicate_results.csv (full file on Zenodo)"})
    with open(out / "FILES.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["path", "bytes", "sha256"]); w.writeheader(); w.writerows(rows)
    with open(out / "path_rewrites.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["public_path", "original_sha256", "public_sha256", "change"])
        w.writeheader(); w.writerows(rewrites)
    return rows, rewrites


def build_zips(items, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    groups = {"downstream": "pm2026_downstream_simulation_runs.zip",
              "operating_audit": "pm2026_operating_study_audit.zip",
              "attenuation_audit": "pm2026_attenuation_study_audit.zip",
              "empirical": "pm2026_empirical_aggregates.zip"}
    rewrites, manifest = [], []
    for prefix, name in groups.items():
        with zipfile.ZipFile(out / name, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for src, rel in items:
                if rel.split("/", 1)[0] != prefix:
                    continue
                data = public_bytes(src, rewrites, rel)
                z.writestr(rel, data)
                manifest.append({"archive": name, "path": rel, "bytes": len(data),
                                 "sha256": hashlib.sha256(data).hexdigest()})
    with open(out / "archive_entry_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["archive", "path", "bytes", "sha256"]); w.writeheader(); w.writerows(manifest)
    with open(out / "path_rewrites.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["public_path", "original_sha256", "public_sha256", "change"])
        w.writeheader(); w.writerows(rewrites)
    with open(out / "SHA256SUMS.txt", "w") as f:
        for p in sorted(out.iterdir()):
            if p.name != "SHA256SUMS.txt":
                f.write(f"{sha256(p)}  {p.name}\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--github-out", type=Path)
    ap.add_argument("--zenodo-out", type=Path)
    ap.add_argument("--extra", action="append", default=[],
                    help="DIR=LABEL: also copy a folder of regenerated check outputs into --github-out")
    a = ap.parse_args()
    if not (a.github_out or a.zenodo_out):
        sys.exit("give --github-out and/or --zenodo-out")
    extra = [tuple(x.split("=", 1)) for x in a.extra]
    if a.github_out:
        if a.github_out.exists():
            shutil.rmtree(a.github_out)
        rows, rw = write_tree(selection(a.root, include_large=False), a.github_out, extra, a.root)
        print(f"GitHub results: {len(rows)} files, {sum(r['bytes'] for r in rows) / 1e6:.1f} MB; "
              f"{len(rw)} manifests rewritten")
    if a.zenodo_out:
        manifest = build_zips(selection(a.root, include_large=True), a.zenodo_out)
        print(f"Zenodo archives: {len(manifest)} entries in {a.zenodo_out}")


if __name__ == "__main__":
    main()
