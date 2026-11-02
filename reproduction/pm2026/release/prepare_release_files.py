#!/usr/bin/env python3
"""Assemble the public result files for the 2026 manuscript from a local project folder.

Two outputs, both built only from an explicit allowlist:
  --github-out DIR   the complete results/pm2026/ tree for the repository: small summaries,
                     the compact replicate table, the check folder given by --checks,
                     FILES.csv (every file with its SHA-256) and path_rewrites.csv
  --zenodo-out DIR   zip archives with the full run outputs, archive_entry_manifest.csv,
                     path_rewrites.csv and SHA256SUMS.txt (no FILES.csv)

The project folder (--root) is the local working folder that holds runs/ and inputs/.
--checks is the folder of check outputs and expected values copied to checks/ (normally the
committed results/pm2026/checks); it must contain every file listed in expected_outputs.py.

Every output folder must be new or empty. All inputs are located and screened before
anything is written; each output is assembled in a temporary sibling folder and moved into
place only after it is complete, so a failed run leaves no partial output and never removes
existing files.
JSON manifests are copied with machine-specific absolute paths rewritten to
repository-relative ones; every rewritten file is listed in path_rewrites.csv with its
original and public SHA-256, so the private originals remain verifiable. The script stops
if any selected file is a restricted input, contains an AABC2 participant identifier, or
still contains an absolute home-directory path after rewriting. It never reads restricted
inputs, schedule exports or notebook extracts.

Screening covers every byte that is published: the text of each selected file, the
uncompressed content of the compact replicate table (screened before it is written and
again after), the decompressed streams of PDF figures and the compressed text chunks of
PNG figures. Archive files (zip, 7z, rar, tar) and other gzip files are never published.
From --checks only the listed expected files and check outputs are copied; any other file
in that folder stops the build (macOS .DS_Store files are ignored).

Usage:
  python3 reproduction/pm2026/release/prepare_release_files.py --root <project folder> \
      --github-out <new dir> --checks results/pm2026/checks
  python3 reproduction/pm2026/release/prepare_release_files.py --root <project folder> \
      --zenodo-out <new dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import gzip
import os
import re
import shutil
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expected_outputs import CHECK_OUTPUTS, EXPECTED_FILES  # noqa: E402

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


ARCHIVE_MAGIC = {b"PK\x03\x04": "zip", b"PK\x05\x06": "zip", b"7z\xbc\xaf\x27\x1c": "7z",
                 b"Rar!": "rar"}
PDF_STREAM_START = re.compile(rb"(?<!end)stream\r?\n")
PDF_IMAGE_FILTERS = {b"DCTDecode", b"JPXDecode", b"CCITTFaxDecode", b"JBIG2Decode"}


def _screen_text(label: str, data: bytes):
    text = data.decode("utf-8", errors="ignore")
    if PARTICIPANT_ID.search(text):
        sys.exit(f"STOP: participant identifier found in {label}")
    if ABSOLUTE_HOME.search(text):
        sys.exit(f"STOP: absolute home-directory path remains in {label}")


def _pdf_streams(label: str, data: bytes):
    """Decoded content of every Flate-compressed PDF stream. Unfiltered streams are already
    covered by the raw-byte screen; image-only codecs carry no text; any other filter stops."""
    for m in PDF_STREAM_START.finditer(data):
        end = data.find(b"endstream", m.end())
        if end < 0:
            sys.exit(f"STOP: unterminated PDF stream in {label}")
        header = data[data.rfind(b"obj", 0, m.start()) + 3:m.start()]
        filters = re.findall(rb"/([A-Za-z0-9]+Decode)", header)
        if not filters or set(filters) <= PDF_IMAGE_FILTERS:
            continue
        if filters != [b"FlateDecode"]:
            sys.exit(f"STOP: PDF stream filter {[f.decode() for f in filters]} cannot be screened in {label}")
        try:
            yield zlib.decompressobj().decompress(data[m.end():end])
        except zlib.error:
            sys.exit(f"STOP: unreadable compressed PDF stream in {label}")


def _png_text_chunks(label: str, data: bytes):
    """Decompressed zTXt and iTXt chunks of a PNG (tEXt is plain and screened as raw bytes)."""
    pos = 8
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        pos += 12 + length
        try:
            if kind == b"zTXt":
                yield zlib.decompress(body.split(b"\x00", 1)[1][1:])
            elif kind == b"iTXt":
                keyword, rest = body.split(b"\x00", 1)
                if rest[:1] == b"\x01":
                    yield zlib.decompress(rest[2:].split(b"\x00", 2)[2])
        except (zlib.error, IndexError, ValueError):
            sys.exit(f"STOP: unreadable compressed PNG text chunk ({kind.decode(errors='ignore')}) in {label}")


def screen_bytes(label: str, data: bytes, allow_gzip: bool = False):
    """Screen everything a reader could recover from data: raw text and compressed content."""
    for magic, kind in ARCHIVE_MAGIC.items():
        if data.startswith(magic):
            sys.exit(f"STOP: {kind} archive selected for publication: {label}")
    if len(data) > 262 and data[257:262] == b"ustar":
        sys.exit(f"STOP: tar archive selected for publication: {label}")
    if data[:2] == b"\x1f\x8b":
        if not allow_gzip:
            sys.exit(f"STOP: gzip file selected for publication: {label}")
        _screen_text(f"{label} (uncompressed)", gzip.decompress(data))
        return
    _screen_text(label, data)
    if data.startswith(b"%PDF"):
        for inner in _pdf_streams(label, data):
            _screen_text(f"{label} (PDF stream)", inner)
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        for inner in _png_text_chunks(label, data):
            _screen_text(f"{label} (PNG text chunk)", inner)


def check_text(path: Path, data: bytes):
    if RESTRICTED_NAMES.search(path.name):
        sys.exit(f"STOP: restricted file selected: {path}")
    screen_bytes(str(path), data)


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


def planned_items(root: Path, include_large: bool):
    """(source path, public relative path) for every selected run file; no existence check."""
    runs = root / "runs"
    items = []
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
    return items


def selection(root: Path, include_large: bool):
    items = planned_items(root, include_large)
    missing = [str(s) for s, _ in items if not s.exists()]
    if not include_large and not (root / COMPACT_SOURCE).exists():
        missing.append(str(root / COMPACT_SOURCE))
    if missing:
        sys.exit("STOP: expected files are missing:\n  " + "\n  ".join(missing))
    return items


COMPACT_SOURCE = "runs/s4_step6_consolidated_refit/summary/operating_study_replicate_results.csv"
COMPACT_COLUMNS = ["cell_id", "replicate", "cell_family", "repair_method", "strict_fit_valid",
                   "mean_Pfull", "mean_Rdirect", "LR_evaluation_replicate", "slope_heterogeneity_supported"]


COMPACT_REL = "operating_audit/s4_step6_consolidated_refit/summary/operating_study_replicate_results_compact.csv.gz"


def compact_replicates(root: Path):
    """Columns of the refit replicate table needed by make_S4_tables.py and make_figure_03.py.

    Built in a private temporary folder (outside every output folder) and screened on its
    uncompressed content before it can be published. Returns (gzip bytes, source SHA-256)."""
    import pandas as pd
    src = root / COMPACT_SOURCE
    d = pd.read_csv(src, usecols=COMPACT_COLUMNS, low_memory=False, float_precision="round_trip")
    with tempfile.TemporaryDirectory(prefix="pm2026_compact_") as tmp:
        dst = Path(tmp) / Path(COMPACT_REL).name
        d[COMPACT_COLUMNS].to_csv(dst, index=False, compression={"method": "gzip", "mtime": 0})
        data = dst.read_bytes()
    screen_bytes(f"{COMPACT_REL} (built from {src})", data, allow_gzip=True)
    return data, sha256(src)


CHECKS_IGNORED = {".DS_Store"}


def check_selection(checks: Path):
    """Exactly the listed expected files and check outputs from --checks, copied to checks/.

    Every listed file is required; any other file in the folder stops the build."""
    required = [f"expected/{f}" for f in EXPECTED_FILES] + CHECK_OUTPUTS
    missing = [str(checks / f) for f in required if not (checks / f).is_file()]
    if missing:
        sys.exit("STOP: check files are missing:\n  " + "\n  ".join(missing))
    present = {p.relative_to(checks).as_posix() for p in checks.rglob("*")
               if (p.is_file() or p.is_symlink()) and p.name not in CHECKS_IGNORED}
    unexpected = sorted(present - set(required))
    if unexpected:
        sys.exit("STOP: unexpected files in the checks folder (only the listed expected files and "
                 "check outputs are published):\n  " + "\n  ".join(str(checks / f) for f in unexpected))
    return [(checks / f, f"checks/{f}") for f in sorted(required, key=lambda f: Path(f).parts)]


def screen(items):
    """Screen every selected file before anything is written."""
    for src, _ in items:
        public_bytes(src, [], "")


def require_new_or_empty(out: Path):
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        sys.exit(f"STOP: output folder {out} exists and is not empty; give a new or empty folder")


def build_atomically(out: Path, build):
    """Run build(tmp) in a temporary sibling folder and move it to out on success."""
    tmp = out.parent / f".{out.name}.partial-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        result = build(tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if out.exists():
        out.rmdir()  # empty, checked by require_new_or_empty
    tmp.rename(out)
    return result


def write_tree(items, out: Path, compact):
    out.mkdir(parents=True, exist_ok=True)
    rewrites, rows = [], []
    for src, rel in items:
        data = public_bytes(src, rewrites, rel)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        rows.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    data, src_hash = compact
    dst = out / COMPACT_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    screen_bytes(str(dst), dst.read_bytes(), allow_gzip=True)  # the written file, as published
    rows.append({"path": COMPACT_REL, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    rewrites.append({"public_path": COMPACT_REL, "original_sha256": src_hash,
                     "public_sha256": hashlib.sha256(data).hexdigest(),
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
    ap.add_argument("--checks", type=Path,
                    help="folder copied to checks/ in --github-out (normally results/pm2026/checks)")
    ap.add_argument("--zenodo-out", type=Path)
    a = ap.parse_args()
    if not (a.github_out or a.zenodo_out):
        sys.exit("give --github-out and/or --zenodo-out")
    if a.github_out and not a.checks:
        sys.exit("--github-out needs --checks (normally results/pm2026/checks)")

    # Locate and screen everything first; nothing is written until all inputs pass.
    plans = []
    if a.github_out:
        require_new_or_empty(a.github_out)
        items = selection(a.root, include_large=False) + check_selection(a.checks)
        screen(items)
        compact = compact_replicates(a.root)
        plans.append(("github", a.github_out, (items, compact)))
    if a.zenodo_out:
        require_new_or_empty(a.zenodo_out)
        items = selection(a.root, include_large=True)
        screen(items)
        plans.append(("zenodo", a.zenodo_out, items))

    for kind, out, items in plans:
        if kind == "github":
            files, compact = items
            rows, rw = build_atomically(out, lambda tmp: write_tree(files, tmp, compact))
            print(f"GitHub results: {len(rows)} files, {sum(r['bytes'] for r in rows) / 1e6:.1f} MB; "
                  f"{len(rw)} rewrites recorded; written to {out}")
        else:
            manifest = build_atomically(out, lambda tmp: build_zips(items, tmp))
            print(f"Zenodo archives: {len(manifest)} entries; written to {out}")


if __name__ == "__main__":
    main()
