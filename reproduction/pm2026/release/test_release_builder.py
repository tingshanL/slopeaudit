#!/usr/bin/env python3
"""Regression checks for prepare_release_files.py. No model fitting; takes a few seconds.

A synthetic project folder is reconstructed from the committed results/pm2026 files (the
compact replicate table stands in for the full one). The checks show that:
  1. a clean build reproduces the committed tree (path_rewrites.csv differs by design,
     because the reconstructed manifests are already repository-relative);
  2. a participant identifier in a retained column of the replicate table stops the build;
  3. an unexpected file in the checks folder, including a zip archive whose compressed
     content holds an identifier, stops the build;
  4. an identifier inside a compressed PDF stream of a listed check output stops the build;
  5. an existing non-empty output folder is refused and left untouched.
Every stopped build must exit non-zero and leave neither the output folder nor a partial
folder behind.

Usage (from the repository root):
  python3 reproduction/pm2026/release/test_release_builder.py
"""
import filecmp
import gzip
import io
import shutil
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "results/pm2026"
BUILDER = HERE / "prepare_release_files.py"
sys.path.insert(0, str(HERE))
import prepare_release_files as builder  # noqa: E402

# Synthetic identifier matching the screening pattern, assembled at run time so that the
# repository itself contains no string a public-file scan would flag.
FAKE_ID = "HC" + "A" + "9" * 7


def reconstruct_root(root: Path):
    # Optional per-run file: selected only when present, so place it before planning.
    for run, label in builder.DOWNSTREAM_RUNS.items():
        public = RESULTS / "downstream" / label / "comparison_with_previous_run.csv"
        if public.exists():
            (root / "runs" / run).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(public, root / "runs" / run / public.name)
    for src, rel in builder.planned_items(root, include_large=False):
        public = RESULTS / rel
        if public.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(public, src)
    full = root / builder.COMPACT_SOURCE
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(gzip.decompress((RESULTS / builder.COMPACT_REL).read_bytes()))


def build(root: Path, out: Path, checks: Path):
    return subprocess.run([sys.executable, str(BUILDER), "--root", str(root), "--github-out", str(out),
                           "--checks", str(checks)], capture_output=True, text=True)


def leftovers(parent: Path, name: str):
    return [p.name for p in parent.iterdir() if p.name == name or p.name.startswith(f".{name}.partial")]


def expect_stop(label: str, r, parent: Path, name: str, message: str):
    ok = r.returncode != 0 and message in (r.stdout + r.stderr) and not leftovers(parent, name)
    detail = (r.stdout + r.stderr).strip().splitlines()[:1]
    return label, ok, detail


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="pm2026_builder_test_") as t:
        t = Path(t)
        root = t / "root"
        reconstruct_root(root)
        checks = RESULTS / "checks"

        # 1. clean build
        r = build(root, t / "clean", checks)
        same = r.returncode == 0
        if same:
            def tree(d):
                return {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}

            built, committed = tree(t / "clean"), tree(RESULTS)
            same = built == committed and all(
                filecmp.cmp(t / "clean" / f, RESULTS / f, shallow=False)
                for f in built if f != "path_rewrites.csv")
        results.append(("clean build reproduces committed results/pm2026", same,
                        (r.stdout + r.stderr).strip().splitlines()[-1:]))

        # 2. identifier in a retained column of the replicate table
        bad_root = t / "root_id"
        shutil.copytree(root, bad_root)
        full = bad_root / builder.COMPACT_SOURCE
        lines = full.read_text().splitlines()
        cols = lines[1].split(",")
        cols[0] = FAKE_ID  # cell_id, a retained column
        lines[1] = ",".join(cols)
        full.write_text("\n".join(lines) + "\n")
        results.append(expect_stop("identifier in retained replicate column is rejected",
                                   build(bad_root, t / "out_id", checks), t, "out_id",
                                   "participant identifier found"))

        # 3a. unexpected zip (with an identifier inside) in the checks folder
        checks_zip = t / "checks_zip"
        shutil.copytree(checks, checks_zip)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("ids.csv", f"id\n{FAKE_ID}\n")
        (checks_zip / "s7" / "extra.zip").write_bytes(buf.getvalue())
        results.append(expect_stop("unexpected zip in checks folder is rejected",
                                   build(root, t / "out_zip", checks_zip), t, "out_zip",
                                   "unexpected files in the checks folder"))

        # 3b. unexpected plain file in the checks folder
        checks_txt = t / "checks_txt"
        shutil.copytree(checks, checks_txt)
        (checks_txt / "expected" / "notes.txt").write_text("not listed\n")
        results.append(expect_stop("unexpected plain file in checks folder is rejected",
                                   build(root, t / "out_txt", checks_txt), t, "out_txt",
                                   "unexpected files in the checks folder"))

        # 4. identifier inside a Flate-compressed stream of a listed PDF
        checks_pdf = t / "checks_pdf"
        shutil.copytree(checks, checks_pdf)
        pdf = checks_pdf / "s7" / "figure_S7_1.pdf"
        payload = zlib.compress(f"BT ({FAKE_ID}) Tj ET".encode())
        pdf.write_bytes(pdf.read_bytes() + b"\n9999 0 obj\n<< /Filter /FlateDecode /Length "
                        + str(len(payload)).encode() + b" >>\nstream\n" + payload + b"\nendstream\nendobj\n")
        results.append(expect_stop("identifier in compressed PDF stream is rejected",
                                   build(root, t / "out_pdf", checks_pdf), t, "out_pdf",
                                   "participant identifier found"))

        # 5. existing non-empty output folder
        keep = t / "keep"
        keep.mkdir()
        (keep / "existing.txt").write_text("keep me\n")
        r = build(root, keep, checks)
        ok = (r.returncode != 0 and "not empty" in r.stdout + r.stderr
              and [p.name for p in keep.iterdir()] == ["existing.txt"]
              and (keep / "existing.txt").read_text() == "keep me\n"
              and not [p for p in t.iterdir() if p.name.startswith(".keep.partial")])
        results.append(("non-empty output folder refused and preserved", ok,
                        (r.stdout + r.stderr).strip().splitlines()[:1]))

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL'}  {detail[0] if detail else ''}"[:200])
    if not all(ok for _, ok, _ in results):
        sys.exit("release-builder regression check FAILED")
    print(f"all {len(results)} release-builder checks PASS")


if __name__ == "__main__":
    main()
