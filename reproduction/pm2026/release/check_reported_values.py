#!/usr/bin/env python3
"""Regenerate the reported tables and summary numbers from the public files in results/pm2026
and compare them with the stored expected outputs (results/pm2026/checks/expected), which were
checked against the manuscript and supplement. No model fitting.

Covers: Supplement S3 tables and main-text Table 1 (with Figure 2), Supplement S4 tables and
Figure 3, the S5.4 attenuation-audit numbers, Table S7.1 with Figure S7.1, and Figure 4.
Usage (from the repository root):
  python3 reproduction/pm2026/release/check_reported_values.py [--out-dir <scratch dir>]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expected_outputs import EXPECTED_FILES, N_REPORTED_GROUPS, S3_TABLES, S4_TABLES  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "reproduction/pm2026"
D = ROOT / "results/pm2026"
E = D / "checks/expected"


def run(args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(map(str, args))}\n{r.stderr[-2000:]}")
    return r.stdout


def same_rows(new: Path, old: Path):
    if not new.is_file():
        return False, 0
    a = [x for x in new.read_text().splitlines() if x.strip()]
    b = [x for x in old.read_text().splitlines() if x.strip()]
    return a == b, len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    missing = [str(E / f) for f in EXPECTED_FILES if not (E / f).is_file()]
    if missing:
        sys.exit("reported-value check FAILED: expected files are missing:\n  " + "\n  ".join(missing))
    out = a.out_dir or Path(tempfile.mkdtemp(prefix="pm2026_check_"))
    runs = {k: D / "downstream" / k for k in ("common_80", "common_50", "irregular_80", "irregular_50")}
    run_args = [x for k, v in runs.items() for x in ("--run", f"{k}={v}")]
    dose, n600 = D / "downstream/dose_A_50", D / "downstream/n600_A_50"
    summary = D / "operating_audit/s4_step6_consolidated_refit/summary"
    compact = summary / "operating_study_replicate_results_compact.csv.gz"
    results = []

    run([P / "s3_downstream/make_S3_tables.py", "--out-dir", out / "s3", *run_args, "--dose", dose, "--n600", n600])
    for name in S3_TABLES:
        results.append((f"S3 {Path(name).stem}", *same_rows(out / "s3" / name, E / "s3_tables" / name)))
    run([P / "s3_downstream/make_figure_02.py", "--out-dir", out / "fig2", *run_args, "--dose", dose, "--n600", n600])
    results.append(("Table 1", *same_rows(out / "fig2/s5_table_main_rows.tex", E / "table1_rows.tex")))

    run([P / "s4_operating_audit/make_S4_tables.py", "--summary", summary, "--replicates", compact,
         "--step7", D / "operating_audit/s4_step7_reporting",
         "--design", D / "operating_audit/operating_study_design.csv", "--out-dir", out / "s4"])
    for name in S4_TABLES:
        results.append((f"S4 {Path(name).stem}", *same_rows(out / "s4" / name, E / "s4_tables" / name)))
    run([P / "s4_operating_audit/make_figure_03.py", "--operating-csv", compact, "--out-dir", out / "fig3"])
    results.append(("Figure 3 (count assertions)", True, 4))

    got = json.loads(run([P / "s5_attenuation_audit/s5_partB_numbers.py", D / "attenuation_audit/partB_by_task.csv"]))
    want = json.loads((E / "s5_attenuation_audit_numbers.json").read_text())
    results.append(("S5.4 audit numbers", got == want, len(want)))

    run([P / "s7_design/make_S7.py", "--aabc2", D / "empirical/s7_aabc2_aggregate", "--out-dir", out / "s7"])
    results.append(("Table S7.1", *same_rows(out / "s7/S7_schedules_rows.tex", E / "s7_schedules_rows.tex")))
    run([P / "s6_empirical/make_figure_04.py", "--sleep", D / "empirical/s7_sleepstudy",
         "--aabc2", D / "empirical/s7_aabc2_aggregate", "--out-dir", out / "fig4"])
    results.append(("Figure 4 (consistency assertions)", True, 1))

    width = max(len(r[0]) for r in results)
    for name, ok, n in results:
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL'}  ({n} rows/items)")
    print(f"outputs in {out}")
    if len(results) != N_REPORTED_GROUPS:
        sys.exit(f"reported-value check FAILED: {len(results)} groups evaluated, {N_REPORTED_GROUPS} expected")
    if not all(ok for _, ok, _ in results):
        sys.exit("reported-value check FAILED")
    print(f"all {N_REPORTED_GROUPS} reported-value groups PASS")


if __name__ == "__main__":
    main()
