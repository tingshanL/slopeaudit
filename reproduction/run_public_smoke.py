#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "reproduction" / "smoke_inputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run public/synthetic software smoke checks; outputs are not "
            "suitable for scientific reporting."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--workflow",
        action="append",
        choices=("s4", "s5", "s6", "s9", "pm2026"),
        help="Restrict to one or more workflows; default runs all public smoke workflows.",
    )
    return parser.parse_args()


def run(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Expected smoke output missing: {path}")


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            "--output-dir must be absent or empty so stale checkpoints/results "
            "cannot satisfy the smoke check."
        )
    output.mkdir(parents=True, exist_ok=True)
    workflows = args.workflow or ["s4", "s5", "s6", "s9", "pm2026"]
    if "pm2026" in workflows:
        # The pm2026 checks refit data sets from the S4 and S5 synthetic smoke outputs.
        workflows = list(dict.fromkeys(["s4", "s5", *workflows]))
    py = sys.executable
    completed = []

    if "s4" in workflows:
        gen = output / "s4" / "generator"
        summary = output / "s4" / "summary"
        run(
            "S4 synthetic generation",
            [
                py,
                "reproduction/s4_operating_study/generate_operating_study.py",
                "--design-manifest",
                "reproduction/smoke_inputs/s4_synthetic_design.csv",
                "--output-dir",
                str(gen),
                "--mode",
                "smoke",
                "--n-jobs",
                "1",
                "--smoke-reps",
                "2",
            ],
        )
        require(gen / "operating_study_replicate_results.csv")
        run(
            "S4 deterministic summarization",
            [
                py,
                "reproduction/s4_operating_study/summarize_operating_study.py",
                "--replicates-csv",
                str(gen / "operating_study_replicate_results.csv"),
                "--manifest-csv",
                str(gen / "operating_study_design.csv"),
                "--output-dir",
                str(summary),
                "--mode",
                "smoke",
            ],
        )
        require(summary / "operating_characteristics.csv")
        completed.append("s4")

    if "s5" in workflows:
        manifest = SMOKE / "s5_synthetic" / "synthetic_anchor_manifest.json"
        gen = output / "s5" / "generator"
        summary = output / "s5" / "summary"
        run(
            "S5 synthetic generation",
            [
                py,
                "reproduction/s5_downstream_simulation/generate_downstream_simulation.py",
                "--anchor-manifest",
                str(manifest),
                "--output-dir",
                str(gen),
                "--mode",
                "smoke",
                "--workers",
                "1",
                "--reps-per-cell",
                "1",
            ],
        )
        require(gen / "task_manifest.csv")
        run(
            "S5 deterministic summarization",
            [
                py,
                "reproduction/s5_downstream_simulation/summarize_downstream_simulation.py",
                "--anchor-manifest",
                str(manifest),
                "--task-manifest",
                str(gen / "task_manifest.csv"),
                "--simulation-replicates-raw",
                str(gen / "simulation_replicate_results_raw.csv"),
                "--model-fit-diagnostics",
                str(gen / "model_fit_diagnostics.csv"),
                "--optimizer-attempts",
                str(gen / "optimizer_attempts.csv"),
                "--output-dir",
                str(summary),
                "--mode",
                "smoke",
            ],
        )
        require(summary / "07_simulation_method_summary.csv")
        completed.append("s5")

    if "s6" in workflows:
        btb_primary = output / "s6" / "btheb_primary"
        btb_logtime = output / "s6" / "btheb_logtime"
        sleep_primary = output / "s6" / "sleepstudy_primary"
        sleep_ar1 = output / "s6" / "sleepstudy_ar1"
        run(
            "S6 Beat the Blues primary synthetic smoke",
            [
                py,
                "reproduction/s6_public_audits/beat_the_blues_primary.py",
                "--output-dir",
                str(btb_primary),
                "--smoke-test",
            ],
        )
        require(btb_primary / "32_repaired_validity_gate.csv")
        run(
            "S6 Beat the Blues log-time synthetic smoke",
            [
                py,
                "reproduction/s6_public_audits/beat_the_blues_logtime.py",
                "--output-dir",
                str(btb_logtime),
                "--primary-results-dir",
                str(btb_primary),
                "--smoke-test",
            ],
        )
        require(btb_logtime / "08_time_estimand_adjudication.csv")
        run(
            "S6 sleepstudy primary synthetic smoke",
            [
                py,
                "reproduction/s6_public_audits/sleepstudy_primary.py",
                "--output-dir",
                str(sleep_primary),
                "--smoke-test",
            ],
        )
        require(sleep_primary / "28_completion_checklist.csv")
        run(
            "S6 sleepstudy AR1 synthetic smoke",
            [
                py,
                "reproduction/s6_public_audits/sleepstudy_ar1_profile.py",
                "--output-dir",
                str(sleep_ar1),
                "--smoke-test",
            ],
        )
        require(sleep_ar1 / "07_profile_adjudication.json")
        completed.append("s6")

    if "s9" in workflows:
        s9 = output / "s9"
        run(
            "S9 deterministic planning",
            [
                py,
                "reproduction/s9_validation_planning/make_planning_table.py",
                "--output-dir",
                str(s9),
            ],
        )
        require(s9 / "practical_design_contract.json")
        completed.append("s9")

    if "pm2026" in workflows:
        pm = output / "pm2026"
        P = "reproduction/pm2026"
        reml = f"{P}/s3_downstream/reml_tools.py"
        pm.mkdir(parents=True, exist_ok=True)
        run("pm2026 S2 identities", [py, f"{P}/s2_identities/s2_numerical_checks.py", str(pm / "s2.json")])
        run("pm2026 S7 design sensitivity", [py, f"{P}/s7_design/make_S7.py",
                                             "--aabc2", "results/pm2026/empirical/s7_aabc2_aggregate",
                                             "--out-dir", str(pm / "s7")])
        run("pm2026 downstream simulation fixtures",
            [py, f"{P}/s3_downstream/downstream_simulation.py", "--stage", "fixtures"])
        run("pm2026 downstream simulation synthetic smoke fit",
            [py, f"{P}/s3_downstream/downstream_simulation.py", "--stage", "common", "--smoke",
             "--constructions", "B", "--c-values", "0.5", "--output-dir", str(pm / "downstream")])
        require(pm / "downstream" / "common_D2_0.80_likelihood_selection_smoke" / "summary_by_method.csv")

        s4_out = output / "s4" / "generator"
        keys = pm / "s4_keys.csv"
        keys.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(s4_out / "operating_study_replicate_results.csv", newline="") as f:
            first = next(csv.DictReader(f))
        keys.write_text(f"cell_id,replicate\n{first['cell_id']},{first['replicate']}\n", encoding="utf-8")
        A = f"{P}/s4_operating_audit"
        run("pm2026 S4 refit (relocated audit path)",
            [py, f"{A}/s4_step4_startpoint_refit.py", "--repo-root", ".", "--reml-tools", reml,
             "--s4-dir", str(s4_out), "--extra-keys", str(keys), "--workers", "1",
             "--out-dir", str(pm / "s4_step4")])
        run("pm2026 S4 random-intercept screen (relocated audit path)",
            [py, f"{A}/s4_step5_null_screen.py", "--repo-root", ".", "--reml-tools", reml,
             "--step4-script", f"{A}/s4_step4_startpoint_refit.py", "--s4-dir", str(s4_out),
             "--replicates-csv", str(pm / "s4_step4" / "operating_study_replicate_results_startpoint_patched.csv"),
             "--step4-audit", str(pm / "s4_step4" / "startpoint_refit_audit.csv"),
             "--families", first["cell_family"], "--schedules", first["schedule"], "--workers", "1",
             "--out-dir", str(pm / "s4_step5")])
        require(pm / "s4_step5" / "step5_screen.csv")

        run("pm2026 attenuation audit with its earlier helper (provenance snapshot)",
            [py, f"{P}/provenance/attenuation_audit/reproduction/s5_downstream_simulation/audit_s5_fit_selection.py",
             "--s5-output", str(output / "s5" / "generator"),
             "--anchor-manifest", "reproduction/smoke_inputs/s5_synthetic/synthetic_anchor_manifest.json",
             "--part", "AB", "--per-stratum", "1", "--include-flagged", "0", "--workers", "1",
             "--output-dir", str(pm / "attenuation_audit")])
        with open(pm / "attenuation_audit" / "partB_candidates.csv", newline="") as f:
            sources = {row["source"] for row in csv.DictReader(f)}
        if "random_intercept_embedded" in sources:
            raise RuntimeError("attenuation audit did not load the earlier likelihood helper")

        run("pm2026 reported values from results/pm2026",
            [py, f"{P}/release/check_reported_values.py", "--out-dir", str(pm / "reported_values")])
        completed.append("pm2026")

    manifest = {
        "schema_version": 1,
        "purpose": "public/synthetic software smoke check",
        "completed_workflows": completed,
        "restricted_s7_s8_execution": "not attempted; authorized AABC2 inputs required",
        "reportable": False,
    }
    (output / "public_smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("\nPUBLIC REPRODUCTION SMOKE: PASS")


if __name__ == "__main__":
    main()
