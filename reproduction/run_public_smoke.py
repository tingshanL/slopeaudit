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
        choices=("s4", "s5", "s6", "s9"),
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
    workflows = args.workflow or ["s4", "s5", "s6", "s9"]
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
