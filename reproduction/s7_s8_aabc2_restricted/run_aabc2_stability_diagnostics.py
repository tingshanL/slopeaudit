#!/usr/bin/env python3
"""Run the validated AABC2 stability-diagnostic pipeline.

This entry point orchestrates three previously validated restricted-data
workflows in dependency order:

1. gap-regime optimizer/covariance diagnostics;
2. quadratic-time-form sensitivity diagnostics;
3. reference-time and covariance-geometry diagnostics.

Participant-linked outputs remain under each component's ``restricted/``
subdirectory. Checkpoints remain under each component's ``checkpoints/``
subdirectory and are not distributed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--input-config", required=True)
parser.add_argument(
    "--bootstrap-metrics",
    required=True,
    help="Full 09_reliability_gap_bootstrap_replicates.csv.",
)
parser.add_argument(
    "--exact-reliability",
    required=True,
    help="Restricted 04_exact_posterior_slope_reliability.csv.",
)
parser.add_argument(
    "--simulation-geometry",
    required=True,
    help="Final simulation 04_model_fit_covariance_diagnostics.csv.",
)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--n-jobs", type=int, default=2)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

ROOT = Path(args.output_dir).expanduser().resolve()
if ROOT.exists() and any(ROOT.iterdir()) and not args.resume:
    raise FileExistsError(
        f"Output directory is non-empty: {ROOT}. Use a fresh directory or --resume."
    )
ROOT.mkdir(parents=True, exist_ok=True)

INPUT_CONFIG = str(Path(args.input_config).expanduser().resolve())
BOOTSTRAP = str(Path(args.bootstrap_metrics).expanduser().resolve())
EXACT = str(Path(args.exact_reliability).expanduser().resolve())
SIM_GEOMETRY = str(Path(args.simulation_geometry).expanduser().resolve())
N_JOBS = max(1, min(int(args.n_jobs), 3))

for required in (INPUT_CONFIG, BOOTSTRAP, EXACT, SIM_GEOMETRY):
    if not Path(required).exists():
        raise FileNotFoundError(required)


def run(script_name: str, argv: list[str]) -> None:
    script = HERE / script_name
    if not script.is_file():
        raise FileNotFoundError(script)
    cmd = [sys.executable, str(script), *argv]
    print()
    print("=" * 80)
    print("RUNNING:", script_name)
    print("=" * 80)
    subprocess.run(cmd, check=True)


def maybe_resume(argv: list[str]) -> list[str]:
    return [*argv, "--resume"] if args.resume else argv


gap_root = ROOT / "gap_optimizer"
run(
    "_aabc2_gap_optimizer_diagnostic.py",
    maybe_resume([
        "--input-config", INPUT_CONFIG,
        "--bootstrap-metrics", BOOTSTRAP,
        "--output-dir", str(gap_root),
        "--n-jobs", str(N_JOBS),
    ]),
)

selection_manifest = gap_root / "public" / "01_selected_gap_replicates.csv"
selected_participants = gap_root / "restricted" / "03_selected_participant_reliability.csv"
representative_fits = gap_root / "public" / "04_replicate_optimizer_agreement.csv"

for required in (selection_manifest, selected_participants, representative_fits):
    if not required.is_file():
        raise RuntimeError(f"Gap diagnostic did not produce required dependency: {required}")

quadratic_root = ROOT / "quadratic_time"
run(
    "_aabc2_quadratic_gap_audit.py",
    maybe_resume([
        "--input-config", INPUT_CONFIG,
        "--selected-manifest", str(selection_manifest),
        "--output-dir", str(quadratic_root),
        "--n-jobs", str(N_JOBS),
    ]),
)

reference_root = ROOT / "reference_time"
run(
    "_aabc2_reference_time_diagnostics.py",
    maybe_resume([
        "--input-config", INPUT_CONFIG,
        "--exact-reliability", EXACT,
        "--gap-bootstrap", BOOTSTRAP,
        "--selected-participants", str(selected_participants),
        "--representative-fits", str(representative_fits),
        "--simulation-geometry", SIM_GEOMETRY,
        "--output-dir", str(reference_root),
        "--n-jobs", str(N_JOBS),
    ]),
)

print()
print("AABC2 STABILITY DIAGNOSTICS COMPLETE")
print("gap optimizer:", gap_root)
print("quadratic-time sensitivity:", quadratic_root)
print("reference-time diagnostics:", reference_root)
