#!/usr/bin/env python3
"""Run targeted AABC2 profile, influence, or adequacy validation.

The validated computational core and aggregate outputs represent the completed
analyses. This entry point reruns diagnostic smoke subsets without repeating
the full historical computation.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--component",choices=("profiles","influence","adequacy"),required=True)
parser.add_argument("--input-config",required=True)
parser.add_argument("--smoke-manifest",required=True)
parser.add_argument("--reference-results",required=True)
parser.add_argument("--output-dir",required=True)
args=parser.parse_args()

HERE=Path(__file__).resolve().parent
script={
    "profiles":"validate_aabc2_advanced_profiles.py",
    "influence":"validate_aabc2_advanced_influence.py",
    "adequacy":"validate_aabc2_advanced_adequacy.py",
}[args.component]

cmd=[
    sys.executable,
    str(HERE/script),
    "--input-config",args.input_config,
    "--smoke-manifest",args.smoke_manifest,
    "--reference-results",args.reference_results,
    "--output-dir",args.output_dir,
]
raise SystemExit(subprocess.call(cmd))
