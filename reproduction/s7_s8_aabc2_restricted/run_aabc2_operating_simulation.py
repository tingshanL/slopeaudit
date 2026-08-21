#!/usr/bin/env python3
"""Preflight or smoke-validate the AABC2 operating simulation."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

parser=argparse.ArgumentParser()
sub=parser.add_subparsers(dest="command",required=True)

p=sub.add_parser("preflight")
p.add_argument("--reference-results",required=True)
p.add_argument("--smoke-manifest-out",required=True)

s=sub.add_parser("repair-smoke")
s.add_argument("--input-config",required=True)
s.add_argument("--smoke-manifest",required=True)
s.add_argument("--reference-results",required=True)
s.add_argument("--output-dir",required=True)

args=parser.parse_args()
HERE=Path(__file__).resolve().parent

if args.command=="preflight":
    cmd=[
        sys.executable,str(HERE/"preflight_aabc2_operating_simulation.py"),
        "--reference-results",args.reference_results,
        "--smoke-manifest-out",args.smoke_manifest_out,
    ]
else:
    cmd=[
        sys.executable,str(HERE/"validate_aabc2_operating_simulation_repairs.py"),
        "--input-config",args.input_config,
        "--smoke-manifest",args.smoke_manifest,
        "--reference-results",args.reference_results,
        "--output-dir",args.output_dir,
    ]
raise SystemExit(subprocess.call(cmd))
