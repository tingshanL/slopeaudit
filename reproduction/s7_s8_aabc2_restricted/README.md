# Restricted AABC2 reproduction workflow

This directory contains the reviewer-facing code and input contracts for the restricted AABC2 analyses. The participant-level AABC2 inputs are not redistributed. Qualified researchers with authorized access can supply the required files through the explicit configuration/input arguments used by these scripts.

## Entry points

The workflow is intentionally split by scientific purpose rather than by historical notebook order:

- `prepare_aabc2_analysis_data.py` reconstructs the six analysis cohorts in memory and emits only aggregate cohort checks.
- `fit_aabc2_outcomes.py` reproduces the observed six-outcome model panels.
- `compute_aabc2_reliability.py` computes the exact participant-level source quantities privately and aggregate reliability summaries separately.
- `run_aabc2_bootstraps.py` reproduces the 499-resample reliability-gap bootstrap for cortical thickness and fluid cognition.
- `run_aabc2_stability_diagnostics.py` orchestrates the covariance-geometry, quadratic-time, and reference-time diagnostic workflows.
- `select_aabc2_advanced_geometry.py`, `run_aabc2_advanced_rlrt.py`, and `summarize_aabc2_advanced_rlrt.py` reproduce the advanced boundary-aware Question 1 workflow.
- `run_aabc2_profiles_influence_adequacy.py` exposes the validated likelihood-profile, influence, and adequacy analysis routes.
- `run_aabc2_operating_simulation.py` and `preflight_aabc2_operating_simulation.py` document the large operating-simulation generator and its zero-refit validation path.
- `make_aabc2_aggregate_exports.py` is the final semantic output router.

Underscore-prefixed modules are implementation modules used by these entry points rather than independent workflows.

## First-submission data boundary

The first-submission default is conservative. Analysis code, schemas, variable definitions, eligibility rules, and independently synthetic examples can be public. AABC2-derived result files are not copied by the release router unless written derivative-export approval has been recorded. Participant-linked source quantities, deletion refits, and raw trajectory/QC products are classified as `restricted_never_export` and are never copied by the router.

`source_anchor_schema.json` is a schema only; it deliberately contains no AABC2-derived anchor values.

The computational scripts retain some frozen internal filenames so that numerical equivalence to the completed analyses can be checked exactly. Reviewer-facing release filenames are semantic and are produced only through `make_aabc2_aggregate_exports.py`; processing-order prefixes are not part of the release names.

## Safe default export check

Running `make_aabc2_aggregate_exports.py --mode manifest ...` creates only `aabc2_export_manifest.csv`. The manifest contains group labels, relative source filenames, sizes, hashes, release classifications, and semantic release names. It does not contain absolute local paths and does not copy any AABC2-derived result file.

## Independent-runnability boundary

The S7/S8 scientific analyses are **restricted-input workflows**, not public
turnkey analyses. A fresh public checkout can inspect the schemas, validate the
explicit input contract, and exercise the safe manifest-only export route, but
it cannot reproduce the AABC2 model fits without authorized AABC2 source files.
The top-level public smoke runner therefore does not substitute restricted-
derived fixtures or fabricated participant data for these analyses. Exact reruns
must use `--input-config` pointing to authorized source files satisfying
`aabc2_input_schema.csv` and the configuration contract in `aabc2_common.py`.
