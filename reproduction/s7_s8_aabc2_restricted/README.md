# Restricted AABC2 reproduction workflow

This directory contains code and input contracts for the restricted AABC2
analyses. Participant-level inputs are not redistributed. Qualified researchers
can supply authorized files through the scripts' explicit configuration and
input arguments.

## Entry points

The workflow is organized by scientific purpose:

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

## Publicly available results

The [AABC2 output manifest](../../results/frozen_summaries/aabc2/aabc2_aggregate_manifest.csv)
lists the aggregate results and validation summaries included in this repository.
Qualified researchers can request access to the participant-level data through
[AABC Release 2 on BALSA](https://balsa.wustl.edu/project?project=AABC2).

## Reproduction requirements

To rerun the AABC2 analyses, pass `--input-config` with paths to authorized
source files. See [the input schema](aabc2_input_schema.csv) and
[the configuration loader](aabc2_common.py).
