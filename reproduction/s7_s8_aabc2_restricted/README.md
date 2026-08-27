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

Participant-level AABC2 inputs are not redistributed. The repository includes
29 reviewed, identifier-free aggregate and validation artifacts listed in
`results/frozen_summaries/aabc2/aabc2_aggregate_manifest.csv`.

These research-result summaries exclude participant identifiers,
participant-linked source quantities, deletion refits, individual trajectories,
exact participant schedules, and raw trajectory/QC products. The influence
summary omits participant-ID and rank fields, and the quadratic-selection audit
omits a local source-path cell; numerical result fields are unchanged.

The automated source router governs copying from an authorized analysis
environment and copies nothing in manifest mode. Public availability is
determined by the reviewed aggregate manifest and privacy review.

This boundary reflects the current public AABC Data Use Terms, which
distinguish research results from redistribution of the underlying Data and
prohibit public association of participant IDs with individual data elements
without approval:

https://www.humanconnectome.org/study/hcp-lifespan-aging/data-use-terms

Researchers remain responsible for any additional requirements in their
institution's applicable agreement. This repository does not authorize
redistribution of AABC2 participant-level data.

## Reproduction requirements

The S7/S8 scientific analyses require restricted inputs. A public checkout can
inspect the schemas, validate the input contract, and exercise the manifest-only
export route, but reproducing the AABC2 model fits requires authorized source
files. Exact reruns use `--input-config` pointing to files satisfying
`aabc2_input_schema.csv` and the configuration contract in `aabc2_common.py`.
