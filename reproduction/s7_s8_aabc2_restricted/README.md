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

Participant-level AABC2 inputs are not redistributed. The first-submission
release includes only the 29 separately reviewed, ID-free aggregate and
validation artifacts listed in
`results/frozen_summaries/aabc2/aabc2_aggregate_manifest.csv`.

These files are research-result summaries rather than copies of participant-
level AABC2 data or alternatively preprocessed participant data. The public
candidate excludes participant identifiers, participant-linked source
quantities, deletion refits, individual trajectories, exact participant
schedules, and raw trajectory/QC products. The influence summary removes the
source participant-ID and rank fields, and the quadratic-selection audit
redacts one local source-path cell; the numerical result fields were not
altered.

The automated source router remains deliberately conservative. Its source
policy governs direct copying from the authorized analysis environment and
does not itself designate an artifact as safe for public release. In manifest
mode it copies nothing. The separately reviewed 29-file public set is governed
by its release manifest and privacy review rather than by automatic copying
from the source directories.

This boundary reflects the current public AABC Data Use Terms, which
distinguish research results from redistribution of the underlying Data and
prohibit public association of participant IDs with individual data elements
without approval:

https://www.humanconnectome.org/study/hcp-lifespan-aging/data-use-terms

Researchers remain responsible for any additional requirements in their
institution's applicable agreement. This release does not authorize
redistribution of AABC2 participant-level data.

## Independent-runnability boundary

The S7/S8 scientific analyses are **restricted-input workflows**, not public
turnkey analyses. A fresh public checkout can inspect the schemas, validate the
explicit input contract, and exercise the safe manifest-only export route, but
it cannot reproduce the AABC2 model fits without authorized AABC2 source files.
The top-level public smoke runner therefore does not substitute restricted-
derived fixtures or fabricated participant data for these analyses. Exact reruns
must use `--input-config` pointing to authorized source files satisfying
`aabc2_input_schema.csv` and the configuration contract in `aabc2_common.py`.
