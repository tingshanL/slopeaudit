# Validation reference fixtures

This directory contains compact reference cases used to validate the
statistical implementation of `slopeaudit`.

These files are validation references rather than package examples or
manuscript-analysis outputs. The R matrix backend is checked against the
governing statistical contract and independent reference calculations.

## Governing specification

Statistical behavior is governed by **Statistical Contract v1.1**. If this directory conflicts with the contract, the contract controls and the fixture must be revised explicitly.

The principal dimensionless identity tolerance is:

- `sqrt(.Machine$double.eps)` ≈ `1.4901161193847656e-08`.

Fitted-model comparisons may later use explicitly documented, test-specific tolerances when optimizer or platform variation makes the contract-level identity tolerance inappropriate.

## Fixture classes

Fixtures fall into three classes:

1. **Analytic / constructed** — expected quantities follow from algebra or a hand-constructed exact case and do not depend on the Python implementation.
2. **Synthetic numerical oracle** — a small hand-specified matrix case whose expected values come from an independent calculation separate from the production backend.
3. **Python-derived public-data reference** — a minimal input/output snapshot from the existing independent Python analyses, used only for cross-implementation validation.

Restricted AABC2 participant-level data must not be stored here.

## Reference fixtures

The reference set includes:

- `known_802060/`: constructed identity with `D_slope2 = 0.80`, `R_direct = 0.20`, and `Delta_fit = 0.60`;
- `irregular_hand_oracle/`: small hand-specified irregular schedule;
- `zero_direct_single_observation/`: participant with one observation;
- `zero_direct_no_time_variation/`: participant with repeated observations but no within-person variation in the target time variable;
- `near_singular_synthetic/`: positive target slope variance with a near-singular random-effects covariance structure;
- `sleepstudy_primary/`: balanced public-data reference from the independent Python analysis;
- `btheb_primary/`: irregular/missing-schedule public-data reference from the independent Python analysis.

The `sleepstudy_ar1` fixture is a second-round check after the independent-residual matrix backend passes.

## File conventions

Constructed fixtures should contain:

- `specification.json`: all matrices, target index, estimand description, and construction notes needed to reproduce the calculation;
- `expected.json`: expected intermediate and final quantities, plus tolerance metadata.

Python-derived public-data fixtures should contain only the minimal information required to reproduce the validation comparison, typically:

- `analysis_data.csv`;
- `fitted_parameters.json`;
- `expected_participant_sources.csv`;
- `expected_summary.json`.

Only the compact files required for each validation comparison belong in this
directory.

## Provenance rule

Every fixture must have one row in `fixture_manifest.csv` containing its source type, origin, purpose, status, and expected files. For Python-derived fixtures, record the exact historical notebook/result files from which the frozen reference was distilled.

## Immutability rule

Once a fixture is marked `frozen`, do not overwrite expected values merely because a new implementation disagrees. A disagreement is investigated against:

1. analytic ground truth where available;
2. Statistical Contract v1.1 and the independently specified matrix algebra;
3. the independent Python implementation;
4. additional high-precision or hand-constructed checks if needed.

If the contract itself changes, create a documented fixture revision rather than silently replacing the old oracle.
## Third-party public-data provenance and licensing

The copied/distilled observation-data inputs in the public-data fixtures retain
their upstream provenance and are not relicensed by `slopeaudit`'s MIT license:

- `btheb_primary/analysis_data.csv` is derived from `HSAUR3::BtheB`. HSAUR3 is
  distributed under GPL-2; the dataset documentation cites Proudfoot et al.
  (2003), *Psychological Medicine*, 33(2), 217--227.
- `sleepstudy_primary/analysis_data.csv` is derived from `lme4::sleepstudy`.
  `lme4` is distributed under GPL (>= 2); the dataset documentation cites
  Belenky et al. (2003), *Journal of Sleep Research*, 12(1), 1--12.

The fitted-parameter and expected-output files in these fixture directories are
project validation artifacts; the upstream-license notice here specifically
covers the copied/distilled observation-data inputs.

Exact source-tree paths and the complete applicable GPL license texts are
centralized in `inst/THIRD_PARTY_NOTICES.md` and `inst/licenses/`.
