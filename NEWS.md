# slopeaudit 0.2.0

## Nuisance share

- `slope_decomposition()`, `slope_design_sensitivity()` and `slope_audit()` now
  report the nuisance share `kappa` for each participant: the share of
  fitted-slope variance attributable to the other random effects after removing
  their slope-predictable part. `sqrt_kappa` is the largest attainable absolute
  correlation between the fitted slope and a participant-level variable
  uncorrelated with the target slope and the residuals.
- `kappa` is computed from the fitted matrices, so it applies to structured
  residual covariances and to models with several nuisance effects. It is zero
  when the model has no nuisance effects and `NA` (status
  `"undefined_zero_determinacy"`) when fitted-slope determinacy is zero.
  Rounding-level negatives are set to zero; material negatives, violations of the
  identity `D_slope2 + kappa + residual share = 1`, and values above the ceilings
  `1 - D_slope2` and `Delta_fit / D_slope2` mark the participant as numerically
  invalid.
- A new `capacity_summary` reports participant-level percentiles of `kappa` and
  `sqrt_kappa`. These are participant-level values at the fitted parameters,
  not bounds for a pooled regression over participants with different
  schedules. `source_summary` is unchanged.

## Reproduction materials

- Added `reproduction/pm2026/` with the code and documentation for the 2026
  manuscript, including exact snapshots of the scripts that produced its
  results (`reproduction/pm2026/provenance/`). The v0.1.0 reproduction folders
  are unchanged.

# slopeaudit 0.1.0

## Reader-facing documentation and worked examples

- Replaced the developer-oriented landing page with a complete `sleepstudy`
  quick start, output interpretation, escalation guidance, and interpretation
  requirements.
- Added worked vignettes for the first audit, the irregular Beat the Blues
  example, prospective design sensitivity, and interpretation boundaries.
- Bundled the 100-participant Beat the Blues teaching fixture and three
  copy-paste R scripts.
- Added smoke tests that verify the example-data schema and ensure every
  installed example remains parseable.
- Kept the four-function v0.1 API and all v14 numerical internals unchanged.

## Public API

- Added the combined `slope_audit()` reader-facing interface.
- Added `test_slope_heterogeneity()` for the validated fixed-usable
  parametric bootstrap with deterministic paired optimizer rescue.
- Added `slope_decomposition()` for the validated `lmerMod` decomposition.
- Added `slope_design_sensitivity()` for prospective matrix-based
  information-source calculations.
- Enforced the six-field target-estimand contract at every public
  decomposition boundary.
- Added stable result classes, compact print methods, and a summary method.
- Added public regression tests for API signatures, reference values, additivity,
  affine invariance, exact-zero boundaries, nuisance-space enforcement,
  plus-one arithmetic, and fixed usable bootstrap denominators.

`slope_stability()` is not included in version 0.1; participant-resampling
stability analyses remain available in the documented reproduction workflows.
