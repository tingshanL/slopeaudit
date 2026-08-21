# slopeaudit 0.1.0

**Release state:** pre-tag first-submission release candidate. The entries below
summarize the API-freeze and reader-documentation work accumulated before the
`v0.1.0` tag; earlier development version numbers are provenance only.

## Reader-facing documentation and worked examples

- Replaced the developer-oriented landing page with a complete `sleepstudy`
  quick start, output interpretation, escalation guidance, and common stops.
- Added worked vignettes for the first audit, the irregular Beat the Blues
  example, prospective design sensitivity, and interpretation boundaries.
- Bundled the 100-participant Beat the Blues teaching fixture and three
  copy-paste R scripts.
- Added smoke tests that freeze the example-data schema and ensure every
  installed example remains parseable.
- Kept the four-function v0.1 API and all v14 numerical internals unchanged.

## Public API freeze

- Added the combined `slope_audit()` reader-facing interface.
- Added `test_slope_heterogeneity()` for the validated fixed-usable
  parametric bootstrap with deterministic paired optimizer rescue.
- Added `slope_decomposition()` for the validated `lmerMod` decomposition.
- Added `slope_design_sensitivity()` for prospective matrix-based
  information-source calculations.
- Enforced the six-field target-estimand contract at every public
  decomposition boundary.
- Added stable result classes, compact print methods, and a summary method.
- Added public regression tests for API signatures, frozen values, additivity,
  affine invariance, exact-zero boundaries, nuisance-space enforcement,
  plus-one arithmetic, and fixed usable bootstrap denominators.

The v14 numerical and bootstrap internals are unchanged. `slope_stability()`
remains deferred until its resampling contract is independently validated.
