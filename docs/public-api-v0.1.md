# Public API contract v0.1

**Status:** frozen for the pre-tag `0.1.0` release candidate\
**Numerical authority:** Statistical Contract v1.1 and the v14-validated
internal engines\
**Change rule:** additions are allowed only after validation; changing a
documented argument, default, output field, or interpretation requires an
explicit API-version decision and regression-test update.

## Exported functions

```r
slope_audit(
  full, null, target, target_metadata,
  B = 999L, seed = 1L, conf_level = 0.95, alpha = 0.05,
  attempt_cap = B + max(50L, ceiling(0.25 * B)),
  retain_replicates = FALSE,
  progress = interactive() && B >= 50L
)

test_slope_heterogeneity(
  full, null, target,
  B = 999L, seed = 1L, conf_level = 0.95, alpha = 0.05,
  attempt_cap = B + max(50L, ceiling(0.25 * B)),
  retain_replicates = FALSE,
  progress = interactive() && B >= 50L
)

slope_decomposition(full, target, target_metadata)

slope_design_sensitivity(
  Z, G, R, target, target_metadata,
  H = NULL, design_id = NULL
)
```

No Python public implementation is part of v0.1.

## Required target metadata

User-facing decomposition requires all six non-empty fields:

1. `basis_type`;
2. `outcome_units`;
3. `time_or_exposure_scale`;
4. `origin_or_reference`;
5. `reference_window`;
6. `estimand`.

For `basis_type` equal to `nonlinear`, `piecewise`, `interaction`, or `other`,
`basis_expression` is additionally required. The public API never guesses or
silently fills these substantive fields. A two-field shortcut is not part of
the frozen contract.

## Question-1 behavior

- Both `full` and `null` are supplied as already fitted models.
- Both must be compatible Gaussian `lmerMod` objects fitted by REML.
- They must use the same retained observations, fixed-effect design, response,
  grouping structure, nuisance random effects, weights, and effective
  optimizer controls.
- The null must remove the target random slope and all covariance parameters
  that disappear with it.
- The supplied observed fits are inspected but never silently reconstructed or
  refitted.
- Bootstrap refits use the v12/v14 validated deterministic hierarchy: paired
  `nloptwrap`, followed when necessary by paired `bobyqa` rescue.
- The primary denominator contains exactly `B` positive/zero LR values. An
  unresolved rescued attempt is retained and deterministically topped up,
  subject to `attempt_cap`.
- The primary p-value is `(k + 1) / (B + 1)` with rejection rule
  `p <= alpha`.
- Worst-case bounds treat excluded attempts as all nonexceeding or all
  exceeding. `decision` is `supported`, `unsupported`,
  `numerically_ambiguous`, or `indeterminate`.

The default `B = 999` is part of the public freeze. Smaller values are useful
for tests and exploratory runs but should not replace the validated manuscript
setting when Monte Carlo resolution matters.

## Returned classes and stable fields

### `slopeaudit_heterogeneity`

Stable reader-facing fields:

- `status`, `decision`, and `target`;
- `observed`;
- `bootstrap`;
- `decision_bounds`;
- `usable_target`, `attempt_cap`, `attempts_evaluated`, `target_reached`;
- `alpha`, `master_seed`, and `execution_mode`.

The complete rescue and replicate provenance remains available when
`retain_replicates = TRUE`.

### `slopeaudit_decomposition`

Stable reader-facing fields:

- `status`;
- `participant_table`;
- `summary`;
- `source_summary` with rows `R_direct`, `D_slope2`, and `Delta_fit` and columns
  `mean`, `median`, `IQR`, and `minimum`;
- `diagnostics`, `metadata`, and normalized `target_metadata`.

### `slopeaudit`

Stable reader-facing fields:

- `status`, `target`, and normalized `target_metadata`;
- `heterogeneity`;
- `decomposition` and `decomposition_status`;
- `participant_table` and `source_summary` convenience fields;
- `interpretation`.

An unsupported Question 1 does not suppress a strictly-positive-variance
decomposition. That decomposition is labelled a conditional description of
the fitted random-slope model. At exact zero fitted target variance,
decomposition quantities are withheld.

### `slopeaudit_design_sensitivity`

Stable reader-facing fields:

- `status`, `target`, `target_index`, and normalized `target_metadata`;
- `design_table`;
- `summary` and `source_summary`;
- `diagnostics` and `matrix_backend`;
- `scope`.

This function evaluates prospective information-source quantities. It does not
claim arbitrary-design Question-1 power.

## Deliberately deferred interface

`slope_stability()` is not exported in v0.1. The current package contains no
validated public resampling contract for participant/bootstrap stability.
Freezing a placeholder would be more damaging than adding the function later
as a validated, additive API extension.

## Final regression gates

The public-interface tests freeze:

- export names and formal argument order;
- strict six-field estimand identification;
- frozen `sleepstudy` decomposition values;
- `.80 = .20 + .60` through the public matrix route;
- aggregate additivity;
- affine-time invariance;
- REML scope and full/null validation;
- fixed usable bootstrap denominator and plus-one arithmetic;
- public object classes, stable field names, summary method, and print methods;
- absence of an unvalidated `slope_stability()` export.

All earlier internal and nested-bootstrap validation tests remain unchanged.
