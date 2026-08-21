# Q1 v13.3.1 Table-1 power diagnostic

## Purpose and inferential boundary

This validation-stage diagnostic compares three statistics for the two
balanced, iid-residual planning rows that the current `lme4` backend can test
without residual-model misspecification:

1. direct-channel slope-variance statistic with known residual variance;
2. covariance-constrained restricted LR statistic, `(1 | id) + (0 + time | id)`;
3. the production Question-1 restricted LR statistic, `(1 + time | id)`.

The diagnostic estimates statistic power under a known DGP. It is not a nested
plug-in parametric-bootstrap experiment and must not be called parametric-
bootstrap power. The subsequent v14 nested-bootstrap validation checked whether
production nested-bootstrap decisions tracked these statistic-level targets.

The 10-day and 30-day EMA rows remain excluded because their planning models
assume AR(1) residuals, whereas the current `lme4` Q1 backend fits iid residuals.

## Changes from v13.2 and v13.3

V13.3.1 preserves the v13.3 estimands, seeds, fitting rules, boundary
adjudication, calibration, and extension design. It corrects one reporting-gate
mismatch: the adjudication helper ignored lme4's expected singular-boundary
message after separately verifying the slope boundary, but the final gate
mistakenly tested the unfiltered convergence-message field. V13.3.1 saves both
the raw and filtered non-singular convergence messages and makes the final gate
test the filtered fields, exactly matching the adjudication rule. Executable
regression checks require an expected singular-boundary message to pass and a
gradient or Hessian warning to fail.

### Narrow optimizer-code boundary adjudication

V13.2 already allowed an otherwise-valid, post-`bobyqa` constrained LR that
was slightly negative within the likelihood-scale tolerance to be recorded as
zero. V13.3 adds a separate constrained-diagnostic-only rule for the four
audited full-run cases in which rescue produced a boundary LR but one optimizer
returned a nonzero code.

The rescued statistic is recorded as zero only when every condition holds:

1. The initial pair is otherwise valid, has the declared optimizer and model
   structure, and has a negative LR within its likelihood-scale tolerance.
2. Both rescue fits exist and use the declared `bobyqa` optimizer.
3. The full/null observations, fixed effects, REML mode, grouping structure,
   and intended random-effects structures match.
4. The sole rescue validity failure is exactly one
   `optimizer_nonzero_code`; no non-singularity convergence message occurs.
5. The rescued LR has absolute value no greater than its likelihood-scale
   tolerance.
6. Each rescued likelihood agrees with or improves on its corresponding
   initial likelihood within tolerance.
7. The fitted constrained slope component is at its numerical boundary.

Raw likelihoods, LR, tolerance, optimizer codes, convergence messages, fitted
slope variance, relative slope SD, seed, and adjudication rule are retained.
This rule never applies to the unstructured production-Q1 statistic.

### Calibration and denominators

Each method is calibrated from its own finite null statistics. A failure in the
constrained bridge cannot remove valid direct or unstructured-Q1 results.
Pairwise method comparisons use only the task-ID intersection for the relevant
two methods.

For `n` usable null statistics, let `k = floor(.05 n)`. The empirical critical
is the order statistic that leaves at most `k` values strictly above it. The
output reports the realized empirical size and the number tied at the cutoff.
Thus, an ensemble of 999 usable null statistics yields a conservative realized
size no greater than .05 instead of an `NA` critical.

The direct-channel statistic is also evaluated at its exact chi-square critical
with 99 degrees of freedom. Its empirical critical is retained for matched-size
comparisons with the mixed-model statistics.

### Null top-up

Repair mode preserves the original 1,000 null datasets per design. Extend mode
adds 9,000 frozen new null datasets per design, for 10,000 per design, while
retaining the original 2,000 null datasets unchanged. New seeds come from a
separate master seed and are checked not to overlap the v13.2 seeds.

### Paired covariance sensitivity

Extend mode adds `rho = .5` for both designs. Each sensitivity dataset reuses
the seed and latent slope draw of its corresponding `rho = 0` alternative. The
residual draws are also identical. Only the random-intercept construction
changes. Consequently, the direct slope statistic must agree within `1e-10`
inside every pair.

The earlier five-wave `rho = .5` cell is retained unchanged as an independent,
unpaired replication and is explicitly labelled as such.

The constrained model remains a mechanism diagnostic. Fixing the covariance to
zero is a substantive assumption and is not advertised as a generally superior
test.

## Modes and frozen work

### `smoke`

Runs 20 null, 20 `rho = 0`, and 20 paired `rho = .5` datasets for each design.
No checkpoint import is accepted. Results are operational checks only.

### `repair`

Requires the v13.2 full checkpoint. It verifies all 5,000 frozen task IDs,
scenarios, replicates, and seeds; retains the 4,996 usable rows; and reruns only
the four audited constrained optimizer-code boundary tasks. It then regenerates
all method-specific summaries and gates.

### `extend`

Requires a completed v13.3.1 repair checkpoint. It retains all 5,000 repaired
rows and runs 20,000 new datasets:

- 9,000 additional null datasets for each of two designs;
- 1,000 paired `rho = .5` alternatives for each of two designs.

The existing 2,000 `rho = 0` alternatives and the existing 1,000 unpaired
five-wave sensitivity datasets are not rerun.

Every mode checkpoints after five tasks per worker. Resume with the identical
mode, output path, import path, and worker count.

## Main outputs

- `q1_table1_power_task_results.csv`: task-level statistics and fit provenance.
- `q1_table1_power_targeted_refit_comparison.csv`: repair-mode proof that the
  four regenerated datasets preserve their direct and unstructured-Q1 results.
- `q1_table1_power_numerical_zero_floors.csv`: both constrained numerical-zero
  rules with raw diagnostic fields.
- `q1_table1_power_null_calibration.csv`: method-specific denominators,
  empirical criticals, realized sizes, ties, and theoretical comparators.
- `q1_table1_power_summary.csv`: empirical-critical power for every method plus
  exact-chi-square direct-channel power.
- `q1_table1_power_paired_differences.csv`: relevant two-method intersections.
- `q1_table1_power_paired_rho_comparison.csv`: paired covariance-channel
  sensitivity and direct-invariance check.
- `q1_table1_power_fit_diagnostics.csv`: rescue and numerical-floor rates.
- `q1_table1_power_candidate_table_rows.csv`: historically named
  manuscript-facing statistic-level reference rows; the constrained bridge is
  omitted from this table.
- `q1_table1_power_operational_gate.csv`: executable operational checks.

## Interpretation

The analytic direct result is called a **direct-channel benchmark**, not an
idealized ceiling. The unstructured Q1 statistic can exceed it when the true
intercept-slope covariance is nonzero because its null removes both slope
variance and covariance. Therefore, Q1 power is reported at `rho = 0` and at
`rho = .5` sensitivity rather than presented as a property of the schedule
alone.

The repaired and extended runs both passed independent audit before their
statistic-level targets were frozen for v14 nested-bootstrap validation. This is
a historical sequencing rule, not pending release work.
