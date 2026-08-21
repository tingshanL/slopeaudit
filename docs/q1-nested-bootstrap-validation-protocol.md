# Q1 v14 nested-bootstrap validation nested-bootstrap protocol

## Purpose

nested-bootstrap validation validates the decision procedure that will be reported for Question 1.
Unlike the v13.3.1 diagnostic, each outer dataset is analyzed with a plug-in
parametric bootstrap under the fitted homogeneous-slope null. The production
unstructured model pair is used throughout:

- full: `y ~ time + (1 + time | id)`;
- null: `y ~ time + (1 | id)`;
- REML in both models;
- initial `nloptwrap`, followed by the frozen paired `bobyqa` rescue when the
  initial full/null pair is not positive/zero usable.

The covariance-constrained comparator is not part of nested-bootstrap validation. It remains a
mechanism diagnostic in the v13.3.1 supplement.

## Eight outer cells

nested-bootstrap validation retains the four historical null-size cells:

1. balanced three-wave schedule, random-intercept variance 0.5;
2. balanced three-wave schedule, random-intercept variance 4;
3. frozen irregular cognition schedule, random-intercept variance 0.5;
4. frozen irregular cognition schedule, random-intercept variance 4.

It evaluates four planning-power cells:

- 3 waves / 2 years, rho = 0;
- 3 waves / 2 years, paired rho = .5;
- 5 waves / 5 years, rho = 0;
- 5 waves / 5 years, paired rho = .5.

Within each rho pair, the outer design seed, participant slope effects,
residual draws, and inner-bootstrap master seed are identical. Only the random
intercept construction changes. The known-error direct statistic is saved for
every planning dataset and must agree within `1e-10` in every pair.

The frozen statistic-level unstructured-Q1 references are `.055`, `.074`,
`.169`, and `.425`, with their independently estimated v13.3.1 binomial
intervals. They are comparison references, not assumed nested-bootstrap truth.
The primary compatibility gate tests interval overlap; it does not pretend the
reference estimate is error-free.

## Fixed usable inner bootstrap

The primary inner execution rule retains exactly `B` positive/zero LR values.
Every initially unusable full/null pair is refit with the declared `bobyqa`
control. If rescue remains unusable, that attempt is retained in provenance
and a new deterministic seed is tried, up to the declared attempt cap.

For full mode, `B = 999` and the default attempt cap is 1,249. The plus-one
decision is `p <= .05`, for an exact alpha rank grid.

The existence of an excluded inner attempt is not itself a failed gate. For
each completed outer run, nested-bootstrap validation additionally reports two worst-case
sensitivities that treat every excluded attempt as either nonexceeding or
exceeding the observed LR. If those cases straddle alpha, the outer decision is
marked ambiguous in the decision bounds. Cell-level support bounds include
these ambiguous decisions.

An unresolved observed full/null pair is never imputed. It remains an unknown
outer decision and enters the cell-level lower/upper support bounds. The
default operational limit is 0.5% unresolved observed pairs per cell. A failure
to obtain `B` usable inner statistics before the attempt cap remains a gate
failure because the declared inner experiment did not complete.

## Modes

### `smoke`

- 2 outer datasets per cell;
- fixed usable `B = 39`;
- all 16 tasks are generated afresh;
- reference checks are descriptive; operational checks alone determine PASS.

### `pilot`

- 20 outer datasets per cell;
- fixed usable `B = 99`;
- imports an authorized private copy of the audited 120-task v12 checkpoint
  after exact task, seed, denominator, cap, alpha, and operational-gate
  verification;
- runs only 40 new paired rho=.5 tasks;
- verifies that all imported scientific fields remain byte-for-value
  identical to the audited v12 rows.

The old v12 analytic-power target gate is intentionally not inherited. Its
failure was scientific evidence that the direct-channel analytic power was not
the production unstructured-Q1 target. V14 replaces that check with the
independently estimated v13.3.1 unstructured-Q1 reference intervals while
retaining every v12 operational gate.

### `full`

- 999 outer datasets per cell;
- fixed usable `B = 999`;
- 7,992 outer tasks and approximately 7.98 million usable inner replicates;
- no smoke or pilot result is silently promoted into the full run.

Every mode checkpoints after the declared batch. Resume only with the
identical mode, output directory, worker count, seed, B, and attempt cap.

## Gates

Operational PASS requires:

- exact and unique task-manifest completion;
- fixed usable `B` in every valid outer run;
- no attempt-cap failure or unhandled Q1 error;
- observed unresolved rate within the declared limit;
- exact reconciliation of attempted, usable, excluded, and top-up counts;
- finite ordered worst-case p-value bounds for every valid outer run;
- identical design/bootstrap seeds and direct-channel statistics inside every
  rho pair;
- all cell-level operational checks;
- pooled null decision bounds compatible with alpha.

Pilot and full additionally require every cell's nested-bootstrap validation decision-bound
interval to be compatible with its frozen reference point or reference
interval. A scientific reference mismatch produces `DIAGNOSE`; it is not
relabelled an optimizer failure.

## Main outputs

- `q1_nested_bootstrap_outer_runs.csv`: one row per outer dataset, including observed
  diagnostics, fixed-usable p-value, top-ups, and worst-case p-value bounds;
- `q1_nested_bootstrap_rescue_attempts.csv`: every inner rescue or unresolved row;
- `q1_nested_bootstrap_summary.csv`: primary estimates plus cell-level decision bounds;
- `q1_nested_bootstrap_paired_rho_comparison.csv`: paired covariance sensitivity and
  exact direct-channel invariance check;
- `q1_nested_bootstrap_pooled_null_summary.csv`: four-cell pooled null result;
- `q1_nested_bootstrap_all_run_diagnostics.csv`: rescue, exclusion, and top-up rates;
- `q1_nested_bootstrap_operational_gate.csv` and `q1_nested_bootstrap_reference_gate.csv`;
- `q1_nested_bootstrap_run_status.csv`;
- checkpoint, manifests, configuration, failure-association diagnostic, and
  session information.

## Run order

Run and audit `smoke`, then run and audit `pilot`, and only then run `full`.
Do not freeze Table 1 or manuscript claims from smoke or pilot estimates.

## Private historical provenance inputs

The exact AABC2-derived irregular schedule and the historical v11/v12
diagnostic payloads are restricted-derived and are not distributed in the
public repository. Authorized historical reproduction supplies them explicitly
through:

- `SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE`;
- `SLOPEAUDIT_Q1_V11_EVIDENCE_DIR`;
- `SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_V12_PILOT_INPUT` for `pilot` mode.

These paths change only provenance/input location; the frozen seeds, model
pair, rescue hierarchy, and decision rules are unchanged.
