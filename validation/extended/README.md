# Question-1 extended validation

These scripts are intentionally **not** ordinary `R CMD check` tests.

The manuscript-scale reference scripts run sequentially and display a base-R
progress bar. Historical v12/v13 validation workflows may parallelize
independent outer datasets. Their default is one worker; use at most two workers
on the 8-GB M2.

**Release-state note.** The v10--v13 sections below are historical validation
provenance. The v14 7,992-task Question 1 bootstrap validation study is
complete. For the pre-tag `0.1.0` release candidate, these expensive historical
runs are not repeated merely to change the package version; release
verification instead checks the preserved provenance/code bridge and the
current release-level smoke/check gates.

## Scripts

- `q1_sleepstudy_reference.R`
  - canonical public sleepstudy model pair;
  - checks the observed LR against the historical ≈42.837 anchor;
  - runs the current R implementation at `SLOPEAUDIT_Q1_B` (default 999);
  - writes replicate and summary outputs for refreezing.

- `q1_btheb_reference.R`
  - reads the already-frozen public Beat-the-Blues primary fixture;
  - fits the homogeneous REML full/null pair;
  - compares observed LR with the historical ≈4.072 anchor;
  - runs the current R bootstrap for refreezing.

- `q1_targeted_stress.R`
  - configurable outer null/strong-signal stress experiment;
  - irregular schedules;
  - uses separate sampled design and bootstrap seeds, avoiding integer seed
    arithmetic and recording both seeds in the replicate output;
  - records outer-run counts of failures, small negatives, material negatives,
    usable rates, and support rates.

- `q1_targeted_stress_diagnostics.R`
  - reads the completed outer-run CSV and reruns only cases with material
    negatives, small negatives, or a prespecified large failure cluster;
  - uses the recorded design and bootstrap seeds, and verifies exact outer-run
    reproduction;
  - saves every retained inner-bootstrap row plus a problem-only table with
    failure reasons, convergence messages, LR magnitudes, and log likelihoods;
  - defaults to a failure-cluster threshold of 10 and can be configured with
    `SLOPEAUDIT_Q1_STRESS_FAILURE_THRESHOLD`.

- `q1_four_design_historical_oracle.R`
  - checks the historical four-design false-support k/n anchors:
    `47/999`, `48/997`, `41/998`, `45/999`;
  - requires the local source path via an environment variable;
  - does **not** claim that the historical calibrated restricted-LR procedure
    is identical to the new Contract-v1.1 bootstrap.

- `q1_four_design_calibration.R`
  - implements the historical v12 deterministic-rescue calibration workflow;
  - runs the genuine four-cell null-size calibration under the recovered DGP;
  - adds two independent-residual planning-power validation cells matching
    the 3-wave/2-year and 5-wave/5-year rows of the practical design table;
  - attempts each observed and inner pair with `nloptwrap`, then refits both
    models with `bobyqa` if the pair is not primary-usable;
  - tops up unresolved inner rescues to a fixed primary usable denominator;
  - uses the inclusive decision rule `p <= alpha`;
  - retains `p_drop`, `p_zero`, and `p_all_zero`;
  - uses separate design/bootstrap seeds, checkpoint/resume, optional outer
    parallelism, exact outer binomial intervals, and two-sided gates;
  - saves every rescue attempt or final problem row, but not ordinary
    initial-usable positive/zero rows;
  - is hard-bound to the exact v10 120-task manifest in `pilot_repeat` mode;
  - repairs all-run diagnostic totals and adds pooled-null/global gates.

- `q1_four_design_calibration_v10.R`
  - preserves the exact failed-pilot implementation for provenance only;
  - should not be used to generate new v12 evidence.

- `q1_optimizer_rescue_diagnostic.R`
  - is the v11 follow-up to the failed v10 pilot gate;
  - reconstructs every retained problem replicate from its frozen outer and
    inner seeds;
  - first reproduces the exact v10 `nloptwrap` result;
  - refits each full/null pair with tighter `nloptwrap`, `bobyqa`, and a
    derivative-disabled diagnostic-only sensitivity;
  - retains likelihood, LR, gradient, Hessian, convergence, singularity, and
    covariance-geometry diagnostics;
  - checkpoints by outer dataset and never freezes a rescue rule
    automatically.

- `q1_table1_power_diagnostic.R`
  - is the v13.3.1 focused diagnostic that precedes nested-bootstrap validation;
  - reconstructs the analytic direct-slope power formula;
  - evaluates the direct, covariance-constrained, and unstructured-Q1
    statistics on the same generated datasets;
  - calibrates each statistic on its method-specific finite null ensemble and
    reports the realized conservative empirical size;
  - reports the direct statistic at both empirical and exact chi-square cutoffs;
  - adds paired nonzero-covariance sensitivities for both designs while holding
    participant slopes and residual draws fixed;
  - adds a narrow constrained-only boundary fallback for a rescued LR at zero
    with one nonzero optimizer code, full diagnostics, and provenance;
  - tops up each null ensemble from 1,000 to 10,000 in `extend` mode;
  - leaves the production unstructured-Q1 rescue/adjudication unchanged;
  - does not run an inner bootstrap or label its result parametric-bootstrap
    decision power;
  - excludes the 10-day and 30-day EMA rows because their AR(1) residual model
    is outside the current lme4 Question-1 backend.

- `q1_nested_bootstrap_validation.R`
  - is the v14 nested-bootstrap validation decision-procedure validation;
  - evaluates only the production unstructured Question-1 model pair;
  - retains the four historical null-size cells and adds rho=0 and paired
    rho=.5 power cells for both planning designs;
  - uses fixed usable inner B with deterministic top-ups;
  - reports worst-case decision bounds for excluded inner attempts and
    unresolved observed fits instead of requiring literally zero numerical
    exclusions across millions of fits;
  - imports the audited v12 120-task checkpoint in `pilot` mode and runs only
    the 40 new covariance-sensitivity tasks;
  - checkpoints and resumes by outer task;
  - is governed by `docs/q1-nested-bootstrap-validation-protocol.md`.

## V13.3.1 Table-1 power diagnostic

First repair the completed v13.2 full run. This retains 4,996 rows and reruns
only the four audited constrained optimizer-code boundary rows:

```bash
SLOPEAUDIT_Q1_POWER_MODE=repair \
SLOPEAUDIT_Q1_POWER_WORKERS=1 \
SLOPEAUDIT_Q1_POWER_IMPORT_CHECKPOINT="validation/extended/results/q1_table1_power_full_v13_2/q1_table1_power_checkpoint.rds" \
SLOPEAUDIT_Q1_POWER_OUT="validation/extended/results/q1_table1_power_repair_v13_3_1" \
Rscript validation/extended/q1_table1_power_diagnostic.R
```

After that repair output has been audited, import its checkpoint in `extend`
mode. This retains all 5,000 repaired rows and runs 20,000 new datasets: 18,000
null top-ups and 2,000 paired covariance sensitivities. See
`docs/q1-v13-table1-power-diagnostic.md` for the exact run contract. The
extended output has now been audited and supplies the frozen nested-bootstrap validation reference
fixture.

## V14 completed nested-bootstrap validation provenance

The smoke, pilot, and full validation sequence is complete. The commands below
document authorized historical reproduction of the pre-full stages; they are
not instructions to restart the expensive validation for release 0.1.0.

Historical smoke command:

```bash
SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_MODE=smoke \
SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_WORKERS=1 \
SLOPEAUDIT_Q1_V11_EVIDENCE_DIR="<PRIVATE_V11_EVIDENCE_DIR>" \
SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE="<PRIVATE_IRREGULAR_SCHEDULE.csv>" \
SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_OUT="validation/extended/results/q1_nested_bootstrap_smoke_v14" \
Rscript validation/extended/q1_nested_bootstrap_validation.R
```

Historical pilot reproduction imports the audited 120 v12 tasks from an
authorized private checkpoint, so only 40 new tasks are fitted. It requires
`SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_V12_PILOT_INPUT`,
`SLOPEAUDIT_Q1_V11_EVIDENCE_DIR`, and `SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE` to point
to the private frozen inputs:

```bash
SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_MODE=pilot \
SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_WORKERS=1 \
SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_V12_PILOT_INPUT="<PRIVATE_V12_PILOT_DIR>" \
SLOPEAUDIT_Q1_V11_EVIDENCE_DIR="<PRIVATE_V11_EVIDENCE_DIR>" \
SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE="<PRIVATE_IRREGULAR_SCHEDULE.csv>" \
SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_OUT="validation/extended/results/q1_nested_bootstrap_pilot_v14" \
Rscript validation/extended/q1_nested_bootstrap_validation.R
```

The full mode subsequently completed all 7,992 outer datasets with fixed
usable inner B=999 and passed the frozen validation gates. Preserve that run as
historical validation provenance; do not relabel it as a `0.1.0` execution.
See the protocol document for the gates and exact interpretation.

## Private irregular-schedule provenance

The exact empirical cognition schedule is derived from restricted AABC2 source
material and is not distributed in the public repository. The one-time freezer
and its frozen output belong to private release provenance. Authorized
historical validation runs must provide the byte-preserved schedule through
`SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE`; do not substitute an approximate schedule.

## V12 completed repeated-pilot provenance

V10's pilot, v11's optimizer diagnostic, and the v12 repeated pilot are
complete. The command below documents authorized historical reproduction of
the same 120 v10 outer tasks under the adjudicated v12 rescue hierarchy:

```bash
SLOPEAUDIT_Q1_CAL_MODE=pilot_repeat \
SLOPEAUDIT_Q1_CAL_WORKERS=1 \
SLOPEAUDIT_Q1_PILOT_INPUT="<PRIVATE_V10_PILOT_DIR>" \
SLOPEAUDIT_Q1_V11_EVIDENCE_DIR="<PRIVATE_V11_EVIDENCE_DIR>" \
SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE="<PRIVATE_IRREGULAR_SCHEDULE.csv>" \
SLOPEAUDIT_Q1_OUT="validation/extended/results/q1_four_design_pilot_repeat_v12" \
Rscript validation/extended/q1_four_design_calibration.R
```

At the time of the repeated pilot, the per-task v10/v12 comparison, observed
and inner rescue records, all-run totals, pooled-null result, power cells, and
global gate were audited before escalation. That adjudication is now historical
provenance; see `docs/q1-v12-rescue-pilot-protocol.md`.

## V11 completed provenance (do not rerun)

The v10 pilot and both v11 diagnostic runs are complete. Preserve their result
archives. The commands below document provenance only; do not rerun them for
v12.

First run the small panel-coverage smoke diagnostic:

```bash
SLOPEAUDIT_Q1_RESCUE_MODE=smoke \
SLOPEAUDIT_Q1_RESCUE_WORKERS=1 \
SLOPEAUDIT_Q1_PILOT_INPUT="<PRIVATE_V10_PILOT_DIR>" \
SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE="<PRIVATE_IRREGULAR_SCHEDULE.csv>" \
SLOPEAUDIT_Q1_OUT="validation/extended/results/q1_optimizer_rescue_smoke_v11" \
Rscript validation/extended/q1_optimizer_rescue_diagnostic.R
```

After its output is audited, run all 908 retained problem replicates:

```bash
SLOPEAUDIT_Q1_RESCUE_MODE=full \
SLOPEAUDIT_Q1_RESCUE_WORKERS=1 \
SLOPEAUDIT_Q1_PILOT_INPUT="<PRIVATE_V10_PILOT_DIR>" \
SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE="<PRIVATE_IRREGULAR_SCHEDULE.csv>" \
SLOPEAUDIT_Q1_OUT="validation/extended/results/q1_optimizer_rescue_full_v11" \
Rscript validation/extended/q1_optimizer_rescue_diagnostic.R
```

The full run performs four full/null refit pairs for each problem row. It is
checkpointed and resumable. `calc.derivs=FALSE` is diagnostic only and is
excluded from every inferential eligibility field. See
`docs/q1-v11-optimizer-rescue-diagnostic.md` before interpreting the outputs.
The audited v10 reconstruction payload is restricted-derived and is not
distributed publicly. Authorized historical reruns must set
`SLOPEAUDIT_Q1_PILOT_INPUT` and `SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE` to private
byte-preserved inputs.

## Manuscript-scale escalation

The package implementation reports whether a run is floor-limited or whether
the Monte Carlo interval crosses the reporting alpha. Manuscript workflows may
then rerun at 4,999 according to the frozen protocol. The package does not
silently change B.


## v3 stress coverage

`q1_targeted_stress.R` now contains four explicit scenarios:
- exact true-zero slope heterogeneity;
- strong slope heterogeneity;
- low-information irregular schedules;
- near-rank-one intercept/slope covariance.

`q1_compare_reference_outputs.R` creates a compact R-versus-historical-Python
comparison table. Observed LR values are direct overlap checks; historical
bootstrap p-values are marked contextual until the R implementation is
formally refrozen because seed/optimizer implementations differ.
