# Q1 v12 deterministic rescue and repeated-pilot protocol

**Historical status:** completed repeated-pilot protocol. V12 repeated the
failed 120-task v10 pilot under the optimizer rule adjudicated from v11 and
supplied the deterministic rescue hierarchy later carried into v14 validation
and the release implementation. It did not itself refreeze the sleepstudy and
Beat-the-Blues manuscript reference analyses.

## Frozen rescue hierarchy

For every observed or bootstrap full/null pair:

1. Fit both models with `nloptwrap`, derivative checks enabled,
   `maxeval=200000`, `restart_edge=TRUE`, and `boundary.tol=1e-5`.
2. Retain that pair only if both fits pass the existing diagnostic contract and
   the LR class is `positive` or `zero`.
3. Otherwise—including fit/convergence failures, small negative LRs, and
   material negative LRs—refit both models with `bobyqa`, derivative checks
   enabled, `maxfun=1000000`, and `rhoend=1e-9`.
4. Retain the `bobyqa` pair only if it is primary-usable.
5. If `bobyqa` also fails, classify the pair as unresolved and top up the inner
   bootstrap, subject to the frozen attempts cap.

The full and null models always use the same optimizer within an attempt. V12
does not mix objectives across controls, select the largest likelihood after
the fact, try tight `nloptwrap` in production, or use `calc.derivs=FALSE`.

## Why this rule is evidence-based

The full v11 diagnostic reconstructed all 908 problematic v10 inner replicates.
`bobyqa` resolved 908/908; tight `nloptwrap` resolved 274/908 and never uniquely
succeeded. Of the 274 jointly usable rows, 219 agreed numerically. In the 55
disagreements, `bobyqa` had the higher full-model likelihood in every case,
while null likelihoods agreed within numerical precision. The 188 singular
`bobyqa` full fits remain eligible because singularity at the null boundary is
not itself a convergence failure.

V12 programmatically checks the minimal frozen v11 summaries supporting
these anchors, but those restricted-derived summaries are not distributed in
the public repository. Authorized historical reproduction supplies them through
`SLOPEAUDIT_Q1_V11_EVIDENCE_DIR`. The original v11 output archive remains the
complete evidence.

## Repeated-pilot identity

`pilot_repeat` is hard-bound to the audited v10 pilot:

- 6 cells x 20 outer datasets = 120 tasks;
- the identical task ordering;
- the identical master, design, and bootstrap seeds;
- fixed usable B = 99;
- attempt cap = 149;
- alpha = .05 and support defined by plus-one p <= .05;
- the identical DGP cells and exact 691-participant/1,805-observation irregular
  schedule fixture.

The script stops if any of these differ. Thus, v10-versus-v12 comparisons are
paired at the outer-dataset and bootstrap-seed levels. Inner seed streams also
remain identical until different stopping behavior ends a task.

## Preserved diagnostics

The outer-run table records initial and rescue diagnostics for the observed
pair. `q1_four_design_rescue_attempts.csv` retains every inner row that triggered
rescue or remained problematic, with initial and rescue statuses, likelihoods,
LRs, optimizer identities, singularity, warnings, and messages kept separately.

All-run summaries include attempt-cap failures and diagnostics from invalid as
well as valid outer runs. They no longer reproduce the v10 reporting bug that
summed problem counts only over valid runs.

## Gates

V12 reports cell-specific and pooled-null results. A global pass requires:

- all 120 outer tasks completed;
- at least 95% valid tasks in every cell;
- no attempt-cap failure, unresolved observed pair, or calibration error;
- no final inner failure or finite-negative LR after rescue (including no
  unresolved `bobyqa` pair);
- no primary-versus-all-negative-zero decision disagreement;
- every cell's exact outer binomial interval contains its target (.05 for null
  cells; the analytic planning value for power cells);
- the pooled-null exact interval contains .05.

These were diagnostic gates for the repeated pilot, not proof of calibrated
size or power from only 20 outer datasets per cell. Their role was to authorize
the next adjudication step; the subsequent v14 validation, not this pilot alone,
provides the release-level nested-bootstrap validation evidence.

## Outputs

V12 adds:

- `q1_four_design_rescue_attempts.csv`;
- `q1_four_design_all_run_diagnostics.csv`;
- `q1_four_design_pooled_null_summary.csv`;
- `q1_four_design_global_gate.csv`;
- `q1_four_design_v10_v12_comparison.csv` in `pilot_repeat` mode.

The preserved v10 implementation is available as
`q1_four_design_calibration_v10.R`; it should not be run to create new v12
evidence.
