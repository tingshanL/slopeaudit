# Q1 v11 optimizer-rescue diagnostic

**Status:** completed and adjudicated. V11 did not change the package bootstrap
engine, rerun the 120-task pilot, or refreeze either manuscript reference
analysis. Its audited output now provides the evidence for v12's deterministic
rescue rule.

## Final adjudication

All 908 retained v10 problem rows reproduced. `bobyqa` produced a
primary-usable pair for 908/908, whereas tight `nloptwrap` did so for 274/908
and was never the only successful rescue. The resolution patterns were 634
`bobyqa_only`, 219 joint agreements, and 55 jointly usable numerical
disagreements. In every disagreement, `bobyqa` attained an equal or higher
full-model likelihood while null likelihoods agreed to numerical precision.

V12 therefore freezes current `nloptwrap` first and paired `bobyqa` rescue only
when the initial pair is not primary-usable. Tight `nloptwrap` and
`calc.derivs=FALSE` do not enter the production hierarchy.

## Why v11 is required

The v10 pilot completed all 120 outer tasks, but five outer datasets exhausted
the attempt cap. Across all attempted inner refits it retained:

- 866 full-model convergence failures;
- 14 small negative likelihood ratios;
- 28 material negative likelihood ratios.

The problems clustered strongly by outer dataset. Consequently, topping up to
99 finite nonnegative likelihood ratios cannot yet be treated as innocuous
replacement of randomly missing bootstrap draws. The optimizer behavior must
be diagnosed before the fixed-usable protocol is accepted or rerun.

## Frozen input

V11 is bound to the already-audited v10 pilot:

- 6 cells x 20 outer datasets = 120 outer tasks;
- 115 completed fixed-usable tests and 5 attempt-cap failures;
- 908 retained problem rows across 95 outer datasets;
- the exact 691-participant/1,805-observation irregular schedule fixture;
- the original design, bootstrap-master, and inner-replicate seeds.

The script stops if these counts, file hashes, task keys, or seed streams do not
reconcile. Smoke mode selects 20 rows covering every observed cell x diagnostic
class combination; full mode selects all 908. It regenerates each outer dataset
and simulated inner response from the saved seeds. The v10 optimizer is rerun
first and must reproduce every selected status, reason class, and available
likelihood quantity before any alternative fit is interpreted.

The audited v10 reconstruction payload is restricted-derived and is not
distributed in the public repository. Authorized historical reproduction must
provide the private frozen directory through `SLOPEAUDIT_Q1_PILOT_INPUT` and
the exact private schedule through `SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE`. The
complete pilot archive remains the authoritative provenance record.

## Optimizer panel

Every retained problem replicate is refitted as a full/null pair under four
explicit controls:

1. `current_nloptwrap`: exact v10 reproduction control; derivative checks on,
   `maxeval=200000`.
2. `tight_nloptwrap`: derivative checks on, `maxeval=1000000`,
   `xtol_abs=1e-10`, and `ftol_abs=1e-10`.
3. `bobyqa`: derivative checks on, `maxfun=1000000`, and `rhoend=1e-9`.
4. `current_no_derivs`: exact v10 optimization settings with
   `calc.derivs=FALSE`, retained only to show what is hidden when derivative
   checks are suppressed. It is never inferentially eligible.

The two rescue candidates are not mixed within a full/null pair. V11 records
log likelihoods, LR classification, optimizer codes, convergence messages,
singularity, raw and scaled gradients, Hessian eigenvalue diagnostics,
fixed/random parameter vectors, and full-model covariance geometry.

## What v11 did not decide at that stage

V11 reports whether tighter `nloptwrap` and `bobyqa` both resolve a row and
whether their full objective, null objective, and LR agree to the declared
diagnostic tolerance. It does not automatically choose the first successful
optimizer, take the largest likelihood across different controls, recode a
failed fit, or adopt `calc.derivs=FALSE`.

Those decisions were subsequently made: v12 froze the deterministic rescue
rule and repeated the same 120 outer datasets, and v14 later validated the
production nested-bootstrap decision procedure. This paragraph is retained as
historical workflow provenance, not as pending release work.

## Outputs

The diagnostic writes:

- an input and method manifest;
- selected task rows and a checkpoint;
- one long result row per problem replicate and optimizer control;
- observed-outer and current-control inner reproduction checks;
- per-task candidate agreement/resolution summaries;
- per-method and per-outer summaries;
- an operational gate and session information.

An operational `PASS` means only that reconstruction and the optimizer panel
completed as specified. It is not approval of a rescue rule or of the v10
fixed-usable bootstrap.
