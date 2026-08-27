# Q1 v10 calibration decisions

**Protocol role:** v10 calibration candidate. Its failed operational gate led
to the v11 optimizer diagnostic, v12 deterministic rescue hierarchy, and v14
nested-bootstrap validation.

## Review points adopted

1. **Fixed usable denominator.** The primary calibration bootstrap draws until
   exactly `B_usable` finite nonnegative LRs have been obtained, subject to a
   declared attempts cap. This removes the large rank-grid change caused by
   losing one replicate at attempted B = 99.

2. **Inclusive rejection rule.** Support is defined as plus-one `p <= alpha`.
   With `p < alpha`, B = 99 would have rank size .04 even with all 99
   replicates usable. At alpha .05, B = 99 with `<=` has 5 rejecting ranks out
   of 100; a variable usable denominator of 98 has only 4 out of 99.

3. **Explicit R control.** The original notebook used statsmodels and an
   optimizer panel, so an equivalent `lmerControl` cannot be recovered. The
   candidate specifies `nloptwrap`, derivative checks on, a 200,000-evaluation
   cap, default edge restart/boundary tolerance, and singular messages. The
   same control is used for observed and bootstrap full/null fits throughout
   the calibration.

4. **Exact DGP or no historical comparison.** All scalar DGP settings were
   recovered from the source notebook. The empirical schedule values
   must be extracted exactly from the original AABC files with the supplied
   provenance utility. Approximate schedules are rejected. If exact extraction
   ultimately fails, the fallback is a newly named/documented irregular DGP
   with the historical comparison removed—not a partial match.

5. **Two power cells.** Two independent-residual rows from the practical design
   table are added: 100 participants with 3 waves/2 years and 5 waves/5 years,
   residual SD 1, slope SD .10 per year. Their empirical bootstrap power is
   compared with the analytic direct-contrast approximation. The 10-day and
   30-day EMA rows are not used because they assume AR(1) residuals that the
   current lme4 backend does not fit.

6. **Third finite-negative convention.** `p_all_zero` retains every finite
   material negative as an explicitly classified optimizer problem while
   assigning LR = 0 only in this sensitivity calculation. `p_drop` remains
   primary. Ordinary failures and nonfinite results are never zero-recoded.

7. **Two-sided operating-characteristic gate.** A null-size cell is diagnosed
   when its exact outer binomial interval excludes .05 in either direction.
   A power cell is diagnosed when the analytic planning value lies outside the
   empirical interval. Low validity, exhausted attempt caps, or consequential
   convention disagreement also trigger diagnosis.

8. **Interval relabeling.** `exceedance_probability_interval_*` identifies the
   exact binomial interval for the fitted-null exceedance probability `k/n`.
   It is not labeled as an interval for `(k+1)/(n+1)`. Old `mc_interval_*`
   fields remain aliases for compatibility with existing output readers.

9. **License.** The MIT copyright holder placeholder is replaced with
   Tingshan Liu.

## Review points qualified

- Top-up restores the intended rank grid only conditional on the retained
  bootstrap distribution. Correlating each outer dataset's issue/top-up count
  with its observed LR is a useful indirect diagnostic, but it cannot test
  independence from the LR that a failed inner fit would have had under
  successful optimization. V10 labels that limitation explicitly and retains
  every problem row.

- Parallelizing independent outer datasets is scientifically safe because all
  design and bootstrap master seeds are fixed before execution. The suggested
  6--8 workers are not assumed safe on an 8-GB laptop, and a timing estimate
  from N = 100 does not project the 600/691-participant null cells. V10 defaults
  to one worker, permits explicit outer parallelism on macOS/Linux, and leaves
  the worker choice to the audited pilot.

- The plus-one rule improves Monte Carlo rank validity; the plug-in parametric
  bootstrap is not described as an exact finite-sample test for unknown mixed-
  model nuisance parameters.

## Consequence for completed reference runs

The B = 4,999 v1.1 results remain valid records of the tested fixed-attempt
implementation:

- sleepstudy: 0/4,936 usable exceedances, plus-one p = .000203;
- Beat the Blues: 365/4,960, plus-one p = .07378.

The corresponding exact binomial intervals describe fitted-null exceedance
probabilities, not plus-one-p intervals. The sleepstudy attainable floor in
that run is 1/4,937 = .000203, not .001.

The v10 pilot did not pass. The later v11--v14 sequence supplied the optimizer
hierarchy and validation used by the current implementation.
Cross-implementation agreement of observed LR statistics remains distinct
from implementation- and protocol-specific bootstrap p-values.
