# Historical manuscript-update notes from the v10 calibration phase

**Historical status:** superseded workflow notes retained for provenance. The
v10 protocol was not the final release protocol; later v11--v14 validation
adjudicated the rescue hierarchy and fixed-usable implementation. The wording
and number corrections below record what was known at this stage.

## Beat the Blues schedule heterogeneity

> Beat the Blues also illustrates why participant-level information should be
> inspected rather than summarized by one number. Three of the 100
> participants contributed only one observation and therefore had
> `R_direct,i = 0`; conditional on the fitted random-slope model, their very
> small fitted-slope determinacy (`D_slope,i^2` approximately .00005) was
> entirely model-assisted. Across participants, mean `R_direct` was .198,
> whereas the median was .334 (Q1--Q3: .026--.334; range: 0--.334), reflecting
> marked differences in the number, timing, and span of observations.

Retain the conditional-model qualification because Question 1 was unsupported
for Beat the Blues.

## Current v1.1 reference-bootstrap wording

- Sleepstudy: report plus-one p = .000203 from 0 exceedances among 4,936 usable
  replicates. Its observed-run floor is .000203, not .001.
- Beat the Blues: report plus-one p = .07378 from 365/4,960 usable replicates.
- Describe the exact binomial interval as Monte Carlo uncertainty for the
  fitted-null exceedance probability, not as a confidence interval for the
  plus-one p-value.
- State plainly that the observed LR statistics reproduce across R and Python
  to the reported numerical tolerance, whereas bootstrap p-values are specific
  to their simulation, optimizer, validity, and denominator protocols.

This final conditional was superseded when v10 did not become the release
protocol. Current manuscript reference values must come from the explicitly
frozen release-verification records rather than mixing historical protocols.
