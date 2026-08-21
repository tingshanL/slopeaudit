# Implementation Decisions

**Status:** frozen coding-facing summary for Statistical Contract v1.1\
**Purpose:** translate the statistical contract into concise implementation rules.\
**Precedence:** if this file conflicts with Statistical Contract v1.1, the contract controls.

## 1. Direct-change information

For participant `i`, `R_direct_i` is based on the **within-participant GLS/BLUE estimator of the target slope** using only that participant's repeated observations.

Other participant-specific random effects are treated as **unknown nuisance constants**, not as random quantities from which information about the target slope may be borrowed.

Operationally, the direct slope contrast must annihilate the nuisance random-effect design. Information transmitted through fitted covariance between the target slope and other random effects belongs to the model-assisted channel, not the direct channel.

The between-participant target-slope variance `tau_s^2` appears in the numerator of `R_direct_i`. Thus `R_direct_i` quantifies direct measurement of a between-person slope difference; it is not a purely within-person quantity.

## 2. Independent computation of the two main quantities

Compute `R_direct_i` and `D_slope2_i` by **independent mathematical routes**.

Do not compute `D_slope2_i` as `R_direct_i + Delta_fit_i`.

After both are available, define:

`Delta_fit_i = D_slope2_i - R_direct_i`.

Then check:

- additivity to numerical tolerance;
- theoretical nonnegativity of `Delta_fit_i`.

Material negative values are diagnostic failures and must not be silently truncated.

## 3. Plug-in conditional-variance convention

`D_slope2_i` uses the conditional variance of the participant's target random slope **conditional on the fitted model parameters**.

Variance-component and fixed-effect estimates are treated as plug-in values. The default decomposition does **not** add sampling uncertainty from estimating `beta`, `G`, or `R` to the participant-level posterior/conditional slope variance.

The `lmerMod` adapter must reproduce the model-conditional quantity implied by the fitted `lme4` model; it must not silently add a fixed-effect-estimation correction.

Sampling uncertainty in aggregate source quantities is a separate problem handled by resampling/stability procedures.

## 4. Participant-level quantities and sample summaries

Participant-level quantities are primary.

When one scalar sample summary is needed, use the **equal-participant arithmetic mean**:

- mean participant-level `R_direct_i`;
- mean participant-level `D_slope2_i`;
- mean participant-level `Delta_fit_i`.

Do not weight participants by number of observations.

This convention preserves exact aggregate additivity:

`mean(D_slope2_i) = mean(R_direct_i) + mean(Delta_fit_i)`.

Retain median, IQR, minimum, and relevant counts as diagnostics. The default output should explicitly report the count of participants with **no direct slope information** when that count is nonzero.

For a completely balanced schedule with common fitted residual structure, participant-level values coincide; in that case mean, median, and individual values are identical.

## 5. Zero direct information

If a participant's retained observations contain no estimable within-person contrast for the target slope, set:

`R_direct_i = 0`.

Examples include:

- a participant with only one usable observation;
- multiple usable observations but no within-person variation in the target time variable.

If the full fitted model is otherwise valid, `D_slope2_i` may still be positive. In that case all fitted determinacy is model-assisted:

`Delta_fit_i = D_slope2_i`.

These cases must be explicitly tested.

## 6. Question 1 unsupported

Question 1 concerns evidence for population heterogeneity in the target slope. It does not automatically suppress computation of Questions 2 and 3.

If:

- the fitted target-slope variance is strictly positive; and
- the supplied full model passes the required numerical checks;

then compute the decomposition even if the heterogeneity test is unsupported.

However, label the quantities as **conditional descriptions of the fitted random-slope model**. Do not describe them as established reliability of a population trait in change, and do not let later questions override an unsupported Question 1.

If the fitted target-slope variance is exactly zero, the ordinary proportion decomposition is undefined/not interpretable and should be withheld.

Global singularity of the random-effects covariance matrix is a separate diagnostic. A globally singular or near-rank-one fit does not by itself imply that the target marginal slope variance is zero.

## 7. Full/null model contract for Question 1

The publication-quality heterogeneity-test interface requires both the full and null fitted models.

No hidden construction or hidden refitting of the null model is permitted.

The full and null models must have the same:

- response;
- observations after model-specific missing-data omission;
- fixed-effect design;
- residual model;
- grouping structure;
- nuisance random effects;
- likelihood convention.

They may differ only by the target random slope and covariance parameters that mathematically disappear when that slope variance is zero.

Under the null, setting the target slope variance to zero also removes every covariance involving that target random effect.

## 8. REML / ML behavior

Never silently convert ML to REML or REML to ML.

For the manuscript's primary Gaussian random-effect comparisons, use the supplied **REML** full/null fits with identical fixed effects, matching the prespecified analysis and bootstrap procedure.

Compatible ML full/null fits may be supported later, but the estimation mode must be retained and reported.

If the two supplied fits use different likelihood conventions, stop with an informative error rather than refitting automatically.

Questions 2 and 3 use the full model exactly as supplied.

## 9. Parametric-bootstrap convention

Question 1 uses a plug-in parametric bootstrap under the fitted null model.

For each replicate:

1. simulate under the fitted null model using the original design and observation schedule;
2. refit both null and full models using the same likelihood mode and prespecified controls;
3. calculate the likelihood-ratio statistic independently from the two refits.

The bootstrap does not propagate uncertainty in the null parameter estimates beyond the plug-in resampling procedure.

Track requested, usable, and failed replicates explicitly.

Small **finite** negative LR values attributable to numerical fitting receive a sensitivity treatment that reports both:

- a convention that drops those finite negative replicates; and
- a convention that sets those finite negative LR values to zero.

Do not reclassify materially invalid, nonfinite, or failed refits as zero evidence. Those remain failed/invalid and are reported separately.

For numerical diagnosis, also report a third `p_all_zero` sensitivity that
leaves every finite negative LR classified as an optimizer problem but assigns
it LR = 0 in that sensitivity distribution. This is never the primary
convention and does not erase the material-negative count or failure reason.
Ordinary refit failures and nonfinite results are excluded from all three
conventions.

The original v1.1 development engine treated `B` as attempted replicates.
The subsequent v10--v14 validation program adopted the release implementation's
fixed-usable execution mode: draw until the primary convention contains exactly
`B_usable` finite nonnegative LRs, subject to a declared attempts cap. At alpha
.05, use `B_usable` values such as 39, 99, 399, or 999 for which
`alpha * (B_usable + 1)` is an integer, with the rejection rule frozen as
plus-one `p <= alpha`.

The fixed-usable rule restores the intended Monte Carlo rank grid conditional
on the retained bootstrap distribution. It cannot prove that optimizer failure
is independent of the unobserved LR. Therefore retain failure classes, top-up
counts, material-negative rows, and an outer-run association diagnostic.

Intervals produced from `k` exceedances among `n` bootstrap draws are exact
binomial confidence intervals for the fitted-null exceedance probability
`Pr(LR* >= LR_observed)`. They are not confidence intervals for the plus-one
Monte Carlo p-value `(k + 1)/(n + 1)`. The v1.1 `mc_interval_*` names remain as
backward-compatible aliases, but new output uses
`exceedance_probability_interval_*`.

The historical v10 calibration candidate froze one explicit R control:
`nloptwrap`, `restart_edge = TRUE`, `boundary.tol = 1e-5`,
`calc.derivs = TRUE`, `maxeval = 200000`, and singular fits reported as
messages. There is no claim that this reproduces the old statsmodels/Powell
control. The later v11--v14 work adjudicated the rescue hierarchy and validated
the fixed-usable release implementation; final manuscript reference outputs
are handled separately by release verification.

The v10 pilot did not pass because initial optimizer problems clustered and
five outer tasks exhausted their attempt caps. The completed v11 diagnostic
therefore freezes a v12 calibration-only rescue hierarchy. Every observed and
bootstrap full/null pair first uses the v10 `nloptwrap` control. If that pair is
not primary-usable—including either finite-negative LR class—both models are
refitted with `bobyqa`, `calc.derivs = TRUE`, `maxfun = 1000000`, and
`rhoend = 1e-9`. A primary-usable `bobyqa` pair replaces the failed initial
pair; otherwise the replicate is unresolved and triggers top-up. No pair mixes
optimizers, tight `nloptwrap` is not a production fallback, and
`calc.derivs = FALSE` remains diagnostic-only. This v12 hierarchy was
subsequently carried into the v14 nested-bootstrap validation and the release
implementation.

The v13.3.1 focused power diagnostic applies this same deterministic optimizer
hierarchy to both an unstructured full model and a covariance-constrained
diagnostic comparator. The identical algorithm defines a reproducible statistic
and prevents post-hoc optimizer selection. It does not make the plug-in
parametric bootstrap an exact finite-sample test when nuisance parameters are
estimated. V13.3 therefore estimates DGP-empirically-calibrated LR-statistic
power without an inner bootstrap; nested-bootstrap validation separately validates the nested
bootstrap decision procedure.

Only within the covariance-constrained diagnostic, every negative initial LR
still triggers paired `bobyqa` rescue, after which an otherwise-valid rescued
`small_negative` LR inside the existing likelihood-scale tolerance is recorded
as the inferential boundary value zero. Its raw LR, tolerance, likelihoods, and
floor flag are retained. Material negatives and invalid rescues remain
unresolved. The unstructured Question-1 statistic and all production/bootstrap
engines retain their v12 adjudication unchanged.

V13.3 adds a second, separate constrained-diagnostic-only boundary rule. A
rescued LR with absolute value inside the likelihood tolerance may be recorded
as zero despite exactly one nonzero optimizer code only when model structures
and optimizers are verified, there is no other convergence message, neither
rescue likelihood deteriorates beyond tolerance, and the fitted constrained
slope component is at its numerical boundary. Full raw provenance is retained.

Empirical null calibration is method-specific. For `n` finite null statistics,
the conservative cutoff leaves at most `floor(alpha * n)` statistics strictly
above it and reports the realized size. Pairwise comparisons use the relevant
two-method intersection. The known-error direct statistic is additionally
reported at its exact chi-square critical.

The covariance-constrained comparator fixes the random intercept-slope
covariance to zero. That is a substantive modeling restriction and is not a
free default merely because the covariance is not the investigator's target.
It may be considered only when the restriction is prespecified and defensible
at the chosen time origin.

The v14 nested-bootstrap validation experiment returns to the production unstructured model pair
and validates nested-bootstrap decisions. It uses the audited v13.3.1
unstructured powers `.055`, `.074`, `.169`, and `.425` as uncertain
statistic-level references, with interval-overlap rather than exact point
equality as the compatibility criterion. The rho=.5 cells are paired to rho=0
by holding slope effects, residual draws, design seeds, and bootstrap seeds
fixed. The constrained comparator is excluded from nested-bootstrap validation.

nested-bootstrap validation retains fixed usable inner B. An unresolved inner rescue is documented
and deterministically topped up; its mere existence is not a gate failure.
Every completed outer run also reports worst-case p-value bounds obtained by
treating excluded attempts as all nonexceeding or all exceeding. Unresolved
observed fits are not imputed and enter cell-level support bounds. Attempt-cap
exhaustion remains a failure because the declared usable denominator was not
reached. See `q1-nested-bootstrap-validation-protocol.md`.

Manuscript analyses may use more bootstrap replicates than the ordinary package default when Monte Carlo resolution could affect interpretation.

## 10. Missing-data behavior

The package does not impute missing repeated outcomes or time values.

The decomposition uses the observations actually supplied by the fitted model after its own omission rules. Full/null comparisons must be based on the same retained observations.

Participant-specific design matrices and residual covariance matrices are constructed from each participant's retained schedule. Unequal schedules are therefore part of the calculation rather than an exception handled by a separate approximation.

## 11. Target slope and time-scale contract

The user must identify a specific target random-effect coefficient whose interpretation is recoverable from the fitted design.

The implementation must preserve and report enough information to state:

- the target time variable/basis;
- its units;
- any centering or scaling;
- the reference point/window relevant to interpretation.

Affine recodings of time should leave the source quantities invariant when they represent the same physical slope after the corresponding parameter transformation.

A nonlinear transformation such as log-time, a piecewise slope, spline basis, or interaction generally defines a different estimand. Such a change must not be treated as a simple invariance check.

If the requested target coefficient cannot be identified unambiguously from the fitted model/design, stop rather than guess.

## 12. Numerical tolerances

Use two conceptually distinct tolerance classes:

1. **identity/equality tolerance** for additivity, analytic checks, and cross-implementation agreement;
2. **fit/boundary diagnostics** obtained from the fitted model and numerical diagnostics rather than a universal absolute variance cutoff.

The initial implementation should define named internal constants for equality checks rather than scattering literals through the code. A provisional default may be `1e-10` for exact linear-algebra identities in deterministic fixtures, with looser test-specific tolerances for fitted-model comparisons where optimizer/platform variation is expected.

Do not define a universal absolute threshold below which the target slope variance is automatically set to zero.

## 13. Authority when implementations disagree

The R matrix backend is intended to become the authoritative package implementation, but it does not win disagreements by declaration.

Discrepancies are adjudicated in this order:

1. **analytic ground truth** where available, especially balanced designs with independent residuals and the known `.80 = .20 + .60` fixture;
2. direct inspection of the frozen statistical contract and matrix algebra;
3. comparison with the independent Python implementation;
4. additional hand-constructed or high-precision numerical cases where necessary.

For irregular designs without a simple analytic closed form, agreement between independently written R and Python implementations is evidence, not a substitute for resolving the underlying mathematics.

No published manuscript number is treated as ground truth merely because it was previously reported.

## 14. First implementation milestone

Before higher-level package work, the backend must pass at least:

- known `.80 = .20 + .60` fixture;
- analytic balanced-design direct-reliability check;
- additivity;
- nonnegativity;
- affine shift/rescaling invariance for the same physical slope;
- one-observation participant;
- no-within-person-time-variation participant;
- irregular-schedule fixture;
- near-singular covariance fixture;
- frozen `sleepstudy` reference comparison;
- frozen Beat the Blues irregular-schedule reference comparison.

Only after this milestone should the project proceed to the full Question-1 bootstrap interface, `slope_audit()`, participant stability, planning utilities, and manuscript-facing R workflows.

## 15. First public API freeze

The v14 nested-bootstrap validation experiment completed all declared operational and scientific
gates. Version 0.1 therefore adds a thin public interface without changing the
validated numerical or optimizer internals. The exported functions are
`slope_audit()`, `test_slope_heterogeneity()`, `slope_decomposition()`, and
`slope_design_sensitivity()`.

The reader-facing `slope_audit()` interface requires all six target metadata
fields specified in Section 11. The requirement is strict; there is no
two-field shortcut. It uses the supplied full/null observed fits without hidden
construction or refitting and delegates bootstrap refits to the fixed-usable,
paired-rescue engine validated in nested-bootstrap validation.

The public matrix planning route reports prospective information-source
quantities only. It does not generalize the two nested-bootstrap validation planning-power cells to
arbitrary schedules or residual covariance structures.

`slope_stability()` remains deferred because a public participant/bootstrap
resampling contract has not yet completed validation. It is not exported as a
placeholder. See `docs/public-api-v0.1.md` for frozen signatures and output
fields.
