# slopeaudit

`slopeaudit` audits what person-specific random slopes in Gaussian linear
mixed models are actually informed by. It separates direct within-person
information about change from additional model-assisted information supplied
by the fitted joint model.

The first public interface is intentionally narrow and frozen:

- `slope_audit()` runs the complete reader-facing audit;
- `test_slope_heterogeneity()` answers whether slope heterogeneity is supported;
- `slope_decomposition()` describes the information sources in a fitted model;
- `slope_design_sensitivity()` evaluates information sources under planned
  designs supplied as matrices.

Version 0.1 targets one scalar Gaussian random slope. It does not automate
generalized outcomes, nonlinear change, external prediction, or a public
participant/bootstrap stability analysis.

## Release state

This working tree is the **pre-tag 0.1.0 release candidate** for the AMPPS
first-submission repository. `DESCRIPTION` is intentionally set to `0.1.0` so
final package checks exercise the intended release version. No Git tag, GitHub
release, archive DOI, or final clean-session verification is asserted yet.

The completed 7,992-task Question 1 bootstrap validation study is retained as
historical validation provenance from development package version `0.0.0.9000`.
It must not be relabeled as a `0.1.0` execution. Final release verification
must instead establish the recorded code/provenance bridge to this release
candidate and run the release-level checks required for the first-submission
freeze.

## Installation

From a source checkout, run either:

```sh
R CMD INSTALL .
```

or, in R:

```r
install.packages("path/to/slopeaudit", repos = NULL, type = "source")
```

The package requires R 4.3 or later and imports `lme4`.

## Reproduction from a fresh checkout

The repository includes separate reproducibility environments for the R package
and the Python reproduction scripts. From a fresh checkout:

```bash
Rscript -e 'renv::restore(prompt = FALSE)'
R CMD INSTALL .
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r environment/requirements.txt
python3 reproduction/run_public_smoke.py --output-dir reproduction_smoke_output
```

`reproduction/run_public_smoke.py` uses only public or synthetic inputs and is
a nonreportable code-path check. Exact historical S4/S5 reproduction and the
AABC2 S7/S8 analyses require authorized restricted inputs; their public
contracts and reconstruction instructions are documented under `reproduction/`.
See `reproduction/README.md` for the workflow-by-workflow boundary.

## A five-minute first example

Fit a full model containing the target random slope and a compatible null
model that removes that slope and its covariance parameters. Both models must
use REML and the same fixed-effect design and retained observations.

```r
library(lme4)
library(slopeaudit)

control <- lmerControl(
  optimizer = "nloptwrap",
  optCtrl = list(maxeval = 200000),
  check.conv.singular = "message"
)

full <- lmer(
  Reaction ~ Days + (1 + Days | Subject),
  data = sleepstudy,
  REML = TRUE,
  control = control
)

null <- lmer(
  Reaction ~ Days + (1 | Subject),
  data = sleepstudy,
  REML = TRUE,
  control = control
)

days_target <- list(
  basis_type = "linear",
  outcome_units = "milliseconds",
  time_or_exposure_scale = "1 day",
  origin_or_reference = "Day 0",
  reference_window = "Days 0--9",
  estimand = "participant-specific linear change in Reaction per day"
)
```

The decomposition is fast and can be inspected before starting the bootstrap:

```r
sources <- slope_decomposition(full, "Days", days_target)
sources
```

The frozen `sleepstudy` reference is:

```text
Random-slope source decomposition
  Target: Days
  Participants: 18
  R_direct  = 0.815 (equal-participant mean)
  D_slope2  = 0.849 (equal-participant mean)
  Delta_fit = 0.033 (equal-participant mean)
```

For a balanced independent-residual design, `R_direct` reduces to the familiar
reliability of an ordinary within-person slope. `D_slope2` is the fitted
random-slope determinacy. Their difference, `Delta_fit`, is model-assisted
information. It is not direct evidence that the participant changed.

Run the complete audit when the bootstrap cost is acceptable:

```r
audit <- slope_audit(
  full = full,
  null = null,
  target = "Days",
  target_metadata = days_target,
  B = 999L,
  seed = 20260803L,
  progress = interactive()
)

audit
summary(audit)
```

`B = 999` is the validated public default and gives a minimum plus-one p-value
of `.001`. Smaller values are useful only for smoke tests or exploration. The
package never silently increases `B`; a manuscript workflow may explicitly
rerun at `B = 4999` when a result is floor-limited or its Monte Carlo interval
crosses the reporting threshold.

## How to read the result

| Output | Reader-facing meaning |
|---|---|
| `decision = "supported"` | The fitted data support nonzero population slope heterogeneity under the declared model and bootstrap procedure. |
| `decision = "unsupported"` | Use **STOP—INDIVIDUAL-SLOPE CLAIM**. Say “unsupported by these data,” not “there is no heterogeneity.” |
| `decision = "numerically_ambiguous"` | Worst-case treatment of excluded attempts changes the decision; do not make an individual-slope claim. |
| `R_direct` | Direct within-person information about the declared change target. |
| `D_slope2` | Total determinacy of the fitted random slope under the fitted joint model. |
| `Delta_fit` | Additional model-assisted information: `D_slope2 - R_direct`. |

When Question 1 is unsupported but the fitted slope variance is positive, the
decomposition is returned only as a conditional description of that fitted
model. At an exact fitted zero-variance boundary, it is withheld.

## Common stops before interpretation

- Define all six target fields. The package will not infer the outcome units,
  time scale, origin, window, or substantive estimand.
- Use identical fixed effects, observations, weights, grouping, nuisance
  random effects, and effective optimizer controls in the full/null pair.
- Remove the target slope and every covariance parameter that disappears with
  it in the null model.
- Do not describe an unsupported result as proof of no heterogeneity.
- Do not interpret `Delta_fit` as observed within-person change.
- Do not claim that an `lme4` fit used AR(1). Correlated-residual sensitivity
  is available only through separately justified `R` matrices in
  `slope_design_sensitivity()`.

## Worked guides

- `vignette("getting-started", package = "slopeaudit")`
- `vignette("beat-the-blues", package = "slopeaudit")`
- `vignette("design-sensitivity", package = "slopeaudit")`
- `vignette("interpretation-guide", package = "slopeaudit")`

Copy-paste scripts are installed under `system.file("examples", package =
"slopeaudit")`. The Beat the Blues teaching fixture is available at
`system.file("extdata", "btheb_primary.csv", package = "slopeaudit")`.

## Third-party public-data fixtures

Some public teaching and validation inputs are derived from datasets distributed
by third-party R packages and are not original `slopeaudit` data:

- Beat the Blues observation data are derived from `HSAUR3::BtheB`. The HSAUR3
  package is distributed under GPL-2, and its dataset documentation cites
  Proudfoot et al. (2003) as the study source.
- `sleepstudy` observation data are derived from `lme4::sleepstudy`. The `lme4`
  package is distributed under GPL (>= 2), and its dataset documentation cites
  Belenky et al. (2003) as the study source.

The MIT license in `DESCRIPTION`/`LICENSE` does not relicense these
third-party-derived observation data. Retain the upstream package licensing and
source attribution when redistributing the derived fixture files. Exact
redistributed paths and full applicable GPL texts are provided in
`inst/THIRD_PARTY_NOTICES.md` and `inst/licenses/`; the validation-fixture
READMEs retain analysis-specific provenance.

## Validation status

The numerical backend and the fixed-usable Question-1 bootstrap completed the
full v14 Question 1 bootstrap validation study: all four null-size cells and all four planning
power cells passed their operational and reference gates. The four-cell pooled
null support rate was 215/3,996 = 0.0538 with exact 95% interval
[0.0470, 0.0613].

The statistical contract, internal validation protocols, and manuscript
reproduction materials remain in the source repository. This reader-facing
layer does not change the v14 numerical engine or the frozen v0.1 API.
