# slopeaudit

`slopeaudit` audits what person-specific random slopes in Gaussian linear
mixed models are actually informed by. It separates direct within-person
information about change from additional model-assisted information supplied
by the fitted joint model, and reports how much of each fitted slope's
variance could come from other person-specific effects (the nuisance share
`kappa`).

The package provides four functions:

- `slope_audit()` runs the complete reader-facing audit;
- `test_slope_heterogeneity()` answers whether slope heterogeneity is supported;
- `slope_decomposition()` describes the information sources in a fitted model;
- `slope_design_sensitivity()` evaluates information sources under planned
  designs supplied as matrices.

The exported API covers one scalar Gaussian random slope in Gaussian linear
mixed models, using compatible `lme4` fits or explicitly supplied design and
covariance matrices.

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

`reproduction/run_public_smoke.py` checks the public execution paths using
public or synthetic inputs. It is a software smoke test rather than a
reproduction of the reported S4, S5, or AABC2 estimates. Input requirements for
each workflow are documented under `reproduction/`.

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

The expected output for the bundled `sleepstudy` example is:

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

The validated default is `B = 999`, corresponding to a minimum plus-one
p-value of `.001`. Use smaller values for exploratory or smoke runs. When
Monte Carlo resolution affects interpretation, rerun explicitly with a larger
value such as `B = 4999`.

## How to read the result

| Output | Reader-facing meaning |
|---|---|
| `decision = "supported"` | The fitted data support nonzero population slope heterogeneity under the declared model and bootstrap procedure. |
| `decision = "unsupported"` | The data do not provide sufficient evidence of population slope heterogeneity under the declared analysis. Any returned decomposition describes the fitted model conditionally. |
| `decision = "numerically_ambiguous"` | Worst-case treatment of excluded attempts changes the decision; do not make an individual-slope claim. |
| `R_direct` | Direct within-person information about the declared change target. |
| `D_slope2` | Total determinacy of the fitted random slope under the fitted joint model. |
| `Delta_fit` | Additional model-assisted information: `D_slope2 - R_direct`. |
| `kappa`, `sqrt_kappa` | Share of a participant's fitted-slope variance due to variation in the other person-specific effects that is uncorrelated with the target slope; `sqrt_kappa` is the largest attainable absolute correlation with a person-level variable uncorrelated with the target slope and the residuals. Participant-level values at the fitted parameters, not a bound for a pooled regression; `NA` when determinacy is zero. |

When Question 1 is unsupported but the fitted slope variance is positive, the
decomposition is returned only as a conditional description of that fitted
model. At an exact fitted zero-variance boundary, it is withheld.

## Requirements for interpretation

- Define all six target fields. The package will not infer the outcome units,
  time scale, origin, window, or substantive estimand.
- Use identical fixed effects, observations, weights, grouping, nuisance
  random effects, and effective optimizer controls in the full/null pair.
- Remove the target slope and every covariance parameter that disappears with
  it in the null model.
- Do not describe an unsupported result as proof of no heterogeneity.
- Do not interpret `Delta_fit` as observed within-person change.
- Do not read a small fitted `kappa` as reassurance near a rank-one fitted
  covariance, where it can be zero while model assistance is large, or as a
  check that a downstream regression on fitted slopes is calibrated.
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
third-party-derived observation data. Redistribution of these fixtures should
preserve the upstream licensing terms and source attribution. Exact
redistributed paths and full applicable GPL texts are provided in
`inst/THIRD_PARTY_NOTICES.md` and `inst/licenses/`; the validation-fixture
READMEs retain analysis-specific provenance.

## 2026 manuscript

Code, exact command lines and small result files for the 2026 manuscript are in
`reproduction/pm2026/` and `results/pm2026/`. To check the reported tables and
figures against the saved outputs without refitting:

```bash
python3 reproduction/pm2026/release/check_reported_values.py
```

The v0.1.0 reproduction folders are unchanged. Their operating-study summaries
predate the likelihood audit described in the manuscript's Supplement S5 and are
superseded for reporting by `results/pm2026/`; see `reproduction/pm2026/README.md`.

## Archived simulation and validation outputs

Large outputs are archived in a Zenodo output dataset
([10.5281/zenodo.22092591](https://doi.org/10.5281/zenodo.22092591)): the v0.1.0
S4 operating-study, S5 downstream-simulation and Question 1 bootstrap-validation
outputs, and, in a later version of the same dataset, the full run outputs for
the 2026 manuscript. That dataset holds outputs only; the code of each tagged
release is archived separately as a Zenodo software record. Code, documentation,
compact summaries and reproduction instructions are maintained in this repository.

## Validation status

The v14 Question 1 validation study comprised 7,992 outer tasks. All four
null-size cells and all four planning-power cells met their prespecified
operational and reference criteria. The pooled null support rate was
215/3,996 = 0.0538, with an exact 95% interval of [0.0470, 0.0613].

The study was conducted during package development under version `0.0.0.9000`;
version `0.1.0` contains the same v14-tested numerical and fixed-usable
bootstrap implementation.
