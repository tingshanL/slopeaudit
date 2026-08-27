# Third-party data redistribution notices

The MIT license for `slopeaudit`-authored code and documentation does not
replace the upstream terms applicable to the third-party-derived observation
data listed below. The notice applies to the five observation-data files listed
below. Project-generated
fitted-parameter files, expected-output files, synthetic data, and other
`slopeaudit` validation artifacts are not relicensed by this notice.

## Beat the Blues / `HSAUR3::BtheB`

Upstream package and version: `HSAUR3` 1.0-15.

Upstream package license: GNU General Public License version 2 (GPL-2).

Upstream dataset documentation identifies the study source as Proudfoot,
Goldberg, Mann, Everitt, Marks, and Gray (2003), *Psychological Medicine*,
33(2), 217--227.

The following redistributed observation-data files are deterministic
long-form/distilled representations derived from `HSAUR3::BtheB`:

- `inst/extdata/btheb_primary.csv`
- `validation/reference/btheb_primary/analysis_data.csv`
- `tests/testthat/fixtures/reference/btheb_primary/analysis_data.csv`

To the extent the upstream package license applies to the redistributed
dataset content, preserve the upstream GPL-2 terms and study attribution.
The complete GPL-2 text is bundled at `inst/licenses/GPL-2.0.txt` in the
source tree (and `licenses/GPL-2.0.txt` after package installation).

## `lme4::sleepstudy`

Upstream package and version: `lme4` 2.0-6.

Upstream package license: GPL (>= 2), represented by CRAN as GPL-2 or GPL-3.

The upstream dataset documentation identifies the study source as Belenky
et al. (2003), *Journal of Sleep Research*, 12(1), 1--12.

The following redistributed observation-data files are distilled from
`lme4::sleepstudy`:

- `validation/reference/sleepstudy_primary/analysis_data.csv`
- `tests/testthat/fixtures/reference/sleepstudy_primary/analysis_data.csv`

To the extent the upstream package license applies to the redistributed
dataset content, preserve the upstream GPL-2-or-later terms and study
attribution. Complete GPL-2 and GPL-3 texts are bundled at
`inst/licenses/GPL-2.0.txt` and `inst/licenses/GPL-3.0.txt` in the source
tree (and under `licenses/` after package installation).

## Scope

These notices apply only to the five exact observation-data paths enumerated
above. The main `slopeaudit` package code remains under its stated MIT license.
