# Beat the Blues teaching fixture

`btheb_primary.csv` is the reference teaching fixture used by the
reader-facing irregular-schedule example and the manuscript's primary
homogeneous-independent-residual reference analysis.

- 100 participants
- 380 retained observations
- scheduled months 0, 2, 3, 5, and 8
- `month_c = month - 4`
- three participants with only one retained observation

Columns:

- `subject`: participant identifier;
- `bdi`: observed Beck Depression Inventory score;
- `month`: elapsed month;
- `month_c`: centered month used for the random slope;
- `occasion`: categorical scheduled occasion used in the fixed effects;
- `treatment`: `TAU` or `BtheB`.

The worked example uses
`bdi ~ treatment * occasion + (1 + month_c | subject)` with REML and
homogeneous independent residuals. It does not represent AR(1),
occasion-specific residual variances, a causal treatment-effect analysis, or
a general claim about attrition.
## Source, attribution, and license

`btheb_primary.csv` is a deterministic long-form distillation of the public
`BtheB` dataset distributed with the HSAUR3 R package. HSAUR3 is distributed
under GPL-2. Its `BtheB` documentation cites the original study as Proudfoot,
Goldberg, Mann, Everitt, Marks, and Gray (2003), *Psychological Medicine*,
33(2), 217--227.

This CSV is third-party-derived observation data, not original `slopeaudit`
data, and is not relicensed under the package's MIT license. Redistribution
should preserve the HSAUR3 GPL-2 terms and the study/source attribution.
The exact path-specific redistribution notice is in
`inst/THIRD_PARTY_NOTICES.md`; the complete GPL-2 text is bundled at
`inst/licenses/GPL-2.0.txt`.
