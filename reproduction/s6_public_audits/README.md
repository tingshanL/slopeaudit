# S6 public audit reproductions

This directory contains the Beat the Blues and `sleepstudy` audit/sensitivity
entry points. Each script writes to an explicit `--output-dir`. The
`--smoke-test` mode uses synthetic data and reduced replication counts; smoke
outputs are nonreportable. Dataset-specific regression guards against the
validated real-data optimum are therefore marked not applicable in smoke mode;
generic model-validity and nested-likelihood checks remain active. Full modes
use pinned public source artifacts and/or explicitly supplied public-data paths.

## Public smoke run

```bash
python3 reproduction/s6_public_audits/beat_the_blues_primary.py \
  --output-dir reproduction_smoke_output/s6/btheb_primary \
  --smoke-test

python3 reproduction/s6_public_audits/beat_the_blues_logtime.py \
  --output-dir reproduction_smoke_output/s6/btheb_logtime \
  --primary-results-dir reproduction_smoke_output/s6/btheb_primary \
  --smoke-test

python3 reproduction/s6_public_audits/sleepstudy_primary.py \
  --output-dir reproduction_smoke_output/s6/sleepstudy_primary \
  --smoke-test

python3 reproduction/s6_public_audits/sleepstudy_ar1_profile.py \
  --output-dir reproduction_smoke_output/s6/sleepstudy_ar1 \
  --smoke-test
```

For full `sleepstudy` runs, the scripts first use a verified local cache or an
explicit `SLEEPSTUDY_CSV` path and otherwise attempt the pinned public download.
The Beat the Blues full analyses use the pinned HSAUR3 source archive and verify
its hashes before analysis.

## Beat the Blues completion statuses

`beat_the_blues_primary.py` reports four distinct categories in
`32_repaired_validity_gate.csv`: execution validity, boundary-bootstrap
usability, scientific model adequacy, and reproduction completion. Scientific
adequacy findings are retained as `WARNING` and are not computational release
blockers. A completed full run can therefore report
`PASS_WITH_SCIENTIFIC_WARNING`; smoke runs report
`NONREPORTABLE_SMOKE_COMPLETE` and remain nonreportable. Boundary-bootstrap
replicates excluded for numerical boundary-floor nesting gaps remain explicitly
counted; usability is judged against the prespecified minimum usable rate.

## Public-data provenance

The full public-data workflows preserve their upstream dataset provenance.
Beat the Blues uses `HSAUR3::BtheB` (HSAUR3 package license GPL-2; study source
Proudfoot et al., 2003). `sleepstudy` uses `lme4::sleepstudy` (`lme4` package
license GPL (>= 2); study source Belenky et al., 2003). These public-data
sources are separate from the restricted AABC2 reproduction boundary.

For the five observation-data files redistributed in the repository, exact
path-specific notices and complete applicable GPL texts are bundled in
`inst/THIRD_PARTY_NOTICES.md` and `inst/licenses/`. This does not change the
separate restricted AABC2 reproduction boundary.
