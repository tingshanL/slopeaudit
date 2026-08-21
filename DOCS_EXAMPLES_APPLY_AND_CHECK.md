# Reader-facing documentation and release checks

This file contains the current reader-facing documentation and release checks
for the complete pre-tag `slopeaudit 0.1.0` first-submission candidate.
Earlier in development, this checklist was introduced while the documentation
and teaching layer was being assembled; it is not a claim that the current
repository contains only documentation changes. The current release tree also
contains the public Python reproduction programs and the restricted-input
stubs documented elsewhere in the repository. The four exported R signatures
and the validated numerical engine remain unchanged by the reader-facing
documentation additions.

Run from the directory containing `slopeaudit/`:

```sh
Rscript -e 'devtools::document("slopeaudit")'
Rscript -e 'devtools::test("slopeaudit")'
R CMD build slopeaudit
R CMD check --as-cran slopeaudit_0.1.0.tar.gz
```

Expected gates:

- the generated `.Rd` files include the new public examples;
- all pre-existing API and numerical tests still pass;
- the 18 new teaching-data and example-parsing expectations pass (expected
  total: `PASS 435`);
- four vignettes build;
- `R CMD check --as-cran` reports 0 errors, 0 warnings, and no notes other
  than the accepted CRAN-incoming `New submission` NOTE for this first release.

The two fitted-model example scripts do not run a bootstrap unless
`SLOPEAUDIT_RUN_BOOTSTRAP=true`. This avoids an accidental long computation
while keeping the public default at the validated `B = 999`.
