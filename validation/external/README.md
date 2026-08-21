# External verification evidence

This directory documents the verification contract for the pre-tag `0.1.0`
first-submission release candidate. Large historical simulation outputs,
restricted AABC2-derived inputs, participant-level diagnostics, and local
release-preparation logs are not distributed here.

The release-preparation evidence archive must retain, at minimum:

- SHA-256 hashes for the exact stageable public tree and every archived
  verification artifact;
- Python and R session/environment records;
- the final package documentation, test, build, and `R CMD check --as-cran`
  logs;
- compact fixed-seed S4 and S5 smoke/repeatability evidence;
- deterministic S4 full-summary equivalence against the frozen
  62,000-replicate operating-study evidence;
- deterministic S5 full-summary equivalence against the frozen 8,000-task
  downstream-simulation evidence;
- deterministic S5 table regeneration/equivalence;
- Figure 2 regeneration from frozen summaries without refitting models;
- final reruns of the public terminology, release-state, path-leakage, and
  independent-runnability validators.

Historical expensive simulations are not rerun merely to prepare the release.
Their frozen outputs remain authoritative scientific evidence; release
verification reruns only compact fixed-seed checks and deterministic
post-processing/equivalence checks.

Restricted/private evidence is referenced only by hashes and abstract evidence
roles. Public verification records must not serialize local absolute paths,
participant-level values, or restricted-derived payloads.

The immutable commit/tag and clean-checkout verification are necessarily
created only after the first release commit exists. Until then the repository
correctly remains a pre-tag release candidate.

## S5 publication-table equivalence

The S5 release verifier distinguishes publication output from archival
floating-point serialization. All 13 regenerated TeX publication tables
must be byte-identical to the frozen references.

For the corresponding 13 CSV tables, row/column structure and all
nonnumeric content must match exactly. Numeric cells are compared with
absolute and relative tolerances of `1e-12`. This narrow tolerance exists
only to accommodate last-bit floating-point representation or reduction
differences; it does not permit changes to counts, categories, ordering,
missingness, interpretation labels, or displayed publication values.

The full recovered S5 analytical summaries remain subject to byte-identical
equivalence.
