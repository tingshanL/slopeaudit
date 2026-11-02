# Documented uses of fitted person-specific slopes

`practice_examples_codebook.csv` codes the 13 published examples of Supplement S9 of the 2026
manuscript (Supplement S3 of v0.1.0). Each row gives the reference and identifiers, the coding
decisions and a short evidence description for each decision.

Columns added in October 2026 for Table S9.1:

| Column | Values |
|---|---|
| `downstream_purpose` | `association`, `prediction`, `index_development`, `method_comparison`, `one_stage` |
| `heterogeneity_category` | `latent_variance` (random-slope variance tested or reported as a result), `fitted_score_spread` (only the spread of fitted slopes described), `modeled_not_reported` |
| `shared_source_inputs` | another downstream input derived from the observations used to estimate the slope (`yes`, `no`, `not_applicable`) |
| `later_same_outcome_endpoint` | downstream outcome was a later measurement of the same variable (`yes`, `no`, `not_applicable`) |
| `coding_revision` | what changed in that row |

`precision_or_reliability_reported` for Hohman et al. (2017) changed from `no` to
`not_applicable`, because no individual slope scores were used. Forthmann et al. (2022) was
recoded from `fitted_score_spread` to `latent_variance` (and `slope_heterogeneity_evaluated`
from `no` to `yes`), because latent slope SDs with 95% credible intervals are reported; no
example now has `fitted_score_spread`. The Kueser et al. (2021) slope description now names the
child-by-word random linear learning slopes on the logit scale.

Counts over the 13 examples: heterogeneity `latent_variance` 4, `modeled_not_reported` 9;
precision or reliability reported 2 (`no` 10, `not_applicable` 1); purpose `association` 7,
`prediction` 3, `method_comparison` 1, `index_development` 1, `one_stage` 1. The earlier binary columns are
kept; `slope_heterogeneity_evaluated = yes` corresponds exactly to
`heterogeneity_category = latent_variance`.

The evidence columns describe the basis for each decision; they do not give page locators.
