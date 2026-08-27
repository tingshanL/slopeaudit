# Reader-facing output and terminology guide

This guide defines terminology for the README, vignettes, examples, and
manuscript-facing output for the version 0.1 API.

## Required terms

| Code term | Reader-facing term | Do not substitute |
|---|---|---|
| `R_direct` | direct within-person slope information or direct slope reliability | observed individual change when the direct channel is absent |
| `D_slope2` | fitted random-slope determinacy | direct reliability |
| `Delta_fit` | additional model-assisted information | extra observed change information |
| `unsupported` | unsupported by these data | no heterogeneity |
| non-supported Q1 branch | insufficient evidence for an individual-slope claim | proof that heterogeneity is absent |

## Required target declaration

Reader-facing code must supply these software field names exactly:

1. `basis_type`;
2. `outcome_units`;
3. `time_or_exposure_scale`;
4. `origin_or_reference`;
5. `reference_window`;
6. `estimand`.

For `nonlinear`, `piecewise`, `interaction`, or `other`, also supply
`basis_expression`.

## Q1 reporting template

> The target was [estimand] in [units] per [scale] over [window], with
> [origin]. We compared the REML models [full formula] and [null formula] using
> a fixed-usable parametric bootstrap (`B = ...`, seed `...`). The observed LR
> was ..., the plus-one p-value was ..., and the worst-case numerical-exclusion
> bounds were [..., ...]. Slope heterogeneity was [supported / unsupported by
> these data / numerically ambiguous].

## Q2 reporting template

> Under the fitted model, the equal-participant mean direct slope reliability
> was ..., total fitted slope determinacy was ..., and additional
> model-assisted information was .... [N] participants had no direct slope
> information. These are plug-in, residual-model-dependent descriptions of the
> analysis sample.

## Escalation language

The validated default is `B = 999`. If that run is floor-limited or its Monte
Carlo interval crosses the reporting alpha, rerun explicitly with a larger
value such as `B = 4999` and report both runs and the reason for escalation.
