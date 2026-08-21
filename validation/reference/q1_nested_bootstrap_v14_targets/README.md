# Frozen nested-bootstrap validation references

The four planning-power references come from the independently audited
v13.3.1 extended unstructured-Q1 statistic diagnostic. They use 1,000
alternative datasets per cell and design-specific empirical null calibration
based on 10,000 null datasets per design (9,999 usable unstructured statistics
for the five-wave null). The one missing five-wave null statistic cannot alter
any alternative rejection count at the bounded critical value.

These values are statistic-level references, not nested-bootstrap truth. nested-bootstrap validation estimates the power of the production unstructured-Q1 nested bootstrap and
tests whether its binomial interval is compatible with the independently
estimated reference interval. The four null-size rows use the nominal alpha of
.05.

The rho=.5 rows are paired to their rho=0 partners: participant slope effects
and residual draws are held fixed, while only the random-intercept construction
changes. The constrained covariance diagnostic is intentionally absent from
nested-bootstrap validation.
