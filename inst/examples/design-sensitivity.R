# Prospective information-source comparison on an annual-slope scale.

library(slopeaudit)

make_Z <- function(time_in_years) {
  cbind(
    intercept = 1,
    time = time_in_years - mean(time_in_years)
  )
}

Z <- list(
  three_waves_two_years = make_Z(c(0, 1, 2)),
  five_waves_five_years = make_Z(seq(0, 5, length.out = 5)),
  ten_daily = make_Z((0:9) / 365),
  thirty_day_EMA = make_Z((0:29) / 365)
)

intercept_variance <- 0.5
slope_variance <- 0.01
rho <- 0.5
covariance <- rho * sqrt(intercept_variance * slope_variance)
G <- matrix(
  c(intercept_variance, covariance, covariance, slope_variance),
  nrow = 2,
  byrow = TRUE,
  dimnames = list(c("intercept", "time"), c("intercept", "time"))
)
R <- lapply(Z, function(z) diag(1, nrow(z)))

annual_target <- list(
  basis_type = "linear",
  outcome_units = "standardized outcome units",
  time_or_exposure_scale = "1 year",
  origin_or_reference = "schedule midpoint",
  reference_window = "design-specific planned follow-up window",
  estimand = "participant-specific linear change per year"
)

design <- slope_design_sensitivity(
  Z = Z,
  G = G,
  R = R,
  target = "time",
  target_metadata = annual_target
)

print(design)
print(
  design$design_table[
    c("design_id", "R_direct", "D_slope2", "Delta_fit")
  ]
)
