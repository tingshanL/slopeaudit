design_estimand_metadata <- function() {
  list(
    basis_type = "linear",
    outcome_units = "outcome units",
    time_or_exposure_scale = "1 planned time unit",
    origin_or_reference = "centered time 0",
    reference_window = "planned observation window",
    estimand = "participant-specific linear change per planned time unit"
  )
}


test_that("public design sensitivity recovers .80 = .20 + .60", {
  Z <- matrix(c(1, -0.5, 1, 0.5), nrow = 2L, byrow = TRUE)
  colnames(Z) <- c("intercept", "time")
  G <- matrix(c(18, sqrt(15), sqrt(15), 1), nrow = 2L, byrow = TRUE)
  dimnames(G) <- list(colnames(Z), colnames(Z))
  R <- diag(2, 2L)

  out <- slope_design_sensitivity(
    Z = list(analytic_001 = Z),
    G = G,
    R = R,
    target = "time",
    target_metadata = design_estimand_metadata()
  )

  expect_s3_class(out, "slopeaudit_design_sensitivity")
  expect_identical(out$target, "time")
  expect_identical(out$design_table$design_id, "analytic_001")
  expect_equal(out$design_table$R_direct, 0.2, tolerance = 1e-10)
  expect_equal(out$design_table$D_slope2, 0.8, tolerance = 1e-10)
  expect_equal(out$design_table$Delta_fit, 0.6, tolerance = 1e-10)
  expect_equal(
    out$design_table$D_slope2,
    out$design_table$R_direct + out$design_table$Delta_fit,
    tolerance = 1e-10
  )
  expect_output(print(out), "Prospective random-slope design sensitivity")
})


test_that("public design sensitivity preserves affine-time invariance", {
  t <- c(0, 1, 4, 7)
  Z <- cbind(intercept = 1, time = t)
  G <- matrix(c(4, 0.5, 0.5, 1), nrow = 2L, byrow = TRUE)
  dimnames(G) <- list(colnames(Z), colnames(Z))
  R <- diag(2, length(t))

  original <- slope_design_sensitivity(
    Z = list(original = Z),
    G = G,
    R = R,
    target = "time",
    target_metadata = design_estimand_metadata()
  )

  a <- 3
  b <- 2
  u <- (t - a) / b
  Z_star <- cbind(intercept = 1, time = u)
  Tmat <- matrix(c(1, 0, a, b), nrow = 2L, byrow = FALSE)
  G_star <- Tmat %*% G %*% t(Tmat)
  dimnames(G_star) <- list(colnames(Z_star), colnames(Z_star))

  recoded <- slope_design_sensitivity(
    Z = list(recoded = Z_star),
    G = G_star,
    R = R,
    target = "time",
    target_metadata = design_estimand_metadata()
  )

  expect_equal(
    recoded$design_table$R_direct,
    original$design_table$R_direct,
    tolerance = 1e-10
  )
  expect_equal(
    recoded$design_table$D_slope2,
    original$design_table$D_slope2,
    tolerance = 1e-10
  )
  expect_equal(
    recoded$design_table$Delta_fit,
    original$design_table$Delta_fit,
    tolerance = 1e-10
  )
})


test_that("public design sensitivity preserves boundary and nuisance contracts", {
  Z <- cbind(intercept = 1, time = c(0, 1, 2))
  R <- diag(1, nrow(Z))

  G_boundary <- diag(c(1, 0))
  dimnames(G_boundary) <- list(colnames(Z), colnames(Z))
  expect_error(
    slope_design_sensitivity(
      Z = list(one = Z),
      G = G_boundary,
      R = R,
      target = "time",
      target_metadata = design_estimand_metadata()
    ),
    class = "slopeaudit_target_boundary_error"
  )

  G <- diag(c(1, 1))
  dimnames(G) <- list(colnames(Z), colnames(Z))
  wrong_H <- list(matrix(numeric(), nrow = nrow(Z), ncol = 0L))
  expect_error(
    slope_design_sensitivity(
      Z = list(one = Z),
      G = G,
      R = R,
      target = "time",
      target_metadata = design_estimand_metadata(),
      H = wrong_H
    ),
    "span exactly the nuisance random-effect space"
  )
})
