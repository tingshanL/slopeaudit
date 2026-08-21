test_that("participant numerical diagnostics are retained and rolled up", {
  G <- matrix(c(2.25, 0.60, 0.60, 0.64), 2, 2)

  times <- list(
    c(-1, 0, 1),
    c(0, 1, 3),
    c(-2, -0.5, 2, 4)
  )

  Z <- lapply(times, function(t) cbind(Intercept = 1, target_slope = t))
  R <- lapply(Z, function(z) diag(nrow(z)))

  out <- slopeaudit:::.sa_matrix_decompose(
    Z = Z,
    G = G,
    R = R,
    target_index = 2
  )

  expect_true(all(c(
    "direct_constraint_failed",
    "additivity_failed",
    "Delta_fit_materially_negative",
    "determinacy_below_zero",
    "determinacy_above_one",
    "numerically_valid"
  ) %in% names(out$participant_table)))

  roll <- out$summary$diagnostics
  expect_equal(roll$n_direct_constraint_failures, 0L)
  expect_equal(roll$n_additivity_failures, 0L)
  expect_equal(roll$n_material_negative_Delta, 0L)
  expect_equal(roll$n_determinacy_below_zero, 0L)
  expect_equal(roll$n_determinacy_above_one, 0L)
  expect_true(roll$numerically_valid)
  expect_identical(out$diagnostics$status, "ok")
})
