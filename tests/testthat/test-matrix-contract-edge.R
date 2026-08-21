test_that("weak but estimable direct information is not thresholded to zero", {
  G <- diag(c(1, 1))
  R <- diag(2)

  for (delta_time in c(1e-4, 1e-12, 1e-20)) {
    Z <- cbind(Intercept = 1, target_slope = c(0, delta_time))

    out <- slopeaudit:::.sa_participant_source_decomposition(
      Z = Z,
      G = G,
      R = R,
      target_index = 2
    )

    expected_J <- delta_time^2 / 2

    expect_true(out$has_direct_slope_information)
    expect_gt(out$direct_information_J, 0)
    expect_equal(
      out$direct_information_J,
      expected_J,
      tolerance = max(1e-15 * expected_J, .Machine$double.xmin)
    )
    expect_gt(out$R_direct, 0)
  }
})


test_that("explicit H must span the nuisance space implied by Z", {
  Z <- cbind(Intercept = 1, target_slope = c(0, 1, 2, 3))
  G <- matrix(c(1, 0.2, 0.2, 1), 2, 2)
  R <- diag(4)

  expect_error(
    slopeaudit:::.sa_participant_source_decomposition(
      Z = Z,
      G = G,
      R = R,
      target_index = 2,
      H = matrix(numeric(0), nrow = 4, ncol = 0)
    ),
    "span exactly"
  )

  # A different basis for the same intercept nuisance space is acceptable.
  out <- slopeaudit:::.sa_participant_source_decomposition(
    Z = Z,
    G = G,
    R = R,
    target_index = 2,
    H = matrix(2, nrow = 4, ncol = 1)
  )
  expect_false(out$diagnostics$Delta_fit_materially_negative)
})


test_that("non-diagonal R and multiple nuisance random effects are supported", {
  time <- c(0, 1, 3, 4)
  nuisance2 <- c(-1, 0.5, 0.25, 1.5)
  Z <- cbind(
    Intercept = 1,
    nuisance2 = nuisance2,
    target_slope = time
  )

  A <- matrix(c(
    1.0, 0.2, 0.1,
    0.2, 0.8, 0.15,
    0.1, 0.15, 0.6
  ), 3, 3, byrow = TRUE)
  G <- A

  rho <- 0.4
  idx <- seq_along(time)
  R <- 1.7 * outer(idx, idx, function(i, j) rho^abs(i - j))

  out <- slopeaudit:::.sa_participant_source_decomposition(
    Z = Z,
    G = G,
    R = R,
    target_index = 3
  )

  # Independent constrained-GLS expression for the direct contrast.
  H <- Z[, 1:2, drop = FALSE]
  s <- Z[, 3]
  Acon <- cbind(H, s)
  d <- c(0, 0, 1)
  Rinv_A <- solve(R, Acon)
  c_alt <- as.numeric(
    Rinv_A %*% solve(crossprod(Acon, Rinv_A), d)
  )
  v_alt <- as.numeric(crossprod(c_alt, R %*% c_alt))
  R_alt <- G[3, 3] / (G[3, 3] + v_alt)

  # Independent full conditional-covariance expression; G is SPD here.
  C_alt <- solve(solve(G) + crossprod(Z, solve(R, Z)))
  D_alt <- 1 - C_alt[3, 3] / G[3, 3]

  expect_equal(out$direct_contrast, c_alt, tolerance = 1e-10)
  expect_equal(out$direct_error_variance, v_alt, tolerance = 1e-10)
  expect_equal(out$R_direct, R_alt, tolerance = 1e-10)
  expect_equal(out$D_slope2, D_alt, tolerance = 1e-10)
  expect_false(out$diagnostics$Delta_fit_materially_negative)
})


test_that("exactly singular G with positive target variance remains computable", {
  Z <- cbind(Intercept = 1, target_slope = c(-1, 0, 1))
  G <- matrix(1, 2, 2) # rank one; target marginal variance is still 1
  R <- diag(3)

  out <- slopeaudit:::.sa_participant_source_decomposition(
    Z = Z,
    G = G,
    R = R,
    target_index = 2
  )

  expect_gt(out$target_slope_variance, 0)
  expect_true(all(is.finite(c(
    out$R_direct,
    out$D_slope2,
    out$Delta_fit,
    out$posterior_slope_variance
  ))))
  expect_equal(out$R_direct, 2 / 3, tolerance = 1e-10)
  expect_equal(out$D_slope2, 5 / 6, tolerance = 1e-10)
  expect_equal(out$Delta_fit, 1 / 6, tolerance = 1e-10)
})


test_that("malformed target index and matrix dimensions fail informatively", {
  Z <- cbind(Intercept = 1, target_slope = c(-1, 0, 1))
  G <- diag(2)
  R <- diag(3)

  expect_error(
    slopeaudit:::.sa_participant_source_decomposition(
      Z, G, R, target_index = NA_real_
    ),
    "target_index"
  )

  expect_error(
    slopeaudit:::.sa_participant_source_decomposition(
      Z, matrix(1, 1, 1), R, target_index = 2
    ),
    "G"
  )
})
