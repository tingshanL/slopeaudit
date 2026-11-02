# Nuisance share kappa (manuscript Supplement S1.5, Proposition 2, Corollary S1).

participant <- function(Z, G, R, target_index = 2L) {
  slopeaudit:::.sa_participant_source_decomposition(
    Z = Z, G = G, R = R, target_index = target_index
  )
}

rirs_closed_form_kappa <- function(t, G, sigma2) {
  # Corollary S1: random-intercept, random-slope model, R = sigma2 * I.
  n <- length(t)
  tau2 <- G[2, 2]
  S <- G[1, 1] - G[1, 2]^2 / tau2
  Z <- cbind(1, t)
  V <- Z %*% G %*% t(Z) + sigma2 * diag(n)
  w <- solve(V, Z %*% G[, 2])
  D2 <- drop(crossprod(w, Z %*% G[, 2])) / tau2
  J <- sum((t - mean(t))^2) / sigma2
  Rd <- tau2 * J / (1 + tau2 * J)
  (D2 - Rd) * (1 - D2) / (D2 * (1 - Rd)) * S / (S + sigma2 / n)
}

test_that("two-occasion constructions give the analytic kappa", {
  t <- c(-0.5, 0.5)
  Z <- cbind(1, t)

  a <- participant(Z, diag(2), diag(2) / 8)
  expect_equal(a$D_slope2, 0.80, tolerance = 1e-12)
  expect_equal(a$kappa, 0, tolerance = 1e-12)

  G_b <- matrix(c(18, sqrt(15), sqrt(15), 1), 2, 2)
  b <- participant(Z, G_b, 2 * diag(2))
  expect_equal(b$R_direct, 0.20, tolerance = 1e-12)
  expect_equal(b$D_slope2, 0.80, tolerance = 1e-12)
  expect_equal(b$kappa, 0.140625, tolerance = 1e-12)
  expect_equal(b$sqrt_kappa, 0.375, tolerance = 1e-12)
  expect_identical(b$kappa_status, "defined")

  # Diagonal G, y0 = a + e0, y1 = a + s + e1: kappa = .10.
  d <- participant(cbind(1, c(0, 1)), diag(2), diag(2))
  expect_equal(d$D_slope2, 0.40, tolerance = 1e-12)
  expect_equal(d$kappa, 0.10, tolerance = 1e-12)
})

test_that("kappa matches the closed form for random-intercept, random-slope schedules", {
  set.seed(20261030)
  err <- 0
  for (i in seq_len(200)) {
    n <- sample(2:6, 1)
    t <- sort(stats::runif(n, 0, 5)) - stats::runif(1, 0, 4)
    g00 <- stats::runif(1, 0.1, 5)
    tau2 <- stats::runif(1, 0.05, 3)
    rho <- stats::runif(1, -0.95, 0.95)
    G <- matrix(c(g00, rho * sqrt(g00 * tau2), rho * sqrt(g00 * tau2), tau2), 2, 2)
    s2 <- stats::runif(1, 0.1, 4)
    out <- participant(cbind(1, t), G, s2 * diag(n))
    err <- max(err, abs(out$kappa - rirs_closed_form_kappa(t, G, s2)))
  }
  expect_lt(err, 1e-9)
})

test_that("composition and ceilings hold with structured residuals and two nuisance effects", {
  set.seed(7)
  for (i in seq_len(100)) {
    n <- sample(3:6, 1)
    t <- sort(stats::runif(n, 0, 5))
    Z <- cbind(1, (t - 2)^2, t - mean(t))
    A <- matrix(stats::rnorm(9), 3, 3)
    G <- crossprod(A) * stats::runif(1, 0.1, 3)
    B <- matrix(stats::rnorm(n * n), n, n) * 0.3
    R <- diag(stats::runif(n, 0.2, 3)) + crossprod(B)
    out <- participant(Z, G, R, target_index = 3L)
    expect_true(out$diagnostics$numerically_valid)
    expect_lt(abs(out$diagnostics$kappa_composition_error), 1e-8)
    expect_lte(out$kappa, min(1 - out$D_slope2, out$Delta_fit / out$D_slope2) + 1e-9)
    expect_gte(out$kappa, 0)
  }
})

test_that("a rank-one covariance gives zero capacity despite model assistance", {
  t <- c(0, 1, 2, 3)
  G <- 2 * matrix(c(1, 0.3, 0.3, 0.09), 2, 2)
  out <- participant(cbind(1, t), G, diag(4))
  expect_gt(out$Delta_fit, 0.01)
  expect_equal(out$kappa, 0, tolerance = 1e-8)
  expect_false(is.na(out$sqrt_kappa))
  expect_false(out$diagnostics$kappa_materially_negative)
})

test_that("kappa is zero without nuisance effects and undefined at exactly zero fitted variance", {
  t <- c(-1, 0, 1)
  only_slope <- participant(matrix(t, ncol = 1), matrix(1, 1, 1), diag(3), target_index = 1L)
  expect_identical(only_slope$kappa_status, "no_nuisance_effects")
  expect_equal(only_slope$kappa, 0)

  # A target column of zeros carries no information: D2 = 0, kappa undefined.
  flat <- participant(cbind(1, c(0, 0, 0)), diag(2), diag(3))
  expect_equal(flat$D_slope2, 0, tolerance = 1e-12)
  expect_identical(flat$kappa_status, "undefined_zero_fitted_variance")
  expect_true(is.na(flat$kappa))
  expect_true(is.na(flat$sqrt_kappa))
  expect_true(flat$diagnostics$numerically_valid)
})

test_that("public outputs report kappa for sleepstudy", {
  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy, REML = TRUE
  )
  out <- slope_decomposition(fit, "Days", sleepstudy_estimand_metadata())
  expect_true(all(c("kappa", "sqrt_kappa", "kappa_status") %in%
    names(out$participant_table)))
  expect_identical(out$capacity_summary$quantity, c("kappa", "sqrt_kappa"))
  expect_equal(out$capacity_summary$median[[2]], 0.170, tolerance = 2e-3)
  expect_equal(out$capacity_summary$n_undefined[[1]], 0L)
  expect_output(print(out), "sqrt\\(kappa\\)")
})

test_that("a singular covariance with two nuisance effects can still give positive kappa", {
  # Level and slope perfectly correlated, curvature independent: G is singular,
  # but the curvature effect leaves slope-unrelated nuisance variation.
  t <- c(0, 1, 2, 3, 4)
  Z <- cbind(1, (t - 2)^2, t - 2)
  G <- matrix(0, 3, 3)
  G[c(1, 3), c(1, 3)] <- matrix(c(4, 1, 1, 0.25), 2, 2)
  G[2, 2] <- 0.5
  expect_lt(min(eigen(G, symmetric = TRUE, only.values = TRUE)$values), 1e-12)
  out <- participant(Z, G, diag(5), target_index = 3L)
  expect_gt(out$kappa, 1e-4)
  expect_true(out$diagnostics$numerically_valid)
})

test_that("design sensitivity summarizes and prints with an undefined kappa", {
  metadata <- list(
    basis_type = "linear",
    outcome_units = "outcome units",
    time_or_exposure_scale = "1 time unit",
    origin_or_reference = "centered time 0",
    reference_window = "planned observations",
    estimand = "participant-specific linear change per time unit"
  )
  Z <- list(
    spread = cbind(intercept = 1, time = c(-1, 0, 1)),
    flat = cbind(intercept = 1, time = c(0, 0, 0))
  )
  # Uncorrelated level and slope: the flat design then has zero determinacy.
  G <- matrix(c(1, 0, 0, 0.5), 2, 2, dimnames = list(c("intercept", "time"), c("intercept", "time")))
  out <- slope_design_sensitivity(Z = Z, G = G, R = diag(3), target = "time", target_metadata = metadata)
  expect_identical(out$design_table$kappa_status, c("defined", "undefined_zero_fitted_variance"))
  expect_equal(out$capacity_summary$n_undefined, c(1L, 1L))
  expect_equal(out$capacity_summary$n_defined, c(1L, 1L))
  expect_output(print(out), "sqrt\\(kappa\\)")
  expect_output(print(out), "not summarized for 1 participant\\(s\\) \\(undefined: 1\\)")
})

test_that("small positive determinacy still gives the defined kappa", {
  # Regression: D2 is about 6.7e-9, far below the identity tolerance, yet kappa
  # is well defined (2/3) because the fitted slope is driven by the intercept.
  Z <- cbind(1, c(0, 0))
  G <- matrix(c(1, 1e-4, 1e-4, 1), 2, 2)
  R <- diag(2)
  out <- participant(Z, G, R)

  V <- Z %*% G %*% t(Z) + R
  w <- solve(V, Z %*% G[, 2])
  var_shat <- drop(crossprod(w, V %*% w))
  S <- G[1, 1] - G[1, 2]^2 / G[2, 2]
  kappa_direct <- drop(crossprod(Z[, 1], w))^2 * S / var_shat

  expect_equal(out$D_slope2, 2e-8 / 3, tolerance = 1e-6)
  expect_lt(out$D_slope2, slopeaudit:::.sa_identity_tol)
  expect_identical(out$kappa_status, "defined")
  expect_equal(out$kappa, kappa_direct, tolerance = 1e-10)
  expect_equal(out$kappa, (2 / 3) * (1 - 1e-8), tolerance = 1e-10)
  expect_equal(out$sqrt_kappa, sqrt(2 / 3), tolerance = 1e-8)
  expect_equal(out$kappa_residual_share, 1 / 3, tolerance = 1e-8)
  expect_true(out$diagnostics$numerically_valid)
})

test_that("rounding-level cross-covariance is reported as unresolved, not as a value", {
  Z <- cbind(1, c(1e-17, -1e-17, 1e-17))
  out <- participant(Z, diag(2), diag(3))
  expect_identical(out$kappa_status, "unresolved_fitted_variance")
  expect_true(is.na(out$kappa))
  expect_true(is.na(out$sqrt_kappa))
})

test_that("materially negative kappa stays reportable and is counted as invalid", {
  # A G that is not positive semidefinite (rejected by the public interfaces) makes
  # the nuisance component materially negative: with these weights the nuisance
  # loading falls on the effect with negative variance.
  t <- c(0, 1, 2, 4)
  Z <- cbind(1, (t - 2)^2, t - 2)
  G <- diag(c(0.01, -1, 1))
  bad <- slopeaudit:::.sa_nuisance_share(
    Z = Z, G = G, R = diag(4), target_index = 3L, slope_weights = c(0, 1, 0, 0),
    D_slope2 = 0.5, Delta_fit = 0.1
  )
  expect_identical(bad$kappa_status, "invalid_negative_share")
  expect_true(bad$kappa_materially_negative)
  expect_lt(bad$kappa, 0)
  expect_true(is.na(bad$sqrt_kappa))

  good <- participant(cbind(1, c(-0.5, 0.5)), matrix(c(18, sqrt(15), sqrt(15), 1), 2, 2), 2 * diag(2))
  table <- data.frame(
    kappa = c(good$kappa, bad$kappa, NA, NA),
    sqrt_kappa = c(good$sqrt_kappa, bad$sqrt_kappa, NA, NA),
    kappa_status = c("defined", bad$kappa_status,
                     "undefined_zero_fitted_variance", "unresolved_fitted_variance"),
    kappa_composition_failed = c(FALSE, bad$kappa_composition_failed, FALSE, FALSE),
    kappa_ceiling_violated = c(FALSE, bad$kappa_ceiling_violated, FALSE, FALSE),
    stringsAsFactors = FALSE
  )
  summary <- slopeaudit:::.sa_kappa_summary(table)
  expect_equal(summary$n_defined, c(1L, 1L))
  expect_equal(summary$n_invalid, c(1L, 1L))
  expect_equal(summary$n_undefined, c(1L, 1L))
  expect_equal(summary$n_unresolved, c(1L, 1L))
  expect_equal(summary$median, c(0.140625, 0.375), tolerance = 1e-12)
  expect_output(
    slopeaudit:::.sa_print_capacity_summary(summary),
    "not summarized for 3 participant\\(s\\) \\(undefined: 1; unresolved: 1; invalid: 1\\)"
  )

  only_bad <- slopeaudit:::.sa_kappa_summary(table[2, , drop = FALSE])
  expect_true(all(is.na(only_bad$median)))
  expect_equal(only_bad$n_invalid, c(1L, 1L))
  expect_output(slopeaudit:::.sa_print_capacity_summary(only_bad), "no participant with a valid value")
})
