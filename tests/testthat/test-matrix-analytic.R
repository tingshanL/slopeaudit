test_that("known .80 = .20 + .60 fixture is recovered", {
  spec <- read_fixture_json("known_802060", "specification.json")
  exp <- read_fixture_json("known_802060", "expected.json")

  Z <- json_matrix(spec$participant$Z_i)
  H <- json_matrix(spec$participant$H_i)
  R <- json_matrix(spec$participant$R_i)
  G <- json_matrix(spec$G)

  out <- slopeaudit:::.sa_participant_source_decomposition(
    Z = Z,
    G = G,
    R = R,
    target_index = 2,
    H = H,
    participant_id = spec$participant$id
  )

  tol <- as.numeric(exp$tolerance$absolute)

  expect_num_equal(
    out$direct_contrast,
    json_vector(exp$expected_intermediate$direct_contrast),
    tol
  )
  expect_num_equal(
    out$direct_error_variance,
    exp$expected_intermediate$direct_error_variance,
    tol
  )
  expect_num_equal(out$R_direct, exp$expected_primary$R_direct_i, tol)
  expect_num_equal(out$D_slope2, exp$expected_primary$D_slope2_i, tol)
  expect_num_equal(out$Delta_fit, exp$expected_primary$Delta_fit_i, tol)

  expected_C <- json_matrix(
    exp$expected_intermediate$conditional_random_effect_covariance
  )
  expect_equal(
    out$conditional_random_effect_covariance,
    expected_C,
    tolerance = tol
  )

  expect_false(out$diagnostics$Delta_fit_materially_negative)
})


test_that("irregular analytic oracle is recovered participant by participant", {
  spec <- read_fixture_json("irregular_hand_oracle", "specification.json")
  exp <- read_fixture_json("irregular_hand_oracle", "expected.json")

  G <- json_matrix(spec$model$G)
  sigma2 <- as.numeric(spec$model$residual_variance)

  Z_list <- lapply(spec$participants, function(p) {
    build_random_intercept_slope_design(json_vector(p$time))
  })
  R_list <- lapply(Z_list, function(Z) diag(sigma2, nrow(Z)))
  ids <- vapply(spec$participants, `[[`, character(1), "id")

  out <- slopeaudit:::.sa_matrix_decompose(
    Z = Z_list,
    G = G,
    R = R_list,
    target_index = 2,
    participant_id = ids
  )

  tol <- as.numeric(exp$tolerance$absolute)

  for (i in seq_along(exp$participants)) {
    e <- exp$participants[[i]]
    got <- out$participants[[i]]

    expect_identical(got$participant_id, e$id)
    expect_num_equal(got$direct_contrast, json_vector(e$direct_contrast), tol)
    expect_num_equal(got$direct_error_variance, e$direct_error_variance, tol)
    expect_num_equal(got$R_direct, e$R_direct, tol)
    expect_num_equal(
      got$posterior_slope_variance,
      e$posterior_slope_variance,
      tol
    )
    expect_num_equal(got$D_slope2, e$D_slope_squared, tol)
    expect_num_equal(got$Delta_fit, e$Delta_fit, tol)
  }

  expect_num_equal(
    out$summary$mean_participant_R_direct,
    exp$equal_participant_means$R_direct,
    tol
  )
  expect_num_equal(
    out$summary$mean_participant_D_slope2,
    exp$equal_participant_means$D_slope_squared,
    tol
  )
  expect_num_equal(
    out$summary$mean_participant_Delta_fit,
    exp$equal_participant_means$Delta_fit,
    tol
  )
})


test_that("zero-direct-information edge cases behave as contracted", {
  ids <- c(
    "zero_direct_single_observation",
    "zero_direct_no_time_variation"
  )

  for (id in ids) {
    spec <- read_fixture_json(id, "specification.json")
    exp <- read_fixture_json(id, "expected.json")

    G <- json_matrix(spec$model$G)
    sigma2 <- as.numeric(spec$model$residual_variance)
    p <- spec$participants[[1]]
    Z <- build_random_intercept_slope_design(json_vector(p$time))

    out <- slopeaudit:::.sa_participant_source_decomposition(
      Z = Z,
      G = G,
      R = diag(sigma2, nrow(Z)),
      target_index = 2,
      participant_id = p$id
    )

    e <- exp$participants[[1]]
    tol <- as.numeric(exp$tolerance$absolute)

    expect_false(out$has_direct_slope_information)
    expect_true(is.infinite(out$direct_error_variance))
    expect_num_equal(out$R_direct, e$R_direct, tol)
    expect_num_equal(out$D_slope2, e$D_slope_squared, tol)
    expect_num_equal(out$Delta_fit, e$Delta_fit, tol)
    expect_num_equal(out$Delta_fit, out$D_slope2, tol)
  }
})


test_that("near-singular positive target variance remains finite", {
  spec <- read_fixture_json("near_singular_synthetic", "specification.json")
  exp <- read_fixture_json("near_singular_synthetic", "expected.json")

  G <- json_matrix(spec$model$G)
  sigma2 <- as.numeric(spec$model$residual_variance)
  p <- spec$participants[[1]]
  Z <- build_random_intercept_slope_design(json_vector(p$time))

  out <- slopeaudit:::.sa_participant_source_decomposition(
    Z = Z,
    G = G,
    R = diag(sigma2, nrow(Z)),
    target_index = 2,
    participant_id = p$id
  )

  e <- exp$participants[[1]]
  tol <- as.numeric(exp$tolerance$absolute)

  expect_true(all(is.finite(c(
    out$R_direct,
    out$D_slope2,
    out$Delta_fit,
    out$posterior_slope_variance
  ))))
  expect_gt(out$target_slope_variance, 0)

  expect_num_equal(out$R_direct, e$R_direct, tol)
  expect_num_equal(out$D_slope2, e$D_slope_squared, tol)
  expect_num_equal(out$Delta_fit, e$Delta_fit, tol)
})


test_that("balanced direct reliability matches the classical closed form", {
  pars <- read_fixture_json("sleepstudy_primary", "fitted_parameters.json")
  dat <- utils::read.csv(
    file.path(fixture_dir("sleepstudy_primary"), "analysis_data.csv"),
    stringsAsFactors = FALSE
  )

  first_id <- unique(dat$Subject)[1]
  d <- dat[dat$Subject == first_id, , drop = FALSE]

  G <- json_matrix(pars$random_effects_covariance_G$matrix)
  sigma2 <- as.numeric(pars$residual$variance)
  tau2 <- G[2, 2]

  Sxx <- sum((d$Days - mean(d$Days))^2)
  v_closed <- sigma2 / Sxx
  R_closed <- tau2 / (tau2 + v_closed)

  Z <- build_random_intercept_slope_design(d$Days)
  out <- slopeaudit:::.sa_participant_source_decomposition(
    Z = Z,
    G = G,
    R = diag(sigma2, nrow(Z)),
    target_index = 2
  )

  expect_equal(out$direct_error_variance, v_closed, tolerance = 1e-10)
  expect_equal(out$R_direct, R_closed, tolerance = 1e-10)
})


test_that("affine recoding preserves source quantities for the same physical slope", {
  spec <- read_fixture_json("irregular_hand_oracle", "specification.json")
  G <- json_matrix(spec$model$G)
  sigma2 <- as.numeric(spec$model$residual_variance)

  t <- json_vector(spec$participants[[2]]$time)
  Z <- build_random_intercept_slope_design(t)
  R <- diag(sigma2, length(t))

  original <- slopeaudit:::.sa_participant_source_decomposition(
    Z = Z,
    G = G,
    R = R,
    target_index = 2
  )

  # t = a + b*u, so b* = T b with
  # T = [[1, a], [0, b]] and G* = T G T'.
  a <- 3
  b <- 2
  u <- (t - a) / b
  Tmat <- matrix(c(1, 0, a, b), nrow = 2, byrow = FALSE)
  G_star <- Tmat %*% G %*% t(Tmat)
  Z_star <- build_random_intercept_slope_design(u)

  recoded <- slopeaudit:::.sa_participant_source_decomposition(
    Z = Z_star,
    G = G_star,
    R = R,
    target_index = 2
  )

  expect_equal(recoded$R_direct, original$R_direct, tolerance = 1e-10)
  expect_equal(recoded$D_slope2, original$D_slope2, tolerance = 1e-10)
  expect_equal(recoded$Delta_fit, original$Delta_fit, tolerance = 1e-10)
})


test_that("all analytic first-milestone fixtures respect nonnegativity", {
  ids <- c(
    "irregular_hand_oracle",
    "zero_direct_single_observation",
    "zero_direct_no_time_variation",
    "near_singular_synthetic"
  )

  for (id in ids) {
    spec <- read_fixture_json(id, "specification.json")
    G <- json_matrix(spec$model$G)
    sigma2 <- as.numeric(spec$model$residual_variance)

    for (p in spec$participants) {
      Z <- build_random_intercept_slope_design(json_vector(p$time))
      out <- slopeaudit:::.sa_participant_source_decomposition(
        Z = Z,
        G = G,
        R = diag(sigma2, nrow(Z)),
        target_index = 2,
        participant_id = p$id
      )
      expect_false(out$diagnostics$Delta_fit_materially_negative)
      expect_gte(out$Delta_fit, -.Machine$double.eps^0.5)
    }
  }
})
