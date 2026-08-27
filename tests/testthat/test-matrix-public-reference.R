test_that("sleepstudy participant and summary references are reproduced", {
  pars <- read_fixture_json("sleepstudy_primary", "fitted_parameters.json")
  exp_summary <- read_fixture_json("sleepstudy_primary", "expected_summary.json")

  dat <- utils::read.csv(
    file.path(fixture_dir("sleepstudy_primary"), "analysis_data.csv"),
    stringsAsFactors = FALSE
  )
  expected <- utils::read.csv(
    file.path(
      fixture_dir("sleepstudy_primary"),
      "expected_participant_sources.csv"
    ),
    stringsAsFactors = FALSE
  )

  G <- json_matrix(pars$random_effects_covariance_G$matrix)
  sigma2 <- as.numeric(pars$residual$variance)

  ids <- unique(as.character(dat$Subject))
  Z_list <- lapply(ids, function(id) {
    d <- dat[as.character(dat$Subject) == id, , drop = FALSE]
    build_random_intercept_slope_design(d$Days)
  })
  R_list <- lapply(Z_list, function(Z) diag(sigma2, nrow(Z)))

  out <- slopeaudit:::.sa_matrix_decompose(
    Z = Z_list,
    G = G,
    R = R_list,
    target_index = 2,
    participant_id = ids
  )

  got <- out$participant_table
  expected$Subject <- as.character(expected$Subject)
  expected <- expected[match(got$participant_id, expected$Subject), ]

  expect_equal(got$R_direct, expected$R_direct, tolerance = 1e-10)
  expect_equal(
    got$D_slope2,
    expected$D_slope_squared,
    tolerance = 1e-10
  )
  expect_equal(got$Delta_fit, expected$Delta_fit, tolerance = 1e-10)
  expect_equal(
    got$direct_error_variance,
    expected$direct_error_variance,
    tolerance = 1e-10
  )

  expect_equal(
    out$summary$mean_participant_R_direct,
    as.numeric(exp_summary$mean_participant_R_direct),
    tolerance = 1e-10
  )
  expect_equal(
    out$summary$mean_participant_D_slope2,
    as.numeric(exp_summary$mean_participant_D_slope_squared),
    tolerance = 1e-10
  )
  expect_equal(
    out$summary$mean_participant_Delta_fit,
    as.numeric(exp_summary$mean_participant_Delta_fit),
    tolerance = 1e-10
  )
  expect_equal(out$summary$n_no_direct_slope_information, 0)
})


test_that("Beat the Blues irregular reference is reproduced", {
  pars <- read_fixture_json("btheb_primary", "fitted_parameters.json")
  exp_summary <- read_fixture_json("btheb_primary", "expected_summary.json")

  dat <- utils::read.csv(
    file.path(fixture_dir("btheb_primary"), "analysis_data.csv"),
    stringsAsFactors = FALSE
  )
  expected <- utils::read.csv(
    file.path(
      fixture_dir("btheb_primary"),
      "expected_participant_sources.csv"
    ),
    stringsAsFactors = FALSE
  )

  G <- json_matrix(pars$random_effects_covariance_G$matrix)
  sigma2 <- as.numeric(pars$residual$variance)

  ids <- unique(as.character(dat$subject))
  Z_list <- lapply(ids, function(id) {
    d <- dat[as.character(dat$subject) == id, , drop = FALSE]
    build_random_intercept_slope_design(d$month_c)
  })
  R_list <- lapply(Z_list, function(Z) diag(sigma2, nrow(Z)))

  out <- slopeaudit:::.sa_matrix_decompose(
    Z = Z_list,
    G = G,
    R = R_list,
    target_index = 2,
    participant_id = ids
  )

  got <- out$participant_table
  expected$subject <- as.character(expected$subject)
  expected <- expected[match(got$participant_id, expected$subject), ]

  expect_equal(got$R_direct, expected$R_direct, tolerance = 1e-10)
  expect_equal(
    got$D_slope2,
    expected$D_slope_squared,
    tolerance = 1e-10
  )
  expect_equal(got$Delta_fit, expected$Delta_fit, tolerance = 1e-10)

  finite_expected <- is.finite(expected$direct_error_variance)
  expect_equal(
    got$direct_error_variance[finite_expected],
    expected$direct_error_variance[finite_expected],
    tolerance = 1e-10
  )
  expect_true(all(is.infinite(
    got$direct_error_variance[!finite_expected]
  )))

  expect_equal(
    out$summary$n_no_direct_slope_information,
    as.integer(exp_summary$participants_with_no_direct_slope_information)
  )
  expect_equal(
    out$summary$mean_participant_R_direct,
    as.numeric(exp_summary$mean_participant_R_direct),
    tolerance = 1e-10
  )
  expect_equal(
    out$summary$mean_participant_D_slope2,
    as.numeric(exp_summary$mean_participant_D_slope_squared),
    tolerance = 1e-10
  )
  expect_equal(
    out$summary$mean_participant_Delta_fit,
    as.numeric(exp_summary$mean_participant_Delta_fit),
    tolerance = 1e-10
  )

  r_desc <- exp_summary$participant_R_direct_descriptives
  expect_equal(
    as.numeric(stats::quantile(got$R_direct, 0.25)),
    as.numeric(r_desc$q1),
    tolerance = 1e-10
  )
  expect_equal(
    stats::median(got$R_direct),
    as.numeric(r_desc$median),
    tolerance = 1e-10
  )
  expect_equal(
    as.numeric(stats::quantile(got$R_direct, 0.75)),
    as.numeric(r_desc$q3),
    tolerance = 1e-10
  )
  expect_equal(
    stats::IQR(got$R_direct),
    as.numeric(r_desc$iqr),
    tolerance = 1e-10
  )
  expect_equal(
    range(got$R_direct),
    c(as.numeric(r_desc$minimum), as.numeric(r_desc$maximum)),
    tolerance = 1e-10
  )

  schedule_counts <- table(expected$n_obs)
  frozen_counts <- unlist(exp_summary$observations_per_participant_counts)
  expect_equal(
    as.integer(schedule_counts[names(frozen_counts)]),
    as.integer(frozen_counts)
  )

  one_observation <- expected$n_obs == 1L
  expect_equal(sum(one_observation), 3L)
  expect_true(all(got$R_direct[one_observation] == 0))
  expect_equal(
    got$D_slope2[one_observation],
    rep(
      as.numeric(exp_summary$single_observation_D_slope_squared),
      3L
    ),
    tolerance = 1e-10
  )
  expect_equal(
    got$Delta_fit[one_observation],
    got$D_slope2[one_observation],
    tolerance = 1e-10
  )
})
