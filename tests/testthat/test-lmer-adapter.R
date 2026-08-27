test_that("sleepstudy lmerMod adapter reproduces fitted model components", {
  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE
  )

  ext <- slopeaudit:::.sa_extract_lmer_matrices(
    fit,
    target = "Days",
    target_metadata = sleepstudy_estimand_metadata()
  )

  expect_identical(ext$metadata$likelihood, "REML")
  expect_identical(ext$metadata$grouping_factor, "Subject")
  expect_identical(ext$target_name, "Days")
  expect_identical(ext$target_index, 2L)

  expect_equal(length(ext$Z), 18L)
  expect_equal(length(ext$R), 18L)
  expect_equal(length(ext$participant_id), 18L)
  expect_equal(sum(vapply(ext$Z, nrow, integer(1))), stats::nobs(fit))

  # VarCorr matrices carry lme4-specific display attributes ("stddev",
  # "correlation"). The adapter intentionally stores G as an ordinary
  # covariance matrix, so compare numerical entries and structural dimnames.
  G_expected <- as.matrix(lme4::VarCorr(fit)[[1]])

  expect_equal(
    as.numeric(ext$G),
    as.numeric(G_expected),
    tolerance = 1e-12
  )
  expect_identical(dim(ext$G), dim(G_expected))
  expect_identical(dimnames(ext$G), dimnames(G_expected))

  sigma2 <- stats::sigma(fit)^2
  for (Ri in ext$R) {
    expect_equal(Ri, diag(sigma2, nrow(Ri)), tolerance = 1e-12)
  }

  expect_true(ext$estimand$identified)
  expect_identical(ext$estimand$basis_type, "linear")
})


test_that("lmerMod adapter posterior variance matches lme4 condVar semantics", {
  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE
  )

  out <- slopeaudit:::.sa_lmer_decompose(
    fit,
    target = "Days",
    target_metadata = sleepstudy_estimand_metadata()
  )

  re <- lme4::ranef(fit, condVar = TRUE)$Subject
  post_var <- attr(re, "postVar")

  ids <- rownames(re)
  got <- out$participant_table
  got <- got[match(ids, got$participant_id), , drop = FALSE]

  posterior_from_lme4 <- vapply(
    seq_along(ids),
    function(i) post_var[2, 2, i],
    numeric(1)
  )

  tau2 <- out$extracted$G[2, 2]
  D_from_lme4 <- 1 - posterior_from_lme4 / tau2

  expect_equal(
    got$posterior_slope_variance,
    posterior_from_lme4,
    tolerance = 1e-8
  )
  expect_equal(
    got$D_slope2,
    D_from_lme4,
    tolerance = 1e-8
  )
})


test_that("sleepstudy adapter agrees with the public reference", {
  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE
  )

  out <- slopeaudit:::.sa_lmer_decompose(
    fit,
    target = "Days",
    target_metadata = sleepstudy_estimand_metadata()
  )

  expected <- utils::read.csv(
    file.path(
      fixture_dir("sleepstudy_primary"),
      "expected_participant_sources.csv"
    ),
    stringsAsFactors = FALSE
  )

  expected$Subject <- as.character(expected$Subject)

  got <- out$participant_table
  expected <- expected[
    match(got$participant_id, expected$Subject),
    ,
    drop = FALSE
  ]

  expect_equal(got$R_direct, expected$R_direct, tolerance = 2e-4)
  expect_equal(
    got$D_slope2,
    expected$D_slope_squared,
    tolerance = 2e-4
  )
  expect_equal(got$Delta_fit, expected$Delta_fit, tolerance = 2e-4)
})


test_that("adapter preserves model-frame omission rather than imputing", {
  dat <- lme4::sleepstudy
  dat$Reaction[c(3, 25, 101)] <- NA_real_

  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = dat,
    REML = TRUE,
    na.action = stats::na.omit
  )

  ext <- slopeaudit:::.sa_extract_lmer_matrices(
    fit,
    target = "Days",
    target_metadata = sleepstudy_estimand_metadata()
  )

  expect_equal(
    sum(vapply(ext$Z, nrow, integer(1))),
    stats::nobs(fit)
  )
  expect_equal(nrow(ext$retained_model_frame), stats::nobs(fit))
  expect_equal(stats::nobs(fit), nrow(dat) - 3L)
})


test_that("target metadata is normalized and does not change calculations", {
  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE
  )

  info <- sleepstudy_estimand_metadata()

  with_info <- slopeaudit:::.sa_lmer_decompose(
    fit,
    target = "Days",
    target_metadata = info
  )

  # For the calculation-only comparison, explicitly bypass the user-facing
  # estimand gate. This is an internal test of whether metadata affects the
  # matrix algebra; it is NOT the intended public behavior.
  without_info <- slopeaudit:::.sa_lmer_decompose(
    fit,
    target = "Days",
    require_identified_estimand = FALSE
  )

  expect_true(with_info$estimand$identified)
  expect_identical(with_info$estimand$target_design_column, "Days")
  expect_identical(with_info$estimand$basis_type, "linear")
  expect_identical(
    with_info$estimand$estimand,
    info$estimand
  )

  expect_equal(
    with_info$participant_table$R_direct,
    without_info$participant_table$R_direct,
    tolerance = 0
  )
  expect_equal(
    with_info$participant_table$D_slope2,
    without_info$participant_table$D_slope2,
    tolerance = 0
  )
  expect_equal(
    with_info$participant_table$Delta_fit,
    without_info$participant_table$Delta_fit,
    tolerance = 0
  )
})
