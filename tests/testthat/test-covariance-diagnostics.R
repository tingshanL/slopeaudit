test_that("near-rank-one G is diagnosed separately from target variance", {
  rho <- 0.999999
  G <- matrix(c(1, rho, rho, 1), 2, 2)

  d <- slopeaudit:::.sa_covariance_diagnostics(
    G = G,
    target_index = 2
  )

  expect_equal(d$target_variance, 1, tolerance = 0)
  expect_gt(d$target_variance, 0)
  expect_lt(d$lambda_min_over_trace, 1e-5)
  expect_gt(d$condition_number, 1e5)
  expect_gt(d$target_max_abs_correlation, 0.999)
})

test_that("exact singular G with positive target variance remains a covariance warning, not target-zero", {
  G <- matrix(1, 2, 2)

  d <- slopeaudit:::.sa_covariance_diagnostics(
    G = G,
    target_index = 2
  )

  expect_true(d$near_rank_deficient_at_rank_tol)
  expect_equal(d$numerical_rank, 1L)
  expect_true(is.infinite(d$condition_number))
  expect_equal(d$target_variance, 1)
})

test_that("lmer extraction carries G diagnostics in metadata", {
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

  expect_true(is.list(ext$G_diagnostics))
  expect_identical(
    ext$metadata$G_diagnostics$target_index,
    ext$target_index
  )
  expect_equal(
    ext$metadata$G_diagnostics$target_variance,
    ext$G[ext$target_index, ext$target_index],
    tolerance = 0
  )
})
