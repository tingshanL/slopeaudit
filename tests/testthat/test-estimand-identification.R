test_that("user-facing lmer decomposition requires identified estimand metadata", {
  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE
  )

  expect_error(
    slopeaudit:::.sa_lmer_decompose(fit, target = "Days"),
    class = "slopeaudit_estimand_identification_error"
  )

  out <- slopeaudit:::.sa_lmer_decompose(
    fit,
    target = "Days",
    target_metadata = sleepstudy_estimand_metadata()
  )

  expect_true(out$estimand$identified)
  expect_identical(out$estimand$target_design_column, "Days")
  expect_identical(out$estimand$basis_type, "linear")
  expect_true(out$estimand$affine_invariance_claim)
})

test_that("transformed target is not silently interpreted as a physical linear rate", {
  fit <- suppressWarnings(
    lme4::lmer(
      Reaction ~ poly(Days, 2) + (poly(Days, 2) | Subject),
      data = lme4::sleepstudy,
      REML = TRUE
    )
  )

  target_names <- lme4::getME(fit, "cnms")[[1]]
  target <- target_names[grepl("poly\\(Days, 2\\)1", target_names)]
  expect_length(target, 1L)

  expect_error(
    slopeaudit:::.sa_lmer_decompose(fit, target = target),
    class = "slopeaudit_estimand_identification_error"
  )

  explicit <- list(
    basis_type = "nonlinear",
    outcome_units = "milliseconds",
    time_or_exposure_scale = "first orthogonal-polynomial basis coordinate",
    origin_or_reference = "defined by stats::poly() on retained Days values",
    reference_window = "Days 0--9",
    estimand = "coefficient on the first orthogonal-polynomial random-effect basis coordinate",
    basis_expression = "poly(Days, 2)[, 1]"
  )

  ext <- slopeaudit:::.sa_extract_lmer_matrices(
    fit,
    target = target,
    target_metadata = explicit
  )

  expect_true(ext$estimand$identified)
  expect_false(ext$estimand$affine_invariance_claim)
  expect_identical(ext$estimand$basis_type, "nonlinear")
})
