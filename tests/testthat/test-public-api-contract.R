test_that("the v0.1 public surface has the documented formal arguments", {
  expect_setequal(
    getNamespaceExports("slopeaudit"),
    c(
      "slope_audit",
      "test_slope_heterogeneity",
      "slope_decomposition",
      "slope_design_sensitivity"
    )
  )

  expect_identical(
    names(formals(slope_audit)),
    c(
      "full", "null", "target", "target_metadata", "B", "seed",
      "conf_level", "alpha", "attempt_cap", "retain_replicates",
      "progress"
    )
  )
  expect_identical(
    names(formals(test_slope_heterogeneity)),
    c(
      "full", "null", "target", "B", "seed", "conf_level", "alpha",
      "attempt_cap", "retain_replicates", "progress"
    )
  )
  expect_identical(
    names(formals(slope_decomposition)),
    c("full", "target", "target_metadata")
  )
  expect_identical(
    names(formals(slope_design_sensitivity)),
    c("Z", "G", "R", "target", "target_metadata", "H", "design_id")
  )

  expect_false(exists("slope_stability", asNamespace("slopeaudit")))
})


test_that("the public decomposition enforces all six estimand fields", {
  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE
  )

  expect_error(
    slope_decomposition(fit, "Days", target_metadata = NULL),
    class = "slopeaudit_estimand_identification_error"
  )

  incomplete <- sleepstudy_estimand_metadata()
  incomplete$reference_window <- NULL
  expect_error(
    slope_decomposition(fit, "Days", target_metadata = incomplete),
    "reference_window",
    class = "slopeaudit_estimand_identification_error"
  )
})


test_that("public sleepstudy decomposition reproduces reference values", {
  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE
  )

  out <- slope_decomposition(
    full = fit,
    target = "Days",
    target_metadata = sleepstudy_estimand_metadata()
  )

  expect_s3_class(out, "slopeaudit_decomposition")
  expect_identical(
    names(out$source_summary),
    c("quantity", "mean", "median", "IQR", "minimum")
  )
  expect_identical(
    out$source_summary$quantity,
    c("R_direct", "D_slope2", "Delta_fit")
  )
  expect_equal(
    out$source_summary$mean,
    c(0.815423918067602, 0.8485308604763783, 0.0331069424087763),
    tolerance = 2e-4
  )
  expect_equal(
    out$source_summary$mean[[2]],
    out$source_summary$mean[[1]] + out$source_summary$mean[[3]],
    tolerance = 1e-10
  )
  expect_output(print(out), "Random-slope source decomposition")
})


test_that("public Question 1 requires the validated REML scope", {
  pair <- q1_sleepstudy_pair(REML = FALSE)

  expect_error(
    test_slope_heterogeneity(
      full = pair$full,
      null = pair$null,
      target = "Days",
      B = 3L,
      seed = 11L,
      progress = FALSE
    ),
    "requires REML"
  )
})
