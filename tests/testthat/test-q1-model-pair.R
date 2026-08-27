test_that("valid sleepstudy full/null pair is accepted", {
  pair <- q1_sleepstudy_pair()

  val <- slopeaudit:::.sa_q1_validate_model_pair(
    full_fit = pair$full,
    null_fit = pair$null,
    target = "Days"
  )

  expect_true(val$validated)
  expect_identical(val$target, "Days")
  expect_identical(val$nuisance_random_effects, "(Intercept)")
  expect_identical(val$likelihood_mode, "REML")
  expect_false(val$target_present_in_null)
})


test_that("mixed REML/ML pair is rejected", {
  ctrl <- q1_sleepstudy_control()

  full <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE,
    control = ctrl
  )

  null <- lme4::lmer(
    Reaction ~ Days + (1 | Subject),
    data = lme4::sleepstudy,
    REML = FALSE,
    control = ctrl
  )

  expect_error(
    slopeaudit:::.sa_q1_validate_model_pair(
      full,
      null,
      target = "Days"
    ),
    "different likelihood"
  )
})


test_that("different retained rows are rejected", {
  ctrl <- q1_sleepstudy_control()
  dat <- lme4::sleepstudy

  full <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = dat,
    REML = TRUE,
    control = ctrl
  )

  null <- lme4::lmer(
    Reaction ~ Days + (1 | Subject),
    data = dat[-1, ],
    REML = TRUE,
    control = ctrl
  )

  expect_error(
    slopeaudit:::.sa_q1_validate_model_pair(
      full,
      null,
      target = "Days"
    ),
    "same retained observations"
  )
})


test_that("different fixed-effect design is rejected", {
  ctrl <- q1_sleepstudy_control()

  full <- lme4::lmer(
    Reaction ~ Days + I(Days^2) + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE,
    control = ctrl
  )

  null <- lme4::lmer(
    Reaction ~ Days + (1 | Subject),
    data = lme4::sleepstudy,
    REML = TRUE,
    control = ctrl
  )

  expect_error(
    slopeaudit:::.sa_q1_validate_model_pair(
      full,
      null,
      target = "Days"
    ),
    "same fixed-effect design"
  )
})


test_that("null that still contains target is rejected", {
  pair <- q1_sleepstudy_pair()

  expect_error(
    slopeaudit:::.sa_q1_validate_model_pair(
      pair$full,
      pair$full,
      target = "Days"
    ),
    "still contains target"
  )
})


test_that("null that changes nuisance random effects is rejected", {
  ctrl <- q1_sleepstudy_control()

  full <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE,
    control = ctrl
  )

  null <- suppressWarnings(
    lme4::lmer(
      Reaction ~ Days + (1 + I(Days^2) | Subject),
      data = lme4::sleepstudy,
      REML = TRUE,
      control = ctrl
    )
  )

  expect_error(
    slopeaudit:::.sa_q1_validate_model_pair(
      full,
      null,
      target = "Days"
    ),
    "differ by more than"
  )
})


test_that("different optimizer convention is rejected", {
  full <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE,
    control = lme4::lmerControl(optimizer = "nloptwrap")
  )

  null <- lme4::lmer(
    Reaction ~ Days + (1 | Subject),
    data = lme4::sleepstudy,
    REML = TRUE,
    control = lme4::lmerControl(optimizer = "bobyqa")
  )

  expect_error(
    slopeaudit:::.sa_q1_validate_model_pair(
      full,
      null,
      target = "Days"
    ),
    "different optimizers"
  )
})


test_that("non-unit prior weights are rejected in Q1 v1", {
  dat <- lme4::sleepstudy
  dat$w <- rep(c(1, 2), length.out = nrow(dat))
  ctrl <- q1_sleepstudy_control()

  full <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = dat,
    weights = w,
    REML = TRUE,
    control = ctrl
  )

  null <- lme4::lmer(
    Reaction ~ Days + (1 | Subject),
    data = dat,
    weights = w,
    REML = TRUE,
    control = ctrl
  )

  expect_error(
    slopeaudit:::.sa_q1_validate_model_pair(
      full,
      null,
      target = "Days"
    ),
    "non-unit prior weights"
  )
})


test_that("custom-control fits require explicit bootstrap refit control", {
  pair <- q1_sleepstudy_pair()

  expect_error(
    slopeaudit:::.sa_q1_resolve_refit_control(
      pair$full,
      pair$null,
      refit_control = NULL
    ),
    "Supply the exact `refit_control`"
  )

  got <- slopeaudit:::.sa_q1_resolve_refit_control(
    pair$full,
    pair$null,
    refit_control = pair$control
  )

  expect_s3_class(got$control, "lmerControl")
  expect_identical(got$source, "explicit_refit_control")
})


test_that("default-control fits can resolve the default bootstrap control", {
  full <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE
  )

  null <- lme4::lmer(
    Reaction ~ Days + (1 | Subject),
    data = lme4::sleepstudy,
    REML = TRUE
  )

  got <- slopeaudit:::.sa_q1_resolve_refit_control(
    full,
    null,
    refit_control = NULL
  )

  expect_s3_class(got$control, "lmerControl")
  expect_identical(got$source, "default_lmerControl")
})


test_that("equivalent controls under different object names are not falsely rejected", {
  ctrl_a <- lme4::lmerControl(
    optimizer = "nloptwrap",
    check.conv.singular = "message"
  )
  ctrl_b <- lme4::lmerControl(
    optimizer = "nloptwrap",
    check.conv.singular = "message"
  )

  full <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE,
    control = ctrl_a
  )

  null <- lme4::lmer(
    Reaction ~ Days + (1 | Subject),
    data = lme4::sleepstudy,
    REML = TRUE,
    control = ctrl_b
  )

  val <- slopeaudit:::.sa_q1_validate_model_pair(
    full,
    null,
    target = "Days"
  )

  expect_true(val$validated)
})


test_that("resolved refit control preserves the common recorded optimizer control", {
  pair <- q1_sleepstudy_pair()

  got <- slopeaudit:::.sa_q1_resolve_refit_control(
    pair$full,
    pair$null,
    refit_control = pair$control
  )

  recorded <- slopeaudit:::.sa_q1_common_recorded_optimizer_control(
    pair$full,
    pair$null
  )

  expect_equal(
    got$control$optCtrl,
    recorded$optimizer_control
  )
  expect_equal(
    got$optimizer_control,
    recorded$optimizer_control
  )
})
