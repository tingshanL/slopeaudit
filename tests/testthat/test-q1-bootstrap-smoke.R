test_that("tiny Q1 bootstrap executes end to end", {
  pair <- q1_sleepstudy_pair()

  out <- slopeaudit:::.sa_q1_bootstrap(
    full_fit = pair$full,
    null_fit = pair$null,
    target = "Days",
    refit_control = pair$control,
    B = 3,
    seed = 20260804,
    retain_replicates = TRUE
  )

  expect_s3_class(out, "slopeaudit_q1_internal")
  expect_identical(out$status, "ok")
  expect_equal(out$B_requested, 3L)
  expect_equal(nrow(out$replicates), 3L)
  expect_false(out$progress_requested)

  expect_equal(
    out$bootstrap$B_requested,
    3L
  )

  expect_equal(
    out$bootstrap$n_positive +
      out$bootstrap$n_zero +
      out$bootstrap$n_small_negative +
      out$bootstrap$n_material_negative +
      out$bootstrap$n_failure,
    3L
  )

  expect_identical(out$likelihood_mode, "REML")
  expect_identical(
    out$simulation_source,
    "plug-in simulation from fitted null parameters"
  )
})


test_that("same master seed gives identical replicate stream and bookkeeping", {
  pair <- q1_sleepstudy_pair()

  a <- slopeaudit:::.sa_q1_bootstrap(
    pair$full,
    pair$null,
    target = "Days",
    refit_control = pair$control,
    B = 3,
    seed = 991,
    retain_replicates = TRUE
  )

  b <- slopeaudit:::.sa_q1_bootstrap(
    pair$full,
    pair$null,
    target = "Days",
    refit_control = pair$control,
    B = 3,
    seed = 991,
    retain_replicates = TRUE
  )

  expect_identical(a$replicate_seeds, b$replicate_seeds)

  cols <- c(
    "replicate",
    "seed",
    "status",
    "reason",
    "LR_raw",
    "LR_zero",
    "logLik_full",
    "logLik_null"
  )

  expect_equal(
    a$replicates[, cols],
    b$replicates[, cols],
    tolerance = 0
  )

  expect_equal(a$bootstrap$p_drop, b$bootstrap$p_drop)
  expect_equal(a$bootstrap$p_zero, b$bootstrap$p_zero)
})


test_that("Q1 bootstrap restores caller RNG state", {
  pair <- q1_sleepstudy_pair()

  set.seed(12345)
  before <- .Random.seed

  invisible(
    slopeaudit:::.sa_q1_bootstrap(
      pair$full,
      pair$null,
      target = "Days",
      refit_control = pair$control,
      B = 2,
      seed = 17,
      retain_replicates = FALSE
    )
  )

  expect_identical(.Random.seed, before)
})


test_that("Q1 refit preserves custom optimizer control explicitly", {
  ctrl <- lme4::lmerControl(
    optimizer = "nloptwrap",
    optCtrl = list(maxeval = 12345),
    check.conv.singular = "message"
  )

  full <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE,
    control = ctrl
  )

  y <- as.numeric(lme4::getME(full, "y"))
  resolved <- slopeaudit:::.sa_q1_resolve_refit_control(
    full,
    full,
    refit_control = ctrl
  )
  refitted <- slopeaudit:::.sa_q1_refit(
    full,
    y,
    resolved$control
  )

  expect_identical(
    as.character(refitted@optinfo$optimizer),
    as.character(full@optinfo$optimizer)
  )
  expect_equal(
    refitted@optinfo$control$maxeval,
    full@optinfo$control$maxeval,
    tolerance = 0
  )
})


test_that("bootstrap summary retains compact LR distributions even without replicate table", {
  pair <- q1_sleepstudy_pair()

  out <- slopeaudit:::.sa_q1_bootstrap(
    pair$full,
    pair$null,
    target = "Days",
    refit_control = pair$control,
    B = 3,
    seed = 121,
    retain_replicates = FALSE
  )

  expect_null(out$replicates)
  expect_true(is.numeric(out$bootstrap$bootstrap_distribution$drop))
  expect_true(is.numeric(out$bootstrap$bootstrap_distribution$zero))
  expect_length(out$replicate_seeds, 3L)
  expect_identical(out$execution_mode, "sequential_v1")
})


test_that("Q1 diagnostics use the singularity tolerance from refit control", {
  pair <- q1_sleepstudy_pair()

  d <- slopeaudit:::.sa_q1_fit_diagnostics(
    pair$full,
    control = pair$control
  )

  expected <- pair$control$checkConv$check.conv.singular$tol
  expect_equal(d$singularity_tolerance, expected, tolerance = 0)
  expect_identical(
    d$singular,
    lme4::isSingular(pair$full, tol = expected)
  )
})


test_that("Q1 simulation/refit preserves retained rows after NA omission", {
  dat <- lme4::sleepstudy
  dat$Reaction[c(3, 25, 101)] <- NA_real_

  ctrl <- q1_sleepstudy_control()

  full <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = dat,
    REML = TRUE,
    na.action = stats::na.omit,
    control = ctrl
  )

  null <- lme4::lmer(
    Reaction ~ Days + (1 | Subject),
    data = dat,
    REML = TRUE,
    na.action = stats::na.omit,
    control = ctrl
  )

  y <- slopeaudit:::.sa_q1_simulate_null(null, seed = 814)

  expect_no_error(
    slopeaudit:::.sa_q1_validate_simulated_response(null, y)
  )

  resolved <- slopeaudit:::.sa_q1_resolve_refit_control(
    full,
    null,
    refit_control = ctrl
  )

  null_b <- slopeaudit:::.sa_q1_refit(
    null,
    y,
    resolved$control
  )
  full_b <- slopeaudit:::.sa_q1_refit(
    full,
    y,
    resolved$control
  )

  expect_equal(stats::nobs(null_b), stats::nobs(null))
  expect_equal(stats::nobs(full_b), stats::nobs(full))

  val <- slopeaudit:::.sa_q1_validate_model_pair(
    full_b,
    null_b,
    target = "Days",
    strict_controls = FALSE
  )
  expect_true(val$validated)
})


test_that("tiny end-to-end Q1 bootstrap works after NA omission", {
  dat <- lme4::sleepstudy
  dat$Reaction[c(3, 25, 101)] <- NA_real_

  ctrl <- q1_sleepstudy_control()

  full <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = dat,
    REML = TRUE,
    na.action = stats::na.omit,
    control = ctrl
  )

  null <- lme4::lmer(
    Reaction ~ Days + (1 | Subject),
    data = dat,
    REML = TRUE,
    na.action = stats::na.omit,
    control = ctrl
  )

  out <- slopeaudit:::.sa_q1_bootstrap(
    full,
    null,
    target = "Days",
    refit_control = ctrl,
    B = 2,
    seed = 815,
    retain_replicates = TRUE
  )

  expect_identical(out$status, "ok")
  expect_equal(out$B_requested, 2L)
  expect_equal(nrow(out$replicates), 2L)
})


test_that("bootstrap rejects invalid B before doing any simulation", {
  pair <- q1_sleepstudy_pair()

  expect_error(
    slopeaudit:::.sa_q1_bootstrap(
      pair$full,
      pair$null,
      target = "Days",
      refit_control = pair$control,
      B = 0,
      seed = 1
    ),
    "positive R integer"
  )
})
