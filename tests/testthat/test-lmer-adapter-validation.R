test_that("adapter rejects unsupported model classes", {
  expect_error(
    slopeaudit:::.sa_extract_lmer_matrices(
      lm(Reaction ~ Days, data = lme4::sleepstudy),
      target = "Days"
    ),
    "lmerMod"
  )
})

test_that("adapter rejects more than one grouping factor", {
  dat <- lme4::sleepstudy
  dat$Batch <- factor(rep(c("A", "B"), length.out = nrow(dat)))

  fit <- suppressWarnings(
    lme4::lmer(
      Reaction ~ Days + (1 + Days | Subject) + (1 | Batch),
      data = dat,
      REML = TRUE
    )
  )

  expect_error(
    slopeaudit:::.sa_extract_lmer_matrices(fit, target = "Days"),
    "exactly one grouping factor"
  )
})

test_that("adapter rejects multiple random-effects terms", {
  fit <- suppressWarnings(
    lme4::lmer(
      Reaction ~ Days + (1 | Subject) + (0 + Days | Subject),
      data = lme4::sleepstudy,
      REML = TRUE
    )
  )

  expect_error(
    slopeaudit:::.sa_extract_lmer_matrices(fit, target = "Days"),
    "exactly one random-effects term"
  )
})

test_that("target coefficient must be explicit and valid", {
  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = TRUE
  )

  expect_error(
    slopeaudit:::.sa_extract_lmer_matrices(fit, target = NULL),
    "non-empty"
  )

  expect_error(
    slopeaudit:::.sa_extract_lmer_matrices(fit, target = "Time"),
    "Available random-effect coefficients"
  )
})

test_that("ML fits are retained as ML and not silently converted", {
  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = FALSE
  )

  ext <- slopeaudit:::.sa_extract_lmer_matrices(fit, target = "Days")
  expect_identical(ext$metadata$likelihood, "ML")
})


test_that("adapter rejects non-unit prior weights rather than mis-specifying R", {
  dat <- lme4::sleepstudy
  dat$w <- rep(c(1, 2), length.out = nrow(dat))

  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = dat,
    weights = w,
    REML = TRUE
  )

  expect_error(
    slopeaudit:::.sa_extract_lmer_matrices(fit, target = "Days"),
    "unit prior weights"
  )
})
