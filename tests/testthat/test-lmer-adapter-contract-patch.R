test_that("identified sleepstudy decomposition still matches lme4 condVar", {
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

  expect_equal(
    got$posterior_slope_variance,
    posterior_from_lme4,
    tolerance = 1e-8
  )
})

test_that("identified irregular schedules still match lme4 condVar", {
  dat <- lme4::sleepstudy
  dat$Reaction[c(2, 3, 24, 25, 26, 108)] <- NA_real_

  fit <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = dat,
    REML = TRUE,
    na.action = stats::na.omit
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

  expect_gt(length(unique(round(posterior_from_lme4, 10))), 1L)
  expect_equal(
    got$posterior_slope_variance,
    posterior_from_lme4,
    tolerance = 1e-8
  )
})
