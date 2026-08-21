test_that("strong synthetic slope heterogeneity yields a positive observed LR", {
  set.seed(8801)

  n_id <- 20
  days <- 0:5
  id <- factor(rep(seq_len(n_id), each = length(days)))
  time <- rep(days, times = n_id)

  b0 <- rnorm(n_id, 0, 2)
  b1 <- rnorm(n_id, 0, 3)

  y <- 10 + 0.5 * time +
    b0[as.integer(id)] +
    b1[as.integer(id)] * time +
    rnorm(length(time), 0, 1)

  dat <- data.frame(y = y, time = time, id = id)

  ctrl <- lme4::lmerControl(
    optimizer = "nloptwrap",
    check.conv.singular = "message"
  )

  full <- lme4::lmer(
    y ~ time + (1 + time | id),
    data = dat,
    REML = TRUE,
    control = ctrl
  )

  null <- lme4::lmer(
    y ~ time + (1 | id),
    data = dat,
    REML = TRUE,
    control = ctrl
  )

  out <- slopeaudit:::.sa_q1_observed_lr(
    full,
    null,
    target = "time"
  )

  expect_true(out$valid_for_inference)
  expect_gt(out$LR_for_inference, 1)
})
