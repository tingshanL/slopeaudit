test_that("observed LR equals direct likelihood difference", {
  pair <- q1_sleepstudy_pair()

  out <- slopeaudit:::.sa_q1_observed_lr(
    full_fit = pair$full,
    null_fit = pair$null,
    target = "Days"
  )

  direct <- 2 * (
    as.numeric(logLik(pair$full)) -
      as.numeric(logLik(pair$null))
  )

  expect_true(out$valid_for_inference)
  expect_identical(out$status, "ok")
  expect_equal(out$LR_raw, direct, tolerance = 1e-12)
  expect_equal(out$LR_for_inference, direct, tolerance = 1e-12)
  expect_gt(out$LR_raw, 0)
})


test_that("LR tolerance follows the frozen likelihood-scale rule", {
  got <- slopeaudit:::.sa_q1_lr_tolerance(
    logLik_full = -100,
    logLik_null = -101
  )

  expected <- 10 * sqrt(.Machine$double.eps) * 101
  expect_equal(got, expected, tolerance = 0)
})


test_that("small finite negative LR is numerical-zero status", {
  tol <- slopeaudit:::.sa_q1_lr_tolerance(-100, -100)

  out <- slopeaudit:::.sa_q1_classify_lr(
    logLik_full = -100 - tol / 4,
    logLik_null = -100
  )

  expect_identical(out$status, "small_negative")
  expect_lt(out$LR_raw, 0)
  expect_equal(out$LR_zero, 0)
})


test_that("material negative LR is separated from small-negative numerical noise", {
  out <- slopeaudit:::.sa_q1_classify_lr(
    logLik_full = -100.01,
    logLik_null = -100
  )

  expect_identical(out$status, "material_negative")
  expect_true(is.na(out$LR_zero))
})
