make_canned_q1_row <- function(replicate, seed, status) {
  lr_raw <- switch(
    status,
    positive = 1,
    zero = 0,
    small_negative = -1e-12,
    material_negative = -0.1,
    failure = NA_real_
  )
  lr_zero <- if (status %in% c("zero", "small_negative")) {
    0
  } else if (status == "positive") {
    lr_raw
  } else {
    NA_real_
  }

  data.frame(
    replicate = as.integer(replicate),
    seed = seed,
    status = status,
    reason = if (status %in% c("positive", "zero")) "" else status,
    LR_raw = lr_raw,
    LR_zero = lr_zero,
    tolerance = 1e-8,
    logLik_full = if (is.finite(lr_raw)) -10 + lr_raw / 2 else NA_real_,
    logLik_null = if (is.finite(lr_raw)) -10 else NA_real_,
    full_singular = status %in% c(
      "small_negative",
      "material_negative"
    ),
    null_singular = FALSE,
    warnings = "",
    messages = "",
    stringsAsFactors = FALSE
  )
}


test_that("fixed-usable bootstrap tops up to its primary denominator", {
  pair <- q1_sleepstudy_pair()
  statuses <- c(
    "failure",
    "small_negative",
    "positive",
    "material_negative",
    "zero"
  )
  canned <- function(replicate, seed, ...) {
    make_canned_q1_row(replicate, seed, statuses[[replicate]])
  }

  out <- slopeaudit:::.sa_q1_bootstrap_fixed_usable(
    pair$full,
    pair$null,
    target = "Days",
    usable_B = 2,
    attempt_cap = 5,
    seed = 481,
    refit_control = pair$control,
    retain_replicates = TRUE,
    progress = FALSE,
    one_replicate_fun = canned
  )

  expect_identical(out$status, "ok")
  expect_true(out$target_reached)
  expect_equal(out$attempts_evaluated, 5L)
  expect_equal(out$top_up_attempts, 3L)
  expect_equal(out$bootstrap$denominator_drop, 2L)
  expect_equal(out$bootstrap$denominator_zero, 3L)
  expect_equal(out$bootstrap$denominator_all_zero, 4L)
  expect_equal(nrow(out$replicates), 5L)
  expect_identical(out$rejection_rule, "plus-one p <= alpha")
})


test_that("fixed-usable bootstrap reports an exhausted attempt cap", {
  pair <- q1_sleepstudy_pair()
  statuses <- c(
    "failure",
    "small_negative",
    "positive",
    "material_negative"
  )
  canned <- function(replicate, seed, ...) {
    make_canned_q1_row(replicate, seed, statuses[[replicate]])
  }

  out <- slopeaudit:::.sa_q1_bootstrap_fixed_usable(
    pair$full,
    pair$null,
    target = "Days",
    usable_B = 3,
    attempt_cap = 4,
    seed = 482,
    refit_control = pair$control,
    retain_replicates = TRUE,
    progress = FALSE,
    one_replicate_fun = canned
  )

  expect_identical(out$status, "attempt_cap_reached")
  expect_false(out$target_reached)
  expect_equal(out$attempts_evaluated, 4L)
  expect_equal(out$bootstrap$denominator_drop, 1L)
})
