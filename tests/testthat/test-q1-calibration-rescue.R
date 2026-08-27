make_rescue_assessment <- function(
    method_id,
    status,
    lr = NA_real_,
    full_logLik = -10,
    null_logLik = -10) {

  usable <- status %in% c("positive", "zero")
  list(
    method_id = method_id,
    primary_usable = usable,
    status = status,
    reason = if (usable) "" else status,
    LR_raw = lr,
    LR_zero = if (usable) lr else NA_real_,
    tolerance = 1e-8,
    logLik_full = full_logLik,
    logLik_null = null_logLik,
    full_valid = usable,
    null_valid = usable,
    full_singular = FALSE,
    null_singular = FALSE,
    full_optimizer = method_id,
    null_optimizer = method_id,
    optimizer_honored = TRUE,
    warnings = "",
    messages = "",
    full_fit = NULL,
    null_fit = NULL
  )
}


test_that("v12 uses the adjudicated optimizer controls", {
  controls <- slopeaudit:::.sa_q1_calibration_rescue_controls()

  expect_identical(as.character(controls$initial$optimizer), "nloptwrap")
  expect_true(controls$initial$calc.derivs)
  expect_equal(controls$initial$optCtrl$maxeval, 200000)

  expect_identical(as.character(controls$rescue$optimizer), "bobyqa")
  expect_true(controls$rescue$calc.derivs)
  expect_equal(controls$rescue$optCtrl$maxfun, 1000000)
  expect_equal(controls$rescue$optCtrl$rhoend, 1e-9)

  altered <- controls$rescue
  altered$calc.derivs <- FALSE
  expect_error(
    slopeaudit:::.sa_q1_validate_calibration_rescue_controls(
      controls$initial,
      altered
    ),
    "differs from the frozen v12 control"
  )
})


test_that("usable initial pair is retained without rescue", {
  initial <- make_rescue_assessment(
    "initial_nloptwrap",
    "positive",
    lr = 1.2,
    full_logLik = -9.4
  )
  out <- slopeaudit:::.sa_q1_adjudicate_pair_rescue(initial)

  expect_identical(out$status, "ok")
  expect_identical(out$source, "initial_nloptwrap")
  expect_false(out$rescue_triggered)
  expect_false(out$rescue_succeeded)
  expect_equal(out$chosen$LR_raw, 1.2)
  expect_null(out$rescue)
})


test_that("unusable initial pair is replaced only by usable bobyqa pair", {
  initial <- make_rescue_assessment(
    "initial_nloptwrap",
    "failure"
  )
  rescue <- make_rescue_assessment(
    "rescue_bobyqa",
    "positive",
    lr = 2.5,
    full_logLik = -8.75
  )
  out <- slopeaudit:::.sa_q1_adjudicate_pair_rescue(initial, rescue)

  expect_identical(out$status, "ok")
  expect_identical(out$source, "rescue_bobyqa")
  expect_true(out$rescue_triggered)
  expect_true(out$rescue_succeeded)
  expect_equal(out$chosen$LR_raw, 2.5)

  row <- slopeaudit:::.sa_q1_rescue_replicate_row(4L, 81L, out)
  expect_identical(row$fit_source, "rescue_bobyqa")
  expect_identical(row$initial_status, "failure")
  expect_identical(row$rescue_status, "positive")
  expect_equal(row$LR_raw, row$rescue_LR_raw)
})


test_that("failed bobyqa pair remains unresolved and cannot enter denominator", {
  initial <- make_rescue_assessment(
    "initial_nloptwrap",
    "small_negative",
    lr = -1e-10
  )
  rescue <- make_rescue_assessment(
    "rescue_bobyqa",
    "failure"
  )
  out <- slopeaudit:::.sa_q1_adjudicate_pair_rescue(initial, rescue)
  row <- slopeaudit:::.sa_q1_rescue_replicate_row(5L, 82L, out)

  expect_identical(out$status, "unresolved_after_bobyqa")
  expect_identical(out$source, "unresolved_after_bobyqa")
  expect_true(out$rescue_triggered)
  expect_false(out$rescue_succeeded)
  expect_null(out$chosen)
  expect_identical(row$status, "failure")
  expect_false(is.finite(row$LR_raw))
})


test_that("fixed-usable rescue engine tops up only unresolved final pairs", {
  pair <- q1_sleepstudy_pair()
  controls <- slopeaudit:::.sa_q1_calibration_rescue_controls()
  observed <- list(
    status = "ok",
    full_fit = pair$full,
    null_fit = pair$null,
    LR_for_inference = 1,
    source = "initial_nloptwrap",
    rescue_triggered = FALSE,
    rescue_succeeded = FALSE,
    chosen = make_rescue_assessment(
      "initial_nloptwrap",
      "positive",
      lr = 1
    )
  )

  canned <- function(replicate, seed, ...) {
    if (replicate == 1L) {
      initial <- make_rescue_assessment("initial_nloptwrap", "failure")
      rescue <- make_rescue_assessment("rescue_bobyqa", "failure")
    } else if (replicate == 2L) {
      initial <- make_rescue_assessment("initial_nloptwrap", "failure")
      rescue <- make_rescue_assessment(
        "rescue_bobyqa",
        "positive",
        lr = 1.5,
        full_logLik = -9.25
      )
    } else {
      initial <- make_rescue_assessment(
        "initial_nloptwrap",
        "zero",
        lr = 0
      )
      rescue <- NULL
    }
    adjudication <- if (is.null(rescue)) {
      slopeaudit:::.sa_q1_adjudicate_pair_rescue(initial)
    } else {
      slopeaudit:::.sa_q1_adjudicate_pair_rescue(initial, rescue)
    }
    slopeaudit:::.sa_q1_rescue_replicate_row(
      replicate,
      seed,
      adjudication
    )
  }

  out <- slopeaudit:::.sa_q1_bootstrap_fixed_usable_rescue(
    observed = observed,
    target = "Days",
    usable_B = 2L,
    attempt_cap = 3L,
    seed = 991L,
    retain_replicates = TRUE,
    initial_control = controls$initial,
    rescue_control = controls$rescue,
    one_replicate_fun = canned
  )

  expect_identical(out$status, "ok")
  expect_equal(out$attempts_evaluated, 3L)
  expect_equal(out$top_up_attempts, 1L)
  expect_equal(out$bootstrap$denominator_drop, 2L)
  expect_equal(out$rescue_attempts, 2L)
  expect_equal(out$rescue_successes, 1L)
  expect_equal(out$unresolved_after_rescue, 1L)
  expect_identical(
    out$execution_mode,
    "fixed_drop_usable_with_bobyqa_rescue_v12"
  )
})
