test_that("forced simulation failure is retained and counted", {
  pair <- q1_sleepstudy_pair()

  fail_sim <- function(null_fit, seed) {
    stop("forced simulation failure")
  }

  out <- slopeaudit:::.sa_q1_bootstrap(
    pair$full,
    pair$null,
    target = "Days",
    refit_control = pair$control,
    B = 2,
    seed = 123,
    retain_replicates = TRUE,
    simulate_fun = fail_sim
  )

  expect_equal(out$bootstrap$n_failure, 2L)
  expect_equal(out$bootstrap$denominator_drop, 0L)
  expect_equal(out$bootstrap$denominator_zero, 0L)
  expect_true(all(out$replicates$status == "failure"))
  expect_true(all(grepl(
    "simulation_error",
    out$replicates$reason
  )))
})


test_that("forced refit failure is retained rather than imputed", {
  pair <- q1_sleepstudy_pair()

  fail_refit <- function(template_fit, new_response, control) {
    stop("forced refit failure")
  }

  out <- slopeaudit:::.sa_q1_bootstrap(
    pair$full,
    pair$null,
    target = "Days",
    refit_control = pair$control,
    B = 2,
    seed = 321,
    retain_replicates = TRUE,
    refit_fun = fail_refit
  )

  expect_equal(out$bootstrap$n_failure, 2L)
  expect_true(all(grepl(
    "null_refit_error",
    out$replicates$reason
  )))
})


test_that("singularity message alone does not make a fitted model invalid", {
  pair <- q1_sleepstudy_pair()

  d <- slopeaudit:::.sa_q1_fit_diagnostics(pair$full)

  # Whatever the substantive singularity status of this particular fit,
  # singularity itself is recorded separately from fit validity.
  if (isTRUE(d$singular)) {
    expect_true(d$valid)
  } else {
    expect_true(is.logical(d$singular))
  }
})
