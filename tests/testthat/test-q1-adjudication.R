test_that("drop and zero conventions use different denominators correctly", {
  reps <- data.frame(
    replicate = 1:5,
    status = c(
      "positive",
      "zero",
      "small_negative",
      "material_negative",
      "failure"
    ),
    LR_raw = c(2, 0, -1e-8, -0.01, NA),
    LR_zero = c(2, 0, 0, NA, NA),
    reason = c(
      "",
      "",
      "small_finite_negative_LR",
      "material_negative_LR",
      "forced_failure"
    ),
    stringsAsFactors = FALSE
  )

  out <- slopeaudit:::.sa_q1_summarize_replicates(
    replicates = reps,
    observed_LR = 1
  )

  expect_equal(out$denominator_drop, 2L)
  expect_equal(out$denominator_zero, 3L)
  expect_equal(out$denominator_all_zero, 4L)
  expect_equal(out$exceedances_drop, 1L)
  expect_equal(out$exceedances_zero, 1L)
  expect_equal(out$exceedances_all_zero, 1L)
  expect_equal(out$p_drop, 2 / 3)
  expect_equal(out$p_zero, 1 / 2)
  expect_equal(out$p_all_zero, 2 / 5)

  expect_identical(
    out$mc_interval_drop,
    out$exceedance_probability_interval_drop
  )

  expect_equal(out$n_small_negative, 1L)
  expect_equal(out$n_material_negative, 1L)
  expect_equal(out$n_failure, 1L)
  expect_true(out$dual_convention_required)
})


test_that("Question-1 rejection includes p exactly equal to alpha", {
  expect_identical(
    slopeaudit:::.sa_q1_reject(c(0.049, 0.05, 0.051, NA_real_)),
    c(TRUE, TRUE, FALSE, FALSE)
  )
})


test_that("fixed usable denominator preserves the intended rank grid", {
  expect_equal(
    slopeaudit:::.sa_q1_nominal_rank_size(99L),
    0.05,
    tolerance = 0
  )
  expect_equal(
    slopeaudit:::.sa_q1_nominal_rank_size(98L),
    4 / 99,
    tolerance = 0
  )
  expect_equal(
    slopeaudit:::.sa_q1_nominal_rank_size(999L),
    0.05,
    tolerance = 0
  )
  expect_equal(
    slopeaudit:::.sa_q1_nominal_rank_size(989L),
    49 / 990,
    tolerance = 0
  )
})


test_that("plus-one floor is .001 with zero exceedances among 999", {
  expect_equal(
    slopeaudit:::.sa_q1_plus_one_p(0, 999),
    0.001,
    tolerance = 0
  )
})


test_that("exact binomial MC interval reproduces the 0/999 anchor", {
  ci <- slopeaudit:::.sa_q1_exact_binomial_interval(
    k = 0,
    n = 999,
    conf_level = 0.95
  )

  exact_high <- 1 - (0.025)^(1 / 999)

  expect_equal(ci[[1]], 0, tolerance = 0)
  expect_equal(ci[[2]], exact_high, tolerance = 1e-12)
  expect_equal(round(ci[[2]], 5), 0.00369, tolerance = 0)
})


test_that("floor-limited result triggers escalation advice", {
  reps <- data.frame(
    replicate = 1:999,
    status = rep("zero", 999),
    LR_raw = rep(0, 999),
    LR_zero = rep(0, 999),
    reason = rep("", 999),
    stringsAsFactors = FALSE
  )

  out <- slopeaudit:::.sa_q1_summarize_replicates(
    replicates = reps,
    observed_LR = 1
  )

  expect_equal(out$exceedances_drop, 0L)
  expect_true(out$escalation$recommended)
  expect_match(out$escalation$reason, "floor_limited")
})


test_that("MC interval crossing alpha triggers escalation advice", {
  reps <- data.frame(
    replicate = 1:99,
    status = rep("positive", 99),
    LR_raw = c(rep(2, 5), rep(0.5, 94)),
    LR_zero = c(rep(2, 5), rep(0.5, 94)),
    reason = rep("", 99),
    stringsAsFactors = FALSE
  )

  out <- slopeaudit:::.sa_q1_summarize_replicates(
    replicates = reps,
    observed_LR = 1,
    alpha = 0.05
  )

  expect_true(out$escalation$recommended)
  expect_match(out$escalation$reason, "MC_interval_crosses_alpha")
})


test_that("out-of-range integer seeds are rejected before coercion", {
  expect_error(
    slopeaudit:::.sa_q1_validate_seed(.Machine$integer.max + 1),
    "representable by R"
  )
})


test_that("Q1 scalar validators reject malformed probabilities and flags", {
  expect_error(
    slopeaudit:::.sa_q1_validate_probability(1, "alpha"),
    "strictly between"
  )

  expect_error(
    slopeaudit:::.sa_q1_validate_flag(NA, "retain_replicates"),
    "TRUE or FALSE"
  )

  expect_error(
    slopeaudit:::.sa_q1_validate_flag(NA, "progress"),
    "TRUE or FALSE"
  )
})
