test_that("public Question 1 reaches a fixed usable denominator", {
  pair <- q1_sleepstudy_pair(REML = TRUE)

  out <- test_slope_heterogeneity(
    full = pair$full,
    null = pair$null,
    target = "Days",
    B = 3L,
    seed = 20260814L,
    attempt_cap = 53L,
    retain_replicates = TRUE,
    progress = FALSE
  )

  expect_s3_class(out, "slopeaudit_heterogeneity")
  expect_identical(out$status, "ok")
  expect_true(out$target_reached)
  expect_equal(out$bootstrap$denominator_drop, 3L)
  expect_equal(out$attempts_evaluated, 3L + out$top_up_attempts)
  expect_equal(
    out$bootstrap$p_drop,
    (out$bootstrap$exceedances_drop + 1) /
      (out$bootstrap$denominator_drop + 1),
    tolerance = 0
  )
  expect_true(isTRUE(out$decision_bounds$decision_stable) ||
    isFALSE(out$decision_bounds$decision_stable))
  expect_output(print(out), "Random-slope heterogeneity test")
})


test_that("slope_audit returns the documented combined object", {
  pair <- q1_sleepstudy_pair(REML = TRUE)

  out <- slope_audit(
    full = pair$full,
    null = pair$null,
    target = "Days",
    target_metadata = sleepstudy_estimand_metadata(),
    B = 3L,
    seed = 20260814L,
    attempt_cap = 53L,
    progress = FALSE
  )

  expect_s3_class(out, "slopeaudit")
  expect_named(
    out,
    c(
      "call", "status", "target", "target_metadata", "heterogeneity",
      "decomposition", "decomposition_status", "decomposition_boundary",
      "participant_table", "source_summary", "capacity_summary",
      "interpretation"
    ),
    ignore.order = FALSE
  )
  expect_s3_class(out$heterogeneity, "slopeaudit_heterogeneity")
  expect_s3_class(out$decomposition, "slopeaudit_decomposition")
  expect_equal(nrow(out$participant_table), 18L)
  expect_equal(
    out$source_summary$mean[[2]],
    out$source_summary$mean[[1]] + out$source_summary$mean[[3]],
    tolerance = 1e-10
  )

  brief <- summary(out)
  expect_named(
    brief,
    c("status", "target", "question1", "question2", "interpretation")
  )
  expect_equal(brief$question1$usable_B, 3L)
  expect_output(print(out), "Question 1 -- slope heterogeneity")
})


test_that("the combined audit fails before bootstrapping unidentified targets", {
  pair <- q1_sleepstudy_pair(REML = TRUE)

  expect_error(
    slope_audit(
      full = pair$full,
      null = pair$null,
      target = "Days",
      target_metadata = list(
        basis_type = "linear",
        estimand = "change per day"
      ),
      B = 3L,
      progress = FALSE
    ),
    class = "slopeaudit_estimand_identification_error"
  )
})
