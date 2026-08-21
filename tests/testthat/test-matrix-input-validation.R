test_that("invalid covariance inputs fail rather than being silently repaired", {
  Z <- cbind(1, c(-1, 0, 1))
  G <- diag(c(1, 1))
  R <- diag(3)

  expect_error(
    slopeaudit:::.sa_participant_source_decomposition(
      Z = Z,
      G = matrix(c(1, 2, 2, 1), 2, 2),
      R = R,
      target_index = 2
    ),
    "positive semidefinite"
  )

  expect_error(
    slopeaudit:::.sa_participant_source_decomposition(
      Z = Z,
      G = G,
      R = matrix(
        c(
          1, 0, 0,
          0, 0, 0,
          0, 0, 1
        ),
        3,
        3,
        byrow = TRUE
      ),
      target_index = 2
    ),
    "positive definite"
  )
})


test_that("exact zero target variance is withheld with typed boundary status", {
  Z <- cbind(1, c(-1, 0, 1))
  G <- matrix(c(1, 0, 0, 0), 2, 2)
  R <- diag(3)

  expect_error(
    slopeaudit:::.sa_participant_source_decomposition(
      Z = Z,
      G = G,
      R = R,
      target_index = 2
    ),
    class = "slopeaudit_target_boundary_error"
  )

  err <- tryCatch(
    slopeaudit:::.sa_participant_source_decomposition(
      Z = Z,
      G = G,
      R = R,
      target_index = 2
    ),
    slopeaudit_target_boundary_error = function(e) e
  )

  expect_identical(err$status, "target_boundary")
  expect_false(err$decomposition_defined)
  expect_equal(err$target_slope_variance, 0)
})
