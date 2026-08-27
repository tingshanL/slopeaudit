test_that("exact target-zero raises a typed boundary condition", {
  Z <- cbind(Intercept = 1, target_slope = c(-1, 0, 1))
  G <- matrix(c(1, 0, 0, 0), 2, 2)
  R <- diag(3)

  err <- tryCatch(
    slopeaudit:::.sa_participant_source_decomposition(
      Z = Z,
      G = G,
      R = R,
      target_index = 2
    ),
    slopeaudit_target_boundary_error = function(e) e
  )

  expect_s3_class(err, "slopeaudit_target_boundary_error")
  expect_identical(err$status, "target_boundary")
  expect_false(err$decomposition_defined)
  expect_equal(err$target_slope_variance, 0)
})

test_that("slope_audit handles target boundaries by class, not message text", {
  handler <- function(expr) {
    tryCatch(
      {
        force(expr)
        list(status = "ok", decomposition_defined = TRUE)
      },
      slopeaudit_target_boundary_error = function(e) {
        list(
          status = e$status,
          decomposition_defined = e$decomposition_defined
        )
      }
    )
  }

  Z <- cbind(Intercept = 1, target_slope = c(-1, 0, 1))
  G <- matrix(c(1, 0, 0, 0), 2, 2)
  R <- diag(3)

  ans <- handler(
    slopeaudit:::.sa_participant_source_decomposition(
      Z, G, R, target_index = 2
    )
  )

  expect_identical(ans$status, "target_boundary")
  expect_false(ans$decomposition_defined)
})
