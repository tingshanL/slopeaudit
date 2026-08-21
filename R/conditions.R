# Typed internal conditions used to keep mathematical failures distinct from
# user-facing audit statuses.

.sa_stop_target_boundary <- function(
    target_slope_variance = 0,
    target_index = NULL,
    target_name = NULL) {

  cond <- structure(
    list(
      message = paste0(
        "The fitted marginal variance of the target random slope is exactly ",
        "zero; R_direct, D_slope2, and Delta_fit are undefined."
      ),
      call = NULL,
      status = "target_boundary",
      decomposition_defined = FALSE,
      target_slope_variance = target_slope_variance,
      target_index = target_index,
      target_name = target_name
    ),
    class = c(
      "slopeaudit_target_boundary_error",
      "slopeaudit_error",
      "error",
      "condition"
    )
  )

  stop(cond)
}

.sa_stop_estimand_identification <- function(message, missing_fields = character()) {
  cond <- structure(
    list(
      message = message,
      call = NULL,
      status = "estimand_unidentified",
      missing_fields = missing_fields
    ),
    class = c(
      "slopeaudit_estimand_identification_error",
      "slopeaudit_error",
      "error",
      "condition"
    )
  )

  stop(cond)
}
