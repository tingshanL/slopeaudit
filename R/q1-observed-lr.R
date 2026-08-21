# Observed Question-1 likelihood-ratio statistic.

.sa_q1_observed_lr <- function(
    full_fit,
    null_fit,
    target,
    diagnostic_control = NULL,
    identity_tol = .sa_identity_tol,
    strict_controls = TRUE) {

  strict_controls <- .sa_q1_validate_flag(
    strict_controls,
    "strict_controls"
  )

  pair <- .sa_q1_validate_model_pair(
    full_fit = full_fit,
    null_fit = null_fit,
    target = target,
    identity_tol = identity_tol,
    strict_controls = strict_controls
  )

  full_diag <- .sa_q1_fit_diagnostics(
    full_fit,
    control = diagnostic_control
  )
  null_diag <- .sa_q1_fit_diagnostics(
    null_fit,
    control = diagnostic_control
  )

  if (!full_diag$valid || !null_diag$valid) {
    reasons <- unique(stats::na.omit(c(
      if (!full_diag$valid) {
        paste0("full:", full_diag$reason)
      },
      if (!null_diag$valid) {
        paste0("null:", null_diag$reason)
      }
    )))

    return(structure(
      list(
        status = "invalid_observed_fit",
        valid_for_inference = FALSE,
        target = target,
        pair = pair,
        logLik_full = full_diag$logLik,
        logLik_null = null_diag$logLik,
        LR_raw = NA_real_,
        LR_for_inference = NA_real_,
        LR_status = "failure",
        tolerance = NA_real_,
        reason = paste(reasons, collapse = "; "),
        full_diagnostics = full_diag,
        null_diagnostics = null_diag
      ),
      class = "slopeaudit_q1_observed"
    ))
  }

  lr <- .sa_q1_classify_lr(
    logLik_full = full_diag$logLik,
    logLik_null = null_diag$logLik
  )

  if (lr$status == "material_negative") {
    status <- "invalid_objective_order"
    valid_for_inference <- FALSE
    LR_for_inference <- NA_real_
  } else if (lr$status == "failure") {
    status <- "invalid_observed_fit"
    valid_for_inference <- FALSE
    LR_for_inference <- NA_real_
  } else {
    status <- if (lr$status == "small_negative") {
      "numerical_zero_warning"
    } else {
      "ok"
    }
    valid_for_inference <- TRUE
    LR_for_inference <- if (lr$status == "small_negative") {
      0
    } else {
      lr$LR_raw
    }
  }

  structure(
    list(
      status = status,
      valid_for_inference = valid_for_inference,
      target = target,
      pair = pair,
      logLik_full = full_diag$logLik,
      logLik_null = null_diag$logLik,
      LR_raw = lr$LR_raw,
      LR_for_inference = LR_for_inference,
      LR_status = lr$status,
      tolerance = lr$tolerance,
      reason = lr$reason,
      full_diagnostics = full_diag,
      null_diagnostics = null_diag
    ),
    class = "slopeaudit_q1_observed"
  )
}
