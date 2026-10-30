# Compact print methods. Complete provenance remains available in the returned
# objects but is omitted from the default display.

.sa_print_source_summary <- function(x, digits = 3L) {
  if (is.null(x) || nrow(x) == 0L) {
    return(invisible(NULL))
  }

  for (i in seq_len(nrow(x))) {
    cat(
      "  ",
      format(x$quantity[[i]], width = 9L, justify = "left"),
      " = ",
      formatC(x$mean[[i]], digits = digits, format = "f"),
      " (equal-participant mean)\n",
      sep = ""
    )
  }
  invisible(NULL)
}

.sa_print_capacity_summary <- function(x, digits = 3L) {
  if (is.null(x) || nrow(x) == 0L) {
    return(invisible(NULL))
  }
  r <- x[x$quantity == "sqrt_kappa", , drop = FALSE]
  if (nrow(r) == 0L || is.na(r$median)) {
    return(invisible(NULL))
  }
  f <- function(v) formatC(v, digits = digits, format = "f")
  cat(
    "  sqrt(kappa) = ", f(r$median), " (median participant; 5th-95th percentile ",
    f(r$p05), "-", f(r$p95), ")\n",
    sep = ""
  )
  invisible(NULL)
}


#' @export
print.slopeaudit_heterogeneity <- function(x, ..., digits = 3L) {
  cat("Random-slope heterogeneity test\n")
  cat("  Target: ", x$target, "\n", sep = "")
  cat("  Status: ", x$status, "\n", sep = "")
  cat("  Decision: ", x$decision, "\n", sep = "")

  if (!is.null(x$observed) &&
      is.finite(x$observed$LR_for_inference)) {
    cat(
      "  Observed LR: ",
      formatC(x$observed$LR_for_inference, digits = digits, format = "f"),
      "\n",
      sep = ""
    )
  }

  if (!is.null(x$bootstrap)) {
    cat(
      "  Plus-one p: ",
      format.pval(x$bootstrap$p_drop, digits = digits, eps = 0),
      " (", x$bootstrap$denominator_drop, " usable replicates)\n",
      sep = ""
    )
    if (!is.null(x$decision_bounds) &&
        isFALSE(x$decision_bounds$decision_stable)) {
      cat("  Numerical-exclusion sensitivity crosses alpha.\n")
    }
  }

  invisible(x)
}


#' @export
print.slopeaudit_decomposition <- function(x, ..., digits = 3L) {
  cat("Random-slope source decomposition\n")
  cat("  Target: ", x$extracted$target_name, "\n", sep = "")
  cat("  Participants: ", x$summary$n_participants, "\n", sep = "")
  .sa_print_source_summary(x$source_summary, digits = digits)
  .sa_print_capacity_summary(x$capacity_summary, digits = digits)

  if (x$summary$n_no_direct_slope_information > 0L) {
    cat(
      "  No direct slope information: ",
      x$summary$n_no_direct_slope_information,
      " participant(s)\n",
      sep = ""
    )
  }
  if (identical(x$diagnostics$status, "warning")) {
    cat("  Diagnostic: fitted random-effects covariance is near rank deficient.\n")
  }

  invisible(x)
}


#' @export
print.slopeaudit_design_sensitivity <- function(x, ..., digits = 3L) {
  cat("Prospective random-slope design sensitivity\n")
  cat("  Target: ", x$target, "\n", sep = "")
  cat("  Design entries: ", nrow(x$design_table), "\n", sep = "")
  .sa_print_source_summary(x$source_summary, digits = digits)
  .sa_print_capacity_summary(x$capacity_summary, digits = digits)
  cat("  Scope: information-source quantities; no arbitrary-design Q1 power.\n")
  invisible(x)
}


#' @export
print.slopeaudit <- function(x, ..., digits = 3L) {
  cat("Random-slope audit\n")
  cat("  Target: ", x$target, "\n", sep = "")
  cat("  Overall status: ", x$status, "\n", sep = "")
  cat("\nQuestion 1 -- slope heterogeneity\n")
  cat("  Decision: ", x$heterogeneity$decision, "\n", sep = "")
  if (!is.null(x$heterogeneity$bootstrap)) {
    cat(
      "  Plus-one p: ",
      format.pval(
        x$heterogeneity$bootstrap$p_drop,
        digits = digits,
        eps = 0
      ),
      "\n",
      sep = ""
    )
  }

  cat("\nQuestion 2 -- fitted information sources\n")
  if (is.null(x$decomposition)) {
    cat("  Withheld: fitted marginal target-slope variance is exactly zero.\n")
  } else {
    .sa_print_source_summary(x$source_summary, digits = digits)
    .sa_print_capacity_summary(x$capacity_summary, digits = digits)
    if (x$decomposition$summary$n_no_direct_slope_information > 0L) {
      cat(
        "  No direct slope information: ",
        x$decomposition$summary$n_no_direct_slope_information,
        " participant(s)\n",
        sep = ""
      )
    }
  }

  cat("\nInterpretation\n")
  cat("  ", x$interpretation$decomposition, ".\n", sep = "")
  invisible(x)
}


#' @export
summary.slopeaudit <- function(object, ...) {
  list(
    status = object$status,
    target = object$target,
    question1 = list(
      status = object$heterogeneity$status,
      decision = object$heterogeneity$decision,
      p_value = if (is.null(object$heterogeneity$bootstrap)) {
        NA_real_
      } else {
        object$heterogeneity$bootstrap$p_drop
      },
      usable_B = if (is.null(object$heterogeneity$bootstrap)) {
        NA_integer_
      } else {
        object$heterogeneity$bootstrap$denominator_drop
      }
    ),
    question2 = object$source_summary,
    interpretation = object$interpretation
  )
}
