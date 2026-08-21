# Helpers for the v11 Question-1 optimizer-rescue diagnostic.
#
# These helpers deliberately live outside the package API. They reconstruct
# the v10 pilot DGP and retain detailed optimizer diagnostics without changing
# the frozen package bootstrap engines.

q1v11_read_integer_env <- function(name, default, minimum = 1L) {
  raw <- Sys.getenv(name, unset = as.character(default))
  value <- suppressWarnings(as.numeric(raw))
  if (length(value) != 1L || !is.finite(value) ||
      value != floor(value) || value < minimum ||
      value > .Machine$integer.max) {
    stop(sprintf("%s must be an integer >= %d.", name, minimum))
  }
  as.integer(value)
}


q1v11_current_control <- function(calc_derivs = TRUE) {
  lme4::lmerControl(
    optimizer = "nloptwrap",
    restart_edge = TRUE,
    boundary.tol = 1e-5,
    calc.derivs = calc_derivs,
    optCtrl = list(maxeval = 200000),
    check.conv.singular = "message"
  )
}


q1v11_control_panel <- function() {
  list(
    current_nloptwrap = list(
      role = "reproduction",
      description = paste0(
        "Exact v10 control: nloptwrap, calc.derivs=TRUE, ",
        "maxeval=200000"
      ),
      control = q1v11_current_control(TRUE)
    ),
    tight_nloptwrap = list(
      role = "rescue_candidate",
      description = paste0(
        "Tighter nloptwrap: calc.derivs=TRUE, maxeval=1000000, ",
        "xtol_abs=1e-10, ftol_abs=1e-10"
      ),
      control = lme4::lmerControl(
        optimizer = "nloptwrap",
        restart_edge = TRUE,
        boundary.tol = 1e-5,
        calc.derivs = TRUE,
        optCtrl = list(
          maxeval = 1000000,
          xtol_abs = 1e-10,
          ftol_abs = 1e-10
        ),
        check.conv.singular = "message"
      )
    ),
    bobyqa = list(
      role = "rescue_candidate",
      description = paste0(
        "Independent bobyqa check: calc.derivs=TRUE, maxfun=1000000, ",
        "rhoend=1e-9"
      ),
      control = lme4::lmerControl(
        optimizer = "bobyqa",
        restart_edge = TRUE,
        boundary.tol = 1e-5,
        calc.derivs = TRUE,
        optCtrl = list(
          maxfun = 1000000,
          rhoend = 1e-9
        ),
        check.conv.singular = "message"
      )
    ),
    current_no_derivs = list(
      role = "diagnostic_only",
      description = paste0(
        "Sensitivity only: v10 nloptwrap optimization with ",
        "calc.derivs=FALSE"
      ),
      control = q1v11_current_control(FALSE)
    )
  )
}


q1v11_cell_manifest <- function() {
  data.frame(
    cell_id = c(
      "null_balanced_q0_0p5",
      "null_balanced_q0_4",
      "null_irregular_q0_0p5",
      "null_irregular_q0_4",
      "power_three_waves_two_years",
      "power_five_waves_five_years"
    ),
    cell_family = c(
      rep("null_size", 4L),
      rep("planning_power_validation", 2L)
    ),
    schedule = c(
      "balanced_three_wave",
      "balanced_three_wave",
      "irregular_fluid_cognition",
      "irregular_fluid_cognition",
      "three_waves_two_years",
      "five_waves_five_years"
    ),
    q0 = c(0.5, 4, 0.5, 4, 0.5, 0.5),
    slope_variance = c(0, 0, 0, 0, 0.01, 0.01),
    residual_variance = rep(1, 6L),
    fixed_slope = rep(0.10, 6L),
    stringsAsFactors = FALSE
  )
}


q1v11_make_balanced_schedule <- function(n_id, times) {
  out <- do.call(
    rbind,
    lapply(seq_len(n_id), function(id) {
      data.frame(id = id, time = times)
    })
  )
  out$id <- factor(out$id)
  rownames(out) <- NULL
  out
}


q1v11_load_schedules <- function(fixture_path) {
  if (!file.exists(fixture_path)) {
    stop("Missing exact irregular schedule fixture: ", fixture_path)
  }

  irregular <- utils::read.csv(
    fixture_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  required <- c("schedule_id", "measurement", "t5_mid")
  if (!identical(names(irregular), required)) {
    stop(
      "Irregular fixture columns must be exactly: ",
      paste(required, collapse = ", ")
    )
  }
  if (nrow(irregular) != 1805L ||
      length(unique(irregular$schedule_id)) != 691L ||
      any(!is.finite(irregular$t5_mid)) ||
      any(table(irregular$schedule_id) < 2L)) {
    stop("Irregular fixture failed the frozen 691/1,805 schedule checks.")
  }

  list(
    balanced_three_wave = q1v11_make_balanced_schedule(
      600L,
      c(-0.5, 0, 0.5)
    ),
    irregular_fluid_cognition = transform(
      irregular,
      id = factor(schedule_id),
      time = t5_mid
    )[, c("id", "time")],
    three_waves_two_years = q1v11_make_balanced_schedule(
      100L,
      c(-1, 0, 1)
    ),
    five_waves_five_years = q1v11_make_balanced_schedule(
      100L,
      seq(0, 5, length.out = 5) - 2.5
    )
  )
}


q1v11_generate_outcome <- function(schedule, cell, seed) {
  set.seed(seed)
  n_id <- nlevels(schedule$id)
  b0 <- stats::rnorm(n_id, 0, sqrt(cell$q0))
  b1 <- if (cell$slope_variance > 0) {
    stats::rnorm(n_id, 0, sqrt(cell$slope_variance))
  } else {
    rep(0, n_id)
  }
  id_index <- as.integer(schedule$id)
  schedule$y <- cell$fixed_slope * schedule$time +
    b0[id_index] +
    b1[id_index] * schedule$time +
    stats::rnorm(
      nrow(schedule),
      0,
      sqrt(cell$residual_variance)
    )
  schedule
}


q1v11_warning_class <- function(status, warnings, messages) {
  if (!identical(status, "failure")) {
    return(status)
  }
  text <- tolower(paste(warnings, messages, sep = " | "))
  if (grepl("degenerate[[:space:]]+hessian|unable to evaluate scaled gradient", text)) {
    return("degenerate_hessian")
  }
  if (grepl("max\\|grad\\|", text)) {
    return("gradient")
  }
  "other_failure"
}


q1v11_smoke_tasks <- function(problems) {
  problems$diagnostic_class <- mapply(
    q1v11_warning_class,
    problems$status,
    problems$warnings,
    problems$messages,
    USE.NAMES = FALSE
  )
  problems$sort_magnitude <- ifelse(
    is.finite(problems$LR_raw),
    abs(problems$LR_raw),
    -Inf
  )
  order_index <- order(
    problems$cell_id,
    problems$diagnostic_class,
    -problems$sort_magnitude,
    problems$outer,
    problems$replicate
  )
  ordered <- problems[order_index, , drop = FALSE]
  group <- paste(
    ordered$cell_id,
    ordered$diagnostic_class,
    sep = "::"
  )
  selected <- ordered[!duplicated(group), , drop = FALSE]
  selected$sort_magnitude <- NULL
  rownames(selected) <- NULL
  selected
}


q1v11_as_character <- function(x) {
  if (is.null(x) || !length(x)) {
    return("")
  }
  paste(unique(as.character(unlist(x, use.names = FALSE))), collapse = " | ")
}


q1v11_vector_label <- function(x) {
  if (is.null(x) || !length(x) || any(!is.finite(x))) {
    return("")
  }
  paste(format(as.numeric(x), digits = 17, scientific = TRUE), collapse = ";")
}


q1v11_hessian_details <- function(fit) {
  derivs <- fit@optinfo$derivs
  gradient <- if (is.null(derivs)) NULL else derivs$gradient
  hessian <- if (is.null(derivs)) NULL else derivs$Hessian

  gradient_ok <- !is.null(gradient) && length(gradient) &&
    all(is.finite(gradient))
  hessian_ok <- !is.null(hessian) && length(hessian) &&
    is.matrix(hessian) && nrow(hessian) == ncol(hessian) &&
    all(is.finite(hessian))

  raw_max <- if (gradient_ok) max(abs(gradient)) else NA_real_
  scaled_max <- NA_real_
  min_eigen <- NA_real_
  max_eigen <- NA_real_
  condition <- NA_real_
  n_negative <- NA_integer_
  n_near_zero <- NA_integer_
  positive_definite <- NA

  if (hessian_ok) {
    sym_hessian <- (hessian + t(hessian)) / 2
    eigenvalues <- tryCatch(
      eigen(sym_hessian, symmetric = TRUE, only.values = TRUE)$values,
      error = function(error) NULL
    )
    if (!is.null(eigenvalues) && all(is.finite(eigenvalues))) {
      eigen_tol <- sqrt(.Machine$double.eps) *
        max(1, max(abs(eigenvalues)))
      min_eigen <- min(eigenvalues)
      max_eigen <- max(eigenvalues)
      n_negative <- sum(eigenvalues < -eigen_tol)
      n_near_zero <- sum(abs(eigenvalues) <= eigen_tol)
      positive_definite <- all(eigenvalues > eigen_tol)
      nonzero <- abs(eigenvalues) > eigen_tol
      if (any(nonzero)) {
        condition <- max(abs(eigenvalues)) / min(abs(eigenvalues[nonzero]))
      }
    }
    if (gradient_ok && length(gradient) == nrow(sym_hessian)) {
      scaled <- tryCatch(
        solve(chol(sym_hessian), gradient),
        error = function(error) NULL
      )
      if (!is.null(scaled) && all(is.finite(scaled))) {
        scaled_max <- max(abs(scaled))
      }
    }
  }

  list(
    derivative_available = gradient_ok && hessian_ok,
    gradient_max_abs = raw_max,
    scaled_gradient_max_abs = scaled_max,
    hessian_min_eigen = min_eigen,
    hessian_max_eigen = max_eigen,
    hessian_condition = condition,
    hessian_n_negative = n_negative,
    hessian_n_near_zero = n_near_zero,
    hessian_positive_definite = positive_definite
  )
}


q1v11_model_details <- function(capture, control) {
  if (!is.null(capture$error)) {
    return(list(
      refit_ok = FALSE,
      error = capture$error,
      valid = FALSE,
      reason = "refit_error",
      logLik = NA_real_,
      optimizer = NA_character_,
      optimizer_code = NA_character_,
      convergence_messages = "",
      singular = NA,
      derivative_available = FALSE,
      gradient_max_abs = NA_real_,
      scaled_gradient_max_abs = NA_real_,
      hessian_min_eigen = NA_real_,
      hessian_max_eigen = NA_real_,
      hessian_condition = NA_real_,
      hessian_n_negative = NA_integer_,
      hessian_n_near_zero = NA_integer_,
      hessian_positive_definite = NA,
      theta = "",
      beta = "",
      captured_warnings = q1v11_as_character(capture$warnings),
      captured_messages = q1v11_as_character(capture$messages),
      fit = NULL
    ))
  }

  fit <- capture$value
  diagnostic <- slopeaudit:::.sa_q1_fit_diagnostics(
    fit,
    control = control
  )
  derivative <- q1v11_hessian_details(fit)
  c(
    list(
      refit_ok = TRUE,
      error = "",
      valid = isTRUE(diagnostic$valid),
      reason = if (is.na(diagnostic$reason)) "" else diagnostic$reason,
      logLik = diagnostic$logLik,
      optimizer = diagnostic$optimizer,
      optimizer_code = paste(diagnostic$optimizer_code, collapse = ";"),
      convergence_messages = q1v11_as_character(
        diagnostic$convergence_messages
      ),
      singular = diagnostic$singular
    ),
    derivative,
    list(
      theta = q1v11_vector_label(lme4::getME(fit, "theta")),
      beta = q1v11_vector_label(lme4::fixef(fit)),
      captured_warnings = q1v11_as_character(capture$warnings),
      captured_messages = q1v11_as_character(capture$messages),
      fit = fit
    )
  )
}


q1v11_covariance_details <- function(fit) {
  empty <- list(
    intercept_variance = NA_real_,
    target_variance = NA_real_,
    intercept_target_correlation = NA_real_,
    lambda_min_over_trace = NA_real_,
    target_max_abs_correlation = NA_real_
  )
  if (is.null(fit)) {
    return(empty)
  }
  covariance <- tryCatch(
    as.matrix(lme4::VarCorr(fit)[[1]]),
    error = function(error) NULL
  )
  if (is.null(covariance) || nrow(covariance) < 2L) {
    return(empty)
  }
  geometry <- tryCatch(
    slopeaudit:::.sa_covariance_diagnostics(
      covariance,
      target_index = 2L
    ),
    error = function(error) NULL
  )
  list(
    intercept_variance = covariance[1, 1],
    target_variance = covariance[2, 2],
    intercept_target_correlation = suppressWarnings(
      covariance[1, 2] /
        sqrt(covariance[1, 1] * covariance[2, 2])
    ),
    lambda_min_over_trace = if (is.null(geometry)) {
      NA_real_
    } else {
      geometry$lambda_min_over_trace
    },
    target_max_abs_correlation = if (is.null(geometry)) {
      NA_real_
    } else {
      geometry$target_max_abs_correlation
    }
  )
}


q1v11_refit_pair <- function(
    full_template,
    null_template,
    response,
    method_id,
    method_spec) {

  control <- method_spec$control
  null_capture <- slopeaudit:::.sa_q1_capture(
    lme4::refit(null_template, newresp = response, control = control)
  )
  full_capture <- slopeaudit:::.sa_q1_capture(
    lme4::refit(full_template, newresp = response, control = control)
  )
  null <- q1v11_model_details(null_capture, control)
  full <- q1v11_model_details(full_capture, control)

  if (!full$refit_ok || !null$refit_ok) {
    status <- "failure"
    reason <- paste(
      c(
        if (!full$refit_ok) paste0("full_refit_error:", full$error),
        if (!null$refit_ok) paste0("null_refit_error:", null$error)
      ),
      collapse = ";"
    )
    lr <- list(
      LR_raw = NA_real_,
      LR_zero = NA_real_,
      tolerance = NA_real_
    )
  } else if (!full$valid || !null$valid) {
    status <- "failure"
    reason <- paste(
      c(
        if (!full$valid) paste0("full:", full$reason),
        if (!null$valid) paste0("null:", null$reason)
      ),
      collapse = ";"
    )
    lr <- list(
      LR_raw = if (is.finite(full$logLik) && is.finite(null$logLik)) {
        2 * (full$logLik - null$logLik)
      } else {
        NA_real_
      },
      LR_zero = NA_real_,
      tolerance = slopeaudit:::.sa_q1_lr_tolerance(
        full$logLik,
        null$logLik
      )
    )
  } else {
    lr <- slopeaudit:::.sa_q1_classify_lr(
      full$logLik,
      null$logLik
    )
    status <- lr$status
    reason <- if (is.na(lr$reason)) "" else lr$reason
  }

  covariance <- q1v11_covariance_details(full$fit)
  derivative_on <- !identical(method_spec$role, "diagnostic_only")
  declared_optimizer <- as.character(method_spec$control$optimizer)
  optimizer_control_honored <-
    (!full$refit_ok || full$optimizer %in% declared_optimizer) &&
    (!null$refit_ok || null$optimizer %in% declared_optimizer)
  primary_usable <- derivative_on && status %in% c("positive", "zero")
  zero_usable <- derivative_on &&
    status %in% c("positive", "zero", "small_negative")
  all_zero_usable <- derivative_on &&
    status %in% c(
      "positive",
      "zero",
      "small_negative",
      "material_negative"
    )

  data.frame(
    method_id = method_id,
    method_role = method_spec$role,
    method_description = method_spec$description,
    optimizer_control_honored = optimizer_control_honored,
    status = status,
    reason = reason,
    LR_raw = lr$LR_raw,
    LR_zero = lr$LR_zero,
    tolerance = lr$tolerance,
    primary_usable = primary_usable,
    zero_sensitivity_usable = zero_usable,
    all_zero_sensitivity_usable = all_zero_usable,
    full_refit_ok = full$refit_ok,
    null_refit_ok = null$refit_ok,
    full_diag_valid = full$valid,
    null_diag_valid = null$valid,
    full_logLik = full$logLik,
    null_logLik = null$logLik,
    full_optimizer = full$optimizer,
    null_optimizer = null$optimizer,
    full_optimizer_code = full$optimizer_code,
    null_optimizer_code = null$optimizer_code,
    full_convergence_messages = full$convergence_messages,
    null_convergence_messages = null$convergence_messages,
    full_singular = full$singular,
    null_singular = null$singular,
    full_derivative_available = full$derivative_available,
    null_derivative_available = null$derivative_available,
    full_gradient_max_abs = full$gradient_max_abs,
    null_gradient_max_abs = null$gradient_max_abs,
    full_scaled_gradient_max_abs = full$scaled_gradient_max_abs,
    null_scaled_gradient_max_abs = null$scaled_gradient_max_abs,
    full_hessian_min_eigen = full$hessian_min_eigen,
    null_hessian_min_eigen = null$hessian_min_eigen,
    full_hessian_max_eigen = full$hessian_max_eigen,
    null_hessian_max_eigen = null$hessian_max_eigen,
    full_hessian_condition = full$hessian_condition,
    null_hessian_condition = null$hessian_condition,
    full_hessian_n_negative = full$hessian_n_negative,
    null_hessian_n_negative = null$hessian_n_negative,
    full_hessian_n_near_zero = full$hessian_n_near_zero,
    null_hessian_n_near_zero = null$hessian_n_near_zero,
    full_hessian_positive_definite = full$hessian_positive_definite,
    null_hessian_positive_definite = null$hessian_positive_definite,
    full_intercept_variance = covariance$intercept_variance,
    full_target_variance = covariance$target_variance,
    full_intercept_target_correlation =
      covariance$intercept_target_correlation,
    full_lambda_min_over_trace = covariance$lambda_min_over_trace,
    full_target_max_abs_correlation =
      covariance$target_max_abs_correlation,
    full_theta = full$theta,
    null_theta = null$theta,
    full_beta = full$beta,
    null_beta = null$beta,
    full_captured_warnings = full$captured_warnings,
    null_captured_warnings = null$captured_warnings,
    full_captured_messages = full$captured_messages,
    null_captured_messages = null$captured_messages,
    full_refit_error = full$error,
    null_refit_error = null$error,
    stringsAsFactors = FALSE
  )
}


q1v11_numeric_match <- function(x, y, relative_tolerance = 1e-7) {
  if (length(x) != 1L || length(y) != 1L) {
    return(FALSE)
  }
  if (is.na(x) && is.na(y)) {
    return(TRUE)
  }
  if (!is.finite(x) || !is.finite(y)) {
    return(FALSE)
  }
  abs(x - y) <= relative_tolerance * max(1, abs(x), abs(y))
}


q1v11_candidate_agreement <- function(x, y, relative_tolerance = 1e-6) {
  q1v11_numeric_match(x, y, relative_tolerance)
}
