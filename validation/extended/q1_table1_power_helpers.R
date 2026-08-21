# Helpers for the v13.3.1 paired Table-1 power diagnostic.
#
# These functions are validation-stage code, not package API.  The diagnostic
# compares three prespecified statistics on every generated dataset:
#
# 1. a direct-channel slope-variance statistic with known residual variance;
# 2. a random-intercept plus independent-random-slope restricted LRT; and
# 3. the current unstructured Question-1 restricted LRT.
#
# Both mixed-model statistics use the v12 deterministic optimizer hierarchy.

q1p_collapse_text <- function(x) {
  if (is.null(x) || !length(x)) {
    return("")
  }
  paste(unique(as.character(unlist(x, use.names = FALSE))), collapse = " | ")
}


q1p_make_balanced_schedule <- function(n_participants, times) {
  if (!is.numeric(n_participants) || length(n_participants) != 1L ||
      !is.finite(n_participants) || n_participants != floor(n_participants) ||
      n_participants < 2L) {
    stop("`n_participants` must be one integer at least two.", call. = FALSE)
  }
  if (!is.numeric(times) || length(times) < 2L ||
      any(!is.finite(times)) || length(unique(times)) < 2L) {
    stop("`times` must contain at least two distinct finite values.",
         call. = FALSE)
  }

  centered <- as.numeric(times - mean(times))
  out <- data.frame(
    id = factor(rep(seq_len(as.integer(n_participants)), each = length(times))),
    time = rep(centered, times = as.integer(n_participants))
  )
  rownames(out) <- NULL
  out
}


q1p_validate_truth <- function(truth) {
  required <- c(
    "random_intercept_variance",
    "random_slope_variance",
    "intercept_slope_correlation",
    "residual_variance",
    "fixed_intercept",
    "fixed_slope"
  )
  if (!is.list(truth) || !all(required %in% names(truth))) {
    stop("`truth` is missing required DGP fields.", call. = FALSE)
  }

  finite <- vapply(truth[required], function(x) {
    is.numeric(x) && length(x) == 1L && is.finite(x)
  }, logical(1))
  if (!all(finite)) {
    stop("Every required DGP field must be one finite number.", call. = FALSE)
  }
  if (truth$random_intercept_variance <= 0 ||
      truth$random_slope_variance < 0 ||
      truth$residual_variance <= 0 ||
      abs(truth$intercept_slope_correlation) >= 1) {
    stop("Invalid variance or correlation in `truth`.", call. = FALSE)
  }
  if (truth$random_slope_variance == 0 &&
      truth$intercept_slope_correlation != 0) {
    stop("Correlation must be zero when slope variance is zero.",
         call. = FALSE)
  }
  invisible(TRUE)
}


q1p_generate_dataset <- function(
    schedule,
    truth,
    seed,
    coupling = c("legacy_hold_intercept", "paired_hold_slope")) {
  q1p_validate_truth(truth)
  seed <- slopeaudit:::.sa_q1_validate_seed(seed)
  coupling <- match.arg(coupling)

  if (!is.data.frame(schedule) ||
      !all(c("id", "time") %in% names(schedule)) ||
      !is.factor(schedule$id) || any(!is.finite(schedule$time))) {
    stop("`schedule` must contain factor `id` and finite numeric `time`.",
         call. = FALSE)
  }

  set.seed(seed)
  n_id <- nlevels(schedule$id)
  z0 <- stats::rnorm(n_id)

  if (truth$random_slope_variance == 0) {
    b0 <- sqrt(truth$random_intercept_variance) * z0
    b1 <- rep(0, n_id)
  } else {
    z1 <- stats::rnorm(n_id)
    rho <- truth$intercept_slope_correlation
    if (identical(coupling, "legacy_hold_intercept")) {
      b0 <- sqrt(truth$random_intercept_variance) * z0
      b1 <- sqrt(truth$random_slope_variance) *
        (rho * z0 + sqrt(1 - rho^2) * z1)
    } else {
      # The paired sensitivity holds the participant slope effects and all
      # residual draws fixed.  Only the intercept construction changes, so
      # rho changes without altering the direct slope channel.
      b1 <- sqrt(truth$random_slope_variance) * z1
      b0 <- sqrt(truth$random_intercept_variance) *
        (sqrt(1 - rho^2) * z0 + rho * z1)
    }
  }

  id_index <- as.integer(schedule$id)
  response <- truth$fixed_intercept +
    truth$fixed_slope * schedule$time +
    b0[id_index] +
    b1[id_index] * schedule$time +
    stats::rnorm(nrow(schedule), sd = sqrt(truth$residual_variance))

  data.frame(
    id = schedule$id,
    time = schedule$time,
    y = response
  )
}


q1p_direct_slope_statistic <- function(data, residual_variance) {
  if (!is.data.frame(data) ||
      !all(c("id", "time", "y") %in% names(data)) ||
      !is.factor(data$id) ||
      any(!is.finite(data$time)) || any(!is.finite(data$y))) {
    stop("Invalid balanced data for the direct-slope statistic.",
         call. = FALSE)
  }
  if (!is.numeric(residual_variance) || length(residual_variance) != 1L ||
      !is.finite(residual_variance) || residual_variance <= 0) {
    stop("`residual_variance` must be one positive finite number.",
         call. = FALSE)
  }

  by_id <- split(data, data$id, drop = TRUE)
  sxx <- vapply(by_id, function(d) {
    sum((d$time - mean(d$time))^2)
  }, numeric(1))
  if (any(!is.finite(sxx)) || any(sxx <= 0) ||
      max(abs(sxx - sxx[[1]])) > 1e-12 * max(1, abs(sxx))) {
    stop("The direct-channel statistic requires one common estimable schedule.",
         call. = FALSE)
  }

  slopes <- vapply(by_id, function(d) {
    centered_time <- d$time - mean(d$time)
    sum(centered_time * d$y) / sum(centered_time^2)
  }, numeric(1))
  direct_error_variance <- residual_variance / sxx[[1]]

  list(
    statistic = sum((slopes - mean(slopes))^2) /
      direct_error_variance,
    df = length(slopes) - 1L,
    direct_error_variance = direct_error_variance,
    participant_slopes = slopes
  )
}


q1p_analytic_values <- function(
    n_participants,
    times,
    random_slope_variance,
    residual_variance,
    alpha = 0.05) {

  centered <- times - mean(times)
  information <- sum(centered^2) / residual_variance
  direct_error_variance <- 1 / information
  r_direct <- random_slope_variance /
    (random_slope_variance + direct_error_variance)
  df <- as.integer(n_participants) - 1L
  critical <- stats::qchisq(1 - alpha, df = df)
  power <- 1 - stats::pchisq(
    critical * direct_error_variance /
      (random_slope_variance + direct_error_variance),
    df = df
  )

  c(
    information = information,
    direct_error_variance = direct_error_variance,
    R_direct = r_direct,
    analytic_direct_power = power,
    df = df,
    theoretical_direct_critical = critical
  )
}


q1p_pair_structure_ok <- function(full_fit, null_fit, pair_kind) {
  if (!inherits(full_fit, "lmerMod") || !inherits(null_fit, "lmerMod")) {
    return(FALSE)
  }

  full_frame <- stats::model.frame(full_fit)
  null_frame <- stats::model.frame(null_fit)
  same_rows <- identical(rownames(full_frame), rownames(null_frame))
  same_response <- identical(
    as.numeric(stats::model.response(full_frame)),
    as.numeric(stats::model.response(null_frame))
  )
  same_reml <- identical(lme4::isREML(full_fit), lme4::isREML(null_fit))
  same_x <- slopeaudit:::.sa_q1_same_numeric_matrix(
    lme4::getME(full_fit, "X"),
    lme4::getME(null_fit, "X")
  )
  full_groups <- as.character(lme4::getME(full_fit, "flist")[[1]])
  null_groups <- as.character(lme4::getME(null_fit, "flist")[[1]])
  same_groups <- identical(full_groups, null_groups)

  full_cnms <- lme4::getME(full_fit, "cnms")
  null_cnms <- lme4::getME(null_fit, "cnms")
  null_ok <- length(null_cnms) == 1L &&
    identical(as.character(null_cnms[[1]]), "(Intercept)")

  full_ok <- if (identical(pair_kind, "constrained")) {
    length(full_cnms) == 2L &&
      all(vapply(full_cnms, length, integer(1)) == 1L) &&
      setequal(
        vapply(full_cnms, function(x) as.character(x[[1]]), character(1)),
        c("(Intercept)", "time")
      )
  } else if (identical(pair_kind, "unstructured")) {
    length(full_cnms) == 1L &&
      setequal(as.character(full_cnms[[1]]), c("(Intercept)", "time"))
  } else {
    FALSE
  }

  isTRUE(same_rows && same_response && same_reml && same_x &&
    same_groups && null_ok && full_ok)
}


q1p_optimizer_code_label <- function(x) {
  if (is.null(x) || !length(x)) {
    return("")
  }
  paste(as.integer(x), collapse = ";")
}


q1p_constrained_slope_geometry <- function(fit, singularity_tolerance) {
  empty <- list(
    slope_variance = NA_real_,
    residual_variance = NA_real_,
    slope_relative_sd = NA_real_,
    slope_at_boundary = FALSE
  )
  if (!inherits(fit, "lmerMod") ||
      !is.numeric(singularity_tolerance) ||
      length(singularity_tolerance) != 1L ||
      !is.finite(singularity_tolerance) ||
      singularity_tolerance <= 0) {
    return(empty)
  }

  components <- tryCatch(
    lapply(lme4::VarCorr(fit), as.matrix),
    error = function(error) NULL
  )
  if (is.null(components)) {
    return(empty)
  }
  slope_components <- Filter(function(component) {
    is.matrix(component) && nrow(component) == 1L &&
      identical(rownames(component), "time") &&
      identical(colnames(component), "time") &&
      is.finite(component[[1]]) && component[[1]] >= 0
  }, components)
  if (length(slope_components) != 1L) {
    return(empty)
  }

  residual_sd <- tryCatch(stats::sigma(fit), error = function(error) NA_real_)
  slope_variance <- as.numeric(slope_components[[1]][[1]])
  residual_variance <- residual_sd^2
  relative_sd <- if (is.finite(residual_sd) && residual_sd > 0) {
    sqrt(slope_variance) / residual_sd
  } else {
    NA_real_
  }
  list(
    slope_variance = slope_variance,
    residual_variance = residual_variance,
    slope_relative_sd = relative_sd,
    slope_at_boundary = is.finite(relative_sd) &&
      relative_sd <= singularity_tolerance
  )
}


q1p_fit_pair_once <- function(
    data,
    full_formula,
    null_formula,
    pair_kind,
    control,
    method_id) {

  full_capture <- slopeaudit:::.sa_q1_capture(
    lme4::lmer(
      formula = full_formula,
      data = data,
      REML = TRUE,
      control = control
    )
  )
  null_capture <- slopeaudit:::.sa_q1_capture(
    lme4::lmer(
      formula = null_formula,
      data = data,
      REML = TRUE,
      control = control
    )
  )

  full_fit <- full_capture$value
  null_fit <- null_capture$value
  full_diag <- slopeaudit:::.sa_q1_fit_diagnostics(
    full_fit,
    control = control
  )
  null_diag <- slopeaudit:::.sa_q1_fit_diagnostics(
    null_fit,
    control = control
  )
  optimizer_honored <- !is.null(full_fit) && !is.null(null_fit) &&
    full_diag$optimizer %in% as.character(control$optimizer) &&
    null_diag$optimizer %in% as.character(control$optimizer)
  structure_ok <- q1p_pair_structure_ok(full_fit, null_fit, pair_kind)
  slope_geometry <- if (identical(pair_kind, "constrained")) {
    q1p_constrained_slope_geometry(
      full_fit,
      full_diag$singularity_tolerance
    )
  } else {
    list(
      slope_variance = NA_real_,
      residual_variance = NA_real_,
      slope_relative_sd = NA_real_,
      slope_at_boundary = FALSE
    )
  }

  if (!is.null(full_capture$error) || !is.null(null_capture$error)) {
    status <- "failure"
    reason <- paste(
      c(
        if (!is.null(full_capture$error)) {
          paste0("full_fit_error:", full_capture$error)
        },
        if (!is.null(null_capture$error)) {
          paste0("null_fit_error:", null_capture$error)
        }
      ),
      collapse = ";"
    )
    lr <- list(LR_raw = NA_real_, tolerance = NA_real_)
  } else if (!isTRUE(structure_ok)) {
    status <- "failure"
    reason <- "full_null_structure_mismatch"
    lr <- list(LR_raw = NA_real_, tolerance = NA_real_)
  } else if (!isTRUE(optimizer_honored)) {
    status <- "failure"
    reason <- "declared_optimizer_not_honored"
    lr <- list(
      LR_raw = 2 * (full_diag$logLik - null_diag$logLik),
      tolerance = slopeaudit:::.sa_q1_lr_tolerance(
        full_diag$logLik,
        null_diag$logLik
      )
    )
  } else if (!isTRUE(full_diag$valid) || !isTRUE(null_diag$valid)) {
    status <- "failure"
    reason <- paste(
      c(
        if (!isTRUE(full_diag$valid)) {
          paste0("full:", full_diag$reason)
        },
        if (!isTRUE(null_diag$valid)) {
          paste0("null:", null_diag$reason)
        }
      ),
      collapse = ";"
    )
    lr <- list(
      LR_raw = if (is.finite(full_diag$logLik) &&
        is.finite(null_diag$logLik)) {
        2 * (full_diag$logLik - null_diag$logLik)
      } else {
        NA_real_
      },
      tolerance = slopeaudit:::.sa_q1_lr_tolerance(
        full_diag$logLik,
        null_diag$logLik
      )
    )
  } else {
    lr <- slopeaudit:::.sa_q1_classify_lr(
      full_diag$logLik,
      null_diag$logLik
    )
    status <- lr$status
    reason <- if (is.na(lr$reason)) "" else lr$reason
  }

  list(
    method_id = method_id,
    primary_usable = status %in% c("positive", "zero"),
    status = status,
    reason = reason,
    LR_raw = lr$LR_raw,
    tolerance = lr$tolerance,
    logLik_full = full_diag$logLik,
    logLik_null = null_diag$logLik,
    full_valid = isTRUE(full_diag$valid),
    null_valid = isTRUE(null_diag$valid),
    full_singular = if (is.logical(full_diag$singular) &&
      length(full_diag$singular) == 1L) full_diag$singular else NA,
    null_singular = if (is.logical(null_diag$singular) &&
      length(null_diag$singular) == 1L) null_diag$singular else NA,
    full_optimizer = full_diag$optimizer,
    null_optimizer = null_diag$optimizer,
    full_optimizer_code = q1p_optimizer_code_label(
      full_diag$optimizer_code
    ),
    null_optimizer_code = q1p_optimizer_code_label(
      null_diag$optimizer_code
    ),
    full_convergence_messages = q1p_collapse_text(
      full_diag$convergence_messages
    ),
    null_convergence_messages = q1p_collapse_text(
      null_diag$convergence_messages
    ),
    full_non_singular_convergence_messages = q1p_collapse_text(
      full_diag$non_singular_convergence_messages
    ),
    null_non_singular_convergence_messages = q1p_collapse_text(
      null_diag$non_singular_convergence_messages
    ),
    optimizer_honored = optimizer_honored,
    structure_ok = structure_ok,
    full_slope_variance = slope_geometry$slope_variance,
    full_residual_variance = slope_geometry$residual_variance,
    full_slope_relative_sd = slope_geometry$slope_relative_sd,
    full_slope_at_boundary = slope_geometry$slope_at_boundary,
    warnings = q1p_collapse_text(c(
      full_capture$warnings,
      null_capture$warnings
    )),
    messages = q1p_collapse_text(c(
      full_capture$messages,
      null_capture$messages
    )),
    full_fit = full_fit,
    null_fit = null_fit
  )
}


q1p_adjudicate_pair <- function(initial, rescue = NULL) {
  if (isTRUE(initial$primary_usable)) {
    return(list(
      status = "ok",
      source = "initial_nloptwrap",
      rescue_triggered = FALSE,
      rescue_succeeded = FALSE,
      chosen = initial,
      initial = initial,
      rescue = NULL
    ))
  }
  if (is.null(rescue)) {
    stop("Rescue assessment is required after an unusable initial pair.",
         call. = FALSE)
  }
  if (isTRUE(rescue$primary_usable)) {
    return(list(
      status = "ok",
      source = "rescue_bobyqa",
      rescue_triggered = TRUE,
      rescue_succeeded = TRUE,
      chosen = rescue,
      initial = initial,
      rescue = rescue
    ))
  }
  list(
    status = "unresolved_after_bobyqa",
    source = "unresolved_after_bobyqa",
    rescue_triggered = TRUE,
    rescue_succeeded = FALSE,
    chosen = NULL,
    initial = initial,
    rescue = rescue
  )
}


q1p_rescued_small_negative_is_floorable <- function(rescue) {
  is.list(rescue) &&
    identical(rescue$status, "small_negative") &&
    identical(rescue$reason, "small_finite_negative_LR") &&
    isTRUE(rescue$full_valid) &&
    isTRUE(rescue$null_valid) &&
    isTRUE(rescue$optimizer_honored) &&
    isTRUE(rescue$structure_ok) &&
    is.numeric(rescue$LR_raw) &&
    length(rescue$LR_raw) == 1L &&
    is.finite(rescue$LR_raw) &&
    rescue$LR_raw < 0 &&
    is.numeric(rescue$tolerance) &&
    length(rescue$tolerance) == 1L &&
    is.finite(rescue$tolerance) &&
    rescue$tolerance >= 0 &&
    rescue$LR_raw >= -rescue$tolerance
}


q1p_rescued_boundary_code_is_adjudicable <- function(initial, rescue) {
  initial_ok <- is.list(initial) &&
    identical(initial$status, "small_negative") &&
    identical(initial$reason, "small_finite_negative_LR") &&
    isTRUE(initial$full_valid) &&
    isTRUE(initial$null_valid) &&
    isTRUE(initial$optimizer_honored) &&
    isTRUE(initial$structure_ok) &&
    is.finite(initial$LR_raw) && initial$LR_raw < 0 &&
    is.finite(initial$tolerance) &&
    initial$LR_raw >= -initial$tolerance
  if (!initial_ok || !is.list(rescue)) {
    return(FALSE)
  }

  reason <- strsplit(rescue$reason, ";", fixed = TRUE)[[1]]
  allowed_reason <- length(reason) == 1L && reason %in% c(
    "full:optimizer_nonzero_code",
    "null:optimizer_nonzero_code"
  )
  rescue_lr_ok <- is.finite(rescue$LR_raw) &&
    is.finite(rescue$tolerance) &&
    abs(rescue$LR_raw) <= rescue$tolerance
  rescue_pair_ok <- isTRUE(rescue$optimizer_honored) &&
    isTRUE(rescue$structure_ok) &&
    nzchar(rescue$full_optimizer_code) &&
    nzchar(rescue$null_optimizer_code) &&
    xor(
      rescue$full_optimizer_code != "0",
      rescue$null_optimizer_code != "0"
    ) &&
    !nzchar(rescue$full_non_singular_convergence_messages) &&
    !nzchar(rescue$null_non_singular_convergence_messages) &&
    isTRUE(rescue$full_slope_at_boundary)

  likelihood_tolerance <- max(
    initial$tolerance,
    rescue$tolerance,
    na.rm = TRUE
  )
  likelihood_not_worse <- all(is.finite(c(
    initial$logLik_full,
    initial$logLik_null,
    rescue$logLik_full,
    rescue$logLik_null
  ))) &&
    rescue$logLik_full >= initial$logLik_full - likelihood_tolerance &&
    rescue$logLik_null >= initial$logLik_null - likelihood_tolerance

  isTRUE(allowed_reason && rescue_lr_ok && rescue_pair_ok &&
    likelihood_not_worse)
}


q1p_adjudicate_constrained_pair <- function(initial, rescue = NULL) {
  if (isTRUE(initial$primary_usable)) {
    return(q1p_adjudicate_pair(initial))
  }
  if (is.null(rescue)) {
    stop("Rescue assessment is required after an unusable initial pair.",
         call. = FALSE)
  }
  if (isTRUE(rescue$primary_usable)) {
    return(q1p_adjudicate_pair(initial, rescue))
  }
  if (q1p_rescued_small_negative_is_floorable(rescue)) {
    return(list(
      status = "ok",
      source = "rescue_bobyqa_numerical_zero",
      rescue_triggered = TRUE,
      rescue_succeeded = TRUE,
      numerical_zero_floored = TRUE,
      numerical_zero_floor_reason = paste0(
        "constrained diagnostic only: otherwise-valid rescued LR was ",
        "negative within the prespecified likelihood-scale tolerance"
      ),
      chosen = rescue,
      initial = initial,
      rescue = rescue
    ))
  }
  if (q1p_rescued_boundary_code_is_adjudicable(initial, rescue)) {
    return(list(
      status = "ok",
      source = "rescue_bobyqa_boundary_zero_optimizer_code",
      rescue_triggered = TRUE,
      rescue_succeeded = TRUE,
      numerical_zero_floored = TRUE,
      numerical_zero_rule = "optimizer_code_boundary_zero",
      boundary_optimizer_code_adjudicated = TRUE,
      numerical_zero_floor_reason = paste0(
        "constrained diagnostic only: rescue returned a within-tolerance ",
        "boundary LR and exactly one nonzero optimizer code, with no other ",
        "convergence failure and no likelihood deterioration"
      ),
      chosen = rescue,
      initial = initial,
      rescue = rescue
    ))
  }
  q1p_adjudicate_pair(initial, rescue)
}


q1p_fit_constrained_pair <- function(data, controls) {
  full_formula <- y ~ time + (1 | id) + (0 + time | id)
  null_formula <- y ~ time + (1 | id)

  initial <- q1p_fit_pair_once(
    data = data,
    full_formula = full_formula,
    null_formula = null_formula,
    pair_kind = "constrained",
    control = controls$initial,
    method_id = "initial_nloptwrap"
  )
  if (isTRUE(initial$primary_usable)) {
    return(q1p_adjudicate_constrained_pair(initial))
  }
  rescue <- q1p_fit_pair_once(
    data = data,
    full_formula = full_formula,
    null_formula = null_formula,
    pair_kind = "constrained",
    control = controls$rescue,
    method_id = "rescue_bobyqa"
  )
  q1p_adjudicate_constrained_pair(initial, rescue)
}


q1p_fit_unstructured_pair <- function(data, controls) {
  slopeaudit:::.sa_q1_fit_observed_pair_rescue(
    full_formula = y ~ time + (1 + time | id),
    null_formula = y ~ time + (1 | id),
    data = data,
    target = "time",
    REML = TRUE,
    initial_control = controls$initial,
    rescue_control = controls$rescue
  )
}


q1p_assessment_field <- function(x, field, default = NA) {
  if (is.null(x) || is.null(x[[field]]) || !length(x[[field]])) {
    return(default)
  }
  x[[field]]
}


q1p_pair_result_fields <- function(pair, prefix) {
  chosen <- pair$chosen
  initial <- pair$initial
  rescue <- pair$rescue
  numerical_zero_floored <- isTRUE(pair$numerical_zero_floored)
  statistic_raw <- if (identical(pair$status, "ok")) {
    q1p_assessment_field(chosen, "LR_raw", NA_real_)
  } else {
    NA_real_
  }
  statistic <- if (identical(pair$status, "ok")) {
    if (numerical_zero_floored) 0 else statistic_raw
  } else {
    NA_real_
  }
  values <- list(
    status = pair$status,
    source = pair$source,
    statistic = statistic,
    statistic_raw = statistic_raw,
    numerical_zero_floored = numerical_zero_floored,
    numerical_zero_rule = q1p_assessment_field(
      pair,
      "numerical_zero_rule",
      if (numerical_zero_floored) "rescued_small_negative" else ""
    ),
    boundary_optimizer_code_adjudicated = isTRUE(
      pair$boundary_optimizer_code_adjudicated
    ),
    numerical_zero_floor_reason = q1p_assessment_field(
      pair,
      "numerical_zero_floor_reason",
      ""
    ),
    numerical_zero_floor_tolerance = if (numerical_zero_floored) {
      q1p_assessment_field(chosen, "tolerance", NA_real_)
    } else {
      NA_real_
    },
    rescue_triggered = pair$rescue_triggered,
    rescue_succeeded = pair$rescue_succeeded,
    initial_status = q1p_assessment_field(initial, "status", "not_run"),
    initial_reason = q1p_assessment_field(initial, "reason", ""),
    initial_LR_raw = q1p_assessment_field(initial, "LR_raw", NA_real_),
    rescue_status = q1p_assessment_field(rescue, "status", "not_run"),
    rescue_reason = q1p_assessment_field(rescue, "reason", ""),
    rescue_LR_raw = q1p_assessment_field(rescue, "LR_raw", NA_real_),
    chosen_logLik_full = q1p_assessment_field(chosen, "logLik_full", NA_real_),
    chosen_logLik_null = q1p_assessment_field(chosen, "logLik_null", NA_real_),
    chosen_tolerance = q1p_assessment_field(chosen, "tolerance", NA_real_),
    chosen_structure_ok = q1p_assessment_field(chosen, "structure_ok", NA),
    chosen_full_optimizer_code = q1p_assessment_field(
      chosen,
      "full_optimizer_code",
      ""
    ),
    chosen_null_optimizer_code = q1p_assessment_field(
      chosen,
      "null_optimizer_code",
      ""
    ),
    chosen_full_convergence_messages = q1p_assessment_field(
      chosen,
      "full_convergence_messages",
      ""
    ),
    chosen_null_convergence_messages = q1p_assessment_field(
      chosen,
      "null_convergence_messages",
      ""
    ),
    chosen_full_non_singular_convergence_messages = q1p_assessment_field(
      chosen,
      "full_non_singular_convergence_messages",
      ""
    ),
    chosen_null_non_singular_convergence_messages = q1p_assessment_field(
      chosen,
      "null_non_singular_convergence_messages",
      ""
    ),
    chosen_full_slope_variance = q1p_assessment_field(
      chosen,
      "full_slope_variance",
      NA_real_
    ),
    chosen_full_slope_relative_sd = q1p_assessment_field(
      chosen,
      "full_slope_relative_sd",
      NA_real_
    ),
    chosen_full_slope_at_boundary = q1p_assessment_field(
      chosen,
      "full_slope_at_boundary",
      NA
    ),
    chosen_full_singular = q1p_assessment_field(
      chosen,
      "full_singular",
      NA
    ),
    chosen_null_singular = q1p_assessment_field(
      chosen,
      "null_singular",
      NA
    ),
    chosen_warnings = q1p_assessment_field(chosen, "warnings", ""),
    chosen_messages = q1p_assessment_field(chosen, "messages", "")
  )
  names(values) <- paste0(prefix, "_", names(values))
  values
}


q1p_run_dataset <- function(
    task,
    schedule,
    truth,
    controls,
    coupling = "legacy_hold_intercept") {

  started <- proc.time()[["elapsed"]]
  data <- q1p_generate_dataset(
    schedule,
    truth,
    task$seed,
    coupling = coupling
  )
  direct <- q1p_direct_slope_statistic(
    data,
    residual_variance = truth$residual_variance
  )
  constrained <- q1p_fit_constrained_pair(data, controls)
  unstructured <- q1p_fit_unstructured_pair(data, controls)

  constrained_fields <- q1p_pair_result_fields(
    constrained,
    "constrained"
  )
  unstructured_fields <- q1p_pair_result_fields(
    unstructured,
    "unstructured"
  )
  all_ok <- identical(constrained$status, "ok") &&
    identical(unstructured$status, "ok")

  core <- list(
    task_id = task$task_id,
    design_id = task$design_id,
    scenario_id = task$scenario_id,
    truth_role = task$truth_role,
    replicate = task$replicate,
    seed = task$seed,
    coupling = task$coupling,
    task_origin = task$task_origin,
    pair_id = task$pair_id,
    status = if (all_ok) "ok" else "diagnose",
    direct_statistic = direct$statistic,
    direct_df = direct$df,
    direct_error_variance = direct$direct_error_variance,
    elapsed_seconds = proc.time()[["elapsed"]] - started
  )
  as.data.frame(
    c(core, constrained_fields, unstructured_fields),
    stringsAsFactors = FALSE
  )
}


q1p_empirical_cutoff <- function(statistic, alpha = 0.05) {
  statistic <- as.numeric(statistic)
  if (!length(statistic) || any(!is.finite(statistic))) {
    stop("Empirical calibration statistics must all be finite.",
         call. = FALSE)
  }
  reject_target <- as.integer(floor(alpha * length(statistic)))
  if (reject_target < 1L || reject_target >= length(statistic)) {
    stop("The null sample is too small for the requested empirical size.",
         call. = FALSE)
  }

  ordered <- sort(statistic)
  critical_index <- length(ordered) - reject_target
  critical <- ordered[[critical_index]]
  n_above <- sum(statistic > critical)
  n_equal <- sum(statistic == critical)

  list(
    critical = critical,
    critical_index = critical_index,
    target_rejections = reject_target,
    n_above = n_above,
    n_equal = n_equal,
    realized_size = n_above / length(statistic),
    exact_without_randomization =
      abs(n_above / length(statistic) - alpha) <
        100 * .Machine$double.eps,
    conservative_without_randomization = n_above / length(statistic) <=
      alpha + 100 * .Machine$double.eps
  )
}


q1p_exact_interval <- function(k, n) {
  slopeaudit:::.sa_q1_exact_binomial_interval(k, n, 0.95)
}


q1p_numerical_zero_scope_ok <- function(results) {
  required <- c(
    "unstructured_numerical_zero_floored",
    "constrained_numerical_zero_rule",
    "constrained_numerical_zero_floored",
    "constrained_status",
    "constrained_rescue_succeeded",
    "constrained_statistic",
    "constrained_statistic_raw",
    "constrained_numerical_zero_floor_tolerance",
    "constrained_source",
    "constrained_rescue_status",
    "constrained_rescue_reason",
    "constrained_boundary_optimizer_code_adjudicated",
    "constrained_chosen_full_slope_at_boundary",
    "constrained_chosen_full_optimizer_code",
    "constrained_chosen_null_optimizer_code",
    "constrained_chosen_full_non_singular_convergence_messages",
    "constrained_chosen_null_non_singular_convergence_messages"
  )
  if (!is.data.frame(results) || !all(required %in% names(results))) {
    return(FALSE)
  }
  if (any(results$unstructured_numerical_zero_floored %in% TRUE)) {
    return(FALSE)
  }

  floored <- results$constrained_numerical_zero_floored %in% TRUE
  if (!any(floored)) {
    return(TRUE)
  }
  rows <- results[floored, , drop = FALSE]
  basic <- rows$constrained_status == "ok" &
    rows$constrained_rescue_succeeded %in% TRUE &
    is.finite(rows$constrained_statistic) &
    rows$constrained_statistic == 0 &
    is.finite(rows$constrained_statistic_raw) &
    is.finite(rows$constrained_numerical_zero_floor_tolerance) &
    abs(rows$constrained_statistic_raw) <=
      rows$constrained_numerical_zero_floor_tolerance

  small_negative <-
    rows$constrained_numerical_zero_rule == "rescued_small_negative" &
    rows$constrained_source == "rescue_bobyqa_numerical_zero" &
    rows$constrained_statistic_raw < 0 &
    rows$constrained_rescue_status == "small_negative" &
    rows$constrained_rescue_reason == "small_finite_negative_LR"

  optimizer_code_boundary <-
    rows$constrained_numerical_zero_rule ==
      "optimizer_code_boundary_zero" &
    rows$constrained_source ==
      "rescue_bobyqa_boundary_zero_optimizer_code" &
    rows$constrained_boundary_optimizer_code_adjudicated %in% TRUE &
    rows$constrained_chosen_full_slope_at_boundary %in% TRUE &
    xor(
      rows$constrained_chosen_full_optimizer_code != "0",
      rows$constrained_chosen_null_optimizer_code != "0"
    ) &
    !nzchar(
      rows$constrained_chosen_full_non_singular_convergence_messages
    ) &
    !nzchar(
      rows$constrained_chosen_null_non_singular_convergence_messages
    )

  isTRUE(all(basic & (small_negative | optimizer_code_boundary)))
}


q1p_run_self_tests <- function() {
  three <- q1p_analytic_values(
    100L,
    c(0, 1, 2),
    0.01,
    1,
    0.05
  )
  five <- q1p_analytic_values(
    100L,
    seq(0, 5, length.out = 5),
    0.01,
    1,
    0.05
  )
  stopifnot(
    abs(three[["R_direct"]] - 0.0196078431372549) < 1e-14,
    abs(three[["analytic_direct_power"]] - 0.0674074967783115) < 1e-12,
    abs(five[["R_direct"]] - 0.135135135135135) < 1e-14,
    abs(five[["analytic_direct_power"]] - 0.283604839704666) < 1e-12
  )

  cutoff <- q1p_empirical_cutoff(seq_len(20L), alpha = 0.05)
  stopifnot(
    cutoff$critical == 19,
    cutoff$n_above == 1L,
    cutoff$exact_without_randomization
  )
  cutoff_999 <- q1p_empirical_cutoff(seq_len(999L), alpha = 0.05)
  stopifnot(
    cutoff_999$critical == 950,
    cutoff_999$target_rejections == 49L,
    cutoff_999$n_above == 49L,
    abs(cutoff_999$realized_size - 49 / 999) < 1e-15,
    !cutoff_999$exact_without_randomization,
    cutoff_999$conservative_without_randomization
  )

  schedule <- q1p_make_balanced_schedule(4L, c(0, 1, 2))
  slopes <- rep(c(-1.5, -0.5, 0.5, 1.5), each = 3L)
  constructed <- transform(schedule, y = slopes * time)
  direct <- q1p_direct_slope_statistic(constructed, residual_variance = 1)
  expected <- sum((c(-1.5, -0.5, 0.5, 1.5))^2) / 0.5
  stopifnot(abs(direct$statistic - expected) < 1e-12)

  assessment <- function(
      status,
      LR_raw,
      tolerance = 1e-6,
      full_optimizer_code = "0",
      null_optimizer_code = "0",
      full_valid = TRUE,
      null_valid = TRUE,
      full_slope_at_boundary = FALSE,
      primary_usable = status %in% c("positive", "zero"),
      reason = if (identical(status, "small_negative")) {
        "small_finite_negative_LR"
      } else if (identical(status, "material_negative")) {
        "material_negative_LR"
      } else {
        ""
      }) {
    list(
      status = status,
      LR_raw = LR_raw,
      tolerance = tolerance,
      primary_usable = primary_usable,
      reason = reason,
      full_valid = full_valid,
      null_valid = null_valid,
      optimizer_honored = TRUE,
      structure_ok = TRUE,
      logLik_full = -100 + LR_raw / 2,
      logLik_null = -100,
      full_singular = FALSE,
      null_singular = FALSE,
      full_optimizer_code = full_optimizer_code,
      null_optimizer_code = null_optimizer_code,
      full_convergence_messages = "",
      null_convergence_messages = "",
      full_non_singular_convergence_messages = "",
      null_non_singular_convergence_messages = "",
      full_slope_variance = if (full_slope_at_boundary) 0 else 0.01,
      full_residual_variance = 1,
      full_slope_relative_sd = if (full_slope_at_boundary) 0 else 0.1,
      full_slope_at_boundary = full_slope_at_boundary,
      warnings = "",
      messages = ""
    )
  }

  initial_small <- assessment("small_negative", -2e-7)
  rescue_small <- assessment("small_negative", -4e-13)
  floored <- q1p_adjudicate_constrained_pair(
    initial_small,
    rescue_small
  )
  floored_fields <- q1p_pair_result_fields(floored, "constrained")
  stopifnot(
    identical(floored$status, "ok"),
    identical(floored$source, "rescue_bobyqa_numerical_zero"),
    isTRUE(floored$rescue_triggered),
    isTRUE(floored$rescue_succeeded),
    isTRUE(floored$numerical_zero_floored),
    identical(floored_fields$constrained_statistic, 0),
    identical(floored_fields$constrained_statistic_raw, -4e-13),
    identical(
      floored_fields$constrained_numerical_zero_floor_tolerance,
      1e-6
    )
  )

  rescue_material <- assessment(
    "material_negative",
    -2e-3,
    tolerance = 1e-6
  )
  unresolved <- q1p_adjudicate_constrained_pair(
    initial_small,
    rescue_material
  )
  stopifnot(
    identical(unresolved$status, "unresolved_after_bobyqa"),
    !isTRUE(unresolved$rescue_succeeded)
  )

  invalid_small <- rescue_small
  invalid_small$full_valid <- FALSE
  stopifnot(!q1p_rescued_small_negative_is_floorable(invalid_small))

  rescue_boundary <- assessment(
    "failure",
    4e-13,
    full_optimizer_code = "1",
    null_optimizer_code = "0",
    full_valid = FALSE,
    null_valid = TRUE,
    full_slope_at_boundary = TRUE,
    primary_usable = FALSE,
    reason = "full:optimizer_nonzero_code"
  )
  rescue_boundary$full_convergence_messages <-
    "boundary (singular) fit: see help('isSingular')"
  boundary <- q1p_adjudicate_constrained_pair(
    initial_small,
    rescue_boundary
  )
  boundary_fields <- q1p_pair_result_fields(boundary, "constrained")
  stopifnot(
    identical(boundary$status, "ok"),
    identical(
      boundary$source,
      "rescue_bobyqa_boundary_zero_optimizer_code"
    ),
    isTRUE(boundary$boundary_optimizer_code_adjudicated),
    identical(boundary_fields$constrained_statistic, 0),
    identical(boundary_fields$constrained_statistic_raw, 4e-13),
    identical(
      boundary_fields$constrained_numerical_zero_rule,
      "optimizer_code_boundary_zero"
    )
  )
  boundary_bad_message <- rescue_boundary
  boundary_bad_message$full_non_singular_convergence_messages <-
    "max|grad| exceeds tolerance"
  stopifnot(!q1p_rescued_boundary_code_is_adjudicable(
    initial_small,
    boundary_bad_message
  ))
  boundary_not_at_edge <- rescue_boundary
  boundary_not_at_edge$full_slope_at_boundary <- FALSE
  stopifnot(!q1p_rescued_boundary_code_is_adjudicable(
    initial_small,
    boundary_not_at_edge
  ))

  boundary_scope <- data.frame(
    unstructured_numerical_zero_floored = FALSE,
    constrained_numerical_zero_rule = "optimizer_code_boundary_zero",
    constrained_numerical_zero_floored = TRUE,
    constrained_status = "ok",
    constrained_rescue_succeeded = TRUE,
    constrained_statistic = 0,
    constrained_statistic_raw = 4e-13,
    constrained_numerical_zero_floor_tolerance = 1e-6,
    constrained_source = "rescue_bobyqa_boundary_zero_optimizer_code",
    constrained_rescue_status = "failure",
    constrained_rescue_reason = "full:optimizer_nonzero_code",
    constrained_boundary_optimizer_code_adjudicated = TRUE,
    constrained_chosen_full_slope_at_boundary = TRUE,
    constrained_chosen_full_optimizer_code = "1",
    constrained_chosen_null_optimizer_code = "0",
    constrained_chosen_full_convergence_messages =
      "boundary (singular) fit: see help('isSingular')",
    constrained_chosen_null_convergence_messages = "",
    constrained_chosen_full_non_singular_convergence_messages = "",
    constrained_chosen_null_non_singular_convergence_messages = "",
    stringsAsFactors = FALSE
  )
  stopifnot(q1p_numerical_zero_scope_ok(boundary_scope))

  boundary_scope_gradient <- boundary_scope
  boundary_scope_gradient[[
    "constrained_chosen_full_non_singular_convergence_messages"
  ]] <- "max|grad| exceeds tolerance"
  stopifnot(!q1p_numerical_zero_scope_ok(boundary_scope_gradient))

  boundary_scope_hessian <- boundary_scope
  boundary_scope_hessian[[
    "constrained_chosen_null_non_singular_convergence_messages"
  ]] <- "Hessian is numerically singular"
  stopifnot(!q1p_numerical_zero_scope_ok(boundary_scope_hessian))

  paired_truth_0 <- list(
    random_intercept_variance = 0.5,
    random_slope_variance = 0.01,
    intercept_slope_correlation = 0,
    residual_variance = 1,
    fixed_intercept = 0,
    fixed_slope = 0.1
  )
  paired_truth_5 <- paired_truth_0
  paired_truth_5$intercept_slope_correlation <- 0.5
  paired_schedule <- q1p_make_balanced_schedule(10L, c(0, 1, 2))
  paired_0 <- q1p_generate_dataset(
    paired_schedule,
    paired_truth_0,
    12345L,
    coupling = "paired_hold_slope"
  )
  paired_5 <- q1p_generate_dataset(
    paired_schedule,
    paired_truth_5,
    12345L,
    coupling = "paired_hold_slope"
  )
  stopifnot(abs(
    q1p_direct_slope_statistic(paired_0, 1)$statistic -
      q1p_direct_slope_statistic(paired_5, 1)$statistic
  ) <= 1e-10)

  invisible(TRUE)
}
