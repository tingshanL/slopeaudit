# Plug-in parametric bootstrap for Question 1.
#
# Simulation occurs under the fitted null model at the fitted null
# parameters. Each simulated response is refitted with the supplied null and
# full model structures.

.sa_q1_validate_simulated_response <- function(fit, y) {
  if (!is.numeric(y)) {
    stop(
      "Question 1 currently requires a numeric Gaussian simulated response.",
      call. = FALSE
    )
  }

  n_retained <- stats::nobs(fit)
  fit_na <- attr(stats::model.frame(fit), "na.action")

  if (is.null(fit_na)) {
    if (length(y) != n_retained || any(!is.finite(y))) {
      stop(
        paste0(
          "Simulation did not preserve the fitted model's retained ",
          "response rows."
        ),
        call. = FALSE
      )
    }

    return(invisible(TRUE))
  }

  omitted <- as.integer(fit_na)
  if (anyNA(omitted) || any(omitted < 1L)) {
    stop(
      "The fitted model has an invalid `na.action` index.",
      call. = FALSE
    )
  }

  n_original <- n_retained + length(omitted)

  if (length(y) == n_retained) {
    if (any(!is.finite(y))) {
      stop(
        "Retained simulated response contains nonfinite values.",
        call. = FALSE
      )
    }

    return(invisible(TRUE))
  }

  if (length(y) != n_original || any(omitted > length(y))) {
    stop(
      paste0(
        "Simulation length is incompatible with the fitted model's ",
        "missing-wave pattern."
      ),
      call. = FALSE
    )
  }

  keep <- setdiff(seq_along(y), omitted)

  if (any(!is.finite(y[keep]))) {
    stop(
      "Simulation contains nonfinite values on retained analysis rows.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


.sa_q1_simulate_null <- function(null_fit, seed) {
  # Match lme4's own bootMer parametric-bootstrap route: simulate from the
  # fitted model while allowing na.exclude to preserve the original
  # missing-wave pattern for refit.merMod().
  ans <- stats::simulate(
    object = null_fit,
    nsim = 1L,
    seed = seed,
    re.form = NA,
    na.action = stats::na.exclude
  )

  if (!is.data.frame(ans) || ncol(ans) != 1L) {
    stop(
      "Null simulation did not return one simulated-response column.",
      call. = FALSE
    )
  }

  y <- ans[[1]]
  .sa_q1_validate_simulated_response(null_fit, y)

  y
}


.sa_q1_refit <- function(template_fit, new_response, control) {
  if (!inherits(control, "lmerControl")) {
    stop("`control` must be an `lmerControl` object.", call. = FALSE)
  }

  lme4::refit(
    object = template_fit,
    newresp = new_response,
    control = control
  )
}


.sa_q1_failure_row <- function(
    replicate,
    seed,
    reason,
    warnings = character(),
    messages = character()) {

  data.frame(
    replicate = as.integer(replicate),
    seed = seed,
    status = "failure",
    reason = as.character(reason),
    LR_raw = NA_real_,
    LR_zero = NA_real_,
    tolerance = NA_real_,
    logLik_full = NA_real_,
    logLik_null = NA_real_,
    full_singular = NA,
    null_singular = NA,
    warnings = paste(unique(warnings), collapse = " | "),
    messages = paste(unique(messages), collapse = " | "),
    stringsAsFactors = FALSE
  )
}


.sa_q1_one_bootstrap_replicate <- function(
    replicate,
    seed,
    full_fit,
    null_fit,
    target,
    identity_tol = .sa_identity_tol,
    strict_controls = TRUE,
    refit_control,
    simulate_fun = .sa_q1_simulate_null,
    refit_fun = .sa_q1_refit) {

  sim <- .sa_q1_capture(
    simulate_fun(null_fit, seed)
  )

  if (!is.null(sim$error)) {
    return(.sa_q1_failure_row(
      replicate = replicate,
      seed = seed,
      reason = paste0("simulation_error:", sim$error),
      warnings = sim$warnings,
      messages = sim$messages
    ))
  }

  y <- sim$value

  sim_valid <- tryCatch(
    {
      .sa_q1_validate_simulated_response(null_fit, y)
      TRUE
    },
    error = function(e) FALSE
  )

  if (!isTRUE(sim_valid)) {
    return(.sa_q1_failure_row(
      replicate = replicate,
      seed = seed,
      reason = "simulation_invalid_response",
      warnings = sim$warnings,
      messages = sim$messages
    ))
  }

  null_refit_capture <- .sa_q1_capture(
    refit_fun(null_fit, y, refit_control)
  )

  if (!is.null(null_refit_capture$error)) {
    return(.sa_q1_failure_row(
      replicate = replicate,
      seed = seed,
      reason = paste0(
        "null_refit_error:",
        null_refit_capture$error
      ),
      warnings = c(
        sim$warnings,
        null_refit_capture$warnings
      ),
      messages = c(
        sim$messages,
        null_refit_capture$messages
      )
    ))
  }

  full_refit_capture <- .sa_q1_capture(
    refit_fun(full_fit, y, refit_control)
  )

  if (!is.null(full_refit_capture$error)) {
    return(.sa_q1_failure_row(
      replicate = replicate,
      seed = seed,
      reason = paste0(
        "full_refit_error:",
        full_refit_capture$error
      ),
      warnings = c(
        sim$warnings,
        null_refit_capture$warnings,
        full_refit_capture$warnings
      ),
      messages = c(
        sim$messages,
        null_refit_capture$messages,
        full_refit_capture$messages
      )
    ))
  }

  null_b <- null_refit_capture$value
  full_b <- full_refit_capture$value

  null_template_check <- .sa_q1_refit_matches_template(
    null_b,
    null_fit
  )
  full_template_check <- .sa_q1_refit_matches_template(
    full_b,
    full_fit
  )

  if (!null_template_check$valid ||
      !full_template_check$valid) {
    reasons <- c(
      if (!null_template_check$valid) {
        paste0("null:", null_template_check$reason)
      },
      if (!full_template_check$valid) {
        paste0("full:", full_template_check$reason)
      }
    )

    return(.sa_q1_failure_row(
      replicate = replicate,
      seed = seed,
      reason = paste(
        "incompatible_refit",
        paste(reasons, collapse = ";"),
        sep = ":"
      ),
      warnings = c(
        sim$warnings,
        null_refit_capture$warnings,
        full_refit_capture$warnings
      ),
      messages = c(
        sim$messages,
        null_refit_capture$messages,
        full_refit_capture$messages
      )
    ))
  }

  pair_check <- tryCatch(
    {
      .sa_q1_validate_model_pair(
        full_fit = full_b,
        null_fit = null_b,
        target = target,
        identity_tol = identity_tol,
        strict_controls = FALSE
      )
      NULL
    },
    error = function(e) conditionMessage(e)
  )

  if (!is.null(pair_check)) {
    return(.sa_q1_failure_row(
      replicate = replicate,
      seed = seed,
      reason = paste0(
        "incompatible_refit_pair:",
        pair_check
      ),
      warnings = c(
        sim$warnings,
        null_refit_capture$warnings,
        full_refit_capture$warnings
      ),
      messages = c(
        sim$messages,
        null_refit_capture$messages,
        full_refit_capture$messages
      )
    ))
  }

  full_diag <- .sa_q1_fit_diagnostics(
    full_b,
    control = refit_control
  )
  null_diag <- .sa_q1_fit_diagnostics(
    null_b,
    control = refit_control
  )

  if (!full_diag$valid || !null_diag$valid) {
    reason <- paste(
      c(
        if (!full_diag$valid) {
          paste0("full:", full_diag$reason)
        },
        if (!null_diag$valid) {
          paste0("null:", null_diag$reason)
        }
      ),
      collapse = ";"
    )

    return(.sa_q1_failure_row(
      replicate = replicate,
      seed = seed,
      reason = reason,
      warnings = c(
        sim$warnings,
        null_refit_capture$warnings,
        full_refit_capture$warnings
      ),
      messages = c(
        sim$messages,
        null_refit_capture$messages,
        full_refit_capture$messages
      )
    ))
  }

  lr <- .sa_q1_classify_lr(
    logLik_full = full_diag$logLik,
    logLik_null = null_diag$logLik
  )

  data.frame(
    replicate = as.integer(replicate),
    seed = seed,
    status = lr$status,
    reason = if (is.na(lr$reason)) "" else lr$reason,
    LR_raw = lr$LR_raw,
    LR_zero = lr$LR_zero,
    tolerance = lr$tolerance,
    logLik_full = full_diag$logLik,
    logLik_null = null_diag$logLik,
    full_singular = isTRUE(full_diag$singular),
    null_singular = isTRUE(null_diag$singular),
    warnings = paste(
      unique(c(
        sim$warnings,
        null_refit_capture$warnings,
        full_refit_capture$warnings
      )),
      collapse = " | "
    ),
    messages = paste(
      unique(c(
        sim$messages,
        null_refit_capture$messages,
        full_refit_capture$messages
      )),
      collapse = " | "
    ),
    stringsAsFactors = FALSE
  )
}


.sa_q1_bootstrap <- function(
    full_fit,
    null_fit,
    target,
    B = 999L,
    seed = 1L,
    conf_level = 0.95,
    alpha = 0.05,
    retain_replicates = FALSE,
    progress = interactive() && B >= 50L,
    identity_tol = .sa_identity_tol,
    strict_controls = TRUE,
    escalation_target_B = 4999L,
    refit_control = NULL,
    simulate_fun = .sa_q1_simulate_null,
    refit_fun = .sa_q1_refit) {

  if (!is.numeric(B) || length(B) != 1L ||
      !is.finite(B) || B != floor(B) ||
      B < 1 || B > .Machine$integer.max) {
    stop("`B` must be one positive R integer.", call. = FALSE)
  }
  B <- as.integer(B)

  seed <- .sa_q1_validate_seed(seed)
  conf_level <- .sa_q1_validate_probability(
    conf_level,
    "conf_level"
  )
  alpha <- .sa_q1_validate_probability(alpha, "alpha")

  if (!is.numeric(escalation_target_B) ||
      length(escalation_target_B) != 1L ||
      !is.finite(escalation_target_B) ||
      escalation_target_B != floor(escalation_target_B) ||
      escalation_target_B < 1 ||
      escalation_target_B > .Machine$integer.max) {
    stop(
      "`escalation_target_B` must be one positive R integer.",
      call. = FALSE
    )
  }
  escalation_target_B <- as.integer(escalation_target_B)

  retain_replicates <- .sa_q1_validate_flag(
    retain_replicates,
    "retain_replicates"
  )
  progress <- .sa_q1_validate_flag(
    progress,
    "progress"
  )
  strict_controls <- .sa_q1_validate_flag(
    strict_controls,
    "strict_controls"
  )

  if (!is.function(simulate_fun)) {
    stop("`simulate_fun` must be a function.", call. = FALSE)
  }
  if (!is.function(refit_fun)) {
    stop("`refit_fun` must be a function.", call. = FALSE)
  }

  resolved_control <- .sa_q1_resolve_refit_control(
    full_fit = full_fit,
    null_fit = null_fit,
    refit_control = refit_control
  )
  frozen_refit_control <- resolved_control$control

  observed <- .sa_q1_observed_lr(
    full_fit = full_fit,
    null_fit = null_fit,
    target = target,
    diagnostic_control = frozen_refit_control,
    identity_tol = identity_tol,
    strict_controls = strict_controls
  )

  if (!isTRUE(observed$valid_for_inference)) {
    return(structure(
      list(
        status = "invalid_observed",
        target = target,
        observed = observed,
        bootstrap = NULL,
        replicates = NULL,
        B_requested = B,
        master_seed = seed,
        simulation_source = "fitted null plug-in",
        likelihood_mode = observed$pair$likelihood_mode
      ),
      class = "slopeaudit_q1_internal"
    ))
  }

  rng_snapshot <- .sa_q1_rng_snapshot()
  on.exit(.sa_q1_rng_restore(rng_snapshot), add = TRUE)

  seeds <- .sa_q1_make_replicate_seeds(
    B = B,
    seed = seed
  )

  rows <- vector("list", B)
  progress_bar <- NULL

  if (isTRUE(progress)) {
    progress_bar <- utils::txtProgressBar(
      min = 0,
      max = B,
      initial = 0,
      style = 3
    )
    on.exit(close(progress_bar), add = TRUE)
  }

  for (b in seq_len(B)) {
    rows[[b]] <- .sa_q1_one_bootstrap_replicate(
      replicate = b,
      seed = seeds[[b]],
      full_fit = full_fit,
      null_fit = null_fit,
      target = target,
      identity_tol = identity_tol,
      strict_controls = strict_controls,
      refit_control = frozen_refit_control,
      simulate_fun = simulate_fun,
      refit_fun = refit_fun
    )

    if (isTRUE(progress)) {
      utils::setTxtProgressBar(progress_bar, b)
    }
  }

  replicates <- do.call(rbind, rows)
  rownames(replicates) <- NULL

  summary <- .sa_q1_summarize_replicates(
    replicates = replicates,
    observed_LR = observed$LR_for_inference,
    conf_level = conf_level,
    alpha = alpha,
    escalation_target_B = escalation_target_B
  )

  structure(
    list(
      status = "ok",
      target = target,
      pair = observed$pair,
      observed = observed,
      bootstrap = summary,
      replicates = if (isTRUE(retain_replicates)) {
        replicates
      } else {
        NULL
      },
      B_requested = B,
      master_seed = seed,
      replicate_seeds = seeds,
      seed_stream = "replicate_seed_stream_v1",
      execution_mode = "sequential_v1",
      progress_requested = isTRUE(progress),
      simulation_source =
        "plug-in simulation from fitted null parameters",
      likelihood_mode = observed$pair$likelihood_mode,
      optimizer = observed$pair$optimizer,
      refit_control_source = resolved_control$source,
      refit_control = frozen_refit_control,
      refit_optimizer_used = resolved_control$optimizer_used,
      refit_optimizer_control = resolved_control$optimizer_control,
      full_call_control = resolved_control$full_call_control,
      null_call_control = resolved_control$null_call_control,
      validity_rules = list(
        small_negative = "dual p_drop / p_zero",
        material_negative = paste0(
          "exclude from p_drop and p_zero; report p_all_zero sensitivity ",
          "with all finite negative LRs set to zero"
        ),
        failed_or_nonfinite = "exclude and count by reason",
        singular_fit = "retain if otherwise converged; flag separately"
      ),
      software = list(
        R = paste(
          R.version$major,
          R.version$minor,
          sep = "."
        ),
        lme4 = as.character(
          utils::packageVersion("lme4")
        )
      )
    ),
    class = "slopeaudit_q1_internal"
  )
}
