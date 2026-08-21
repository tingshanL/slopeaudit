# Deterministic optimizer rescue for the validated Question-1 bootstrap.
#
# This is the internal rescue hierarchy adopted by the release implementation.
# A full/null pair is always attempted with
# the declared primary control first. Both models are refitted with the declared
# rescue control only when that pair is not primary-usable. Optimizers are never
# mixed within a pair.

.sa_q1_calibration_rescue_controls <- function() {
  list(
    initial = lme4::lmerControl(
      optimizer = "nloptwrap",
      restart_edge = TRUE,
      boundary.tol = 1e-5,
      calc.derivs = TRUE,
      optCtrl = list(maxeval = 200000),
      check.conv.singular = "message"
    ),
    rescue = lme4::lmerControl(
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
  )
}


.sa_q1_validate_calibration_rescue_controls <- function(
    initial_control,
    rescue_control) {

  if (!inherits(initial_control, "lmerControl") ||
      !inherits(rescue_control, "lmerControl")) {
    stop(
      "Both calibration controls must be `lmerControl` objects.",
      call. = FALSE
    )
  }
  expected <- .sa_q1_calibration_rescue_controls()
  if (!isTRUE(all.equal(
    initial_control,
    expected$initial,
    check.attributes = TRUE
  ))) {
    stop(
      "The initial calibration control differs from the frozen v12 control.",
      call. = FALSE
    )
  }
  if (!isTRUE(all.equal(
    rescue_control,
    expected$rescue,
    check.attributes = TRUE
  ))) {
    stop(
      "The rescue calibration control differs from the frozen v12 control.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


.sa_q1_capture_text <- function(x) {
  if (is.null(x) || !length(x)) {
    return("")
  }
  paste(unique(as.character(unlist(x, use.names = FALSE))), collapse = " | ")
}


.sa_q1_pair_assessment <- function(
    full_capture,
    null_capture,
    target,
    control,
    method_id,
    identity_tol = .sa_identity_tol) {

  full_fit <- full_capture$value
  null_fit <- null_capture$value
  full_error <- full_capture$error
  null_error <- null_capture$error
  warnings <- .sa_q1_capture_text(c(
    full_capture$warnings,
    null_capture$warnings
  ))
  messages <- .sa_q1_capture_text(c(
    full_capture$messages,
    null_capture$messages
  ))

  empty_diag <- list(
    valid = FALSE,
    reason = "fit_unavailable",
    logLik = NA_real_,
    optimizer = NA_character_,
    optimizer_code = NA_integer_,
    convergence_messages = character(),
    singular = NA
  )
  full_diag <- empty_diag
  null_diag <- empty_diag
  pair_error <- ""

  if (is.null(full_error) && is.null(null_error)) {
    pair_error <- tryCatch(
      {
        .sa_q1_validate_model_pair(
          full_fit = full_fit,
          null_fit = null_fit,
          target = target,
          identity_tol = identity_tol,
          strict_controls = FALSE
        )
        ""
      },
      error = function(error) conditionMessage(error)
    )
    full_diag <- .sa_q1_fit_diagnostics(full_fit, control = control)
    null_diag <- .sa_q1_fit_diagnostics(null_fit, control = control)
  }

  declared <- as.character(control$optimizer)
  optimizer_honored <- is.null(full_error) && is.null(null_error) &&
    full_diag$optimizer %in% declared && null_diag$optimizer %in% declared

  if (!is.null(full_error) || !is.null(null_error)) {
    status <- "failure"
    reason <- paste(
      c(
        if (!is.null(full_error)) paste0("full_refit_error:", full_error),
        if (!is.null(null_error)) paste0("null_refit_error:", null_error)
      ),
      collapse = ";"
    )
    lr <- list(
      LR_raw = NA_real_,
      LR_zero = NA_real_,
      tolerance = NA_real_
    )
  } else if (nzchar(pair_error)) {
    status <- "failure"
    reason <- paste0("incompatible_refit_pair:", pair_error)
    lr <- list(
      LR_raw = NA_real_,
      LR_zero = NA_real_,
      tolerance = NA_real_
    )
  } else if (!isTRUE(optimizer_honored)) {
    status <- "failure"
    reason <- "declared_optimizer_not_honored"
    lr <- list(
      LR_raw = if (is.finite(full_diag$logLik) &&
        is.finite(null_diag$logLik)) {
        2 * (full_diag$logLik - null_diag$logLik)
      } else {
        NA_real_
      },
      LR_zero = NA_real_,
      tolerance = .sa_q1_lr_tolerance(
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
      LR_zero = NA_real_,
      tolerance = .sa_q1_lr_tolerance(
        full_diag$logLik,
        null_diag$logLik
      )
    )
  } else {
    lr <- .sa_q1_classify_lr(
      logLik_full = full_diag$logLik,
      logLik_null = null_diag$logLik
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
    LR_zero = lr$LR_zero,
    tolerance = lr$tolerance,
    logLik_full = full_diag$logLik,
    logLik_null = null_diag$logLik,
    full_valid = isTRUE(full_diag$valid),
    null_valid = isTRUE(null_diag$valid),
    full_singular = if (isTRUE(full_diag$valid)) {
      isTRUE(full_diag$singular)
    } else {
      NA
    },
    null_singular = if (isTRUE(null_diag$valid)) {
      isTRUE(null_diag$singular)
    } else {
      NA
    },
    full_optimizer = full_diag$optimizer,
    null_optimizer = null_diag$optimizer,
    optimizer_honored = optimizer_honored,
    warnings = warnings,
    messages = messages,
    full_fit = full_fit,
    null_fit = null_fit
  )
}


.sa_q1_refit_pair_attempt <- function(
    full_template,
    null_template,
    response,
    target,
    control,
    method_id,
    identity_tol = .sa_identity_tol,
    refit_fun = .sa_q1_refit) {

  null_capture <- .sa_q1_capture(
    refit_fun(null_template, response, control)
  )
  full_capture <- .sa_q1_capture(
    refit_fun(full_template, response, control)
  )

  .sa_q1_pair_assessment(
    full_capture = full_capture,
    null_capture = null_capture,
    target = target,
    control = control,
    method_id = method_id,
    identity_tol = identity_tol
  )
}


.sa_q1_adjudicate_pair_rescue <- function(initial, rescue = NULL) {
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
    stop("A rescue assessment is required after an unusable initial pair.")
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


.sa_q1_assessment_value <- function(x, name, default = NA) {
  if (is.null(x) || is.null(x[[name]]) || length(x[[name]]) == 0L) {
    return(default)
  }
  x[[name]]
}


.sa_q1_rescue_replicate_row <- function(
    replicate,
    seed,
    adjudication) {

  initial <- adjudication$initial
  rescue <- adjudication$rescue
  chosen <- adjudication$chosen
  final_ok <- identical(adjudication$status, "ok")

  data.frame(
    replicate = as.integer(replicate),
    seed = as.integer(seed),
    status = if (final_ok) chosen$status else "failure",
    reason = if (final_ok) {
      chosen$reason
    } else {
      paste0(
        "unresolved_after_bobyqa:",
        .sa_q1_assessment_value(rescue, "status", "not_run"),
        ":",
        .sa_q1_assessment_value(rescue, "reason", "")
      )
    },
    LR_raw = if (final_ok) chosen$LR_raw else NA_real_,
    LR_zero = if (final_ok) chosen$LR_zero else NA_real_,
    tolerance = if (final_ok) chosen$tolerance else NA_real_,
    logLik_full = if (final_ok) chosen$logLik_full else NA_real_,
    logLik_null = if (final_ok) chosen$logLik_null else NA_real_,
    full_singular = if (final_ok) chosen$full_singular else NA,
    null_singular = if (final_ok) chosen$null_singular else NA,
    warnings = if (final_ok) chosen$warnings else
      .sa_q1_assessment_value(rescue, "warnings", ""),
    messages = if (final_ok) chosen$messages else
      .sa_q1_assessment_value(rescue, "messages", ""),
    fit_source = adjudication$source,
    rescue_triggered = adjudication$rescue_triggered,
    rescue_succeeded = adjudication$rescue_succeeded,
    initial_status = initial$status,
    initial_reason = initial$reason,
    initial_LR_raw = initial$LR_raw,
    initial_tolerance = initial$tolerance,
    initial_logLik_full = initial$logLik_full,
    initial_logLik_null = initial$logLik_null,
    initial_full_valid = initial$full_valid,
    initial_null_valid = initial$null_valid,
    initial_full_singular = initial$full_singular,
    initial_null_singular = initial$null_singular,
    initial_full_optimizer = initial$full_optimizer,
    initial_null_optimizer = initial$null_optimizer,
    initial_optimizer_honored = initial$optimizer_honored,
    initial_warnings = initial$warnings,
    initial_messages = initial$messages,
    rescue_status = .sa_q1_assessment_value(rescue, "status", "not_run"),
    rescue_reason = .sa_q1_assessment_value(rescue, "reason", ""),
    rescue_LR_raw = .sa_q1_assessment_value(
      rescue,
      "LR_raw",
      NA_real_
    ),
    rescue_tolerance = .sa_q1_assessment_value(
      rescue,
      "tolerance",
      NA_real_
    ),
    rescue_logLik_full = .sa_q1_assessment_value(
      rescue,
      "logLik_full",
      NA_real_
    ),
    rescue_logLik_null = .sa_q1_assessment_value(
      rescue,
      "logLik_null",
      NA_real_
    ),
    rescue_full_valid = .sa_q1_assessment_value(
      rescue,
      "full_valid",
      NA
    ),
    rescue_null_valid = .sa_q1_assessment_value(
      rescue,
      "null_valid",
      NA
    ),
    rescue_full_singular = .sa_q1_assessment_value(
      rescue,
      "full_singular",
      NA
    ),
    rescue_null_singular = .sa_q1_assessment_value(
      rescue,
      "null_singular",
      NA
    ),
    rescue_full_optimizer = .sa_q1_assessment_value(
      rescue,
      "full_optimizer",
      ""
    ),
    rescue_null_optimizer = .sa_q1_assessment_value(
      rescue,
      "null_optimizer",
      ""
    ),
    rescue_optimizer_honored = .sa_q1_assessment_value(
      rescue,
      "optimizer_honored",
      NA
    ),
    rescue_warnings = .sa_q1_assessment_value(
      rescue,
      "warnings",
      ""
    ),
    rescue_messages = .sa_q1_assessment_value(
      rescue,
      "messages",
      ""
    ),
    stringsAsFactors = FALSE
  )
}


.sa_q1_simulation_failure_rescue_row <- function(
    replicate,
    seed,
    reason,
    warnings = character(),
    messages = character()) {

  empty <- list(
    primary_usable = FALSE,
    status = "failure",
    reason = reason,
    LR_raw = NA_real_,
    LR_zero = NA_real_,
    tolerance = NA_real_,
    logLik_full = NA_real_,
    logLik_null = NA_real_,
    full_valid = FALSE,
    null_valid = FALSE,
    full_singular = NA,
    null_singular = NA,
    full_optimizer = NA_character_,
    null_optimizer = NA_character_,
    optimizer_honored = NA,
    warnings = .sa_q1_capture_text(warnings),
    messages = .sa_q1_capture_text(messages)
  )
  adjudication <- list(
    status = "unresolved_after_bobyqa",
    source = "simulation_failure",
    rescue_triggered = FALSE,
    rescue_succeeded = FALSE,
    chosen = NULL,
    initial = empty,
    rescue = NULL
  )
  row <- .sa_q1_rescue_replicate_row(replicate, seed, adjudication)
  row$reason <- reason
  row$fit_source <- "simulation_failure"
  row
}


.sa_q1_one_bootstrap_replicate_rescue <- function(
    replicate,
    seed,
    full_fit,
    null_fit,
    target,
    initial_control,
    rescue_control,
    identity_tol = .sa_identity_tol,
    simulate_fun = .sa_q1_simulate_null,
    refit_fun = .sa_q1_refit) {

  .sa_q1_validate_calibration_rescue_controls(
    initial_control,
    rescue_control
  )
  sim <- .sa_q1_capture(simulate_fun(null_fit, seed))
  if (!is.null(sim$error)) {
    return(.sa_q1_simulation_failure_rescue_row(
      replicate = replicate,
      seed = seed,
      reason = paste0("simulation_error:", sim$error),
      warnings = sim$warnings,
      messages = sim$messages
    ))
  }

  response_valid <- tryCatch(
    {
      .sa_q1_validate_simulated_response(null_fit, sim$value)
      TRUE
    },
    error = function(error) FALSE
  )
  if (!isTRUE(response_valid)) {
    return(.sa_q1_simulation_failure_rescue_row(
      replicate = replicate,
      seed = seed,
      reason = "simulation_invalid_response",
      warnings = sim$warnings,
      messages = sim$messages
    ))
  }

  initial <- .sa_q1_refit_pair_attempt(
    full_template = full_fit,
    null_template = null_fit,
    response = sim$value,
    target = target,
    control = initial_control,
    method_id = "initial_nloptwrap",
    identity_tol = identity_tol,
    refit_fun = refit_fun
  )

  if (isTRUE(initial$primary_usable)) {
    adjudication <- .sa_q1_adjudicate_pair_rescue(initial)
  } else {
    rescue <- .sa_q1_refit_pair_attempt(
      full_template = full_fit,
      null_template = null_fit,
      response = sim$value,
      target = target,
      control = rescue_control,
      method_id = "rescue_bobyqa",
      identity_tol = identity_tol,
      refit_fun = refit_fun
    )
    adjudication <- .sa_q1_adjudicate_pair_rescue(initial, rescue)
  }

  row <- .sa_q1_rescue_replicate_row(
    replicate = replicate,
    seed = seed,
    adjudication = adjudication
  )
  row$warnings <- .sa_q1_capture_text(c(sim$warnings, row$warnings))
  row$messages <- .sa_q1_capture_text(c(sim$messages, row$messages))
  row
}


.sa_q1_fit_observed_pair_rescue <- function(
    full_formula,
    null_formula,
    data,
    target,
    REML,
    initial_control,
    rescue_control,
    identity_tol = .sa_identity_tol) {

  .sa_q1_validate_calibration_rescue_controls(
    initial_control,
    rescue_control
  )

  fit_once <- function(control, method_id) {
    full_capture <- .sa_q1_capture(
      lme4::lmer(
        formula = full_formula,
        data = data,
        REML = REML,
        control = control
      )
    )
    null_capture <- .sa_q1_capture(
      lme4::lmer(
        formula = null_formula,
        data = data,
        REML = REML,
        control = control
      )
    )
    .sa_q1_pair_assessment(
      full_capture = full_capture,
      null_capture = null_capture,
      target = target,
      control = control,
      method_id = method_id,
      identity_tol = identity_tol
    )
  }

  initial <- fit_once(initial_control, "initial_nloptwrap")
  adjudication <- if (isTRUE(initial$primary_usable)) {
    .sa_q1_adjudicate_pair_rescue(initial)
  } else {
    rescue <- fit_once(rescue_control, "rescue_bobyqa")
    .sa_q1_adjudicate_pair_rescue(initial, rescue)
  }

  if (!identical(adjudication$status, "ok")) {
    return(c(
      adjudication,
      list(
        full_fit = NULL,
        null_fit = NULL,
        LR_for_inference = NA_real_
      )
    ))
  }

  c(
    adjudication,
    list(
      full_fit = adjudication$chosen$full_fit,
      null_fit = adjudication$chosen$null_fit,
      LR_for_inference = adjudication$chosen$LR_raw
    )
  )
}


.sa_q1_bootstrap_fixed_usable_rescue <- function(
    observed,
    target,
    usable_B = 999L,
    attempt_cap = usable_B + max(50L, ceiling(0.25 * usable_B)),
    seed = 1L,
    conf_level = 0.95,
    alpha = 0.05,
    retain_replicates = FALSE,
    progress = FALSE,
    identity_tol = .sa_identity_tol,
    escalation_target_B = 4999L,
    initial_control,
    rescue_control,
    simulate_fun = .sa_q1_simulate_null,
    one_replicate_fun = .sa_q1_one_bootstrap_replicate_rescue) {

  if (!is.list(observed) || !identical(observed$status, "ok") ||
      !inherits(observed$full_fit, "lmerMod") ||
      !inherits(observed$null_fit, "lmerMod") ||
      !is.finite(observed$LR_for_inference)) {
    stop("`observed` must be one resolved calibration model pair.", call. = FALSE)
  }
  if (!is.numeric(usable_B) || length(usable_B) != 1L ||
      !is.finite(usable_B) || usable_B != floor(usable_B) || usable_B < 1L) {
    stop("`usable_B` must be one positive integer.", call. = FALSE)
  }
  usable_B <- as.integer(usable_B)
  if (!is.numeric(attempt_cap) || length(attempt_cap) != 1L ||
      !is.finite(attempt_cap) || attempt_cap != floor(attempt_cap) ||
      attempt_cap < usable_B) {
    stop("`attempt_cap` must be an integer at least `usable_B`.", call. = FALSE)
  }
  attempt_cap <- as.integer(attempt_cap)
  seed <- .sa_q1_validate_seed(seed)
  conf_level <- .sa_q1_validate_probability(conf_level, "conf_level")
  alpha <- .sa_q1_validate_probability(alpha, "alpha")
  retain_replicates <- .sa_q1_validate_flag(
    retain_replicates,
    "retain_replicates"
  )
  progress <- .sa_q1_validate_flag(progress, "progress")
  if (!is.numeric(escalation_target_B) ||
      length(escalation_target_B) != 1L ||
      !is.finite(escalation_target_B) ||
      escalation_target_B != floor(escalation_target_B) ||
      escalation_target_B < 1L ||
      escalation_target_B > .Machine$integer.max) {
    stop(
      "`escalation_target_B` must be one positive R integer.",
      call. = FALSE
    )
  }
  escalation_target_B <- as.integer(escalation_target_B)
  if (!is.function(simulate_fun)) {
    stop("`simulate_fun` must be a function.", call. = FALSE)
  }
  .sa_q1_validate_calibration_rescue_controls(
    initial_control,
    rescue_control
  )
  if (!is.function(one_replicate_fun)) {
    stop("`one_replicate_fun` must be a function.", call. = FALSE)
  }

  rng_snapshot <- .sa_q1_rng_snapshot()
  on.exit(.sa_q1_rng_restore(rng_snapshot), add = TRUE)

  seeds <- .sa_q1_make_replicate_seeds(B = attempt_cap, seed = seed)
  rows <- vector("list", attempt_cap)
  attempts_evaluated <- 0L
  usable <- 0L
  progress_bar <- NULL
  if (isTRUE(progress)) {
    progress_bar <- utils::txtProgressBar(
      min = 0,
      max = usable_B,
      initial = 0,
      style = 3
    )
    on.exit(close(progress_bar), add = TRUE)
  }

  for (attempt in seq_len(attempt_cap)) {
    row <- one_replicate_fun(
      replicate = attempt,
      seed = seeds[[attempt]],
      full_fit = observed$full_fit,
      null_fit = observed$null_fit,
      target = target,
      initial_control = initial_control,
      rescue_control = rescue_control,
      identity_tol = identity_tol,
      simulate_fun = simulate_fun
    )
    if (!is.data.frame(row) || nrow(row) != 1L ||
        !all(c("status", "rescue_triggered", "rescue_succeeded") %in%
          names(row))) {
      stop(
        "`one_replicate_fun` must return one rescue-aware replicate row.",
        call. = FALSE
      )
    }
    rows[[attempt]] <- row
    attempts_evaluated <- attempt
    if (row$status[[1]] %in% c("positive", "zero")) {
      usable <- usable + 1L
      if (isTRUE(progress)) {
        utils::setTxtProgressBar(progress_bar, usable)
      }
    }
    if (usable >= usable_B) {
      break
    }
  }

  replicates <- do.call(rbind, rows[seq_len(attempts_evaluated)])
  rownames(replicates) <- NULL
  summary <- .sa_q1_summarize_replicates(
    replicates = replicates,
    observed_LR = observed$LR_for_inference,
    conf_level = conf_level,
    alpha = alpha,
    escalation_target_B = escalation_target_B
  )
  target_reached <- identical(as.integer(summary$denominator_drop), usable_B)

  structure(
    list(
      status = if (target_reached) "ok" else "attempt_cap_reached",
      target = target,
      observed = observed,
      bootstrap = summary,
      replicates = if (isTRUE(retain_replicates)) replicates else NULL,
      usable_target = usable_B,
      attempt_cap = attempt_cap,
      attempts_evaluated = as.integer(attempts_evaluated),
      top_up_attempts = as.integer(attempts_evaluated - usable_B),
      target_reached = target_reached,
      rescue_attempts = sum(replicates$rescue_triggered),
      rescue_successes = sum(replicates$rescue_succeeded),
      unresolved_after_rescue = sum(
        replicates$fit_source == "unresolved_after_bobyqa"
      ),
      alpha = alpha,
      rank_grid_value = alpha * (usable_B + 1),
      exact_alpha_rank_grid = abs(
        alpha * (usable_B + 1) - round(alpha * (usable_B + 1))
      ) <= 100 * .Machine$double.eps,
      rejection_rule = "plus-one p <= alpha",
      master_seed = seed,
      replicate_seeds = seeds[seq_len(attempts_evaluated)],
      seed_stream = "replicate_seed_stream_v1",
      execution_mode = "fixed_drop_usable_with_bobyqa_rescue_v12",
      rescue_rule = paste0(
        "nloptwrap full/null first; if pair is not positive/zero usable, ",
        "refit both with bobyqa; unresolved bobyqa pairs are excluded and ",
        "trigger top-up"
      ),
      initial_control = initial_control,
      rescue_control = rescue_control,
      software = list(
        R = paste(R.version$major, R.version$minor, sep = "."),
        lme4 = as.character(utils::packageVersion("lme4"))
      )
    ),
    class = "slopeaudit_q1_internal"
  )
}
