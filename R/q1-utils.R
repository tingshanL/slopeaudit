# Internal utilities for Question 1: random-slope heterogeneity.
#
# Question 1 tests whether the marginal variance of one explicitly named
# random-effect coordinate is supported above the exact boundary value zero.
# These functions remain internal until the Question-1 subsystem is frozen.

.sa_q1_lr_tolerance <- function(logLik_full, logLik_null) {
  vals <- c(logLik_full, logLik_null)

  if (length(vals) != 2L || any(!is.finite(vals))) {
    return(NA_real_)
  }

  10 * sqrt(.Machine$double.eps) * max(1, abs(vals))
}


.sa_q1_classify_lr <- function(logLik_full, logLik_null) {
  if (!is.finite(logLik_full) || !is.finite(logLik_null)) {
    return(list(
      status = "failure",
      LR_raw = NA_real_,
      LR_zero = NA_real_,
      tolerance = NA_real_,
      reason = "nonfinite_logLik"
    ))
  }

  lr <- 2 * (logLik_full - logLik_null)
  tol <- .sa_q1_lr_tolerance(logLik_full, logLik_null)

  if (!is.finite(lr)) {
    return(list(
      status = "failure",
      LR_raw = NA_real_,
      LR_zero = NA_real_,
      tolerance = tol,
      reason = "nonfinite_LR"
    ))
  }

  if (lr > 0) {
    status <- "positive"
    lr_zero <- lr
    reason <- NA_character_
  } else if (lr == 0) {
    status <- "zero"
    lr_zero <- 0
    reason <- NA_character_
  } else if (lr >= -tol) {
    status <- "small_negative"
    lr_zero <- 0
    reason <- "small_finite_negative_LR"
  } else {
    status <- "material_negative"
    lr_zero <- NA_real_
    reason <- "material_negative_LR"
  }

  list(
    status = status,
    LR_raw = lr,
    LR_zero = lr_zero,
    tolerance = tol,
    reason = reason
  )
}


.sa_q1_exact_binomial_interval <- function(k, n, conf_level = 0.95) {
  if (!is.numeric(k) || length(k) != 1L || !is.finite(k) ||
      k != floor(k) || k < 0) {
    stop("`k` must be one nonnegative integer.", call. = FALSE)
  }

  if (!is.numeric(n) || length(n) != 1L || !is.finite(n) ||
      n != floor(n) || n < 0 || k > n) {
    stop("`n` must be an integer with n >= k >= 0.", call. = FALSE)
  }

  if (!is.numeric(conf_level) || length(conf_level) != 1L ||
      !is.finite(conf_level) || conf_level <= 0 || conf_level >= 1) {
    stop("`conf_level` must lie strictly between 0 and 1.", call. = FALSE)
  }

  if (n == 0L) {
    return(c(low = NA_real_, high = NA_real_))
  }

  alpha <- 1 - conf_level

  low <- if (k == 0L) {
    0
  } else {
    stats::qbeta(alpha / 2, k, n - k + 1)
  }

  high <- if (k == n) {
    1
  } else {
    stats::qbeta(1 - alpha / 2, k + 1, n - k)
  }

  c(low = as.numeric(low), high = as.numeric(high))
}


.sa_q1_plus_one_p <- function(exceedances, denominator) {
  if (!is.numeric(exceedances) || length(exceedances) != 1L ||
      !is.finite(exceedances) || exceedances < 0 ||
      exceedances != floor(exceedances)) {
    stop("`exceedances` must be one nonnegative integer.", call. = FALSE)
  }

  if (!is.numeric(denominator) || length(denominator) != 1L ||
      !is.finite(denominator) || denominator < 0 ||
      denominator != floor(denominator) ||
      exceedances > denominator) {
    stop(
      "`denominator` must be an integer >= `exceedances`.",
      call. = FALSE
    )
  }

  if (denominator == 0L) {
    return(NA_real_)
  }

  (exceedances + 1) / (denominator + 1)
}


.sa_q1_reject <- function(p, alpha = 0.05) {
  alpha <- .sa_q1_validate_probability(alpha, "alpha")

  if (!is.numeric(p)) {
    stop("`p` must be numeric.", call. = FALSE)
  }

  is.finite(p) & p <= alpha
}


.sa_q1_nominal_rank_size <- function(usable_B, alpha = 0.05) {
  if (!is.numeric(usable_B) || length(usable_B) != 1L ||
      !is.finite(usable_B) || usable_B != floor(usable_B) ||
      usable_B < 1 || usable_B > .Machine$integer.max) {
    stop("`usable_B` must be one positive R integer.", call. = FALSE)
  }
  alpha <- .sa_q1_validate_probability(alpha, "alpha")

  total_ranks <- as.integer(usable_B) + 1
  floor(alpha * total_ranks) / total_ranks
}


.sa_q1_validate_seed <- function(seed) {
  if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed) ||
      seed != floor(seed) ||
      seed < -.Machine$integer.max ||
      seed > .Machine$integer.max) {
    stop(
      "`seed` must be one finite integer representable by R.",
      call. = FALSE
    )
  }

  as.integer(seed)
}


.sa_q1_validate_probability <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
      x <= 0 || x >= 1) {
    stop(
      sprintf("`%s` must lie strictly between 0 and 1.", name),
      call. = FALSE
    )
  }

  as.numeric(x)
}


.sa_q1_validate_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(
      sprintf("`%s` must be TRUE or FALSE.", name),
      call. = FALSE
    )
  }

  isTRUE(x)
}


.sa_q1_rng_snapshot <- function() {
  list(
    kind = RNGkind(),
    seed_exists = exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE),
    seed = if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    } else {
      NULL
    }
  )
}


.sa_q1_rng_restore <- function(snapshot) {
  do.call(
    RNGkind,
    list(
      kind = snapshot$kind[[1]],
      normal.kind = snapshot$kind[[2]],
      sample.kind = snapshot$kind[[3]]
    )
  )

  if (isTRUE(snapshot$seed_exists)) {
    assign(".Random.seed", snapshot$seed, envir = .GlobalEnv)
  } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    rm(".Random.seed", envir = .GlobalEnv)
  }

  invisible(TRUE)
}


.sa_q1_make_replicate_seeds <- function(B, seed) {
  if (!is.numeric(B) || length(B) != 1L || !is.finite(B) ||
      B != floor(B) || B < 1 || B > .Machine$integer.max) {
    stop("`B` must be one positive R integer.", call. = FALSE)
  }

  seed <- .sa_q1_validate_seed(seed)

  snapshot <- .sa_q1_rng_snapshot()
  on.exit(.sa_q1_rng_restore(snapshot), add = TRUE)

  set.seed(seed)

  sample.int(
    n = .Machine$integer.max,
    size = as.integer(B),
    replace = FALSE
  )
}


.sa_q1_capture <- function(expr) {
  warnings <- character()
  messages <- character()

  value <- tryCatch(
    withCallingHandlers(
      expr,
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      message = function(m) {
        messages <<- c(messages, conditionMessage(m))
        invokeRestart("muffleMessage")
      }
    ),
    error = function(e) e
  )

  list(
    value = if (inherits(value, "error")) NULL else value,
    error = if (inherits(value, "error")) conditionMessage(value) else NULL,
    warnings = unique(warnings),
    messages = unique(messages)
  )
}


.sa_q1_singular_message <- function(x) {
  grepl(
    "boundary \\(singular\\) fit",
    x,
    ignore.case = TRUE
  )
}


.sa_q1_normalize_opt_control <- function(x) {
  if (is.null(x)) {
    return(list())
  }

  x <- as.list(x)

  # lme4 may store optimizer-internal scaling entries that should not define
  # the user's fitting-control contract.
  x[c("xst", "xt")] <- NULL
  x
}


.sa_q1_recorded_control_signature <- function(fit) {
  .sa_validate_lmer_fit(fit)

  list(
    optimizer_used = as.character(fit@optinfo$optimizer),
    optimizer_control = .sa_q1_normalize_opt_control(
      fit@optinfo$control
    )
  )
}


.sa_q1_call_control_label <- function(fit) {
  call <- stats::getCall(fit)
  expr <- call$control

  if (is.null(expr)) {
    "<default lmerControl()>"
  } else {
    paste(deparse(expr), collapse = " ")
  }
}


.sa_q1_validate_refit_control <- function(fit, control) {
  if (!inherits(control, "lmerControl")) {
    stop(
      "`refit_control` must be an `lmerControl` object.",
      call. = FALSE
    )
  }

  used_optimizer <- as.character(fit@optinfo$optimizer)
  declared <- as.character(control$optimizer)

  if (length(used_optimizer) != 1L ||
      !used_optimizer %in% declared) {
    stop(
      paste0(
        "`refit_control` does not include the optimizer recorded in the ",
        "observed fitted model."
      ),
      call. = FALSE
    )
  }

  control
}


.sa_q1_common_recorded_optimizer_control <- function(full_fit, null_fit) {
  full_sig <- .sa_q1_recorded_control_signature(full_fit)
  null_sig <- .sa_q1_recorded_control_signature(null_fit)

  if (!identical(
    full_sig$optimizer_used,
    null_sig$optimizer_used
  )) {
    stop(
      paste0(
        "Full and null models were fitted with different optimizers. ",
        "No hidden optimizer switch is permitted."
      ),
      call. = FALSE
    )
  }

  if (!isTRUE(all.equal(
    full_sig$optimizer_control,
    null_sig$optimizer_control,
    check.attributes = TRUE
  ))) {
    stop(
      paste0(
        "Full and null models do not have the same recorded effective ",
        "optimizer-control settings."
      ),
      call. = FALSE
    )
  }

  list(
    optimizer_used = full_sig$optimizer_used,
    optimizer_control = full_sig$optimizer_control
  )
}


.sa_q1_user_optctrl_matches_recorded <- function(
    user_optctrl,
    recorded_optctrl) {

  user_optctrl <- .sa_q1_normalize_opt_control(user_optctrl)
  recorded_optctrl <- .sa_q1_normalize_opt_control(recorded_optctrl)

  if (length(user_optctrl) == 0L) {
    return(TRUE)
  }

  for (nm in names(user_optctrl)) {
    if (!nm %in% names(recorded_optctrl)) {
      return(FALSE)
    }

    if (!isTRUE(all.equal(
      user_optctrl[[nm]],
      recorded_optctrl[[nm]],
      check.attributes = TRUE
    ))) {
      return(FALSE)
    }
  }

  TRUE
}


.sa_q1_resolve_refit_control <- function(
    full_fit,
    null_fit,
    refit_control = NULL) {

  common <- .sa_q1_common_recorded_optimizer_control(
    full_fit,
    null_fit
  )

  full_custom <- !is.null(stats::getCall(full_fit)$control)
  null_custom <- !is.null(stats::getCall(null_fit)$control)

  if (is.null(refit_control)) {
    if (full_custom || null_custom) {
      stop(
        paste0(
          "At least one observed model used a custom `control` argument. ",
          "Supply the exact `refit_control` used for the observed fits. ",
          "Question 1 will not reconstruct a mutable control expression."
        ),
        call. = FALSE
      )
    }

    control <- lme4::lmerControl()
    source <- "default_lmerControl"
  } else {
    control <- .sa_q1_validate_refit_control(
      full_fit,
      refit_control
    )
    invisible(.sa_q1_validate_refit_control(
      null_fit,
      refit_control
    ))

    if (!.sa_q1_user_optctrl_matches_recorded(
      control$optCtrl,
      common$optimizer_control
    )) {
      stop(
        paste0(
          "`refit_control$optCtrl` disagrees with optimizer-control ",
          "settings recorded in the observed full/null fits."
        ),
        call. = FALSE
      )
    }

    source <- "explicit_refit_control"
  }

  # lme4::refit() fills an empty optCtrl from object@optinfo$control.
  # Populate it here with the already-validated common effective control so
  # full and null refits cannot silently receive object-specific controls.
  control$optCtrl <- common$optimizer_control

  list(
    control = control,
    source = source,
    optimizer_used = common$optimizer_used,
    optimizer_control = common$optimizer_control,
    full_call_control = .sa_q1_call_control_label(full_fit),
    null_call_control = .sa_q1_call_control_label(null_fit)
  )
}


.sa_q1_singularity_tolerance <- function(control = NULL) {
  if (is.null(control)) {
    control <- lme4::lmerControl()
  }

  tol <- control$checkConv$check.conv.singular$tol

  if (!is.numeric(tol) || length(tol) != 1L ||
      !is.finite(tol) || tol <= 0) {
    stop(
      "Could not recover a valid lme4 singularity tolerance.",
      call. = FALSE
    )
  }

  as.numeric(tol)
}


.sa_q1_fit_diagnostics <- function(fit, control = NULL) {
  if (!inherits(fit, "lmerMod")) {
    return(list(
      valid = FALSE,
      reason = "refit_not_lmerMod",
      logLik = NA_real_,
      optimizer = NA_character_,
      optimizer_code = NA_integer_,
      convergence_messages = character(),
      singular = NA,
      singularity_tolerance = NA_real_
    ))
  }

  ll <- suppressWarnings(as.numeric(stats::logLik(fit)))
  optinfo <- fit@optinfo

  opt_code <- optinfo$conv$opt
  if (is.null(opt_code)) {
    opt_code <- 0L
  }
  opt_code <- as.integer(opt_code)

  conv_messages <- optinfo$conv$lme4$messages
  if (is.null(conv_messages)) {
    conv_messages <- character()
  } else {
    conv_messages <- as.character(unlist(conv_messages, use.names = FALSE))
  }

  non_singular_messages <- conv_messages[
    !.sa_q1_singular_message(conv_messages)
  ]

  optimizer_ok <- length(opt_code) > 0L && all(opt_code == 0L)
  convergence_ok <- length(non_singular_messages) == 0L
  likelihood_ok <- length(ll) == 1L && is.finite(ll)

  reason <- if (!likelihood_ok) {
    "nonfinite_logLik"
  } else if (!optimizer_ok) {
    "optimizer_nonzero_code"
  } else if (!convergence_ok) {
    "lme4_convergence_message"
  } else {
    NA_character_
  }

  singular_tol <- .sa_q1_singularity_tolerance(control)

  list(
    valid = likelihood_ok && optimizer_ok && convergence_ok,
    reason = reason,
    logLik = if (likelihood_ok) ll else NA_real_,
    optimizer = paste(as.character(optinfo$optimizer), collapse = "+"),
    optimizer_code = opt_code,
    convergence_messages = conv_messages,
    non_singular_convergence_messages = non_singular_messages,
    singularity_tolerance = singular_tol,
    singular = lme4::isSingular(fit, tol = singular_tol)
  )
}


.sa_q1_same_numeric_matrix <- function(a, b, tol = .sa_identity_tol) {
  a <- as.matrix(a)
  b <- as.matrix(b)

  if (!identical(dim(a), dim(b))) {
    return(FALSE)
  }

  if (!identical(colnames(a), colnames(b))) {
    return(FALSE)
  }

  scale <- max(1, abs(a), abs(b))
  max(abs(a - b)) <= tol * scale
}


.sa_q1_prior_weights <- function(fit) {
  w <- stats::weights(fit, type = "prior")

  if (is.null(w)) {
    w <- rep(1, stats::nobs(fit))
  }

  as.numeric(w)
}


.sa_q1_offset <- function(fit) {
  mf <- stats::model.frame(fit)
  off <- stats::model.offset(mf)

  if (is.null(off)) {
    rep(0, nrow(mf))
  } else {
    as.numeric(off)
  }
}


.sa_q1_refit_matches_template <- function(refit, template) {
  if (!inherits(refit, "lmerMod")) {
    return(list(valid = FALSE, reason = "refit_not_lmerMod"))
  }

  if (!identical(lme4::isREML(refit), lme4::isREML(template))) {
    return(list(valid = FALSE, reason = "refit_changed_likelihood_mode"))
  }

  if (!identical(
    as.character(refit@optinfo$optimizer),
    as.character(template@optinfo$optimizer)
  )) {
    return(list(valid = FALSE, reason = "refit_changed_optimizer"))
  }

  list(valid = TRUE, reason = NA_character_)
}
