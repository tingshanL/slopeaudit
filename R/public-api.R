# Public API wrappers for validated slopeaudit numerical engines.
#
# These functions validate inputs, organize output objects, and delegate
# numerical calculations to the internal engines.

.sa_public_decomposition_summary <- function(x) {
  s <- x$summary

  data.frame(
    quantity = c("R_direct", "D_slope2", "Delta_fit"),
    mean = c(
      s$mean_participant_R_direct,
      s$mean_participant_D_slope2,
      s$mean_participant_Delta_fit
    ),
    median = c(
      s$median_participant_R_direct,
      s$median_participant_D_slope2,
      s$median_participant_Delta_fit
    ),
    IQR = c(
      s$IQR_participant_R_direct,
      s$IQR_participant_D_slope2,
      s$IQR_participant_Delta_fit
    ),
    minimum = c(
      s$min_participant_R_direct,
      s$min_participant_D_slope2,
      s$min_participant_Delta_fit
    ),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}


.sa_public_q1_decision_bounds <- function(
    p_drop,
    usable_B,
    attempts_evaluated,
    alpha) {

  if (!is.finite(p_drop) || !is.finite(usable_B) ||
      !is.finite(attempts_evaluated) || usable_B < 1L ||
      attempts_evaluated < usable_B) {
    return(list(
      exceedances = NA_integer_,
      excluded_attempts = NA_integer_,
      p_if_excluded_all_nonexceed = NA_real_,
      p_if_excluded_all_exceed = NA_real_,
      decision_stable = NA
    ))
  }

  exceedances <- round(p_drop * (usable_B + 1) - 1)
  reconstructed <- (exceedances + 1) / (usable_B + 1)
  if (abs(p_drop - reconstructed) > 1e-10) {
    stop(
      "The primary p-value is not on its declared plus-one grid.",
      call. = FALSE
    )
  }

  excluded <- attempts_evaluated - usable_B
  lower <- (exceedances + 1) / (attempts_evaluated + 1)
  upper <- (exceedances + excluded + 1) / (attempts_evaluated + 1)

  list(
    exceedances = as.integer(exceedances),
    excluded_attempts = as.integer(excluded),
    p_if_excluded_all_nonexceed = lower,
    p_if_excluded_all_exceed = upper,
    decision_stable = identical(lower <= alpha, upper <= alpha)
  )
}


.sa_public_q1_decision <- function(x) {
  if (!identical(x$status, "ok") || !isTRUE(x$target_reached) ||
      is.null(x$bootstrap)) {
    return("indeterminate")
  }

  if (!isTRUE(x$decision_bounds$decision_stable)) {
    return("numerically_ambiguous")
  }

  if (x$bootstrap$p_drop <= x$alpha) "supported" else "unsupported"
}


.sa_public_validate_q1_arguments <- function(
    B,
    seed,
    conf_level,
    alpha,
    attempt_cap,
    retain_replicates,
    progress) {

  if (!is.numeric(B) || length(B) != 1L || !is.finite(B) ||
      B != floor(B) || B < 1L || B > .Machine$integer.max) {
    stop("`B` must be one positive R integer.", call. = FALSE)
  }
  B <- as.integer(B)

  if (!is.numeric(attempt_cap) || length(attempt_cap) != 1L ||
      !is.finite(attempt_cap) || attempt_cap != floor(attempt_cap) ||
      attempt_cap < B || attempt_cap > .Machine$integer.max) {
    stop(
      "`attempt_cap` must be an R integer at least as large as `B`.",
      call. = FALSE
    )
  }

  list(
    B = B,
    seed = .sa_q1_validate_seed(seed),
    conf_level = .sa_q1_validate_probability(conf_level, "conf_level"),
    alpha = .sa_q1_validate_probability(alpha, "alpha"),
    attempt_cap = as.integer(attempt_cap),
    retain_replicates = .sa_q1_validate_flag(
      retain_replicates,
      "retain_replicates"
    ),
    progress = .sa_q1_validate_flag(progress, "progress")
  )
}


.sa_public_observed_pair <- function(full, null, target, controls) {
  observed <- .sa_q1_observed_lr(
    full_fit = full,
    null_fit = null,
    target = target,
    diagnostic_control = controls$initial,
    identity_tol = .sa_identity_tol,
    strict_controls = TRUE
  )

  if (!isTRUE(observed$valid_for_inference)) {
    return(list(observed = observed, resolved = NULL))
  }

  resolved <- list(
    status = "ok",
    source = "supplied_fits",
    rescue_triggered = FALSE,
    rescue_succeeded = FALSE,
    full_fit = full,
    null_fit = null,
    LR_for_inference = observed$LR_for_inference,
    observed_diagnostics = observed
  )

  list(observed = observed, resolved = resolved)
}


#' Test target random-slope heterogeneity
#'
#' Run the validated fixed-usable parametric-bootstrap test for Question 1.
#' Both already fitted models must use REML, the same retained observations,
#' fixed-effect design, grouping structure, nuisance random effects, and
#' effective optimizer controls.  The null must omit the target random slope.
#' The supplied observed fits define the model structures used for bootstrap
#' refitting.
#'
#' @param full A supported Gaussian `lme4::lmer()` full model.
#' @param null A supported Gaussian `lme4::lmer()` null model.
#' @param target One non-empty target random-effect coefficient name.
#' @param B Number of usable positive/zero bootstrap LR statistics.  The
#'   validated default is 999.
#' @param seed One integer master seed.
#' @param conf_level Confidence level for exact binomial intervals describing
#'   the fitted-null exceedance probability.
#' @param alpha Decision threshold for the plus-one Monte Carlo p-value.
#' @param attempt_cap Maximum attempted bootstrap replicates.  Unresolved
#'   rescued attempts are retained in provenance and trigger deterministic
#'   top-up until `B` usable values are obtained or this cap is reached.
#' @param retain_replicates Logical; retain the replicate-level table.
#' @param progress Logical; display a text progress bar.
#'
#' @return An object of class `slopeaudit_heterogeneity`.
#' @examples
#' if (interactive()) {
#'   control <- lme4::lmerControl(optimizer = "nloptwrap")
#'   full <- lme4::lmer(
#'     Reaction ~ Days + (1 + Days | Subject),
#'     data = lme4::sleepstudy, REML = TRUE, control = control
#'   )
#'   null <- lme4::lmer(
#'     Reaction ~ Days + (1 | Subject),
#'     data = lme4::sleepstudy, REML = TRUE, control = control
#'   )
#'   test_slope_heterogeneity(
#'     full, null, target = "Days", B = 999L, seed = 20260803L
#'   )
#' }
#' @export
test_slope_heterogeneity <- function(
    full,
    null,
    target,
    B = 999L,
    seed = 1L,
    conf_level = 0.95,
    alpha = 0.05,
    attempt_cap = B + max(50L, ceiling(0.25 * B)),
    retain_replicates = FALSE,
    progress = interactive() && B >= 50L) {

  validated <- .sa_public_validate_q1_arguments(
    B = B,
    seed = seed,
    conf_level = conf_level,
    alpha = alpha,
    attempt_cap = attempt_cap,
    retain_replicates = retain_replicates,
    progress = progress
  )
  B <- validated$B
  seed <- validated$seed
  conf_level <- validated$conf_level
  alpha <- validated$alpha
  attempt_cap <- validated$attempt_cap
  retain_replicates <- validated$retain_replicates
  progress <- validated$progress

  .sa_validate_lmer_fit(full)
  .sa_validate_lmer_fit(null)

  if (!isTRUE(lme4::isREML(full)) || !isTRUE(lme4::isREML(null))) {
    stop(
      paste0(
        "The version 0.1 Question-1 interface requires REML ",
        "full/null fits with identical fixed-effect designs."
      ),
      call. = FALSE
    )
  }

  controls <- .sa_q1_calibration_rescue_controls()
  observed_pair <- .sa_public_observed_pair(
    full = full,
    null = null,
    target = target,
    controls = controls
  )

  if (is.null(observed_pair$resolved)) {
    out <- list(
      call = match.call(),
      status = "invalid_observed",
      decision = "indeterminate",
      target = target,
      observed = observed_pair$observed,
      bootstrap = NULL,
      decision_bounds = NULL,
      usable_target = as.integer(B),
      attempt_cap = as.integer(attempt_cap),
      attempts_evaluated = 0L,
      target_reached = FALSE,
      alpha = alpha,
      master_seed = seed,
      execution_mode = "fixed_drop_usable_with_bobyqa_rescue_v12"
    )
    class(out) <- c("slopeaudit_heterogeneity", "slopeaudit_q1_internal")
    return(out)
  }

  out <- .sa_q1_bootstrap_fixed_usable_rescue(
    observed = observed_pair$resolved,
    target = target,
    usable_B = B,
    attempt_cap = attempt_cap,
    seed = seed,
    conf_level = conf_level,
    alpha = alpha,
    retain_replicates = retain_replicates,
    progress = progress,
    identity_tol = .sa_identity_tol,
    escalation_target_B = 4999L,
    initial_control = controls$initial,
    rescue_control = controls$rescue
  )

  out$call <- match.call()
  out$observed$observed_diagnostics <- observed_pair$observed
  out$decision_bounds <- if (isTRUE(out$target_reached)) {
    .sa_public_q1_decision_bounds(
      p_drop = out$bootstrap$p_drop,
      usable_B = out$usable_target,
      attempts_evaluated = out$attempts_evaluated,
      alpha = out$alpha
    )
  } else {
    .sa_public_q1_decision_bounds(
      p_drop = NA_real_,
      usable_B = out$usable_target,
      attempts_evaluated = out$attempts_evaluated,
      alpha = out$alpha
    )
  }
  out$decision <- .sa_public_q1_decision(out)
  class(out) <- c("slopeaudit_heterogeneity", class(out))
  out
}


#' Decompose fitted random-slope information sources
#'
#' Compute participant-level direct-change reliability, total fitted-slope
#' determinacy, and their model-assisted difference from one supported
#' Gaussian `lme4::lmer()` fit.
#'
#' @param full A supported Gaussian `lme4::lmer()` full model.
#' @param target One non-empty target random-effect coefficient name.
#' @param target_metadata A named list containing the six required fields
#'   `basis_type`, `outcome_units`, `time_or_exposure_scale`,
#'   `origin_or_reference`, `reference_window`, and `estimand`.  Transformed,
#'   piecewise, interaction, and other non-affine bases additionally require
#'   `basis_expression`.
#'
#' @return An object of class `slopeaudit_decomposition`.
#' @examples
#' full <- lme4::lmer(
#'   Reaction ~ Days + (1 + Days | Subject),
#'   data = lme4::sleepstudy,
#'   REML = TRUE
#' )
#' target_metadata <- list(
#'   basis_type = "linear",
#'   outcome_units = "milliseconds",
#'   time_or_exposure_scale = "1 day",
#'   origin_or_reference = "Day 0",
#'   reference_window = "Days 0--9",
#'   estimand = "participant-specific linear change in Reaction per day"
#' )
#' slope_decomposition(full, "Days", target_metadata)
#' @export
slope_decomposition <- function(full, target, target_metadata) {
  out <- .sa_lmer_decompose(
    fit = full,
    target = target,
    target_metadata = target_metadata,
    require_identified_estimand = TRUE,
    identity_tol = .sa_identity_tol,
    rank_tol = .sa_rank_tol
  )

  out$status <- out$diagnostics$status
  out$call <- match.call()
  out$source_summary <- .sa_public_decomposition_summary(out)
  class(out) <- c("slopeaudit_decomposition", "slopeaudit_result")
  out
}


.sa_public_resolve_matrix_target <- function(Z, G, target) {
  if (!is.list(Z) || length(Z) == 0L) {
    stop("`Z` must be a non-empty list of design matrices.", call. = FALSE)
  }

  G <- as.matrix(G)
  .sa_assert_numeric_matrix(G, "G")
  q <- ncol(G)

  if (is.numeric(target) && length(target) == 1L &&
      is.finite(target) && target == floor(target)) {
    target_index <- .sa_validate_target_index(as.integer(target), q)
  } else if (is.character(target) && length(target) == 1L &&
      !is.na(target) && nzchar(target)) {
    candidate_names <- colnames(as.matrix(G))
    if (is.null(candidate_names)) {
      candidate_names <- colnames(as.matrix(Z[[1L]]))
    }
    if (is.null(candidate_names)) {
      stop(
        "A character `target` requires random-effect column names in `G` or `Z`.",
        call. = FALSE
      )
    }
    matches <- which(candidate_names == target)
    if (length(matches) != 1L) {
      stop(
        sprintf("Target `%s` must identify exactly one random-effect column.", target),
        call. = FALSE
      )
    }
    target_index <- as.integer(matches)
  } else {
    stop(
      "`target` must be one random-effect column name or one-based index.",
      call. = FALSE
    )
  }

  target_name <- if (is.character(target)) {
    target
  } else {
    candidate_names <- colnames(as.matrix(G))
    if (is.null(candidate_names)) {
      candidate_names <- colnames(as.matrix(Z[[1L]]))
    }
    if (is.null(candidate_names)) {
      paste0("random_effect_", target_index)
    } else {
      candidate_names[[target_index]]
    }
  }

  list(index = target_index, name = target_name)
}


#' Evaluate prospective random-slope information for supplied designs
#'
#' Apply the matrix backend to one or more prospective
#' participant schedules.  This function evaluates information-source
#' quantities; it does not claim to provide Question-1 power for arbitrary
#' designs or residual covariance models.
#'
#' @param Z A non-empty list of participant/design random-effects matrices.
#' @param G The planned random-effects covariance matrix.
#' @param R One residual covariance matrix shared by all entries in `Z`, or a
#'   list with one residual covariance matrix per entry.
#' @param target One random-effect column name or one-based column index.
#' @param target_metadata The same strict target-estimand metadata required by
#'   `slope_decomposition()`.
#' @param H `NULL`, or a list whose entries span exactly the corresponding
#'   nuisance random-effect column spaces.
#' @param design_id Optional labels, one per entry of `Z`.
#'
#' @return An object of class `slopeaudit_design_sensitivity`.
#' @examples
#' Z <- matrix(c(1, -0.5, 1, 0.5), nrow = 2, byrow = TRUE)
#' colnames(Z) <- c("intercept", "time")
#' G <- matrix(c(18, sqrt(15), sqrt(15), 1), nrow = 2, byrow = TRUE)
#' dimnames(G) <- list(colnames(Z), colnames(Z))
#' metadata <- list(
#'   basis_type = "linear",
#'   outcome_units = "outcome units",
#'   time_or_exposure_scale = "1 time unit",
#'   origin_or_reference = "centered time 0",
#'   reference_window = "two planned observations",
#'   estimand = "participant-specific linear change per time unit"
#' )
#' slope_design_sensitivity(
#'   Z = list(example = Z), G = G, R = diag(2),
#'   target = "time", target_metadata = metadata
#' )
#' @export
slope_design_sensitivity <- function(
    Z,
    G,
    R,
    target,
    target_metadata,
    H = NULL,
    design_id = NULL) {

  resolved <- .sa_public_resolve_matrix_target(Z, G, target)
  estimand <- .sa_validate_target_metadata(
    target = resolved$name,
    target_metadata = target_metadata,
    require_identified = TRUE
  )

  if (is.null(design_id)) {
    design_id <- names(Z)
    if (is.null(design_id) || any(!nzchar(design_id))) {
      design_id <- as.character(seq_along(Z))
    }
  }

  backend <- .sa_matrix_decompose(
    Z = Z,
    G = G,
    R = R,
    target_index = resolved$index,
    H = H,
    participant_id = design_id,
    identity_tol = .sa_identity_tol,
    rank_tol = .sa_rank_tol
  )

  design_table <- backend$participant_table
  names(design_table)[names(design_table) == "participant_id"] <- "design_id"

  out <- list(
    call = match.call(),
    status = backend$diagnostics$status,
    target = resolved$name,
    target_index = resolved$index,
    target_metadata = estimand,
    design_table = design_table,
    summary = backend$summary,
    source_summary = .sa_public_decomposition_summary(backend),
    diagnostics = backend$diagnostics,
    matrix_backend = backend,
    scope = paste0(
      "Prospective information-source quantities only; no arbitrary-design ",
      "Question-1 power claim."
    )
  )
  class(out) <- c("slopeaudit_design_sensitivity", "slopeaudit_result")
  out
}


#' Audit a target random slope
#'
#' Run the validated Question-1 heterogeneity test and the fitted-model source
#' decomposition through a single interface.
#'
#' @inheritParams test_slope_heterogeneity
#' @param target_metadata The strict target-estimand metadata described in
#'   `slope_decomposition()`.
#'
#' @return An object of class `slopeaudit`.
#' @examples
#' if (interactive()) {
#'   full <- lme4::lmer(
#'     Reaction ~ Days + (1 + Days | Subject),
#'     data = lme4::sleepstudy, REML = TRUE
#'   )
#'   null <- lme4::lmer(
#'     Reaction ~ Days + (1 | Subject),
#'     data = lme4::sleepstudy, REML = TRUE
#'   )
#'   metadata <- list(
#'     basis_type = "linear",
#'     outcome_units = "milliseconds",
#'     time_or_exposure_scale = "1 day",
#'     origin_or_reference = "Day 0",
#'     reference_window = "Days 0--9",
#'     estimand = "participant-specific linear change in Reaction per day"
#'   )
#'   slope_audit(
#'     full, null, "Days", metadata, B = 999L, seed = 20260803L
#'   )
#' }
#' @export
slope_audit <- function(
    full,
    null,
    target,
    target_metadata,
    B = 999L,
    seed = 1L,
    conf_level = 0.95,
    alpha = 0.05,
    attempt_cap = B + max(50L, ceiling(0.25 * B)),
    retain_replicates = FALSE,
    progress = interactive() && B >= 50L) {

  # Validate the estimand before starting bootstrap work. The decomposition
  # validates it again at its own boundary.
  normalized_estimand <- .sa_validate_target_metadata(
    target = target,
    target_metadata = target_metadata,
    require_identified = TRUE
  )

  decomposition_attempt <- tryCatch(
    list(
      value = slope_decomposition(
        full = full,
        target = target,
        target_metadata = normalized_estimand
      ),
      boundary = NULL
    ),
    slopeaudit_target_boundary_error = function(error) {
      list(value = NULL, boundary = error)
    }
  )
  decomposition <- decomposition_attempt$value
  decomposition_boundary <- decomposition_attempt$boundary

  heterogeneity <- test_slope_heterogeneity(
    full = full,
    null = null,
    target = target,
    B = B,
    seed = seed,
    conf_level = conf_level,
    alpha = alpha,
    attempt_cap = attempt_cap,
    retain_replicates = retain_replicates,
    progress = progress
  )

  decomposition_status <- if (is.null(decomposition)) {
    "target_boundary"
  } else {
    decomposition$status
  }

  out <- list(
    call = match.call(),
    status = if (identical(heterogeneity$status, "ok") &&
      decomposition_status %in% c("ok", "warning")) {
      decomposition_status
    } else if (identical(decomposition_status, "target_boundary")) {
      "target_boundary"
    } else {
      "diagnose"
    },
    target = target,
    target_metadata = normalized_estimand,
    heterogeneity = heterogeneity,
    decomposition = decomposition,
    decomposition_status = decomposition_status,
    decomposition_boundary = decomposition_boundary,
    participant_table = if (is.null(decomposition)) {
      NULL
    } else {
      decomposition$participant_table
    },
    source_summary = if (is.null(decomposition)) {
      NULL
    } else {
      decomposition$source_summary
    },
    interpretation = list(
      question1 = heterogeneity$decision,
      decomposition = if (is.null(decomposition)) {
        paste0(
          "withheld because the fitted marginal target-slope variance is ",
          "exactly zero"
        )
      } else if (identical(heterogeneity$decision, "unsupported")) {
        paste0(
          "conditional description of the fitted random-slope model; ",
          "population heterogeneity was not supported"
        )
      } else {
        "conditional fitted-model source decomposition"
      }
    )
  )
  class(out) <- c("slopeaudit", "slopeaudit_result")
  out
}
