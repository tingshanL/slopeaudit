# Explicit full/null validation for Question 1.
#
# The interface accepts already fitted full and null models and evaluates them
# as supplied.

.sa_q1_response_label <- function(fit) {
  paste(deparse(stats::formula(fit)[[2]]), collapse = " ")
}


.sa_q1_validate_unit_weights <- function(
    fit,
    label,
    tol = .sa_identity_tol) {

  w <- .sa_q1_prior_weights(fit)

  if (length(w) != stats::nobs(fit) ||
      any(!is.finite(w)) ||
      any(w <= 0)) {
    stop(
      sprintf("The %s model has invalid prior weights.", label),
      call. = FALSE
    )
  }

  if (any(abs(w - 1) > tol)) {
    stop(
      paste0(
        "Question 1 v1 supports only ordinary unweighted `lmer` fits. ",
        "The ", label, " model has non-unit prior weights."
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


.sa_q1_validate_model_pair <- function(
    full_fit,
    null_fit,
    target,
    identity_tol = .sa_identity_tol,
    strict_controls = TRUE) {

  strict_controls <- .sa_q1_validate_flag(
    strict_controls,
    "strict_controls"
  )

  .sa_validate_lmer_fit(full_fit)
  .sa_validate_lmer_fit(null_fit)

  .sa_q1_validate_unit_weights(
    full_fit,
    "full",
    tol = identity_tol
  )
  .sa_q1_validate_unit_weights(
    null_fit,
    "null",
    tol = identity_tol
  )

  full_frame <- stats::model.frame(full_fit)
  null_frame <- stats::model.frame(null_fit)

  full_rows <- rownames(full_frame)
  null_rows <- rownames(null_frame)

  if (!identical(full_rows, null_rows)) {
    stop(
      paste0(
        "Full and null models must use exactly the same retained ",
        "observations in the same order."
      ),
      call. = FALSE
    )
  }

  if (stats::nobs(full_fit) != stats::nobs(null_fit)) {
    stop(
      "Full and null models have different observation counts.",
      call. = FALSE
    )
  }

  full_response <- stats::model.response(full_frame)
  null_response <- stats::model.response(null_frame)

  if (!is.numeric(full_response) ||
      !is.numeric(null_response) ||
      !identical(
        as.numeric(full_response),
        as.numeric(null_response)
      )) {
    stop(
      "Full and null models must use the same numeric response values.",
      call. = FALSE
    )
  }

  if (!identical(
    .sa_q1_response_label(full_fit),
    .sa_q1_response_label(null_fit)
  )) {
    stop(
      "Full and null models must use the same response expression.",
      call. = FALSE
    )
  }

  X_full <- as.matrix(lme4::getME(full_fit, "X"))
  X_null <- as.matrix(lme4::getME(null_fit, "X"))

  if (!.sa_q1_same_numeric_matrix(
    X_full,
    X_null,
    tol = identity_tol
  )) {
    stop(
      "Full and null models must have the same fixed-effect design matrix.",
      call. = FALSE
    )
  }

  off_full <- .sa_q1_offset(full_fit)
  off_null <- .sa_q1_offset(null_fit)

  if (length(off_full) != length(off_null) ||
      max(abs(off_full - off_null)) >
        identity_tol *
          max(1, abs(off_full), abs(off_null))) {
    stop(
      "Full and null models must use the same offset.",
      call. = FALSE
    )
  }

  reml_full <- lme4::isREML(full_fit)
  reml_null <- lme4::isREML(null_fit)

  if (!identical(reml_full, reml_null)) {
    stop(
      paste0(
        "Full and null models use different likelihood conventions. ",
        "No hidden REML/ML conversion is permitted."
      ),
      call. = FALSE
    )
  }

  full_flist <- lme4::getME(full_fit, "flist")
  null_flist <- lme4::getME(null_fit, "flist")

  full_group_name <- names(full_flist)[1]
  null_group_name <- names(null_flist)[1]

  if (!identical(full_group_name, null_group_name)) {
    stop(
      "Full and null models must use the same grouping factor.",
      call. = FALSE
    )
  }

  full_group <- as.character(full_flist[[1]])
  null_group <- as.character(null_flist[[1]])

  if (!identical(full_group, null_group)) {
    stop(
      paste0(
        "Full and null models must have identical grouping membership ",
        "on every retained row."
      ),
      call. = FALSE
    )
  }

  full_cnms <- as.character(lme4::getME(full_fit, "cnms")[[1]])
  null_cnms <- as.character(lme4::getME(null_fit, "cnms")[[1]])

  target_index <- .sa_match_lmer_target(full_cnms, target)

  if (target %in% null_cnms) {
    stop(
      paste0(
        "The null model still contains target random-effect coefficient `",
        target,
        "`. The Question-1 null must remove it."
      ),
      call. = FALSE
    )
  }

  nuisance_names <- full_cnms[-target_index]

  if (length(nuisance_names) == 0L) {
    stop(
      paste0(
        "The lmer Question-1 interface requires at least ",
        "one nuisance random effect to remain under the null."
      ),
      call. = FALSE
    )
  }

  if (!setequal(nuisance_names, null_cnms)) {
    stop(
      paste0(
        "Full and null random-effects structures differ by more than ",
        "the target slope. The null nuisance coefficients must equal ",
        "the full-model coefficients after removing the target."
      ),
      call. = FALSE
    )
  }

  Z_full <- as.matrix(lme4::getME(full_fit, "mmList")[[1]])
  Z_null <- as.matrix(lme4::getME(null_fit, "mmList")[[1]])

  colnames(Z_full) <- full_cnms
  colnames(Z_null) <- null_cnms

  Z_full_nuisance <- Z_full[, nuisance_names, drop = FALSE]
  Z_null_reordered <- Z_null[, nuisance_names, drop = FALSE]

  if (!.sa_q1_same_numeric_matrix(
    Z_full_nuisance,
    Z_null_reordered,
    tol = identity_tol
  )) {
    stop(
      paste0(
        "Full and null models do not preserve the same nuisance ",
        "random-effect design."
      ),
      call. = FALSE
    )
  }

  weights_full <- .sa_q1_prior_weights(full_fit)
  weights_null <- .sa_q1_prior_weights(null_fit)

  if (max(abs(weights_full - weights_null)) >
      identity_tol *
        max(1, abs(weights_full), abs(weights_null))) {
    stop(
      "Full and null models must use the same residual-weight structure.",
      call. = FALSE
    )
  }

  controls_common <- NULL

  if (isTRUE(strict_controls)) {
    controls_common <- .sa_q1_common_recorded_optimizer_control(
      full_fit,
      null_fit
    )
  }

  structure(
    list(
      validated = TRUE,
      target = target,
      target_index_full = target_index,
      target_present_in_null = FALSE,
      nuisance_random_effects = nuisance_names,
      grouping_factor = full_group_name,
      n_observations = stats::nobs(full_fit),
      retained_rows = full_rows,
      likelihood_mode = if (reml_full) "REML" else "ML",
      fixed_effect_names = colnames(X_full),
      full_random_effect_names = full_cnms,
      null_random_effect_names = null_cnms,
      full_formula = paste(
        deparse(stats::formula(full_fit)),
        collapse = " "
      ),
      null_formula = paste(
        deparse(stats::formula(null_fit)),
        collapse = " "
      ),
      optimizer = paste(
        as.character(full_fit@optinfo$optimizer),
        collapse = "+"
      ),
      control_signature = if (isTRUE(strict_controls)) {
        controls_common
      } else {
        NULL
      },
      full_call_control = .sa_q1_call_control_label(full_fit),
      null_call_control = .sa_q1_call_control_label(null_fit),
      strict_controls = isTRUE(strict_controls),
      null_definition = paste0(
        "Target random slope `",
        target,
        "` is absent under H0; all covariance parameters involving ",
        "that target therefore disappear, while nuisance random effects ",
        "remain."
      )
    ),
    class = "slopeaudit_q1_model_pair"
  )
}
