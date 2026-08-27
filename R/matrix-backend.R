# Participant-level and sample-level matrix backend.

.sa_participant_source_decomposition <- function(
    Z,
    G,
    R,
    target_index,
    H = NULL,
    participant_id = NA_character_,
    identity_tol = .sa_identity_tol,
    rank_tol = .sa_rank_tol) {

  Z <- as.matrix(Z)
  G <- as.matrix(G)
  R <- as.matrix(R)

  .sa_assert_numeric_matrix(Z, "Z")

  n <- nrow(Z)
  q <- ncol(Z)
  if (n == 0L || q == 0L) {
    stop("`Z` must have at least one row and one column.", call. = FALSE)
  }

  target_index <- .sa_validate_target_index(target_index, q)

  .sa_assert_numeric_matrix(G, "G", nrow_ = q, ncol_ = q)
  .sa_assert_psd(G, "G", tol = identity_tol)
  .sa_assert_numeric_matrix(R, "R", nrow_ = n, ncol_ = n)
  .sa_assert_spd(R, "R")

  tau_s2 <- G[target_index, target_index]
  if (!is.finite(tau_s2)) {
    stop("The target random-effect variance is non-finite.", call. = FALSE)
  }
  if (tau_s2 == 0) {
    .sa_stop_target_boundary(
      target_slope_variance = 0,
      target_index = target_index
    )
  }
  if (tau_s2 < 0) {
    stop("The target random-effect variance is negative.", call. = FALSE)
  }

  nuisance_index <- setdiff(seq_len(q), target_index)
  H_expected <- Z[, nuisance_index, drop = FALSE]

  if (is.null(H)) {
    H <- H_expected
  } else {
    H <- as.matrix(H)
    .sa_assert_numeric_matrix(H, "H", nrow_ = n)

    if (!.sa_same_column_space(H, H_expected, tol = rank_tol)) {
      stop(
        paste0(
          "`H` must span exactly the nuisance random-effect space defined ",
          "by `Z` after removing the target column."
        ),
        call. = FALSE
      )
    }
  }

  s <- Z[, target_index]

  direct <- .sa_direct_slope_information(
    H = H,
    s = s,
    R = R,
    tau_s2 = tau_s2,
    rank_tol = rank_tol,
    identity_tol = identity_tol
  )

  full <- .sa_full_slope_determinacy(
    Z = Z,
    G = G,
    R = R,
    target_index = target_index,
    identity_tol = identity_tol
  )

  Delta_fit <- full$D_slope2 - direct$R_direct
  additivity_error <- full$D_slope2 - direct$R_direct - Delta_fit

  direct_constraint_failed <-
    direct$constraint_nuisance_failed ||
    direct$constraint_target_failed

  diagnostics <- list(
    direct_constraint_error_nuisance =
      direct$constraint_error_nuisance,
    direct_constraint_error_target =
      direct$constraint_error_target,
    direct_constraint_tolerance_nuisance =
      direct$constraint_tolerance_nuisance,
    direct_constraint_tolerance_target =
      direct$constraint_tolerance_target,
    direct_constraint_failed =
      direct_constraint_failed,
    additivity_error =
      additivity_error,
    additivity_failed =
      abs(additivity_error) > identity_tol,
    Delta_fit_materially_negative =
      Delta_fit < -identity_tol,
    determinacy_below_zero =
      full$determinacy_below_zero,
    determinacy_above_one =
      full$determinacy_above_one
  )

  diagnostics$numerically_valid <- !any(c(
    diagnostics$direct_constraint_failed,
    diagnostics$additivity_failed,
    diagnostics$Delta_fit_materially_negative,
    diagnostics$determinacy_below_zero,
    diagnostics$determinacy_above_one
  ))

  list(
    participant_id = as.character(participant_id),
    has_direct_slope_information = direct$has_direct_slope_information,
    direct_contrast = direct$contrast,
    direct_information_J = direct$direct_information_J,
    direct_error_variance = direct$direct_error_variance,
    R_direct = direct$R_direct,
    conditional_random_effect_covariance =
      full$conditional_random_effect_covariance,
    posterior_slope_variance = full$posterior_slope_variance,
    D_slope2 = full$D_slope2,
    Delta_fit = Delta_fit,
    target_slope_variance = full$target_slope_variance,
    diagnostics = diagnostics
  )
}

.sa_expand_R <- function(R, n_participants) {
  if (is.list(R)) {
    if (length(R) != n_participants) {
      stop("When `R` is a list, it must have one matrix per participant.",
           call. = FALSE)
    }
    return(R)
  }

  rep(list(as.matrix(R)), n_participants)
}

.sa_matrix_decompose <- function(
    Z,
    G,
    R,
    target_index,
    H = NULL,
    participant_id = NULL,
    identity_tol = .sa_identity_tol,
    rank_tol = .sa_rank_tol) {

  if (!is.list(Z) || length(Z) == 0L) {
    stop("`Z` must be a non-empty list of participant design matrices.",
         call. = FALSE)
  }

  G <- as.matrix(G)
  target_index <- .sa_validate_target_index(target_index, ncol(G))

  G_diagnostics <- .sa_covariance_diagnostics(
    G = G,
    target_index = target_index,
    rank_tol = rank_tol,
    identity_tol = identity_tol
  )

  if (G_diagnostics$target_variance == 0) {
    .sa_stop_target_boundary(
      target_slope_variance = 0,
      target_index = target_index
    )
  }

  n_participants <- length(Z)
  R_list <- .sa_expand_R(R, n_participants)

  if (is.null(H)) {
    H_list <- rep(list(NULL), n_participants)
  } else {
    if (!is.list(H) || length(H) != n_participants) {
      stop("`H` must be NULL or a list with one matrix per participant.",
           call. = FALSE)
    }
    H_list <- H
  }

  if (is.null(participant_id)) {
    participant_id <- as.character(seq_len(n_participants))
  }
  if (length(participant_id) != n_participants) {
    stop("`participant_id` must have one value per participant.",
         call. = FALSE)
  }

  out <- vector("list", n_participants)

  for (i in seq_len(n_participants)) {
    out[[i]] <- .sa_participant_source_decomposition(
      Z = Z[[i]],
      G = G,
      R = R_list[[i]],
      target_index = target_index,
      H = H_list[[i]],
      participant_id = participant_id[[i]],
      identity_tol = identity_tol,
      rank_tol = rank_tol
    )
  }

  participant_table <- data.frame(
    participant_id =
      vapply(out, `[[`, character(1), "participant_id"),
    has_direct_slope_information =
      vapply(out, `[[`, logical(1), "has_direct_slope_information"),
    direct_information_J =
      vapply(out, `[[`, numeric(1), "direct_information_J"),
    direct_error_variance =
      vapply(out, `[[`, numeric(1), "direct_error_variance"),
    R_direct =
      vapply(out, `[[`, numeric(1), "R_direct"),
    posterior_slope_variance =
      vapply(out, `[[`, numeric(1), "posterior_slope_variance"),
    D_slope2 =
      vapply(out, `[[`, numeric(1), "D_slope2"),
    Delta_fit =
      vapply(out, `[[`, numeric(1), "Delta_fit"),
    direct_constraint_error_nuisance =
      vapply(out, function(x) {
        x$diagnostics$direct_constraint_error_nuisance
      }, numeric(1)),
    direct_constraint_error_target =
      vapply(out, function(x) {
        x$diagnostics$direct_constraint_error_target
      }, numeric(1)),
    direct_constraint_failed =
      vapply(out, function(x) {
        x$diagnostics$direct_constraint_failed
      }, logical(1)),
    additivity_error =
      vapply(out, function(x) {
        x$diagnostics$additivity_error
      }, numeric(1)),
    additivity_failed =
      vapply(out, function(x) {
        x$diagnostics$additivity_failed
      }, logical(1)),
    Delta_fit_materially_negative =
      vapply(out, function(x) {
        x$diagnostics$Delta_fit_materially_negative
      }, logical(1)),
    determinacy_below_zero =
      vapply(out, function(x) {
        x$diagnostics$determinacy_below_zero
      }, logical(1)),
    determinacy_above_one =
      vapply(out, function(x) {
        x$diagnostics$determinacy_above_one
      }, logical(1)),
    numerically_valid =
      vapply(out, function(x) {
        x$diagnostics$numerically_valid
      }, logical(1)),
    stringsAsFactors = FALSE
  )

  summary <- .sa_aggregate_source_decomposition(
    participant_table,
    identity_tol = identity_tol
  )

  overall_status <- if (!summary$diagnostics$numerically_valid) {
    "invalid"
  } else if (G_diagnostics$near_rank_deficient_at_rank_tol) {
    "warning"
  } else {
    "ok"
  }

  list(
    participants = out,
    participant_table = participant_table,
    summary = summary,
    diagnostics = list(
      status = overall_status,
      G = G_diagnostics,
      participant_rollup = summary$diagnostics
    )
  )
}

.sa_aggregate_source_decomposition <- function(
    participant_table,
    identity_tol = .sa_identity_tol) {

  required <- c(
    "has_direct_slope_information",
    "R_direct",
    "D_slope2",
    "Delta_fit",
    "direct_constraint_error_nuisance",
    "direct_constraint_error_target",
    "direct_constraint_failed",
    "additivity_error",
    "additivity_failed",
    "Delta_fit_materially_negative",
    "determinacy_below_zero",
    "determinacy_above_one",
    "numerically_valid"
  )

  missing <- setdiff(required, names(participant_table))
  if (length(missing) > 0L) {
    stop(
      sprintf(
        "`participant_table` is missing: %s",
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  mean_R <- mean(participant_table$R_direct)
  mean_D <- mean(participant_table$D_slope2)
  mean_Delta <- mean(participant_table$Delta_fit)

  diagnostic_rollup <- list(
    n_direct_constraint_failures =
      sum(participant_table$direct_constraint_failed),
    n_additivity_failures =
      sum(participant_table$additivity_failed),
    n_material_negative_Delta =
      sum(participant_table$Delta_fit_materially_negative),
    n_determinacy_below_zero =
      sum(participant_table$determinacy_below_zero),
    n_determinacy_above_one =
      sum(participant_table$determinacy_above_one),
    max_direct_constraint_error_nuisance =
      max(participant_table$direct_constraint_error_nuisance),
    max_direct_constraint_error_target =
      max(participant_table$direct_constraint_error_target),
    max_abs_additivity_error =
      max(abs(participant_table$additivity_error)),
    numerically_valid =
      all(participant_table$numerically_valid)
  )

  list(
    n_participants = nrow(participant_table),
    n_no_direct_slope_information =
      sum(!participant_table$has_direct_slope_information),
    mean_participant_R_direct = mean_R,
    mean_participant_D_slope2 = mean_D,
    mean_participant_Delta_fit = mean_Delta,
    median_participant_R_direct =
      stats::median(participant_table$R_direct),
    median_participant_D_slope2 =
      stats::median(participant_table$D_slope2),
    median_participant_Delta_fit =
      stats::median(participant_table$Delta_fit),
    IQR_participant_R_direct =
      stats::IQR(participant_table$R_direct),
    IQR_participant_D_slope2 =
      stats::IQR(participant_table$D_slope2),
    IQR_participant_Delta_fit =
      stats::IQR(participant_table$Delta_fit),
    min_participant_R_direct =
      min(participant_table$R_direct),
    min_participant_D_slope2 =
      min(participant_table$D_slope2),
    min_participant_Delta_fit =
      min(participant_table$Delta_fit),
    aggregate_additivity_error =
      mean_D - mean_R - mean_Delta,
    any_material_negative_Delta =
      any(participant_table$Delta_fit_materially_negative),
    diagnostics = diagnostic_rollup
  )
}
