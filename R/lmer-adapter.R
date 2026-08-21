# Narrow adapter from a supported Gaussian lme4::lmer fit to the
# authoritative slopeaudit matrix backend.

.sa_validate_lmer_fit <- function(fit) {
  if (!inherits(fit, "lmerMod")) {
    stop("`fit` must be a Gaussian `lme4::lmer()` fit (`lmerMod`).",
         call. = FALSE)
  }

  flist <- lme4::getME(fit, "flist")
  cnms <- lme4::getME(fit, "cnms")
  mm_list <- lme4::getME(fit, "mmList")

  if (length(flist) != 1L) {
    stop(
      paste0(
        "The first lmerMod adapter supports exactly one grouping factor. ",
        "Use the matrix backend directly for more complex models."
      ),
      call. = FALSE
    )
  }

  if (length(cnms) != 1L || length(mm_list) != 1L) {
    stop(
      paste0(
        "The first lmerMod adapter supports exactly one random-effects term. ",
        "Use the matrix backend directly for multiple random-effects terms."
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}

.sa_match_lmer_target <- function(random_effect_names, target) {
  if (!is.character(target) || length(target) != 1L ||
      is.na(target) || !nzchar(target)) {
    stop("`target` must be one non-empty random-effect coefficient name.",
         call. = FALSE)
  }

  matches <- which(random_effect_names == target)

  if (length(matches) == 0L) {
    stop(
      sprintf(
        "Target `%s` was not found. Available random-effect coefficients: %s.",
        target,
        paste(random_effect_names, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (length(matches) > 1L) {
    stop(sprintf("Target `%s` is ambiguous in the random-effects term.", target),
         call. = FALSE)
  }

  as.integer(matches)
}

.sa_lmer_model_metadata <- function(
    fit,
    grouping_name,
    target,
    G_diagnostics,
    estimand) {

  list(
    model_class = class(fit),
    formula = paste(deparse(stats::formula(fit)), collapse = " "),
    likelihood = if (lme4::isREML(fit)) "REML" else "ML",
    n_observations = stats::nobs(fit),
    grouping_factor = grouping_name,
    target_random_effect = target,
    lme4_global_singularity = lme4::isSingular(fit),
    G_diagnostics = G_diagnostics,
    estimand_identified = isTRUE(estimand$identified),
    residual_model = "homogeneous independent Gaussian",
    residual_covariance_rule =
      "unit prior weights required; R_i = sigma(fit)^2 * I",
    conditional_variance_convention = paste0(
      "Plug-in model-conditional random-effect variance at fitted parameters; ",
      "no additional fixed-effect or variance-component estimation uncertainty."
    )
  )
}

.sa_extract_lmer_matrices <- function(
    fit,
    target,
    target_metadata = NULL,
    identity_tol = .sa_identity_tol,
    rank_tol = .sa_rank_tol) {

  .sa_validate_lmer_fit(fit)

  flist <- lme4::getME(fit, "flist")
  cnms <- lme4::getME(fit, "cnms")
  mm_list <- lme4::getME(fit, "mmList")

  grouping_name <- names(flist)[1]
  group <- flist[[1]]
  if (is.null(grouping_name) || !nzchar(grouping_name)) {
    grouping_name <- "group"
  }

  random_effect_names <- as.character(cnms[[1]])
  target_index <- .sa_match_lmer_target(random_effect_names, target)

  Z_obs <- as.matrix(mm_list[[1]])
  .sa_assert_numeric_matrix(Z_obs, "random-effects model matrix")

  if (nrow(Z_obs) != length(group)) {
    stop(
      "Internal lme4 extraction mismatch between random-effects rows and grouping factor.",
      call. = FALSE
    )
  }

  if (ncol(Z_obs) != length(random_effect_names)) {
    stop(
      "Internal lme4 extraction mismatch in random-effect coefficient count.",
      call. = FALSE
    )
  }

  colnames(Z_obs) <- random_effect_names

  vc <- lme4::VarCorr(fit)
  if (length(vc) != 1L) {
    stop("Expected one random-effects covariance block from the supported lmer fit.",
         call. = FALSE)
  }

  G <- as.matrix(vc[[1]])
  G <- G[random_effect_names, random_effect_names, drop = FALSE]
  .sa_assert_psd(G, "G extracted from lmerMod", tol = identity_tol)

  G_diagnostics <- .sa_covariance_diagnostics(
    G = G,
    target_index = target_index,
    rank_tol = rank_tol,
    identity_tol = identity_tol
  )

  sigma_hat <- stats::sigma(fit)
  if (!is.numeric(sigma_hat) || length(sigma_hat) != 1L ||
      is.na(sigma_hat) || !is.finite(sigma_hat) || sigma_hat <= 0) {
    stop("The fitted lmer residual SD is not strictly positive and finite.",
         call. = FALSE)
  }
  sigma2 <- sigma_hat^2

  prior_weights <- stats::weights(fit, type = "prior")
  if (is.null(prior_weights)) {
    prior_weights <- rep(1, nrow(Z_obs))
  }
  prior_weights <- as.numeric(prior_weights)

  if (length(prior_weights) != nrow(Z_obs) ||
      any(!is.finite(prior_weights)) || any(prior_weights <= 0)) {
    stop("The fitted lmer prior weights are invalid for this adapter.",
         call. = FALSE)
  }

  if (any(abs(prior_weights - 1) > identity_tol)) {
    stop(
      paste0(
        "The first lmerMod adapter supports only unit prior weights, so ",
        "R_i = sigma(fit)^2 I. Use the matrix backend directly for ",
        "weighted residual covariance matrices."
      ),
      call. = FALSE
    )
  }

  participant_ids <- levels(group)
  if (is.null(participant_ids)) {
    participant_ids <- unique(as.character(group))
  }
  participant_ids <- as.character(participant_ids)

  Z_list <- vector("list", length(participant_ids))
  R_list <- vector("list", length(participant_ids))
  retained_row_index <- vector("list", length(participant_ids))

  group_chr <- as.character(group)

  for (i in seq_along(participant_ids)) {
    idx <- which(group_chr == participant_ids[i])

    if (length(idx) == 0L) {
      stop(sprintf("Grouping level `%s` has no retained observations.",
                   participant_ids[i]),
           call. = FALSE)
    }

    Zi <- Z_obs[idx, , drop = FALSE]
    Z_list[[i]] <- Zi
    R_list[[i]] <- diag(sigma2, nrow(Zi))
    retained_row_index[[i]] <- idx
  }

  names(Z_list) <- participant_ids
  names(R_list) <- participant_ids
  names(retained_row_index) <- participant_ids

  model_frame <- stats::model.frame(fit)
  if (nrow(model_frame) != nrow(Z_obs)) {
    stop("Internal lme4 extraction mismatch with the retained model frame.",
         call. = FALSE)
  }

  estimand <- .sa_validate_target_metadata(
    target = target,
    target_metadata = target_metadata,
    require_identified = FALSE
  )

  list(
    Z = Z_list,
    G = G,
    R = R_list,
    G_diagnostics = G_diagnostics,
    target_index = target_index,
    target_name = target,
    random_effect_names = random_effect_names,
    participant_id = participant_ids,
    retained_row_index = retained_row_index,
    retained_model_frame = model_frame,
    sigma = sigma_hat,
    sigma2 = sigma2,
    prior_weights = prior_weights,
    estimand = estimand,
    target_metadata = if (isTRUE(estimand$identified)) estimand else NULL,
    metadata = .sa_lmer_model_metadata(
      fit = fit,
      grouping_name = grouping_name,
      target = target,
      G_diagnostics = G_diagnostics,
      estimand = estimand
    )
  )
}

.sa_lmer_decompose <- function(
    fit,
    target,
    target_metadata = NULL,
    require_identified_estimand = TRUE,
    identity_tol = .sa_identity_tol,
    rank_tol = .sa_rank_tol) {

  extracted <- .sa_extract_lmer_matrices(
    fit = fit,
    target = target,
    target_metadata = target_metadata,
    identity_tol = identity_tol,
    rank_tol = rank_tol
  )

  if (require_identified_estimand) {
    .sa_require_identified_estimand(extracted$estimand)
  }

  decomposition <- .sa_matrix_decompose(
    Z = extracted$Z,
    G = extracted$G,
    R = extracted$R,
    target_index = extracted$target_index,
    participant_id = extracted$participant_id,
    identity_tol = identity_tol,
    rank_tol = rank_tol
  )

  list(
    source = "lmerMod_adapter",
    status = "ok",
    decomposition_defined = TRUE,
    matrix_backend = decomposition,
    participant_table = decomposition$participant_table,
    summary = decomposition$summary,
    diagnostics = decomposition$diagnostics,
    extracted = extracted,
    metadata = extracted$metadata,
    estimand = extracted$estimand,
    target_metadata = extracted$target_metadata
  )
}
