# Global random-effects covariance diagnostics.
#
# These diagnostics are separate from the target marginal
# variance. A singular/near-rank-one G can still have positive target variance.

.sa_covariance_diagnostics <- function(
    G,
    target_index,
    rank_tol = .sa_rank_tol,
    identity_tol = .sa_identity_tol) {

  G <- as.matrix(G)
  .sa_assert_numeric_matrix(G, "G")
  .sa_assert_psd(G, "G", tol = identity_tol)

  q <- nrow(G)
  if (ncol(G) != q) {
    stop("`G` must be square.", call. = FALSE)
  }

  target_index <- .sa_validate_target_index(target_index, q)

  eig_raw <- sort(
    eigen(.sa_symmetrize(G), symmetric = TRUE, only.values = TRUE)$values,
    decreasing = TRUE
  )

  # Tiny negative values admitted by the PSD numerical tolerance are reported
  # as zero for conditioning summaries, while the raw minimum is retained.
  eig <- pmax(eig_raw, 0)
  lambda_max <- max(eig)
  lambda_min <- min(eig)
  trace_G <- sum(diag(G))

  numerical_rank <- if (lambda_max > 0) {
    sum(eig > rank_tol * lambda_max)
  } else {
    0L
  }

  lambda_min_over_trace <- if (trace_G > 0) {
    lambda_min / trace_G
  } else {
    NA_real_
  }

  condition_number <- if (lambda_max == 0) {
    NA_real_
  } else if (lambda_min == 0) {
    Inf
  } else {
    lambda_max / lambda_min
  }

  variances <- diag(G)
  sds <- sqrt(pmax(variances, 0))
  target_sd <- sds[target_index]

  target_correlations <- rep(NA_real_, q)
  names(target_correlations) <- colnames(G)

  if (is.null(names(target_correlations))) {
    names(target_correlations) <- as.character(seq_len(q))
  }

  if (target_sd > 0) {
    ok <- sds > 0
    target_correlations[ok] <-
      G[target_index, ok] / (target_sd * sds[ok])
  }

  other <- setdiff(seq_len(q), target_index)
  target_max_abs_correlation <- if (length(other) == 0L) {
    NA_real_
  } else {
    vals <- abs(target_correlations[other])
    if (all(is.na(vals))) NA_real_ else max(vals, na.rm = TRUE)
  }

  list(
    dimension = q,
    eigenvalues = eig,
    minimum_raw_eigenvalue = min(eig_raw),
    lambda_max = lambda_max,
    lambda_min = lambda_min,
    trace = trace_G,
    lambda_min_over_trace = lambda_min_over_trace,
    condition_number = condition_number,
    numerical_rank = as.integer(numerical_rank),
    rank_tolerance = rank_tol,
    near_rank_deficient_at_rank_tol = numerical_rank < q,
    target_index = target_index,
    target_variance = G[target_index, target_index],
    target_sd = target_sd,
    target_correlations = target_correlations,
    target_max_abs_correlation = target_max_abs_correlation
  )
}
